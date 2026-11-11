"""Minimal BC-only DP3 (3D-Diffusion-Policy) policy for the ARX L5 grasp task.

This is a self-contained behavioral-cloning implementation of the DP3 policy
from the 3D-Diffusion-Policy repo. It keeps the exact same module layout as the
external training run so that the bundled `epoch=0175` checkpoint
(``state_dicts['model']`` / ``state_dicts['ema_model']``) can be loaded directly
with ``torch.load(..., pickle_module=dill)`` followed by ``load_state_dict``.

Module attribute names (``obs_encoder``, ``model``, ``normalizer``,
``mask_generator``) and the top-level ``_dummy_variable`` are required for
strict state_dict compatibility.
"""
import copy
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce
from termcolor import cprint

from rl_100.common.pytorch_util import dict_apply
from rl_100.model.common.module_attr_mixin import ModuleAttrMixin
from rl_100.model.common.normalizer import LinearNormalizer
from rl_100.model.diffusion.conditional_unet1d import ConditionalUnet1D
from rl_100.model.diffusion.mask_generator import LowdimMaskGenerator
from rl_100.model.vision.pointnet_extractor import DP3Encoder


class DP3Policy(ModuleAttrMixin):
    """Diffusion BC policy conditioned on point cloud + agent state (DP3)."""

    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 num_inference_steps=None,
                 obs_as_global_cond=True,
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,
                 n_groups=8,
                 condition_type="film",
                 use_down_condition=True,
                 use_mid_condition=True,
                 use_up_condition=True,
                 encoder_output_dim=256,
                 crop_shape=None,
                 use_pc_color=False,
                 pointnet_type="pointnet",
                 pointcloud_encoder_cfg=None,
                 ss_loss_cfg=None,
                 **kwargs):
        super().__init__()

        self.condition_type = condition_type
        # Control-aware auxiliary loss: penalise the servo error predicted
        # after executing the action chunk (the "steady-state" error).
        # Config comes from training.steady_state in the yaml; empty/None
        # disables it (old checkpoints keep the exact old objective).
        self.ss_loss_cfg = dict(ss_loss_cfg or {})

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:  # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta['obs']
        # shape_meta is plain python dicts ({k: {'shape': [...], 'type': ...}}),
        # so extract each obs key's shape directly (no DictConfig semantics).
        obs_dict = {k: v['shape'] for k, v in obs_shape_meta.items()}

        obs_encoder = DP3Encoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DP3Policy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DP3Policy] pointnet_type: {self.pointnet_type}", "yellow")

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler

        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        # unknown config keys (e.g. use_point_crop) are accepted but unused
        self.kwargs = dict(kwargs)

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference ============
    def conditional_sample(self,
                           condition_data, condition_mask,
                           local_cond=None, global_cond=None,
                           generator=None,
                           **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            model_output = model(sample=trajectory,
                                 timestep=t,
                                 local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + self.obs_feature_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred,
        }
        return result

    # ========= training ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, sample_weights=None):
        # normalize input
        # sample_weights: optional (B,) per-sample weights (FIX_IQL P2 AWR —
        # weighted diffusion-BC; caller pre-normalizes to mean 1).
        # Top-level 'feat': pre-encoded frozen-encoder features (B, To*D).
        # Used by the rollout-weighted BC fine-tune, which trains on cached
        # features instead of re-reading point clouds.  Normal BC paths keep
        # passing batch['obs'] and are unchanged.
        if 'feat' in batch:
            nobs = None
        else:
            nobs = self.normalizer.normalize(batch['obs'])
            if not self.use_pc_color:
                nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        nactions = self.normalizer['action'].normalize(batch['action'])

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if 'feat' in batch:
            nobs_features = batch['feat']
            vib_kl = None
            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(
                    batch_size, self.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(batch_size, -1)
        elif self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                                   lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # VIB KL term (weighted by beta_kl inside the encoder, if enabled)
            vib_kl = getattr(self.obs_encoder, 'kl_loss', None)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            vib_kl = getattr(self.obs_encoder, 'kl_loss', None)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(sample=noisy_trajectory,
                          timestep=timesteps,
                          local_cond=local_cond,
                          global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t = self.noise_scheduler.alpha_t[timesteps]
            sigma_t = self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')  # (B, T*Da)
        if sample_weights is not None:
            # per-sample mean (matches BC's loss.mean() semantics), then
            # weighted batch mean; caller pre-normalizes weights to mean 1
            loss = (loss.mean(dim=1) * sample_weights.type(loss.dtype)).mean()
        else:
            loss = loss.mean()

        bc_loss_value = float(loss.item())
        ss_loss_value = 0.0
        ss_w = float(self.ss_loss_cfg.get('weight', 0.0))
        if ss_w > 0.0 and 'feat' not in batch:
            cfg = self.ss_loss_cfg
            n_steps = int(cfg.get('n_steps', self.n_action_steps))
            act_scale = float(cfg.get('action_scale', 0.10))
            w_xy = float(cfg.get('w_xy', 10.0))
            w_z = float(cfg.get('w_z', 4.0))
            # ``pred`` is the clean action chunk in NORMALIZED action space
            # (scheduler prediction_type == 'sample').  Undo the dataset
            # normalizer, clip exactly like the env (a in [-1, 1]), and
            # integrate the executed chunk into an EE displacement.
            a_raw = self.normalizer['action'].unnormalize(pred)
            delta = a_raw[:, :n_steps, :3].clamp(-1.0, 1.0).sum(dim=1)
            delta = delta * act_scale
            agent = batch['obs']['agent_pos']
            e_t = agent[:, -1, 9:12].to(delta.dtype)       # metres
            valid = agent[:, -1, 12].clamp(0.0, 1.0).to(delta.dtype)
            stage = agent[:, -1, 13].round().long().clamp(0, 1)
            stage_w = torch.tensor(
                cfg.get('stage_weights', [1.0, 1.0]),
                device=delta.device, dtype=delta.dtype)[stage]
            # the target (ball / pre-drop) is static over one chunk, so the
            # residual after the chunk is e_t - commanded displacement
            e_ss = e_t - delta
            per = (w_xy * (e_ss[:, 0] ** 2 + e_ss[:, 1] ** 2)
                   + w_z * e_ss[:, 2] ** 2)
            if cfg.get('hinge_to_expert', False):
                # Safe variant: only penalise a residual WORSE than the
                # expert's own residual.  A direct-to-zero ss loss on the
                # shared network dominated the BC gradient and destroyed the
                # grasp (v3/v4); this hinge is zero on the expert trajectory.
                a_exp = self.normalizer['action'].unnormalize(nactions)
                delta_exp = (a_exp[:, :n_steps, :3].clamp(-1.0, 1.0)
                             .sum(dim=1) * act_scale)
                e_exp = e_t - delta_exp
                per_exp = (w_xy * (e_exp[:, 0] ** 2 + e_exp[:, 1] ** 2)
                           + w_z * e_exp[:, 2] ** 2)
                per = torch.relu(per - per_exp)
            mask = valid * stage_w
            t_max = int(cfg.get('t_max', -1))
            if t_max >= 0:
                # x0 predictions are only reliable at low diffusion noise
                mask = mask * (timesteps <= t_max).to(delta.dtype)
            loss_ss = (per * mask).sum() / mask.sum().clamp_min(1e-6)
            loss = loss + ss_w * loss_ss
            ss_loss_value = float(loss_ss.detach())

        if vib_kl is not None:
            loss = loss + vib_kl

        loss_dict = {
            'bc_loss': bc_loss_value,
            'ss_loss': ss_loss_value,
            'kl_loss': float(vib_kl) if vib_kl is not None else 0.0,
        }
        return loss, loss_dict

    def reset(self):
        pass

    # ========= RL (offline BPPO) ============
    def get_unet_timesteps(self, timesteps):
        # DDIM timesteps are already integers; pass through.
        return timesteps

    def obs2feature(self, obs_dict, fix_encoder=False):
        """Mirror predict_action's conditioning block, optionally with a frozen
        (eval + no-grad) encoder. Returns (cond_data, cond_mask, local_cond,
        global_cond, nobs_features[B, n_obs*Do])."""
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps
        device = self.device
        dtype = self.dtype
        local_cond = None
        global_cond = None
        if 'feat' in obs_dict:
            # Pre-encoded frozen-encoder features from the RAM feature cache
            # (scripts/cache_features.py).  The cache stores exactly
            # obs_encoder(normalizer.normalize(state, point_cloud)) per frame,
            # so re-running the normalize+encoder block below would recompute
            # the identical tensor -- skip it.  feat is (B, horizon, D); the
            # policy conditions on the first n_obs_steps frames, matching the
            # x[:, :To] slice of the raw path.
            feat = obs_dict['feat']
            B = feat.shape[0]
            nobs_features = feat[:, :To, ...].reshape(B * To, -1)
        else:
            nobs = self.normalizer.normalize(obs_dict)
            if not self.use_pc_color:
                nobs['point_cloud'] = nobs['point_cloud'][..., :3]
            value = next(iter(nobs.values()))
            B = value.shape[0]
            was_training = self.obs_encoder.training
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            if fix_encoder:
                self.obs_encoder.eval()
                with torch.no_grad():
                    nobs_features = self.obs_encoder(this_nobs)
            else:
                nobs_features = self.obs_encoder(this_nobs)
            if fix_encoder and was_training:
                self.obs_encoder.train()
        if self.obs_as_global_cond:
            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(B, -1)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:  # impainting
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + self.obs_feature_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True
        return cond_data, cond_mask, local_cond, global_cond, nobs_features.reshape(B, -1)

    def all_step_logprob(self, state_dict, fix_encoder=True):
        """Full DDIM sampling chain with per-step (detached) log-probs. Requires
        noise_scheduler to be a DDIMSchedulerExtended. Returns (all_x, all_next_x,
        all_logprob, cond_data, cond_mask, local_cond, global_cond, nobs_features)."""
        cond_data, cond_mask, local_cond, global_cond, nobs_features = \
            self.obs2feature(state_dict, fix_encoder=fix_encoder)
        scheduler = self.noise_scheduler
        trajectory = torch.randn(size=cond_data.shape, dtype=cond_data.dtype,
                                 device=cond_data.device)
        scheduler.set_timesteps(self.num_inference_steps)
        all_x, all_logprob = [], []
        for t in scheduler.timesteps:
            trajectory[cond_mask] = cond_data[cond_mask]
            timesteps = t
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long,
                                         device=trajectory.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(trajectory.device)
            timesteps = timesteps.expand(trajectory.shape[0])
            unet_timesteps = self.get_unet_timesteps(timesteps)
            model_output = self.model(sample=trajectory, timestep=unet_timesteps,
                                      local_cond=local_cond, global_cond=global_cond)
            all_x.append(trajectory)
            trajectory, log_prob = scheduler.step_logprob(
                model_output, timesteps, trajectory)
            trajectory = trajectory.prev_sample
            all_logprob.append(log_prob)
        trajectory[cond_mask] = cond_data[cond_mask]
        all_x.append(trajectory)
        return all_x[:-1], all_x[1:], all_logprob, cond_data, cond_mask, \
            local_cond, global_cond, nobs_features
