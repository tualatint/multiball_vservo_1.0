"""Rebuild a bundled DP3 policy from an external checkpoint.

The bundled `checkpoints/epoch=0175-test_mean_score=-0.001.ckpt` comes from the
3D-Diffusion-Policy repo: its `payload['cfg']` is a Hydra/OmegaConf config and
the weights are under `state_dicts['model']` / `state_dicts['ema_model']`.
Because this folder's ``DP3Policy`` keeps the exact same module layout, the
state_dict loads directly -- no dependency on the external repo.
"""
import datetime
import dill
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from rl_100.policy.dp3_pc import DP3Policy

# The external ckpt's cfg uses Hydra `${eval:...}` / `${now:...}` interpolations;
# register resolvers so om_to_py can dereference them.
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver(
    "now", lambda fmt: datetime.datetime.now().strftime(fmt), replace=True)


def om_to_py(obj):
    """Recursively convert an OmegaConf/DictConfig/ListConfig to plain python."""
    if isinstance(obj, DictConfig):
        return {k: om_to_py(v) for k, v in obj.items()}
    if isinstance(obj, ListConfig):
        return [om_to_py(v) for v in obj]
    if hasattr(obj, '__iter__') and hasattr(obj, 'keys'):
        return {k: om_to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(om_to_py(v) for v in obj)
    return obj


def build_policy_from_ckpt(ckpt_path, device, use_ema=True):
    """Rebuild the DP3 policy from the checkpoint's cfg + state_dict.

    Returns (policy, cfg). ``cfg`` is a plain-dict version of the original
    Hydra config, so the caller can read n_obs_steps/n_action_steps/...
    """
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = om_to_py(payload['cfg'])

    pol_cfg = cfg['policy']
    shape_meta = cfg['task']['shape_meta']
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=pol_cfg['noise_scheduler']['num_train_timesteps'],
        beta_start=pol_cfg['noise_scheduler']['beta_start'],
        beta_end=pol_cfg['noise_scheduler']['beta_end'],
        beta_schedule=pol_cfg['noise_scheduler']['beta_schedule'],
        clip_sample=pol_cfg['noise_scheduler']['clip_sample'],
        set_alpha_to_one=pol_cfg['noise_scheduler']['set_alpha_to_one'],
        steps_offset=pol_cfg['noise_scheduler']['steps_offset'],
        prediction_type=pol_cfg['noise_scheduler']['prediction_type'],
    )

    policy = DP3Policy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=cfg['horizon'],
        n_action_steps=cfg['n_action_steps'],
        n_obs_steps=cfg['n_obs_steps'],
        num_inference_steps=pol_cfg['num_inference_steps'],
        obs_as_global_cond=cfg.get('obs_as_global_cond', True),
        diffusion_step_embed_dim=pol_cfg['diffusion_step_embed_dim'],
        down_dims=tuple(pol_cfg['down_dims']),
        kernel_size=pol_cfg['kernel_size'],
        n_groups=pol_cfg['n_groups'],
        condition_type=pol_cfg['condition_type'],
        use_down_condition=pol_cfg['use_down_condition'],
        use_mid_condition=pol_cfg['use_mid_condition'],
        use_up_condition=pol_cfg['use_up_condition'],
        encoder_output_dim=pol_cfg['encoder_output_dim'],
        pointnet_type=pol_cfg['pointnet_type'],
        use_pc_color=pol_cfg['use_pc_color'],
        pointcloud_encoder_cfg=dict(pol_cfg['pointcloud_encoder_cfg']),
    )

    state_dict_key = ('ema_model' if (use_ema and 'ema_model' in payload['state_dicts']
                                      and payload['state_dicts']['ema_model'] is not None)
                      else 'model')
    missing, unexpected = policy.load_state_dict(
        payload['state_dicts'][state_dict_key], strict=True)
    policy.eval()
    policy.to(device)
    print(f"[INFO] Loaded DP3 policy from {state_dict_key} of {ckpt_path} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return policy, cfg
