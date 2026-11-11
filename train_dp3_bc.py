#!/usr/bin/env python
"""Minimal DP3 BC training for the ARX L5 grasp task.

Self-contained behavioral-cloning trainer for the bundled ARX5 grasp dataset.
Loads config from ``config/dp3_bc.yaml`` (paths relative to this repo root).

Usage (from this folder):
    python train_dp3_bc.py --config config/dp3_bc.yaml [--device cuda:0]
"""
import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import yaml
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from termcolor import cprint

from rl_100.common.pytorch_util import dict_apply
from rl_100.dataset.arx5_dataset import Arx5GraspDataset
from rl_100.model.diffusion.ema_model import EMAModel
from rl_100.policy.dp3_pc import DP3Policy

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def build_shape_meta(cfg=None):
    """Interface shapes for the DP3 policy.

    Two supported interfaces:
      * ``metaworld`` (EE-space, new): declared via an optional ``shape_meta``
        section in the config yaml -- 9-dim agent_pos (ee + left/right
        fingertip xyz), 4-dim action (dx, dy, dz, gripper), 512-point cloud.
      * ``joint`` (legacy, default): 10-dim agent_pos (6 joints + 3 ee pos +
        1 gripper), 7-dim action (6 normalized joints + 1 gripper), 512-point
        xyz cloud. Unchanged behaviour when no ``shape_meta`` section exists.
    """
    if cfg is not None and 'shape_meta' in cfg:
        sm = cfg['shape_meta']
        return {
            'obs': {
                'point_cloud': {'shape': list(sm['obs']['point_cloud']),
                                'type': 'point_cloud'},
                'agent_pos': {'shape': list(sm['obs']['agent_pos']),
                              'type': 'low_dim'},
            },
            'action': {'shape': list(sm['action'])},
        }
    # joint-space interface: 10-dim agent_pos (6 joints + 3 ee pos + 1 gripper),
    # 7-dim action (6 normalized joints + 1 gripper), 512-point xyz cloud.
    return {
        'obs': {
            'point_cloud': {'shape': [512, 3], 'type': 'point_cloud'},
            'agent_pos': {'shape': [10], 'type': 'low_dim'},
        },
        'action': {'shape': [7]},
    }


def build_scheduler(cfg, optimizer, num_training_steps):
    """Cosine LR schedule with linear warmup (matches the external DP3 run)."""
    warmup = int(cfg['training']['lr_warmup_steps'])

    def lr_lambda(step):
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = float(step - warmup) / max(1, num_training_steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Train DP3 BC for ARX5 grasp")
    parser.add_argument("--config", type=str, default=os.path.join(REPO_ROOT, "config", "dp3_bc.yaml"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--init-ckpt", type=str, default="",
                        help="optional checkpoint to warm-start from (distillation / "
                             "fine-tune). Loads its state dict INCLUDING its normalizer, "
                             "so the policy keeps the exact action scaling the data was "
                             "generated under. --init-from-ema picks ema_model instead.")
    parser.add_argument("--init-from-ema", action="store_true",
                        help="load state_dicts['ema_model'] from --init-ckpt (default: "
                             "prefer ema_model if present, else model)")
    parser.add_argument("--init-no-normalizer", action="store_true",
                        help="keep the dataset-fitted normalizer instead of the "
                             "init checkpoint's (needed when the new skill's action "
                             "range differs, e.g. lift/drop vs insert)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = args.device or cfg['training']['device']
    seed = int(cfg['training']['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ---- dataset ----
    # The hanoi collector writes independent shards (one per worker); the
    # loader accepts a list and assembles them in RAM via
    # ReplayBuffer.load_members, so no merge pass is needed.
    zp = cfg['dataset']['zarr_path']
    if isinstance(zp, (list, tuple)):
        zarr_path = [p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)
                     for p in zp]
    else:
        zarr_path = zp if os.path.isabs(zp) else os.path.join(REPO_ROOT, zp)
    ds_cfg = cfg['dataset']
    # BC only consumes state / action / point_cloud; leave next_* and reward
    # arrays on the slow FUSE disk instead of loading a second copy of the
    # (1024x8) clouds.  Configs can override with dataset.obs_keys.
    obs_keys = ds_cfg.get('obs_keys')
    if obs_keys is None:
        obs_keys = ['state', 'action', 'point_cloud', 'reward']
    rs_cfg = ds_cfg.get('residual_servo') or {}
    dataset = Arx5GraspDataset(
        zarr_path=zarr_path,
        horizon=ds_cfg['horizon'],
        pad_before=ds_cfg['pad_before'],
        pad_after=ds_cfg['pad_after'],
        seed=ds_cfg['seed'],
        val_ratio=ds_cfg['val_ratio'],
        max_train_episodes=ds_cfg['max_train_episodes'],
        succ_weight=ds_cfg.get('succ_weight'),
        obs_keys=obs_keys,
        zero_mask_channels=ds_cfg.get('zero_mask_channels', False),
        residual_gain=float(rs_cfg.get('gain', 0.0)),
        action_scale=float(rs_cfg.get('action_scale', 0.10)),
    )
    val_dataset = dataset.get_validation_dataset()
    cprint(f"[DATA] train samples: {len(dataset)}, val samples: {len(val_dataset)}", "green")

    normalizer = dataset.get_normalizer(mode='limits')

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg['dataloader']['batch_size'],
        shuffle=True, num_workers=cfg['dataloader']['num_workers'],
        persistent_workers=False, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg['dataloader']['batch_size'],
        shuffle=False, num_workers=cfg['dataloader']['num_workers'],
        persistent_workers=False, pin_memory=True)

    # ---- policy ----
    shape_meta = build_shape_meta(cfg)
    # Expose the resolved interface shapes on the saved cfg so
    # rl_100.common.ckpt_loader.build_policy_from_ckpt can rebuild this exact
    # policy at eval time (same layout as the external 3D-DP checkpoints).
    cfg.setdefault('task', {})['shape_meta'] = shape_meta
    pol_cfg = cfg['policy']
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
        obs_as_global_cond=cfg['obs_as_global_cond'],
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
        # optional control-aware auxiliary loss (training.steady_state)
        ss_loss_cfg=cfg.get('training', {}).get('steady_state'),
    )
    # NOTE: set_normalizer MUST run before .to(device). DictOfTensorMixin's
    # _load_from_state_dict REPLACES params_dict with tensors cloned on the
    # source (CPU) device, so fitting the normalizer after the device move
    # strands it on CPU and _normalize() drags every batch back to CPU
    # (same ordering as the working external 3D-Diffusion-Policy train.py).
    policy.set_normalizer(normalizer)

    # Optional warm start (distillation / fine-tune).  Do this BEFORE
    # .to(device): DictOfTensorMixin._load_from_state_dict replaces the
    # normalizer's params_dict with tensors cloned on the SOURCE (CPU) device,
    # so loading after the device move would strand the normalizer on CPU.
    if args.init_ckpt:
        ck = torch.load(os.path.abspath(args.init_ckpt), map_location="cpu")
        sd = ck.get("state_dicts", ck)
        if args.init_from_ema:
            src = sd.get("ema_model") or sd["model"]
        else:
            src = sd.get("ema_model") or sd.get("model") or sd
        src = {k: v for k, v in src.items() if not k.endswith("_dummy_variable")}
        missing, unexpected = policy.load_state_dict(src, strict=False)
        if args.init_no_normalizer:
            # The init state dict overwrote the normalizer; restore the one
            # fitted to THIS dataset (different action range).
            policy.set_normalizer(normalizer)
        cprint(f"[INIT] warm-started from {args.init_ckpt} "
               f"(missing={len(missing)}, unexpected={len(unexpected)})", "yellow")
        if missing:
            cprint(f"[INIT] missing keys (first 8): {list(missing)[:8]}", "yellow")

    policy.to(device)
    cprint(f"[MODEL] DP3 BC params: {sum(p.numel() for p in policy.parameters()) / 1e6:.2f}M", "green")

    ema_model = None
    ema = None
    if cfg['training']['use_ema']:
        ema_model = copy.deepcopy(policy)
        ema_model.to(device)
        ema_cfg = cfg['training']['ema']
        ema = EMAModel(
            model=ema_model,
            update_after_step=ema_cfg['update_after_step'],
            inv_gamma=ema_cfg['inv_gamma'],
            power=ema_cfg['power'],
            min_value=ema_cfg['min_value'],
            max_value=ema_cfg['max_value'],
        )

    # ---- optimizer / lr ----
    opt_cfg = cfg['optimizer']
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=opt_cfg['lr'],
        betas=tuple(opt_cfg['betas']),
        eps=opt_cfg['eps'],
        weight_decay=opt_cfg['weight_decay'],
    )
    # Optional epoch-step cap: when training.steps_per_epoch is set, each epoch
    # only draws that many batches (dataset may be much larger, e.g. the
    # success-weighted merged zarr), so total optimizer steps =
    # steps_per_epoch * num_epochs stays on the intended budget (round-1's
    # ~14.5k) instead of sweeping the whole 247k-window dataset per epoch.
    steps_per_epoch = max(1, len(dataloader))
    cap_steps = cfg['training'].get('steps_per_epoch')
    if cap_steps is not None:
        steps_per_epoch = min(int(cap_steps), steps_per_epoch)
    num_training_steps = steps_per_epoch * int(cfg['training']['num_epochs'])
    lr_scheduler = build_scheduler(cfg, optimizer, num_training_steps)

    # ---- train ----
    ckpt_dir = os.path.join(REPO_ROOT, cfg['output']['ckpt_dir'])
    os.makedirs(ckpt_dir, exist_ok=True)
    global_step = 0
    best_val_loss = float('inf')
    use_ema = cfg['training']['use_ema']

    for epoch in range(int(cfg['training']['num_epochs'])):
        policy.train()
        t0 = time.time()
        total_loss = 0.0
        total_bc = 0.0
        total_ss = 0.0
        n_batches = 0
        for batch in dataloader:
            batch = dict_apply(batch, lambda x: x.to(device))
            loss, loss_dict = policy.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if ema is not None:
                ema.step(policy)

            total_loss += loss.item()
            total_bc += float(loss_dict.get('bc_loss', 0.0))
            total_ss += float(loss_dict.get('ss_loss', 0.0))
            n_batches += 1
            global_step += 1

            if n_batches >= steps_per_epoch:
                break

            if global_step % int(cfg['training']['log_every']) == 0:
                lr_now = optimizer.param_groups[0]['lr']
                print(f"[train] step {global_step}  loss {loss.item():.4f}  "
                      f"bc {loss_dict.get('bc_loss', 0.0):.4f}  "
                      f"ss {loss_dict.get('ss_loss', 0.0):.4f}  "
                      f"lr {lr_now:.2e}", flush=True)

        avg_loss = total_loss / max(1, n_batches)
        avg_bc = total_bc / max(1, n_batches)
        avg_ss = total_ss / max(1, n_batches)

        # validation (EMA model)
        val_loss = None
        if use_ema:
            ema_model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    batch = dict_apply(batch, lambda x: x.to(device))
                    vloss, _ = ema_model.compute_loss(batch)
                    val_losses.append(vloss.item())
            val_loss = float(np.mean(val_losses))

        msg = (f"[epoch {epoch + 1}/{cfg['training']['num_epochs']}] "
               f"train_loss {avg_loss:.4f}  bc {avg_bc:.4f}  ss {avg_ss:.4f}")
        if val_loss is not None:
            msg += f"  val_loss {val_loss:.4f}"
        msg += f"  ({time.time() - t0:.1f}s)"
        print(msg, flush=True)

        # save checkpoints
        if (epoch + 1) % int(cfg['training']['checkpoint_every']) == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch={epoch + 1:04d}-train_loss={avg_loss:.3f}.ckpt")
            payload = {
                'cfg': cfg,
                'state_dicts': {
                    'model': policy.state_dict(),
                    'ema_model': ema_model.state_dict() if ema_model is not None else None,
                    'optimizer': optimizer.state_dict(),
                },
                'pickles': {'global_step': global_step, 'epoch': epoch},
            }
            torch.save(payload, ckpt_path)
            cprint(f"[save] {ckpt_path}", "green")
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(ckpt_dir, "best.ckpt")
                torch.save(payload, best_path)
                cprint(f"[save] best val -> {best_path}", "green")

    cprint(f"[done] training finished. checkpoints in {os.path.abspath(ckpt_dir)}", "green")


if __name__ == "__main__":
    main()
