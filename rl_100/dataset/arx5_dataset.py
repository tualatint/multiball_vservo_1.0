"""Minimal BC-only dataset loader for the ARX L5 grasp zarr dataset.

Self-contained: only depends on ``ReplayBuffer``/``SequenceSampler`` in this
folder (no RewardScaling, no base_dataset, no return computation). This is
deliberately the smallest thing needed to train DP3 BC on
``arx5_grasp_pc_v2_512.zarr``.
"""
from typing import Dict

import copy
import os
import numpy as np
import torch

from rl_100.common.pytorch_util import dict_apply
from rl_100.common.replay_buffer import ReplayBuffer
from rl_100.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from rl_100.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def _resolve_feature_cache(feature_cache, zarr_path):
    """Sidecar path for the frozen-encoder feature cache, or None.

    ``feature_cache`` may be None (off), True/'auto' (<zarr>.feats.zarr) or an
    explicit path.  Only single-zarr datasets are supported.
    """
    if feature_cache is None or feature_cache is False:
        return None
    if isinstance(zarr_path, (list, tuple)):
        return None
    if feature_cache is True or feature_cache == 'auto':
        return str(zarr_path).rstrip('/') + '.feats.zarr'
    return str(feature_cache)


def _inject_feature_cache(rb, fc_path):
    """Attach feat/next_feat from the sidecar into the in-RAM buffer."""
    import zarr
    g = zarr.open_group(fc_path, mode='r')
    feat = np.asarray(g['data']['feat'][:], dtype=np.float32)
    nfeat = np.asarray(g['data']['next_feat'][:], dtype=np.float32)
    n = int(rb.n_steps)
    if len(feat) != n or len(nfeat) != n:
        raise ValueError(
            f'feature cache {fc_path} has {len(feat)}/{len(nfeat)} frames but '
            f'buffer has {n}; rebuild the cache for this dataset')
    rb.data['feat'] = feat
    rb.data['next_feat'] = nfeat
    print(f'[dataset] feature cache ON: {fc_path} (feat {feat.shape}); '
          f'point clouds skipped')


def _fit_pc_with_passthrough(pc, n_extra=2):
    """Gaussian-fit the geometric/colour channels, identity for extras.

    ``pc`` is (N, P, 6 + n_extra); the trailing channels are binary target
    masks and must stay 0/1 at the policy input.
    """
    base = SingleFieldLinearNormalizer.create_fit(
        pc[..., :6], last_n_dims=1, mode='gaussian')
    scale = base.params_dict['scale'].detach().float()
    offset = base.params_dict['offset'].detach().float()
    ones = torch.ones(n_extra, dtype=torch.float32)
    zeros = torch.zeros(n_extra, dtype=torch.float32)
    scale = torch.cat([scale, ones])
    offset = torch.cat([offset, zeros])
    stats = {}
    for k, v in base.params_dict['input_stats'].items():
        v = v.detach().float()
        tail = ones if k == 'std' else (
            zeros if k in ('min', 'mean') else torch.ones(n_extra))
        stats[k] = torch.cat([v, tail])
    return SingleFieldLinearNormalizer.create_manual(scale, offset, stats)


class Arx5GraspDataset:
    """BC dataset for the ARX5 grasp task (joint-space, 512-point cloud)."""

    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 reward_field='reward',
                 hold_ratio=None,
                 succ_weight=None,
                 obs_keys=None,
                 feature_cache=None,
                 zero_mask_channels=False,
                 residual_gain=0.0,
                 action_scale=0.10,
                 ):
        self.zarr_path = zarr_path
        # Ablation: feed the semantic-mask point cloud but zero channels 6+
        # (same 1024-point cloud, no target semantics).
        self.zero_mask_channels = bool(zero_mask_channels)
        # Explicit residual servo: train the policy on
        #   a_res = a_expert - K * e / scale
        # where e = state[..., 9:12] is the D3 servo error (metres).  At
        # deployment the caller executes clip(a_res + K*e/scale).  gain=0
        # leaves the raw-action objective unchanged.
        self.residual_gain = float(residual_gain)
        self.action_scale = float(action_scale)

        # Optional frozen-encoder feature cache (scripts/cache_features.py).
        self.feature_cache_path = _resolve_feature_cache(feature_cache, zarr_path)
        self.use_feature_cache = bool(
            self.feature_cache_path
            and os.path.exists(os.path.join(self.feature_cache_path, 'data', 'feat')))

        # obs_keys=None -> the default heavy schema (point clouds included);
        # callers that only need episode_ends / sampler indices (e.g.
        # train_dynamics, whose features come from a separate encode_frames
        # pass) may pass a light key list so the ~15 GB point-cloud arrays are
        # NOT loaded a second time from the slow FUSE zarr.
        if obs_keys is None:
            obs_keys = ['state', 'action', 'next_state', 'next_action',
                        'reward', 'done', 'timeout', 'return',
                        'point_cloud', 'next_point_cloud']
        else:
            obs_keys = list(obs_keys)
        # FIX_IQL P1b: optional shaped-reward field (an ADDITIVE zarr array,
        # e.g. 'reward_shaped'; 'reward' itself is never modified).
        if reward_field not in obs_keys:
            obs_keys.append(reward_field)
        self.reward_field = reward_field
        if self.use_feature_cache:
            obs_keys = [k for k in obs_keys
                        if k not in ('point_cloud', 'next_point_cloud')]
        if isinstance(zarr_path, (list, tuple)):
            # Constant-size flywheel: assemble several member zarrs in RAM
            # (ReplayBuffer.load_members) instead of one merged blob, so no
            # round has to rewrite the accumulated dataset on the slow
            # NTFS-over-FUSE volume.
            self.replay_buffer = ReplayBuffer.load_members(
                list(zarr_path), keys=obs_keys)
        else:
            self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=obs_keys)
        if self.use_feature_cache:
            _inject_feature_cache(self.replay_buffer, self.feature_cache_path)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        # FIX_BC_ROUND2: optional success-weighted episode sampling for IL
        # re-training on a large merged dataset (e.g. the 71%-fail composed_r2
        # zarr). Round-2 collapsed because the fresh BC policy was dominated by
        # failure windows. succ_weight>1 oversamples windows belonging to
        # successful episodes (episode return at last step > 0.5) by that
        # factor while keeping all failure windows (so the policy still sees
        # recoverable states). Windows are permuted so a plain shuffled
        # dataloader sees a balanced mix every epoch.
        self.succ_weight = succ_weight
        if succ_weight is not None:
            self._apply_success_weight(succ_weight, seed)
        # FIX_IQL §5: optional hold-window down-weighting for Stage A. A
        # window is HOLD iff done=1 at its chunk-start frame (frame 1 — the
        # same classification _probe_gate_a uses): the fixed post-lift
        # plateau where not_done=0 regresses Q to a constant (§1.2).
        # hold_ratio is the TARGET hold:approach count ratio (1.0 = 1:1);
        # hold windows are randomly subsampled down to it. Train sampler
        # only — get_validation_dataset() rebuilds an unfiltered sampler.
        self.hold_ratio = hold_ratio
        if hold_ratio is not None:
            self._subsample_hold_windows(hold_ratio, seed)

    def _apply_success_weight(self, succ_weight, seed):
        idx = self.sampler.indices  # (buffer_start, buffer_end, sample_start, sample_end)
        if len(idx) == 0:
            return
        ee = self.replay_buffer.episode_ends[:]
        ret = self.replay_buffer['return'][:]
        # map each window to its episode index
        ep_idx = np.searchsorted(ee, idx[:, 0], side='right')
        # episode success: return at the episode's last step > 0.5
        ep_succ = np.array([ret[ee[i] - 1] > 0.5 for i in range(len(ee))])
        win_succ = ep_succ[ep_idx]
        n_succ = int(win_succ.sum())
        n_fail = len(idx) - n_succ
        # duplicate the success windows (floor + random remainder)
        rng = np.random.default_rng(seed=seed)
        reps = int(succ_weight)
        add = int(round((succ_weight - reps) * n_succ))
        sel = np.concatenate([
            np.repeat(np.flatnonzero(win_succ), reps),
            rng.choice(np.flatnonzero(win_succ), size=add, replace=False),
        ])
        new_idx = np.concatenate([idx, idx[sel]])
        # permute so a shuffled dataloader mixes succ/fail every epoch
        new_idx = new_idx[rng.permutation(len(new_idx))]
        self.sampler.indices = new_idx
        print(f"[dataset] success-weighted sampling: succ windows={n_succ} "
              f"fail={n_fail} (fail frac {n_fail / len(idx):.3f}) x{succ_weight} "
              f"-> total windows {len(new_idx)} "
              f"(succ frac {int(win_succ.sum()) * succ_weight / len(new_idx):.3f})")

    def _subsample_hold_windows(self, hold_ratio, seed):
        done = self.replay_buffer['done'][:]
        idx = self.sampler.indices  # (buffer_start, buffer_end, sample_start, sample_end)
        # frame 1 of the window = buffer frame buffer_start + (1 - sample_start)
        # (pad_before <= 1 gives sample_start in {0, 1}; max() also covers
        # larger pads, where frame 1 is the edge-padded first buffer frame)
        frame1 = idx[:, 0] + np.maximum(1 - idx[:, 2], 0)
        is_hold = done[frame1].astype(bool)
        n_app = int((~is_hold).sum())
        n_hold = int(is_hold.sum())
        n_keep = min(n_hold, int(round(hold_ratio * n_app)))
        keep = np.ones(len(idx), dtype=bool)
        if n_keep < n_hold:
            rng = np.random.default_rng(seed=seed)
            drop = rng.choice(np.flatnonzero(is_hold),
                              size=n_hold - n_keep, replace=False)
            keep[drop] = False
        kept_hold = int((keep & is_hold).sum())
        self.sampler.indices = idx[keep]
        print(f"[dataset] hold down-weight: approach={n_app} hold={n_hold}"
              f" -> kept approach={n_app} hold={kept_hold}"
              f" (achieved ratio {kept_hold / max(n_app, 1):.3f},"
              f" windows {len(idx)} -> {int(keep.sum())})")

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def _to_residual_action(self, action, state):
        """a_res = clip(a_expert - K * e / scale, -1, 1) on the xyz dims.

        The servo subtraction is applied in the PRE-DROP stage only
        (state[..., 13] >= 0.5).  In the grasp stage the expert deliberately
        approaches 12 cm above the ball, so subtracting the full ball-error
        feedback there makes the policy descend before centring and destroys
        the grasp (v3/v4 and the first residual run).  Stage 0 keeps the raw
        expert action; deployment adds the same stage-1-only feedback.
        """
        action = np.asarray(action, dtype=np.float32)
        if self.residual_gain <= 0.0:
            return action
        state = np.asarray(state, dtype=np.float32)
        e = state[..., 9:12]
        stage1 = (state[..., 13] >= 0.5).astype(np.float32)[..., None]
        servo = (self.residual_gain / self.action_scale) * e * stage1
        out = action.copy()
        out[..., :3] = np.clip(out[..., :3] - servo, -1.0, 1.0)
        return out

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self._to_residual_action(
                self.replay_buffer['action'], self.replay_buffer['state']),
            'agent_pos': self.replay_buffer['state'][..., :],
        }
        normalizer = LinearNormalizer()
        # actions are bounded to [-1,1], agent state has well-defined ranges
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        # Re-fit the point cloud with gaussian mode. In limits mode a few
        # scene outliers (e.g. an object knocked off the table) blow up the
        # [min,max] range and collapse the valid object-position deltas, making
        # the object invisible to the policy. Gaussian keeps the object cluster
        # distinguishable.
        #
        # Semantic-mask clouds have 8 channels: fit xyz+rgb (0-5) only and pass
        # the binary target-mask channels (6-7) through unchanged, so the
        # policy sees exact 0/1 indicators instead of standardised versions.
        for key in ['point_cloud']:
            pc = self.replay_buffer[key]
            n_ch = int(pc.shape[-1])
            if n_ch > 6:
                pc_norm = _fit_pc_with_passthrough(pc, n_extra=n_ch - 6)
            else:
                pc_norm = SingleFieldLinearNormalizer.create_fit(
                    pc, last_n_dims=1, mode='gaussian')
            normalizer[key] = pc_norm
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32)
        point_cloud = sample['point_cloud'][:,].astype(np.float32)
        if self.zero_mask_channels and point_cloud.shape[-1] > 6:
            point_cloud = point_cloud.copy()
            point_cloud[..., 6:] = 0.0
        data = {
            'obs': {
                'agent_pos': agent_pos,
                'point_cloud': point_cloud,
            },
            'action': self._to_residual_action(
                sample['action'], sample['state']),
        }
        return data

    def get_shape_info(self, n_action_steps, n_obs_steps):
        sample = self.sampler.sample_sequence(10)
        agent_pos = sample['state'][:,].astype(np.float32)
        point_cloud = sample['point_cloud'][:,].astype(np.float32)

        obs_shapes = {
            'agent_pos': (n_obs_steps,) + agent_pos.shape[1:],
            'point_cloud': (n_obs_steps,) + point_cloud.shape[1:],
        }
        shape_info = {
            'obs': obs_shapes,
            'action': (n_action_steps, sample['action'].shape[-1]),
        }
        return shape_info

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


class Arx5GraspRLDataset(Arx5GraspDataset):
    """RL emission: full-window obs/next_obs/action/reward/not_done. With
    horizon=9, pad_before=1, pad_after=0 a window at idx covers s[idx-1..idx+7]:
    obs=window[0:2]; action[1:9]=8-step chunk; next_obs from next_state/
    next_point_cloud same window, critic takes [:, -2:]; reward full window,
    critic slices [1:9]; not_done=1-done, critic takes index 8.

    need_next_point_cloud=False drops the next_point_cloud array from the
    loaded schema (and the next_obs field from the emitted sample). BPPO Stage
    B consumes ONLY ``obs`` + ``action`` (bppo.py step()), so the ~11 GB
    next_point_cloud array is dead weight there; on the NTFS-over-FUSE zarr
    (multiball AUDIT section 4b) skipping it halves the per-phase load. Any
    consumer that bootstraps through next_obs (Stage A critic, Gate A) must
    leave this True (the default)."""
    def __init__(self, *args, need_next_point_cloud=True, **kwargs):
        self.need_next_point_cloud = need_next_point_cloud
        if not need_next_point_cloud and 'obs_keys' not in kwargs:
            kwargs['obs_keys'] = ['state', 'action', 'next_state',
                                  'next_action', 'reward', 'done', 'timeout',
                                  'return', 'point_cloud']
        super().__init__(*args, **kwargs)

    def _sample_to_data(self, sample):
        reward = np.asarray(sample[self.reward_field], dtype=np.float32)
        not_done = 1.0 - np.asarray(sample['done'], dtype=np.float32)
        if reward.ndim == 1:
            reward = reward[:, None]
        if not_done.ndim == 1:
            not_done = not_done[:, None]
        # return-to-go (Monte-Carlo discounted return per window frame) is
        # already in the zarr; emit it so an AWR / weighted-BC extraction can
        # use the OUTCOME signal directly instead of a weak critic advantage.
        _ret = np.asarray(sample['return'], dtype=np.float32)
        if _ret.ndim == 1:
            _ret = _ret[:, None]
        if self.use_feature_cache:
            data = {
                'obs': {'feat': sample['feat'].astype(np.float32)},
                'action': sample['action'].astype(np.float32),
                'reward': reward,
                'not_done': not_done,
                'return': _ret,
            }
            if self.need_next_point_cloud:
                data['next_obs'] = {
                    'feat': sample['next_feat'].astype(np.float32)}
            return data
        agent_pos = sample['state'].astype(np.float32)
        point_cloud = sample['point_cloud'].astype(np.float32)
        data = {
            'obs': {'agent_pos': agent_pos, 'point_cloud': point_cloud},
            'action': sample['action'].astype(np.float32),
            'reward': reward,
            'not_done': not_done,
            'return': _ret,
        }
        if self.need_next_point_cloud:
            data['next_obs'] = {
                'agent_pos': sample['next_state'].astype(np.float32),
                'point_cloud': sample['next_point_cloud'].astype(np.float32),
            }
        return data
