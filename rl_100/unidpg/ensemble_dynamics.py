"""MOPO-style ensemble dynamics in DP3 latent space -- faithful port of the
official RL-100 ``robot_gym/rl100/dynamics`` module
(ensemble_dynamics_model.py + ensemble_linear.py + the chunk-mode paths of
ensemble_dynamics_batch.py), adapted to arx5_regen conventions:

- Operates on PRECOMPUTED frozen-encoder features (128-d per frame) instead
  of raw obs dicts: no normalizer / obs encoder / point-cloud stack.
- Chunk-as-single-action mode only (matches our IQL critic: state = 2-frame
  obs window flattened (B, 256), action = flattened 8x7 chunk (B, 56),
  Q(s, a_chunk) = value of executing the whole chunk).
- 'last' prediction mode: the dynamics maps last-frame features (B, 128)
  + chunk (B, 56) -> next last-frame features (B, 128) + chunk reward.
- predict_r=True: the ensemble predicts the discounted chunk reward as an
  extra output column. Target convention (official _discounted_chunk_rewards):
  sum_k gamma^k * r_k * prior_not_done_k -- rewards after the FIRST done in
  the chunk are zeroed (the chunk stays inside one episode in our windows,
  so only the final step's done can mask).
- terminal_fn: always-False (official ``termination_fn_robot_gym`` default
  for latent-space rollouts; true terminals live in the replay buffer).
- Uncertainty penalty (aleatoric max-ensemble-std L2, official
  ``penalty_coef`` 0.5) subtracted from the predicted reward inside step(),
  the MOPO/AM-Q conservative penalty used by the OPE gate.

Kept faithful from the official code: MBPO probabilistic ensemble (7
members, 5 elites, Swish, [400x4] hidden, per-layer weight decays), the
variance-weighted NLL training loss with learnable logvar bounds
(max_logvar 0.5 / min_logvar -10, soft-clamped), per-member holdout-based
elite snapshot/rollback early stopping, one-random-elite-per-sample
stepping, and the GAE-based ``rollout`` OPE used by train.py's gate
(train.py L1230-1250: refresh the behavior policy ONLY if the dynamics
rollout's mean Q improves).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONST_EPS = 1e-10


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def soft_clamp(x, lo, hi):
    """Soft clamp (official ensemble_dynamics_model.soft_clamp): first the
    upper bound x = hi - softplus(hi - x) (= x - softplus(x - hi)), then the
    lower bound x = lo + softplus(x - lo)."""
    x = x - F.softplus(x - hi)
    x = lo + F.softplus(x - lo)
    return x


class EnsembleLinear(nn.Module):
    """Batched linear layer for the ensemble (official ensemble_linear.py).
    weight (E, in, out), bias (E, 1, out); forward accepts (B, in) (same
    input to every member) or (E, B, in). ``saved_weight``/``saved_bias``
    hold the best-so-far snapshots for elite rollback."""

    def __init__(self, in_dim, out_dim, num_ensemble, weight_decay=0.0):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.weight_decay = weight_decay
        w = torch.empty(num_ensemble, in_dim, out_dim)
        stdv = 1.0 / (2.0 * np.sqrt(in_dim))
        torch.nn.init.trunc_normal_(w, a=-2 * stdv, b=2 * stdv, std=stdv)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(num_ensemble, 1, out_dim))
        # registered buffers would be dropped by load with elite state;
        # the official code registers them as parameters -- keep that so
        # state_dict round-trips identically.
        self.saved_weight = nn.Parameter(self.weight.detach().clone(),
                                         requires_grad=False)
        self.saved_bias = nn.Parameter(self.bias.detach().clone(),
                                       requires_grad=False)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0).expand(self.weight.shape[0], -1, -1)
        assert x.ndim == 3, \
            f'EnsembleLinear got {tuple(x.shape)}'
        y = torch.einsum('ebi,eio->ebo',
                         x.reshape(x.shape[0], -1, self.in_dim), self.weight)
        return y + self.bias

    def get_decay_loss(self):
        # official: weight_decay * 0.5 * (weight ** 2).sum() (no ensemble div)
        return self.weight_decay * 0.5 * (self.weight ** 2).sum()


class EnsembleDynamicsModel(nn.Module):
    """MBPO probabilistic ensemble (official ensemble_dynamics_model.py).

    Each member: [obs_features, action_embed] -> hidden... -> (mean, logvar)
    of (next_obs_features [+ reward]). With ``use_action_embed`` the action
    is encoded by Linear(action_dim, obs_dim) + ReLU so the first hidden
    width is obs_dim + obs_dim (official behavior)."""

    def __init__(self, obs_dim, action_dim,
                 hidden_dims=(400, 400, 400, 400),
                 num_ensemble=7, num_elites=5,
                 weight_decays=(2.5e-5, 5.0e-5, 7.5e-5, 7.5e-5, 1.0e-4),
                 with_reward=True, use_action_embed=True,
                 device='cpu'):
        super().__init__()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.num_ensemble, self.num_elites = num_ensemble, num_elites
        self.with_reward = with_reward
        self.use_action_embed = use_action_embed
        self.device = device
        self.act = Swish()
        out_dim = obs_dim + (1 if with_reward else 0)

        if use_action_embed:
            self.action_encoder = nn.Sequential(
                nn.Linear(action_dim, obs_dim), nn.ReLU())
            in_dim = obs_dim + obs_dim
        else:
            self.action_encoder = None
            in_dim = obs_dim + action_dim

        self.hidden_dims = list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * (len(self.hidden_dims) + 1)
        assert len(weight_decays) == len(self.hidden_dims) + 1
        self.weight_decays = list(weight_decays)

        dims = [in_dim] + self.hidden_dims
        self.layers = nn.ModuleList([
            EnsembleLinear(dims[i], dims[i + 1], num_ensemble,
                           weight_decay=self.weight_decays[i])
            for i in range(len(dims) - 1)])
        self.output_layer = EnsembleLinear(
            self.hidden_dims[-1], 2 * out_dim, num_ensemble,
            weight_decay=self.weight_decays[-1])

        # learnable logvar bounds (official: max 0.5 / min -10)
        self.max_logvar = nn.Parameter(
            torch.full((1, out_dim), 0.5, device=device), requires_grad=True)
        self.min_logvar = nn.Parameter(
            torch.full((1, out_dim), -10.0, device=device), requires_grad=True)
        self.elites = nn.Parameter(
            torch.arange(num_elites, dtype=torch.long), requires_grad=False)
        self.to(device)

    # ------------------------------------------------------------------
    # elite machinery (official)
    # ------------------------------------------------------------------
    def set_elites(self, indexes):
        assert len(indexes) <= self.num_elites
        self.elites.data = torch.as_tensor(
            np.array(indexes), dtype=torch.long, device=self.elites.device)

    def update_save(self, indexes):
        for i in indexes:
            self.layers.__dict__  # no-op keeps parity with official lints
            for layer in list(self.layers) + [self.output_layer]:
                layer.saved_weight.data[i] = layer.weight.data[i].clone()
                layer.saved_bias.data[i] = layer.bias.data[i].clone()

    def load_save(self):
        for layer in list(self.layers) + [self.output_layer]:
            layer.weight.data = layer.saved_weight.data.clone()
            layer.bias.data = layer.saved_bias.data.clone()

    def random_elite_idxs(self, batch_size):
        return np.random.choice(self.elites.cpu().numpy(),
                                size=batch_size, replace=True)

    def encode_action(self, a):
        return self.action_encoder(a) if self.action_encoder is not None else a

    def get_decay_loss(self):
        decay = sum(l.get_decay_loss() for l in self.layers)
        return decay + self.output_layer.get_decay_loss()

    def forward(self, obs_act):
        # obs_act: (B, obs_dim + action_dim) raw features ++ chunk. The
        # action is encoded INSIDE forward (consistent single encode site --
        # functionally equivalent to the official caller-pre-encodes design).
        if obs_act.ndim == 2:
            obs_act = obs_act.unsqueeze(0).expand(self.num_ensemble, -1, -1)
        if self.action_encoder is not None:
            obs, act = obs_act[..., :self.obs_dim], obs_act[..., self.obs_dim:]
            h = torch.cat([obs, self.action_encoder(act)], dim=-1)
        else:
            h = obs_act
        for layer in self.layers:
            h = self.act(layer(h))
        out = self.output_layer(h)
        mean, logvar = out.chunk(2, dim=-1)
        # official soft_clamp(logvar, min_logvar, max_logvar)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar


class LatentChunkDynamics:
    """Chunk-mode EnsembleDynamicsBatch port on precomputed features.

    Training: ``learn(inputs, targets)`` with inputs (B, obs_dim+action_dim)
    [last-frame features ++ chunk] and targets (B, obs_dim(+1))
    [delta features (+ discounted chunk reward)]; MBPO var-weighted NLL.
    Inference: ``step(last_feat, chunk)`` -> (next_feat, reward, terminal);
    one random elite per sample, predict_delta added back, aleatoric
    penalty on the reward."""

    LOGVAR_COEF = 0.01

    def __init__(self, model, gamma=0.99, penalty_coef=0.5,
                 uncertainty_mode='aleatoric', device='cpu'):
        self.model = model
        self.gamma = gamma
        self.penalty_coef = penalty_coef
        self.uncertainty_mode = uncertainty_mode
        self.device = device
        # holdout / early-stop state (official _update_holdout_and_log)
        self.holdout_losses = [1e10] * model.num_ensemble
        self.cnt = 0

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def learn(self, inputs, targets):
        self.model.train()
        mean, logvar = self.model(inputs)
        inv_var = torch.exp(-logvar)
        mse_loss_inv = (
            (torch.pow(mean - targets, 2) * inv_var).mean(dim=(1, 2)))
        var_loss = logvar.mean(dim=(1, 2))
        loss = mse_loss_inv.sum() + var_loss.sum()
        loss = loss + self.model.get_decay_loss() \
            + self.LOGVAR_COEF * self.model.max_logvar.sum() \
            - self.LOGVAR_COEF * self.model.min_logvar.sum()
        return loss

    def optimize(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    @torch.no_grad()
    def validate(self, inputs, targets):
        self.model.eval()
        mean, _ = self.model(inputs)
        return ((mean - targets) ** 2).mean(dim=(1, 2)).tolist()

    def update_holdout(self, new_holdout_losses, max_epochs_since_update=50):
        """Official _update_holdout_and_log: snapshot any member that improved
        >1%; stop when no member improved for max_epochs_since_update."""
        for i in range(len(self.holdout_losses)):
            old, new = self.holdout_losses[i], new_holdout_losses[i]
            improvement = (old - new) / max(old, 1e-12)
            if improvement > 0.01:
                self.holdout_losses[i] = new
                self.model.update_save([i])
        if all((self.holdout_losses[i] - new_holdout_losses[i])
               / max(self.holdout_losses[i], 1e-12) <= 0.01
               for i in range(len(self.holdout_losses))):
            self.cnt += 1
        else:
            self.cnt = 0
        return self.cnt >= max_epochs_since_update

    def post_well_learned(self):
        self.model.set_elites(
            np.argsort(self.holdout_losses)[:self.model.num_elites].tolist())
        self.model.load_save()
        self.model.eval()
        return self.model.elites.tolist()

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, last_feat, chunk):
        """(B, obs_dim) last-frame features + (B, 56) chunk -> next features,
        chunk reward, terminal(False), info. One random elite per sample,
        Gaussian sample, predict_delta added back, penalty on reward."""
        B = last_feat.shape[0]
        model = self.model
        model.eval()
        obs_act = torch.cat([last_feat, chunk], dim=-1)
        mean, logvar = model(obs_act)
        # mean/logvar: (E, B, out); predict_delta on the feature part only
        mean = mean.clone()
        mean[..., :model.obs_dim] += last_feat.unsqueeze(0)
        std = torch.sqrt(torch.exp(logvar))
        samples = mean + torch.randn_like(mean) * std
        idxs = model.random_elite_idxs(B)
        samples = samples[idxs, np.arange(B)]
        next_feat = samples[..., :model.obs_dim]
        reward = samples[..., model.obs_dim:model.obs_dim + 1]
        if model.with_reward and self.penalty_coef != 0:
            if self.uncertainty_mode == 'aleatoric':
                penalty = std.norm(dim=-1).max(dim=0).values  # (B,)
            elif self.uncertainty_mode == 'pairwise-diff':
                em = mean.mean(dim=0, keepdim=True)
                penalty = (mean - em).norm(dim=-1).max(dim=0).values
            else:  # ensemble_std
                penalty = mean.std(dim=0).mean(dim=-1) * \
                    np.sqrt(model.num_ensemble)
            reward = reward - self.penalty_coef * penalty.unsqueeze(-1)
        terminal = torch.zeros(B, 1, device=reward.device)
        info = {'penalty': penalty.unsqueeze(-1) if self.penalty_coef != 0
                else None}
        return next_feat, reward, terminal, info

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path, extra=None):
        payload = {'model': self.model.state_dict(),
                   'gamma': self.gamma,
                   'penalty_coef': self.penalty_coef,
                   'uncertainty_mode': self.uncertainty_mode}
        if extra is not None:
            payload['extra'] = extra
        torch.save(payload, path)

    @classmethod
    def load(cls, path, device='cpu'):
        payload = torch.load(path, map_location=device)
        sd = payload['model']
        # num_ensemble from the layer weights; 'elites' holds num_elites only
        num_ensemble = sd['layers.0.weight'].shape[0]
        num_elites = sd['elites'].shape[0] if 'elites' in sd else 5
        # infer dims from the action encoder
        obs_dim = sd['max_logvar'].shape[-1] - 1
        action_dim = sd['action_encoder.0.weight'].shape[-1] \
            if 'action_encoder.0.weight' in sd else None
        n_hidden = sum(k.startswith('layers.') and k.endswith('.weight')
                       for k in sd)
        hidden = [sd[f'layers.{i}.weight'].shape[-1] for i in range(n_hidden)]
        model = EnsembleDynamicsModel(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dims=hidden,
            num_ensemble=num_ensemble, num_elites=num_elites,
            weight_decays=[0.0] * (len(hidden) + 1), device=device)
        model.load_state_dict(sd)
        dyn = cls(model, gamma=payload['gamma'],
                  penalty_coef=payload['penalty_coef'],
                  uncertainty_mode=payload['uncertainty_mode'],
                  device=device)
        dyn.extra = payload.get('extra')
        dyn.model.eval()
        return dyn


# ----------------------------------------------------------------------
# OPE rollout (official EnsembleDynamicsBatch.rollout, chunk mode)
# ----------------------------------------------------------------------
def sample_chunk_from_features(policy, window):
    """Sample a normalized action chunk (B, n_action_steps, Da) from the
    policy conditioned on window features (B, n_obs_steps, D). Mirrors the
    official policy.sample_action(features): global_cond from the frozen
    encoder features, empty cond_data, full denoising chain."""
    B = window.shape[0]
    window = window.to(policy.device)
    global_cond = window.reshape(B, -1)
    T, Da = policy.horizon, policy.action_dim
    cond_data = torch.zeros(size=(B, T, Da), device=policy.device,
                            dtype=policy.dtype)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    nsample = policy.conditional_sample(
        cond_data, cond_mask, local_cond=None, global_cond=global_cond)
    start = policy.n_obs_steps - 1
    return nsample[:, start:start + policy.n_action_steps]


@torch.no_grad()
def ope_rollout(policy, iql, dynamics, windows, rollout_length,
                gamma=0.99, lamda=0.95, use_gae=True):
    """Official dynamics.rollout (chunk mode) -- the AM-Q OPE used to gate
    the BPPO behavior-policy refresh (train.py L1230-1250).

    windows: (B, n_obs_steps, D) initial feature windows (dataset states).
    Each of rollout_length iterations: sample a chunk from the policy,
    evaluate iql.minQ(flat window, chunk), step the dynamics, slide the
    window. With use_gae: q_evaluation = mean over GAE advantages with
    gamma**n_action_steps per chunk (official); else mean of the Qs."""
    n_obs_steps = windows.shape[1]
    n_action_steps = policy.n_action_steps
    B = windows.shape[0]
    window = windows.to(dynamics.device)
    flat = window.reshape(B, -1)
    qs, rewards, terminals = [], [], []
    for _ in range(int(rollout_length)):
        chunk = sample_chunk_from_features(policy, window)   # (B, 8, 7)
        chunk_flat = chunk.reshape(B, -1)
        qs.append(iql.minQ(flat, chunk_flat))                 # (B, 1)
        next_feat, reward, terminal, _ = dynamics.step(
            window[:, -1, :], chunk_flat)
        # slide: drop oldest frame, append the prediction ('last' mode)
        window = torch.cat(
            [window[:, 1:, :], next_feat.unsqueeze(1)], dim=1)
        flat = window.reshape(B, -1)
        rewards.append(reward)
        terminals.append(terminal)

    if use_gae:
        final_chunk = sample_chunk_from_features(policy, window)
        final_bootstrap = iql.minQ(flat, final_chunk.reshape(B, -1))
        gamma_k = gamma ** n_action_steps
        # _return_and_gae (terminals always false here)
        deltas = [rewards[t] + gamma_k * (qs[t + 1] if t + 1 < len(qs)
                                          else final_bootstrap) - qs[t]
                  for t in range(len(qs))]
        gae = torch.zeros_like(deltas[-1])
        gae_advs = []
        for delta in reversed(deltas):
            gae = delta + gamma_k * lamda * gae
            gae_advs.insert(0, gae)
        q_evaluation = torch.stack(gae_advs).mean()
    else:
        q_evaluation = torch.stack(qs).mean()
    mean_reward = torch.stack(rewards).mean().item()
    return q_evaluation, mean_reward


@torch.no_grad()
def ope_rollout_lowdim(policy, iql, dynamics, lat_windows, ld_windows,
                       rollout_length, gamma=0.99, lamda=0.95,
                       use_gae=True):
    """Low-dim-critic variant of ope_rollout. The policy conditions on the
    LATENT window (B, n_obs_steps, D_lat) but the critic and the dynamics
    live in low-dim space (ld_windows (B, n_obs_steps, D_ld)); the dynamics
    (joint latent+lowdim output, obs_dim = D_lat + D_ld) advances BOTH
    windows each rollout step."""
    n_action_steps = policy.n_action_steps
    B = lat_windows.shape[0]
    D_lat = lat_windows.shape[-1]
    lat_win = lat_windows.to(dynamics.device)
    ld_win = ld_windows.to(dynamics.device)
    ld_flat = ld_win.reshape(B, -1)
    qs, rewards, terminals = [], [], []
    for _ in range(int(rollout_length)):
        chunk = sample_chunk_from_features(policy, lat_win)   # (B, 8, 7)
        chunk_flat = chunk.reshape(B, -1)
        qs.append(iql.minQ(ld_flat, chunk_flat))                # (B, 1)
        last_joint = torch.cat([lat_win[:, -1, :], ld_win[:, -1, :]],
                               dim=-1)
        next_joint, reward, terminal, _ = dynamics.step(last_joint,
                                                        chunk_flat)
        next_lat = next_joint[..., :D_lat]
        next_ld = next_joint[..., D_lat:]
        lat_win = torch.cat([lat_win[:, 1:, :], next_lat.unsqueeze(1)], dim=1)
        ld_win = torch.cat([ld_win[:, 1:, :], next_ld.unsqueeze(1)], dim=1)
        ld_flat = ld_win.reshape(B, -1)
        rewards.append(reward)
        terminals.append(terminal)

    if use_gae:
        final_chunk = sample_chunk_from_features(policy, lat_win)
        final_bootstrap = iql.minQ(ld_flat, final_chunk.reshape(B, -1))
        gamma_k = gamma ** n_action_steps
        deltas = [rewards[t] + gamma_k * (qs[t + 1] if t + 1 < len(qs)
                                          else final_bootstrap) - qs[t]
                  for t in range(len(qs))]
        gae = torch.zeros_like(deltas[-1])
        gae_advs = []
        for delta in reversed(deltas):
            gae = delta + gamma_k * lamda * gae
            gae_advs.insert(0, gae)
        q_evaluation = torch.stack(gae_advs).mean()
    else:
        q_evaluation = torch.stack(qs).mean()
    mean_reward = torch.stack(rewards).mean().item()
    return q_evaluation, mean_reward
