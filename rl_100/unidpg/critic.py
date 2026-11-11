"""Slim IQL critic (chunk-actions-as-single-action) for offline RL Stage A.

Mirrors RL-100's ``rl_100/unidpg/critic.py`` ``IQL_Q_V_no`` (shared frozen DP3
encoder, double-Q, expectile V) minus the dynamics / action-embed /
encoder-sharing stack. Verified against the old source:
- V loss target: min of the TARGET Q network at (s, a).
- Q loss target: r_chunk + gamma^n_action_steps * not_done * V(s')  with the
  ONLINE V (old code L825-832; there is no V_target in the old framework).
- Polyak: Q -> target_Q ONLY, every ``polyak_every`` updates (old: 2), tau 0.005.
- Update order: V loss (backward retain_graph=True) -> Q loss -> polyak.
MLP shapes mirror old ``unidpg/net.py``: depth=2 -> ONE hidden layer of
``hidden`` units (MLP receives depth-1).
"""
import torch
import torch.nn as nn

from rl_100.common.pytorch_util import dict_apply

CONST_EPS = 1e-10  # old unidpg/utils.py L6; inlined (new framework has no utils.py)


def MLP(input_dim, hidden_dim, depth, output_dim):
    """Mirror of old unidpg/net.py MLP (ReLU, no layer norm)."""
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class ValueMLP(nn.Module):
    """V(s): state_dim -> (depth-1) hidden layers -> 1."""

    def __init__(self, state_dim, hidden_dim, depth):
        super().__init__()
        self.state_dim = state_dim
        self._net = MLP(state_dim, hidden_dim, depth - 1, 1)

    def forward(self, state):
        return self._net(state.reshape(-1, self.state_dim))


class DoubleQMLP(nn.Module):
    """Q(s, a): raw concat [s, a] -> two heads; forward returns (Q1, Q2)."""

    def __init__(self, state_dim, action_dim, hidden_dim, depth):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._net1 = MLP(state_dim + action_dim, hidden_dim, depth - 1, 1)
        self._net2 = MLP(state_dim + action_dim, hidden_dim, depth - 1, 1)

    def forward(self, s, a):
        a = a.reshape(-1, self.action_dim)
        sa = torch.cat([s.reshape(-1, self.state_dim), a], dim=1)
        return self._net1(sa), self._net2(sa)

    def forward_all(self, s, a):
        """Uniform ensemble interface: stacked head outputs, shape (B, 2)."""
        q1, q2 = self.forward(s, a)
        return torch.stack([q1.squeeze(-1), q2.squeeze(-1)], dim=1)


class EnsembleQMLP(nn.Module):
    """FIX_IQL P1 escalation: N independent Q heads. ``forward_all`` stacks
    the head outputs to (B, N) so min/mean over heads is one op. min over
    independently-initialized heads is PESSIMISTIC where the data never pinned
    Q (off-manifold head disagreement) — the direction Gate A's dQ@noise
    metric demands. Used for n_ensemble > 2; n_ensemble == 2 keeps
    DoubleQMLP so existing 2-head checkpoints still load."""

    def __init__(self, state_dim, action_dim, hidden_dim, depth, n_heads):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_heads = n_heads
        self._nets = nn.ModuleList(
            MLP(state_dim + action_dim, hidden_dim, depth - 1, 1)
            for _ in range(n_heads))

    def forward_all(self, s, a):
        a = a.reshape(-1, self.action_dim)
        sa = torch.cat([s.reshape(-1, self.state_dim), a], dim=1)
        return torch.stack(
            [net(sa).squeeze(-1) for net in self._nets], dim=1)  # (B, N)


class IQL_Q_V_v2(nn.Module):
    """Clean v2 IQL critic (paper-faithful) for sphere-box Stage A.

    Fixes over the v1 slim port (IQL_Q_V_no), which mirrored RL-100's
    implementation where the V expectile target was the LAGGING target-Q
    network (target_minQ). The IQL paper fits V(s) toward the ONLINE Q at the
    DATA action (stop-grad):

      V loss:  expectile( minQ(s, a_data) - V(s) )     <- ONLINE Q, detached
      Q loss:  ( minQ-heads(s, a_data) - (r_chunk + gamma^n * not_done
               * V(s')) )^2                             <- ONLINE V, detached
      adv:     minQ(s, a) - V(s)   (min over 2 double-Q heads)

    Using the online Q (instead of the lagging target_Q) ties V to the
    CURRENT Q surface, so the expectile tracks the improving critic instead
    of a stale snapshot -- the ranking signal BPPO optimizes should follow
    the actual Q(s, a_data) mass rather than a polyak-delayed copy.

    Standard double-Q (n_ensemble=2, min over heads) -- the 8-head ensemble
    pessimism crutch from the grid search is dropped; correctness comes from
    the fixed sign (omega=0.1) + online-Q V target + clean labels.
    """

    def __init__(self,
                 obs_encoder,
                 normalizer,
                 omega=0.1,
                 gamma=0.99,
                 tau=0.005,
                 lr=1.4e-4,
                 hidden=512,
                 depth=3,
                 n_ensemble=2,
                 state_dim=128,
                 action_dim=32,
                 fix_encoder=True,
                 polyak_every=2,
                 n_obs_steps=2,
                 n_action_steps=8,
                 use_pc_color=False,
                 cql_alpha=0.0,
                 cql_scales=(0.05, 0.1, 0.2, 0.4),
                 cql_n_uniform=2,
                 device=None):
        super().__init__()
        self._device = device
        self._omega = omega
        self._gamma = gamma
        self._tau = tau
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fix_encoder = fix_encoder
        self.use_pc_color = use_pc_color
        # CQL(H)-style conservative penalty (round-3 fix for the action-blind
        # advantage): with one action per state Q(s,.) is unconstrained
        # off-data -> flat -> BPPO gets no within-state ranking signal (and
        # exp_fix1 showed an off-manifold Q BUMP, corr(A, ||a-a_data||)=+0.83).
        # The penalty logsumexp_k Q(s, a_cf_k) - Q(s, a_data) suppresses Q at
        # counterfactual actions (gaussian perturbations at the BPPO movement
        # scales + uniform-in-cube) while the TD term anchors data actions.
        # cql_alpha=0.0 reproduces the old update exactly.
        self._cql_alpha = cql_alpha
        self._cql_scales = tuple(cql_scales)
        self._cql_n_uniform = cql_n_uniform
        # chunk window: with horizon=5, pad_before=1 -> action[1:5]
        self.action_start = n_obs_steps - 1
        self.action_end = self.action_start + n_action_steps

        self.obs_encoder = obs_encoder
        self.normalizer = normalizer

        self.n_ensemble = n_ensemble
        if n_ensemble and n_ensemble > 2:
            self._Q = EnsembleQMLP(state_dim, action_dim, hidden, depth,
                                   n_ensemble)
            self._target_Q = EnsembleQMLP(state_dim, action_dim, hidden, depth,
                                          n_ensemble)
        else:
            self._Q = DoubleQMLP(state_dim, action_dim, hidden, depth)
            self._target_Q = DoubleQMLP(state_dim, action_dim, hidden, depth)
        self._target_Q.load_state_dict(self._Q.state_dict())
        self._value = ValueMLP(state_dim, hidden, depth)
        if device is not None:
            self._Q = self._Q.to(device)
            self._target_Q = self._target_Q.to(device)
            self._value = self._value.to(device)

        self._v_optimizer = torch.optim.Adam(self._value.parameters(), lr=lr)
        self._q_optimizer = torch.optim.Adam(self._Q.parameters(), lr=lr)
        self._total_update_step = 0
        self._target_update_freq = polyak_every

    def train(self, mode: bool = True):
        super().train(mode)
        if self.fix_encoder and self.obs_encoder is not None:
            self.obs_encoder.eval()
        return self

    def _encode(self, obs_dict, last=False):
        """obs_dict -> (B, state_dim) encoder features. ``last`` takes the
        LAST n_obs_steps frames (next_obs pair after the chunk)."""
        if 'feat' in obs_dict:
            # frozen-encoder feature cache (scripts/cache_features.py):
            # feat is (B, n_obs_steps, D); no normalization/encoder needed.
            _f = obs_dict['feat']
            _f = _f[:, -self.n_obs_steps:, ...] if last else _f[:, :self.n_obs_steps, ...]
            return _f.reshape(-1, self.state_dim)
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        if last:
            this_nobs = dict_apply(
                nobs, lambda x: x[:, -self.n_obs_steps:, ...].reshape(-1, *x.shape[2:]))
        else:
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        if self.fix_encoder:
            self.obs_encoder.eval()
            with torch.no_grad():
                feats = self.obs_encoder(this_nobs)
        else:
            feats = self.obs_encoder(this_nobs)
        return feats.reshape(-1, self.state_dim)

    def expectile_loss(self, loss):
        weight = torch.where(loss > 0, self._omega, (1 - self._omega))
        return weight * (loss ** 2)

    @staticmethod
    def _as_column(tensor, name):
        tensor = tensor.reshape(tensor.shape[0], -1)
        if tensor.shape[1] != 1:
            raise ValueError(f"{name} must be scalar per sample, got shape {tensor.shape}")
        return tensor

    def minQ(self, s, a):
        return self._Q.forward_all(s, a).min(dim=1, keepdim=True).values

    def update(self, batch):
        if self.fix_encoder and self.obs_encoder is not None:
            self.obs_encoder.eval()
        batch_size = batch['action'].shape[0]

        s = self._encode(batch['obs'])
        sp = self._encode(batch['next_obs'], last=True)
        a = self.normalizer['action'].normalize(
            batch['action'][:, self.action_start:self.action_end]
        ).reshape(batch_size, self.action_dim)

        # discounted chunk reward: sum_i gamma^i * r_{t+i}, i = 0..n-1
        reward_chunk = batch['reward'][:, self.action_start:self.action_end]
        reward_chunk = reward_chunk.reshape(batch_size, self.n_action_steps, -1)
        if reward_chunk.shape[-1] != 1:
            raise ValueError(
                f"chunk reward must be scalar per step, got shape {reward_chunk.shape}")
        reward_chunk = reward_chunk.squeeze(-1)
        gamma_weights = torch.pow(
            self._gamma,
            torch.arange(self.n_action_steps, device=reward_chunk.device,
                         dtype=reward_chunk.dtype))
        r = torch.sum(reward_chunk * gamma_weights, dim=-1, keepdim=True)
        not_done = self._as_column(
            batch['not_done'][:, self.action_end - 1], 'not_done')

        # 1) V loss: expectile toward the ONLINE min-Q at (s, a_data),
        #    detached (paper-faithful -- V tracks the CURRENT Q surface).
        with torch.no_grad():
            target_q = self.minQ(s, a)
        value = self._value(s)
        value_loss = self.expectile_loss(target_q - value).mean()
        self._v_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self._v_optimizer.step()

        # 2) Q loss: r_chunk + gamma^n * not_done * V(s') with the ONLINE V
        with torch.no_grad():
            self._value.eval()
            next_v = self._value(sp)
        q_target = r + not_done * (self._gamma ** self.n_action_steps) * next_v
        q_heads = self._Q.forward_all(s, a)  # (B, n_ensemble)
        q_loss = (q_heads - q_target).pow(2).sum(dim=1).mean()

        # 2b) CQL conservative term: suppress Q off-data so the advantage has
        #     within-state action gradient (see __init__ note).
        cql_loss = None
        if self._cql_alpha > 0:
            cfs = [a + sc * torch.randn_like(a) for sc in self._cql_scales]
            cfs += [torch.empty_like(a).uniform_(-1.0, 1.0)
                    for _ in range(self._cql_n_uniform)]
            # the DATA action must be inside the exp-set: without it the
            # softmax weights over counterfactuals always sum to 1, so the
            # downward pressure never vanishes and off-data Q slides to
            # -inf (round-3 v1: margin +770 cliff, residual/Q 0.315). With
            # a_data included, penalty >= 0 and self-limits once q_data is
            # the max (softmax weight on the rest vanishes).
            cfs = [a] + cfs
            K = len(cfs)
            a_cf = torch.cat(cfs, dim=0)                    # (K*B, Da)
            s_cf = s.repeat(K, 1)                           # (K*B, Ds)
            q_cf = self._Q.forward_all(s_cf, a_cf) \
                .reshape(K, batch_size, -1)                 # (K, B, H)
            # per-head logsumexp over counterfactuals minus Q at data action
            cql_loss = (torch.logsumexp(q_cf, dim=0) - q_heads).mean()
            q_loss = q_loss + self._cql_alpha * cql_loss

        self._q_optimizer.zero_grad()
        q_loss.backward()
        self._q_optimizer.step()

        # 3) polyak Q -> target_Q every polyak_every updates (kept only for
        #    checkpoint layout compatibility; v2 never reads target_Q)
        self._total_update_step += 1
        if self._total_update_step % self._target_update_freq == 0:
            for param, target_param in zip(self._Q.parameters(),
                                           self._target_Q.parameters()):
                target_param.data.copy_(
                    self._tau * param.data + (1 - self._tau) * target_param.data)

        return q_loss.detach().cpu().numpy(), value_loss.detach().cpu().numpy()

    def get_advantage(self, s, a):
        """s: (B, state_dim) encoder features; a: chunk actions ALREADY in
        normalized action space. Returns minQ(s, a) - V(s)."""
        q = self.minQ(s, a)
        v = self._value(s)
        return q - v

    def save(self, path, cfg=None):
        payload = {
            'state_dicts': {
                'Q': self._Q.state_dict(),
                'V': self._value.state_dict(),
                'target_Q': self._target_Q.state_dict(),
            },
            'cfg': cfg,
            'version': 2,
        }
        torch.save(payload, path)

    def load(self, path, device=None):
        payload = torch.load(path, map_location=device, weights_only=False)
        self._Q.load_state_dict(payload['state_dicts']['Q'])
        self._value.load_state_dict(payload['state_dicts']['V'])
        self._target_Q.load_state_dict(payload['state_dicts']['target_Q'])
        return payload.get('cfg')


class IQL_Q_V_no(nn.Module):
    """IQL with a frozen shared DP3 encoder (registered as submodule so it
    moves with .to(device) but is NEVER in any optimizer)."""

    def __init__(self,
                 obs_encoder,
                 normalizer,
                 omega=0.7,
                 gamma=0.99,
                 tau=0.005,
                 lr=1.4e-4,
                 hidden=512,
                 depth=2,
                 n_ensemble=2,
                 state_dim=128,
                 action_dim=32,
                 fix_encoder=True,
                 polyak_every=2,
                 n_obs_steps=2,
                 n_action_steps=8,
                 use_pc_color=False,
                 device=None):
        super().__init__()
        self._device = device
        self._omega = omega
        self._gamma = gamma
        self._tau = tau
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fix_encoder = fix_encoder
        self.use_pc_color = use_pc_color
        # chunk window: with horizon=9, pad_before=1 -> action[1:9]
        self.action_start = n_obs_steps - 1
        self.action_end = self.action_start + n_action_steps

        self.obs_encoder = obs_encoder
        self.normalizer = normalizer

        self.n_ensemble = n_ensemble
        if n_ensemble and n_ensemble > 2:
            # FIX_IQL P1 escalation (see FIX_IQL.md §3.1): N independent Q
            # heads; advantage = min over heads - V (pessimism off-manifold).
            self._Q = EnsembleQMLP(state_dim, action_dim, hidden, depth,
                                   n_ensemble)
            self._target_Q = EnsembleQMLP(state_dim, action_dim, hidden, depth,
                                          n_ensemble)
        else:
            self._Q = DoubleQMLP(state_dim, action_dim, hidden, depth)
            self._target_Q = DoubleQMLP(state_dim, action_dim, hidden, depth)
        self._target_Q.load_state_dict(self._Q.state_dict())
        self._value = ValueMLP(state_dim, hidden, depth)
        if device is not None:
            self._Q = self._Q.to(device)
            self._target_Q = self._target_Q.to(device)
            self._value = self._value.to(device)

        # optimizers over V / Q params ONLY (encoder frozen, never included)
        self._v_optimizer = torch.optim.Adam(self._value.parameters(), lr=lr)
        self._q_optimizer = torch.optim.Adam(self._Q.parameters(), lr=lr)
        self._total_update_step = 0
        self._target_update_freq = polyak_every

    def train(self, mode: bool = True):
        super().train(mode)
        if self.fix_encoder and self.obs_encoder is not None:
            self.obs_encoder.eval()
        return self

    def _encode(self, obs_dict, last=False):
        """obs_dict -> (B, state_dim) encoder features. ``last`` takes the
        LAST n_obs_steps frames (next_obs pair after the chunk)."""
        if 'feat' in obs_dict:
            # frozen-encoder feature cache (scripts/cache_features.py):
            # feat is (B, n_obs_steps, D); no normalization/encoder needed.
            _f = obs_dict['feat']
            _f = _f[:, -self.n_obs_steps:, ...] if last else _f[:, :self.n_obs_steps, ...]
            return _f.reshape(-1, self.state_dim)
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        if last:
            this_nobs = dict_apply(
                nobs, lambda x: x[:, -self.n_obs_steps:, ...].reshape(-1, *x.shape[2:]))
        else:
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        if self.fix_encoder:
            self.obs_encoder.eval()
            with torch.no_grad():
                feats = self.obs_encoder(this_nobs)
        else:
            feats = self.obs_encoder(this_nobs)
        return feats.reshape(-1, self.state_dim)

    def expectile_loss(self, loss):
        weight = torch.where(loss > 0, self._omega, (1 - self._omega))
        return weight * (loss ** 2)

    @staticmethod
    def _as_column(tensor, name):
        tensor = tensor.reshape(tensor.shape[0], -1)
        if tensor.shape[1] != 1:
            raise ValueError(f"{name} must be scalar per sample, got shape {tensor.shape}")
        return tensor

    def minQ(self, s, a):
        return self._Q.forward_all(s, a).min(dim=1, keepdim=True).values

    def target_minQ(self, s, a):
        return self._target_Q.forward_all(s, a).min(dim=1, keepdim=True).values

    def update(self, batch):
        if self.fix_encoder and self.obs_encoder is not None:
            self.obs_encoder.eval()
        batch_size = batch['action'].shape[0]

        s = self._encode(batch['obs'])
        sp = self._encode(batch['next_obs'], last=True)
        a = self.normalizer['action'].normalize(
            batch['action'][:, self.action_start:self.action_end]
        ).reshape(batch_size, self.action_dim)

        # discounted chunk reward: sum_i gamma^i * r_{t+i}, i = 0..7
        reward_chunk = batch['reward'][:, self.action_start:self.action_end]
        reward_chunk = reward_chunk.reshape(batch_size, self.n_action_steps, -1)
        if reward_chunk.shape[-1] != 1:
            raise ValueError(
                f"chunk reward must be scalar per step, got shape {reward_chunk.shape}")
        reward_chunk = reward_chunk.squeeze(-1)
        gamma_weights = torch.pow(
            self._gamma,
            torch.arange(self.n_action_steps, device=reward_chunk.device,
                         dtype=reward_chunk.dtype))
        r = torch.sum(reward_chunk * gamma_weights, dim=-1, keepdim=True)
        not_done = self._as_column(
            batch['not_done'][:, self.action_end - 1], 'not_done')

        # 1) V loss: expectile toward min target_Q(s, a)
        with torch.no_grad():
            self._target_Q.eval()
            target_q = self.target_minQ(s, a)
        value = self._value(s)
        value_loss = self.expectile_loss(target_q - value).mean()
        self._v_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)  # mirrors old code ordering
        self._v_optimizer.step()

        # 2) Q loss: r + gamma^n * not_done * V(s') with the ONLINE V
        with torch.no_grad():
            self._value.eval()
            next_v = self._value(sp)
        q_target = r + not_done * (self._gamma ** self.n_action_steps) * next_v
        q_heads = self._Q.forward_all(s, a)  # (B, n_ensemble)
        # sum of per-head squared TD errors, batch-averaged; for
        # n_ensemble == 2 identical to the old (q1-t)^2 + (q2-t)^2 mean
        # (q_target is (B, 1) and broadcasts against (B, N))
        q_loss = (q_heads - q_target).pow(2).sum(dim=1).mean()
        self._q_optimizer.zero_grad()
        q_loss.backward()
        self._q_optimizer.step()

        # 3) polyak Q -> target_Q every polyak_every updates
        self._total_update_step += 1
        if self._total_update_step % self._target_update_freq == 0:
            for param, target_param in zip(self._Q.parameters(),
                                           self._target_Q.parameters()):
                target_param.data.copy_(
                    self._tau * param.data + (1 - self._tau) * target_param.data)

        return q_loss.detach().cpu().numpy(), value_loss.detach().cpu().numpy()

    def get_advantage(self, s, a):
        """s: (B, state_dim) encoder features; a: (B, n_action_steps, Da) or
        (B, action_dim) chunk actions ALREADY in normalized action space (the
        diffusion chain samples in normalized space — old uni_ppo.py passes
        the sampled chunk straight through, no re-normalization)."""
        q = self.minQ(s, a)
        v = self._value(s)
        return q - v

    def save(self, path, cfg=None):
        payload = {
            'state_dicts': {
                'Q': self._Q.state_dict(),
                'V': self._value.state_dict(),
                'target_Q': self._target_Q.state_dict(),
            },
            'cfg': cfg,
        }
        torch.save(payload, path)

    def load(self, path, device=None):
        payload = torch.load(path, map_location=device, weights_only=False)
        self._Q.load_state_dict(payload['state_dicts']['Q'])
        self._value.load_state_dict(payload['state_dicts']['V'])
        self._target_Q.load_state_dict(payload['state_dicts']['target_Q'])
        return payload.get('cfg')
