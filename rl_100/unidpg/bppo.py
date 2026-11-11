"""Slim BPPO (diffusion policy extraction) for offline RL Stage B.

Mirrors RL-100's ``unidpg/uni_ppo.py`` ``update_distribution``
(final_reward / per-step ratio path, L968-1001 — the path the old configs
actually exercised via ``chunk_as_single_action: False``) minus the
dynamics / GAE / debug stack.  Verified against the old source:
- Advantage is computed ONCE per batch on the OLD POLICY'S SAMPLED chunk
  ``old_all_next_x[-1][:, 1:9]`` (NOT the dataset action), normalized
  (adv - mean) / (std + CONST_EPS), then broadcast to every action step
  (old L989-990 pattern; slim port has no dynamics -> no per-step GAE).
- Per-step ratio: slice log-probs to the chunk ``[:, 1:9]``, sum event
  dims PER ACTION STEP -> (B, 8) ratios of Da=4 dims each.  The 32-dim
  scalar chunk sum (old L912-935, requires chunk_as_single_action: True,
  which no old yaml sets) saturates: late denoise steps have
  sigma = clip_std_min = 0.0067 -> 1/sigma^2 ~ 22,000 -> the summed
  log-prob delta explodes under microscopic UNet drift -> ratio far
  outside the clip band -> clip thrash -> random-walk destruction.
- Per-denoise-step PPO: model call on the NEW policy with cond from the OLD
  chain (encoder frozen -> identical weights in both copies); the
  ``step_forward_logprob`` is called on the NEW policy's scheduler.
  Mean over (B, n_action_steps) == old ``sum_j mean_B / opt_steps``.
- Clip linear decay frozen after ``clip_decay_until`` steps; lr linear decay
  assigned at batch end; grad-clip 0.5 over UNet params; NO KL / entropy.
- Old-policy snapshot refreshed every ``old_policy_refresh`` steps
  UNCONDITIONALLY (replaces old dynamics-OPE gating).

``cfg.bppo.diag = True`` turns on the O1 trust-region diagnostic: every step
records, per denoise step, the PPO ratio quantiles / fraction clipped / fraction
of samples receiving ZERO gradient, plus (once per batch) the Q/V/advantage
decomposition.  Recording is read-only -- it is wrapped in ``no_grad`` and never
touches a tensor the loss depends on, so ``diag=False`` and ``diag=True`` are
bit-identical policies.  Consumed by ``scripts/diag_bppo_trust_region.py``.
"""
import torch
from copy import deepcopy
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

CONST_EPS = 1e-10  # old unidpg/utils.py L6; inlined (new framework has no utils.py)


class BPPO:

    def __init__(self, policy, critic, cfg):
        # policy: DP3Policy whose noise_scheduler is ALREADY a
        # DDIMSchedulerExtended (swap happens in the train script, §4.5)
        self._policy = policy
        self._old_policy = deepcopy(policy)
        self.critic = critic
        self.cfg = cfg
        self.optimizer = Adam(self._policy.model.parameters(),
                              lr=cfg.bppo.lr)  # UNet ONLY (encoder frozen)
        self.n_obs_steps = policy.n_obs_steps
        self.n_action_steps = policy.n_action_steps
        self.action_start = self.n_obs_steps - 1
        self.action_end = self.action_start + self.n_action_steps
        # O1 diagnostic (see module docstring); default OFF.
        self.diag = bool(cfg.bppo.get('diag', False))
        self.diag_records = []
        # O2 per-denoise-step loss weighting (see `_denoise_sigma`).  'none'
        # is the shipped/official behaviour (all 10 steps averaged equally);
        # 'sigma2' down-weights step i by sigma_i^2, cancelling the 1/sigma^2
        # factor that otherwise lets the near-deterministic last DDIM step
        # (sigma=0.0067 -> 1/sigma^2=22,277) supply nearly all of the
        # gradient (O1: delta_std 28.7 at step 9 vs 0.06 at step 0).
        self.step_weight = str(cfg.bppo.get('step_weight', 'none'))
        # sigma at denoise step 0 (sigma^2 normaliser for 'sigma2' weighting),
        # and the sigma floor for 'skip_late'.
        self.step_weight_floor = float(cfg.bppo.get('step_weight_floor', 0.05))
        # sigma^2 at denoise step 0 of the CURRENT batch (set in step()).
        self._sigma0_sq = None
        # RL-100 paper Eq. (18): an action chunk is ONE decision, so the
        # per-denoise-step importance ratio sums the log-probs over the whole
        # chunk (all action steps and action dims).  The old slim port always
        # used the `chunk_as_single_action=False` per-action-step path.
        self.chunk_ratio = str(cfg.bppo.get('chunk_ratio', 'per_step'))
        # one-shot warning when a driver exceeds the configured decay horizon
        self._lr_floor_warned = False

    def _denoise_sigma(self, scheduler, t):
        """The sigma_t that ``step_forward_logprob`` actually uses at denoise
        step ``t`` (eta * sqrt(variance), then the scheduler's clip_std_min /
        clip_std_max clamp).  Read-only: only used to weight the per-step
        surrogate when ``cfg.bppo.step_weight`` is set."""
        ac = scheduler.alphas_cumprod
        num_train = int(scheduler.config.num_train_timesteps)
        prev_t = int(t) - num_train // int(scheduler.num_inference_steps)
        a_t = float(ac[int(t)])
        a_prev = (float(ac[prev_t]) if prev_t >= 0
                  else float(scheduler.final_alpha_cumprod))
        # DDIM sigma_t(eta=1): sqrt((1-a_prev)/(1-a_t) * (1 - a_t/a_prev))
        var = ((1.0 - a_prev) / (1.0 - a_t)) * (1.0 - a_t / a_prev)
        sigma = float(self.cfg.bppo.get('eta', 1.0)) * max(var, 0.0) ** 0.5
        lo = getattr(scheduler, 'clip_std_min', 0.0)
        hi = getattr(scheduler, 'clip_std_max', None)
        if lo:
            sigma = max(sigma, float(lo))
        if hi is not None:
            sigma = min(sigma, float(hi))
        return sigma

    def refresh_old_policy(self):
        self._old_policy = deepcopy(self._policy)

    def revert_to_old_policy(self):
        """Algorithm 1: a REJECTED inner update is DISCARDED -- revert the
        numerator (policy being optimized) to the last OPE-ACCEPTED behavior
        policy so the next inner window restarts from the accepted state
        (near-monotonic improvement). Keeps the Adam state on the same
        parameter objects (standard PPO revert practice)."""
        self._policy.load_state_dict(self._old_policy.state_dict())

    # ---- O1 diagnostic helpers (read-only, never touch the loss) --------
    @staticmethod
    def _dstat(x):
        x = x.detach().double().reshape(-1)
        return {'mean': float(x.mean()), 'std': float(x.std(unbiased=False))}

    @staticmethod
    def _dq(x, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
        x = x.detach().double().reshape(-1)
        return {f'q{int(q * 100):02d}': float(torch.quantile(x, q))
                for q in qs}

    @staticmethod
    def _dcorr(x, y):
        x = x.detach().double().reshape(-1)
        y = y.detach().double().reshape(-1)
        x = x - x.mean()
        y = y - y.mean()
        d = (x.std(unbiased=False) * y.std(unbiased=False))
        if float(d) == 0.0:
            return 0.0
        return float((x * y).mean() / d)

    @staticmethod
    def _dspearman(x, y):
        def rank(v):
            order = torch.argsort(torch.argsort(v))
            return order.double()
        return BPPO._dcorr(rank(x.detach().double().reshape(-1)),
                           rank(y.detach().double().reshape(-1)))

    @torch.no_grad()
    def _diag_advantage(self, k, s, chunk_actions, batch, B, adv_final):
        """Per-batch Q / V / advantage decomposition.

        The point is to test the hypothesis behind the r11-r17 plateau: that
        the *between-state* component of Q dominates, so Q-V is a poor
        advantage, while Q(s,a_pol)-Q(s,a_data) cancels it.  We therefore
        report both parameterisations plus their rank agreement, and the
        variance split var(Q) = var(V) + 2cov(V, Q-V) + var(Q-V)."""
        crit = self.critic
        q_pol = crit.minQ(s, chunk_actions).detach().float().reshape(B)
        a_data = batch['action'][:, self.action_start:self.action_end] \
            .reshape(B, -1)
        q_data = crit.minQ(
            s, crit.normalizer['action'].normalize(a_data)
        ).detach().float().reshape(B)
        v = crit._value(s).detach().float().reshape(B)
        d_rel = q_pol - q_data          # state-cancelling (our adv_mode)
        d_qv = q_pol - v                # the paper's IQL advantage
        var_v = float(v.double().var(unbiased=False))
        var_q = float(q_pol.double().var(unbiased=False))
        var_dqv = float(d_qv.double().var(unbiased=False))
        cov = float(((v.double() - v.double().mean()) *
                     (d_qv.double() - d_qv.double().mean())).mean())
        rec = {
            'kind': 'adv', 'k': int(k),
            'var_q_pol': var_q, 'var_v': var_v, 'var_d_qv': var_dqv,
            'var_d_rel': float(d_rel.double().var(unbiased=False)),
            'cov_v_d_qv': cov,
            # share of the Q spread that is pure state identity
            'between_state_frac': (var_v / var_q) if var_q > 0 else float('nan'),
            'within_state_frac': (var_dqv / var_q) if var_q > 0 else float('nan'),
            'corr_v_d_qv': self._dcorr(v, d_qv),
            'corr_v_d_rel': self._dcorr(v, d_rel),
            'corr_d_rel_d_qv': self._dcorr(d_rel, d_qv),
            'spearman_d_rel_d_qv': self._dspearman(d_rel, d_qv),
            # sign agreement: do the two advantage definitions even point the
            # same way for the SAME sample?
            'sign_agree_d_rel_d_qv': float(
                ((d_rel > 0) == (d_qv > 0)).double().mean()),
            'adv_final_std': float(adv_final.detach().double().std(unbiased=False)),
            'frac_pos_adv_final': float((adv_final > 0).float().mean()),
            'frac_pos_d_rel': float((d_rel > 0).float().mean()),
            'frac_pos_d_qv': float((d_qv > 0).float().mean()),
        }
        for nm, t in (('q_pol', q_pol), ('q_data', q_data), ('v', v),
                      ('d_rel', d_rel), ('d_qv', d_qv)):
            rec[f'{nm}_mean'] = self._dstat(t)['mean']
            rec[f'{nm}_std'] = self._dstat(t)['std']
        self.diag_records.append(rec)

    @torch.no_grad()
    def _diag_ratio(self, k, i, ratio, old_lp, new_lp, adv, clip,
                    old_full, new_full):
        """Per-denoise-step trust-region state.

        ``frac_grad_zero`` is the decisive number: the clipped-surrogate
        gradient is exactly zero for a sample iff (A>0 and ratio>1+clip) or
        (A<0 and ratio<1-clip).  If that fraction is ~1, the update is a no-op
        regardless of how the loss looks."""
        r = ratio.detach().float().reshape(-1)
        a = adv.detach().float().reshape(-1)[:, None].expand(
            -1, self.n_action_steps).reshape(-1)
        d = (new_lp - old_lp).detach().float().reshape(-1)
        d_elem = (new_full - old_full).detach().float().reshape(-1)
        zero_up = (a > 0) & (r > 1 + clip)
        zero_lo = (a < 0) & (r < 1 - clip)
        rec = {
            'kind': 'ratio', 'k': int(k), 'denoise_step': int(i),
            'clip': float(clip),
            'ratio_mean': float(r.mean()),
            'ratio_std': float(r.std(unbiased=False)),
            'frac_clipped': float(((r - 1).abs() > clip).float().mean()),
            'frac_clipped_upper': float((r > 1 + clip).float().mean()),
            'frac_clipped_lower': float((r < 1 - clip).float().mean()),
            'frac_grad_zero': float((zero_up | zero_lo).float().mean()),
            'frac_grad_zero_upper': float(zero_up.float().mean()),
            'frac_grad_zero_lower': float(zero_lo.float().mean()),
            'delta_mean': float(d.mean()),
            'delta_std': float(d.std(unbiased=False)),
            'old_lp_mean': float(old_lp.detach().float().mean()),
            'new_lp_mean': float(new_lp.detach().float().mean()),
            'old_lp_std': float(old_lp.detach().float().std(unbiased=False)),
            # MC estimate of KL(old || new) on the old policy's own samples
            'kl_mc': float(d.mean()),
            'delta_elem_std': float(d_elem.std(unbiased=False)),
            'delta_elem_absmax': float(d_elem.abs().max()),
        }
        rec.update({f'ratio_{q}': v for q, v in self._dq(r).items()})
        rec.update({f'delta_{q}': v for q, v in self._dq(d).items()})
        self.diag_records.append(rec)

    def step(self, k, batch):
        """One BPPO update on a batch. ``k`` = 1-based batch index within the
        run (drives clip / lr decay). Returns the mean per-step loss."""
        obs = batch['obs']
        # obs is either the raw schema ({'agent_pos','point_cloud'} -- the
        # policy encoder runs inside obs2feature) or the RAM feature-cache
        # schema ({'feat'} -- encoder already applied).  Derive B generically.
        B = next(iter(obs.values())).shape[0]

        # 1) old-policy sampling chain (detached log-probs) + scalar_iql
        #    advantage, computed ONCE PER BATCH on the OLD POLICY'S SAMPLED
        #    chunk (diffusion chain output is already in normalized action
        #    space -> critic.get_advantage must NOT re-normalize).
        with torch.no_grad():
            old_all_x, old_all_next_x, old_all_logprob, cond_data, cond_mask, \
                local_cond, global_cond, nobs_features = \
                self._old_policy.all_step_logprob(obs, fix_encoder=True)
            chunk_actions = old_all_next_x[-1][:, self.action_start:self.action_end]
            adv_mode = self.cfg.bppo.get('adv_mode', 'scalar')
            if adv_mode == 'relative':
                # RELATIVE advantage: Q(s, a_policy) - Q(s, a_data) at the SAME
                # state. Cancels the between-state component of Q exactly (no
                # V baseline needed), so the signal is "is the policy chunk
                # better than the demo chunk HERE". The scalar Q - V mode is
                # dominated by state identity (sphere-box within/between adv
                # std ~0.09): its batch attractor sits on the approach phase,
                # and BPPO chases it -> the policy stops grasping (step-500
                # lift=0). Relative mode reuses the same frozen critic.
                q_pol = self.critic.minQ(nobs_features, chunk_actions)
                a_data = batch['action'][:, self.action_start:self.action_end] \
                    .reshape(B, -1)
                q_data = self.critic.minQ(
                    nobs_features,
                    self.critic.normalizer['action'].normalize(a_data))
                adv = q_pol - q_data
                # The CQL-sharpened critic has huge within-state Q cliffs
                # (std ~24 at sigma=0.1); raw relative adv would be unstable.
                # Respect the same adv_normalize flag as scalar mode.
                if self.cfg.bppo.get('adv_normalize', True):
                    adv = (adv - adv.mean()) / (adv.std() + CONST_EPS)
            else:
                adv = self.critic.get_advantage(nobs_features, chunk_actions)
                # Per-batch z-score was the default (old uni_ppo pattern). It
                # is DESTRUCTIVE on state-dominated critics: the between-state
                # component of Q-V dwarfs the within-state signal, so
                # z-scoring turns the below-median phases (e.g. the approach)
                # NEGATIVE and BPPO pushes the policy AWAY from them -> the
                # policy stops approaching/grasping by step 500. With
                # adv_normalize: false the (mostly positive, omega=0.1 ->
                # mean A > 0) raw advantage acts as return-weighted BC.
                if self.cfg.bppo.get('adv_normalize', True):
                    adv = (adv - adv.mean()) / (adv.std() + CONST_EPS)
            adv = adv.reshape(B)

        # O1 diagnostic: Q / V / advantage decomposition (read-only)
        if self.diag:
            self._diag_advantage(k, nobs_features, chunk_actions, batch, B, adv)

        # 2) MANDATORY: set timesteps on the NEW policy's scheduler
        #    (step_forward_logprob raises if num_inference_steps is None;
        #     set_timesteps is pure attribute assignment, no RNG).
        scheduler = self._policy.noise_scheduler
        scheduler.set_timesteps(self._policy.num_inference_steps)

        # 3) clip linear decay, set ONCE PER BATCH; frozen after
        #    clip_decay_until steps (old launcher froze both decays at 200).
        #    Clamped at 0: when a driver runs past cfg.bppo.steps the raw
        #    linear schedule went negative, which silently flipped PPO into
        #    ascent (Stage-3 multiball collapse at update 1200 with steps=1000).
        steps = max(1, int(self.cfg.bppo.steps))
        clip = self.cfg.bppo.clip_coef * max(
            0.0, 1 - min(k, self.cfg.bppo.clip_decay_until) / steps)

        # 4) per-denoise-step PPO update (num_inference_steps optimizer steps)
        losses = []
        base_lr = self.optimizer.param_groups[0]['lr']
        for i, t in enumerate(scheduler.timesteps):
            timesteps = t
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long,
                                         device=old_all_x[i].device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(old_all_x[i].device)
            timesteps = timesteps.expand(B)
            unet_timesteps = self._policy.get_unet_timesteps(timesteps)

            model_output = self._policy.model(
                sample=old_all_x[i], timestep=unet_timesteps,
                local_cond=local_cond, global_cond=global_cond)
            log_prob = scheduler.step_forward_logprob(
                model_output, timesteps, old_all_x[i], old_all_next_x[i],
                eta=self.cfg.bppo.eta)

            # PER-STEP joint action ratio (old uni_ppo.py L968-1001 — the path
            # the old configs actually exercised via chunk_as_single_action:
            # False).  Sum event dims PER ACTION STEP -> (B, n_action_steps):
            # each ratio uses only Da=4 log-prob dims instead of the 32-dim
            # chunk sum.  The 32-dim scalar sum (old L912-935) amplifies the
            # late-step sigma^-2 (~22,000) sensitivity 8x and saturates the
            # ratio far outside the clip band after ONE batch (probe T2:
            # max ratio 13 at step 1, |dlp|~33 by step 30) -> clip thrash ->
            # random-walk destruction of the UNet.  Scalar advantage is
            # broadcast to every action step (old L989-990 pattern); the
            # mean over (B, n_action_steps) equals the old
            # sum_j mean_B(...) / opt_steps exactly.
            if self.chunk_ratio == 'scalar':
                # Paper Eq. (18) / chunk-as-single-action: one ratio per
                # denoise step, joint over the entire action chunk.
                old_lp = old_all_logprob[i][
                    :, self.action_start:self.action_end
                ].reshape(B, -1).sum(dim=1)                       # (B,)
                new_lp = log_prob[
                    :, self.action_start:self.action_end
                ].reshape(B, -1).sum(dim=1)                       # (B,)
                ratio = (new_lp - old_lp).exp()
                s1 = ratio * adv
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
                loss = -torch.min(s1, s2).mean()
            else:
                old_lp = old_all_logprob[i][:, self.action_start:self.action_end] \
                    .reshape(B, self.n_action_steps, -1).sum(dim=2)   # (B, 8)
                new_lp = log_prob[:, self.action_start:self.action_end] \
                    .reshape(B, self.n_action_steps, -1).sum(dim=2)   # (B, 8)
                ratio = (new_lp - old_lp).exp()

                s1 = ratio * adv[:, None]
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[:, None]
                loss = -torch.min(s1, s2).mean()
            # O2: per-denoise-step weighting.  NOTE this must act on the
            # OPTIMIZER STEP, not on the loss: Adam is invariant to a constant
            # rescaling of the gradient, and `clip_grad_norm_` below would
            # rescale it back anyway, so `loss * w` is a no-op.  The 10
            # denoise steps are 10 separate optimizer steps (one per iteration
            # of this loop), so the honest knob is the LR of step i.
            w = 1.0
            if self.step_weight != 'none':
                sig = self._denoise_sigma(scheduler, t)
                if i == 0:
                    self._sigma0_sq = sig * sig
                if self.step_weight == 'sigma2':
                    # cancels the 1/sigma_t^2 factor in d(log p)/d(theta)
                    w = (sig * sig) / self._sigma0_sq
                elif self.step_weight == 'skip_late':
                    w = 0.0 if sig < self.step_weight_floor else 1.0

            # O1 diagnostic: trust-region state at this denoise step
            if self.diag and self.chunk_ratio != 'scalar':
                self._diag_ratio(k, i, ratio, old_lp, new_lp, adv, clip,
                                 old_all_logprob[i], log_prob)

            # NO retain_graph: per-step graphs are independent (all_x[i]
            # detached; each step's loss only flows through model(all_x[i])).
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self._policy.model.parameters(),
                            self.cfg.bppo.grad_clip)
            if w != 1.0:
                # per-denoise-step effective step size (Adam is scale
                # invariant, so only the LR -- not the gradient scale -- can
                # express this)
                for g in self.optimizer.param_groups:
                    g['lr'] = base_lr * w
            if w > 0.0:
                self.optimizer.step()
            if w != 1.0:
                for g in self.optimizer.param_groups:
                    g['lr'] = base_lr
            losses.append((w, loss.item()))

        # 5) lr linear decay to ~0, ASSIGNED AT BATCH END (old ordering ->
        #    batch k actually trains with lr*(1-(k-1)/steps)).  Clamped at 0;
        #    see the clip note above.  The online driver now sizes
        #    cfg.bppo.steps to iters*steps_per_iter so the decay actually
        #    reaches 0 on the last update instead of going negative.
        if k > steps and not self._lr_floor_warned:
            print(f'[BPPO] k={k} > bppo.steps={steps}: lr schedule at floor 0. '
                  f'Set bppo.steps to the driver total for a full decay.',
                  flush=True)
            self._lr_floor_warned = True
        for g in self.optimizer.param_groups:
            g['lr'] = self.cfg.bppo.lr * max(0.0, 1 - k / steps)

        wsum = float(sum(w for w, _ in losses))
        if not losses:
            return 0.0
        if wsum <= 0:
            return sum(l for _, l in losses) / len(losses)
        return sum(w * l for w, l in losses) / wsum

    @torch.no_grad()
    def predict_action(self, obs_dict):
        """Delegate eval-time inference to the live policy."""
        return self._policy.predict_action(obs_dict)

    def save(self, path, bc_cfg, step, strict_sr, policy=None):
        """Payload mirrors the BC ckpt layout so build_policy_from_ckpt can
        reload it (falls back to 'model' when 'ema_model' is absent).
        ``policy`` defaults to the live numerator (self._policy); when the
        OPE gate is active, pass the accepted behavior policy
        (self._old_policy) to persist the Algorithm-1 deployable policy."""
        import dill
        p = self._policy if policy is None else policy
        payload = {
            'cfg': bc_cfg,
            'state_dicts': {
                'model': p.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            },
            'pickles': {'step': step, 'strict_sr': strict_sr},
        }
        torch.save(payload, open(path, 'wb'), pickle_module=dill)
