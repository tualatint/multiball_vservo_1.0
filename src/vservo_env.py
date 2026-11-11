"""VservoMultiballEnv -- multiball sphere-in-box with two virtual targets.

Port of the ``hanoi_vservo_2026`` ghost/virtual-target method to the multiball
pick-and-place task.  Three things are added on top of the read-only
``SphereBoxMultiballEnv``; all three use the same depth+RGB camera that
produces the policy's point cloud:

1. **Virtual target points** appended to the cloud (7th channel = ghost flag):
   * stage 0 ("grasp"): a magenta halo at the GREEN ball position;
   * stage 1 ("pre-drop"): a magenta ball-sized shell at the ball-centre pose
     directly above the box opening, at the expert carry height.
2. **Servo error** appended to ``agent_pos``:
   ``[ex, ey, ez, valid, stage]`` where the error is
   ``target_ball_pose - (finger_site_pose - [0, 0, EE_TO_BALL_Z])``.
   The finger site sits ~11 mm above a held ball's centre, so subtracting it
   makes both stages object-centric (where the ball is / should be).
3. **Dense virtual-target reward**: a potential on the same error, with a
   two-phase shape (centre laterally before descending in the grasp stage;
   lift clear of the rim before translating in the pre-drop stage) and a
   one-time centred bonus per stage.

The stage switches when the vision-estimated ball rises ``LIFT_EPS`` above
its rest height (or is forced by the demo collector at the expert lift
transition).  Simulator state is used only for the episode-level task events
(``in_box`` / ``is_lifted`` for the info dict and the one-time grasp bonus),
never for the servo signal, matching the Hanoi implementation.

The decoder/estimator lives in ``vision_pose.py``.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from gym import spaces

from rl_100.env.arx5.arx5_env import _fps_downsample
from multiball_env import SphereBoxMultiballEnv

from vision_pose import (
    BALL_RADIUS, BOX_RIM_Z, EE_TO_BALL_Z, TABLE_SURFACE_Z,
    estimate_box, estimate_green_ball, fixture_box, sample_sphere_shell)

GHOST_G = 64
GHOST_RGB = np.array([255.0, 0.0, 255.0], dtype=np.float32)   # magenta
# stage-0 halo is 1 cm larger than the real ball so it stays visible around it
BALL_HALO_RADIUS = BALL_RADIUS + 0.010


class VservoMultiballEnv(SphereBoxMultiballEnv):
    """Multiball env + two vision-only virtual targets, servo error, reward."""

    N_GHOST = GHOST_G
    N_ERR = 3
    N_VALID = 1
    N_STAGE = 1

    # ---- reward weights (metres) ----
    W_XY = 10.0
    W_Z = 4.0
    SAFE_XY_TOL = 0.004          # lateral tolerance before z progress counts
    SOFT_XY = 0.015              # lateral width of the smooth low-carry penalty
    SOFT_Z = 0.030               # height width of the smooth low-carry penalty
    CENTER_XY_TOL = 0.006
    CENTER_Z_TOL = 0.010
    CENTER_BONUS = 0.20
    DESCENT_PENALTY = 0.010
    LOW_CARRY_PENALTY = 0.010
    APPROACH_GUARD = 0.030       # do not descend below ball+3 cm off-centre
    SAFE_CARRY_CLEARANCE = 0.040  # ball must clear rim+4 cm before translating

    # ---- stage switch ----
    LIFT_EPS = 0.030             # ball rise above its rest height -> stage 1
    LIFT_CONFIRM_FRAMES = 2

    # ---- optional deployment grasp guard (training data is untouched) ----
    # The BC policy samples a stochastic gripper action and can (a) descend
    # below the grasp height and hover with the gripper open, or (b) close,
    # lift 2-3 cm, then re-open before the vision stage switch.  The guard is
    # a gripper phase machine + a descent clamp: open until the grasp window,
    # close and latch through the carry, then open at the pre-drop window.
    # Tight grasp window: close only near the ball equator.  Closing at the
    # old 15 mm z window pinched the ball above/below the equator, where the
    # cross-section is narrower, and the ball popped out during the lift.
    GRASP_XY_TOL = 0.010
    GRASP_Z_TOL = 0.006
    GRASP_Z_CLAMP_XY = 0.025
    # Lateral servo while approaching the grasp (xy only; the z stays with
    # the policy + clamp).  This centres the gripper on the ball so the
    # tighter window is reachable even when the policy stalls slightly off.
    GRASP_LATERAL_GAIN = 0.20
    GRASP_Z_GAIN = 0.15
    GRASP_CENTER_RANGE = 0.040
    # If the ball is not lifted within this many steps after the close, the
    # grasp is not firm -> open and re-approach in the same episode.
    GRASP_TIMEOUT_STEPS = 20
    # Close the pads gradually while holding the arm still.  Snapping the
    # gripper shut at the equator can squeeze an off-centre ball out with a
    # 2+ m/s squirt while the lift starts (the "ball flies out during lift").
    GRASP_RAMP_STEPS = int(os.environ.get("VSERVO_GRASP_RAMP", "4"))
    # After the ramp, keep the arm still for a further settle so the pads
    # actually reach the ball and the grip force builds BEFORE the lift
    # starts.  With only the ramp, the stage-1 lift began while the pads were
    # still ~1.4 mm from the ball; the fast upward move then rammed the pad
    # edge through the ball against the table and the solver launched it.
    GRASP_SETTLE_STEPS = int(os.environ.get("VSERVO_GRASP_SETTLE", "4"))
    # Close target.  Full close (a3=+1, the expert's action) is needed: the
    # calibrated ~5 N soft close still let the ball roll out of the pinch.
    # The 40 N squeeze only became unstable when the lift started before the
    # pads touched the ball (see GRASP_SETTLE_STEPS); with the settle hold
    # the ball is already gripped when the lift begins.  Env override for
    # ablation.
    GRASP_CLOSE_CMD = float(os.environ.get("VSERVO_CLOSE_CMD", "1.0"))
    # Do not command a downward step while the open gripper is laterally
    # farther than this from the ball: the descending open finger pads can
    # bump the ball off the table before the grasp window is reached.
    # 0 disables the gate (the validated guard5 behaviour).
    GRASP_DESCENT_XY_TOL = float(
        os.environ.get("VSERVO_DESCENT_XY_TOL", "0.0"))
    # The re-approach after an in-episode recovery starts from a pose the
    # stage-0 policy did not see in the demos (the ball has rolled to a new
    # spot), so the open gripper can descend off-centre and knock the ball
    # away.  Use the descent gate only for recovery approaches.
    RECOVERY_DESCENT_XY_TOL = float(
        os.environ.get("VSERVO_RECOVERY_DESCENT_XY_TOL", "0.015"))
    # Carry speed cap: the pinched ball is only held by the 40 mm pad faces
    # (the ball is 70 mm), so a fast lateral acceleration lets it roll/slip
    # along the pad and be squeezed out at the edge.  Scale the stage-1 xyz
    # command (1.0 = policy speed, the validated guard5 behaviour).
    CARRY_ACTION_SCALE = float(os.environ.get("VSERVO_CARRY_SCALE", "1.0"))
    # After an in-episode re-grasp the carry is slower: a recovery has
    # already shown that this episode's grasp is marginal, and the second
    # carry is the one that used to drop the ball outside the re-grasp band.
    RECOVERY_CARRY_SCALE = float(
        os.environ.get("VSERVO_RECOVERY_CARRY_SCALE", "0.5"))
    # Test lift: after the settle, verify that the ball actually rises with
    # the hand.  A marginal pinch can let the ball slip out during the first
    # centimetres of the lift and then fall while the (occluded) vision
    # estimate is invalid, so the drop is only detected after the arm has
    # already reached the pre-drop pose.  Checking the vision ball rise
    # against the EE rise during the first steps catches the slip while the
    # ball is still on the table and a clean re-grasp is possible.
    GRIP_CHECK_STEPS = int(os.environ.get("VSERVO_GRIP_CHECK_STEPS", "8"))
    GRIP_CHECK_RISE_FRAC = float(
        os.environ.get("VSERVO_GRIP_CHECK_RISE_FRAC", "0.3"))
    GRIP_CHECK_RISE_SLACK = 0.004
    # After a re-grasp is triggered, first raise the open gripper to a
    # stand-off above the (fallen) ball with a scripted visual servo, then
    # hand control back to the stage-0 policy.  Starting the re-approach
    # directly from the old grasp pose is out of the policy's demo
    # distribution and can knock the ball away.
    RECOVER_STANDOFF = float(os.environ.get("VSERVO_RECOVER_STANDOFF", "0.14"))
    RECOVER_GAIN = float(os.environ.get("VSERVO_RECOVER_GAIN", "0.5"))
    RECOVER_XY_TOL = 0.010
    RECOVER_Z_TOL = 0.012
    DROP_XY_TOL = 0.015
    DROP_Z_TOL = 0.020
    # release settle: near the pre-drop pose, damp the lateral/vertical
    # motion and open only when the post-settle lateral command is small, so
    # the ball leaves the gripper with little horizontal velocity instead of
    # hitting the box wall and being ejected by the stiff contact solver
    DROP_SETTLE_XY = 0.040
    DROP_SETTLE_Z = 0.030
    DROP_SETTLE_LATERAL = 0.20
    DROP_SETTLE_MAX_Z = 0.05
    DROP_LATERAL_SPEED_TOL = 0.08
    # once the release window is reached, hold the arm still for a couple of
    # steps before opening, so the ball leaves with (near) zero velocity
    DROP_HOLD_STEPS = 4

    # ---- deployment drop detection + in-episode re-grasp ----
    # If the vision ball falls >4 cm below the EE-implied held-ball pose for
    # 3 frames (or stays invisible for 10 frames) in the carry stage, the
    # grasp has failed.  Revert to the grasp stage and re-approach the ball
    # in the SAME episode (no environment reset).  Capped per episode.
    DROP_DZ = 0.060
    DROP_STREAK = 3
    DROP_INVALID_STREAK = 25
    RECOVERY_MAX = 2
    # arm only once the ball is clearly above the table; a drop is detected
    # when it is back on the table (or >6 cm below the EE-implied pose), so a
    # transient lag/dip during a normal lift is not mistaken for a drop
    LIFT_ARM_CLEARANCE = 0.070     # above table + radius
    BALL_ON_TABLE_CLEARANCE = 0.020
    # lift-first: no lateral command while the held ball is below rim+4 cm
    LIFT_FIRST_LATERAL_SCALE = 0.25
    LIFT_FIRST_MIN_Z = 0.12

    # ---- temporal filter ----
    ALPHA_BALL = 0.50

    def __init__(self, *args, vision_source: str = "vision",
                 switch_mode: str = "vision", pre_drop_clearance: float = 0.095,
                 ball_radius: float = BALL_RADIUS, lift_eps: float = 0.030,
                 grasp_guard: bool = False, **kwargs):
        if vision_source not in ("vision", "gt"):
            raise ValueError(f"vision_source must be vision/gt, got "
                             f"{vision_source!r}")
        if switch_mode not in ("vision", "gt"):
            raise ValueError(f"switch_mode must be vision/gt, got "
                             f"{switch_mode!r}")
        # must exist before super().__init__ in case the base calls self.seed
        self._episode_seed = None
        kwargs.setdefault("use_point_cloud", True)
        super().__init__(*args, **kwargs)

        self.vision_source = str(vision_source)
        self.switch_mode = str(switch_mode)
        self.pre_drop_clearance = float(pre_drop_clearance)
        self.ball_radius = float(ball_radius)
        # Stage-switch threshold on the vision ball rise.  The demo collector
        # switches at the expert LIFT phase (first lift action); a 3 cm
        # deployment threshold leaves the stage-0 policy in control while the
        # ball is already rising, which is where it samples the release.
        self.lift_eps = float(lift_eps)
        self.grasp_guard = bool(grasp_guard)
        self.pc_channels = 7          # xyz + rgb + ghost flag

        # Deployment-only contact softening (training collection leaves the
        # scene untouched: the default is 0).  The ball / finger-pad contacts
        # use solref 0.004 s, i.e. a time constant at the 2 ms integration
        # limit; when the pads squeeze the ball the stiff constraint solver
        # can inject a huge impulse (measured: 47 m/s ball after one step).
        # A softer contact (0.02 s) bounds that impulse while leaving the
        # grasp mechanics (friction, geometry) unchanged.
        # Deployment-only pad rolling/torsional resistance (0 = unchanged).
        # The pad geoms have rolling 0.05 / torsional 0.02; a 70 mm sphere
        # pinched between the 40 mm pad faces is effectively a roller
        # bearing, so during the carry the ball walks along the pad and is
        # squeezed out at the edge.  Raising the pad's rolling resistance
        # makes the pinch behave like a rubber gripper instead.  Only
        # applied in guard (deployment) mode so demo collection, which does
        # not use the guard, keeps the original physics.
        self.pad_rolling_friction = float(os.environ.get(
            "VSERVO_PAD_ROLL", "0.5" if self.grasp_guard else "0"))
        if self.pad_rolling_friction > 0.0:
            _tors = float(os.environ.get("VSERVO_PAD_TORS", "0.1"))
            for gi in range(self.model.ngeom):
                if self.model.geom(gi).name in ("left_finger_pad",
                                                "right_finger_pad"):
                    f = self.model.geom_friction[gi].copy()
                    self.model.geom_friction[gi] = [
                        f[0], self.pad_rolling_friction, _tors]
            print(f"[VservoMultiballEnv] pad rolling friction -> "
                  f"{self.pad_rolling_friction}/{_tors}", flush=True)

        self.contact_solref = float(os.environ.get("VSERVO_SOLREF", "0"))
        if self.contact_solref > 0.0:
            _soft = {"sphere_visual", "sphere_d1_visual", "sphere_d2_visual",
                     "left_finger_pad", "right_finger_pad"}
            _n = 0
            for gi in range(self.model.ngeom):
                if self.model.geom(gi).name in _soft:
                    self.model.geom_solref[gi] = [self.contact_solref, 1.0]
                    _n += 1
            print(f"[VservoMultiballEnv] softened {_n} geoms to "
                  f"solref={self.contact_solref}", flush=True)

        self._vision_rng = np.random.default_rng(7)
        self._ghost_halo_local = sample_sphere_shell(
            BALL_HALO_RADIUS, n=self.N_GHOST, rng=self._vision_rng)
        self._ghost_ball_local = sample_sphere_shell(
            self.ball_radius, n=self.N_GHOST, rng=self._vision_rng)
        self._reset_vision()

        n_agent = 9 + self.N_ERR + self.N_VALID + self.N_STAGE
        obs_space = {
            "agent_pos": spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(n_agent,), dtype=np.float32),
            "point_cloud": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.n_points, self.pc_channels), dtype=np.float32),
        }
        if "image" in getattr(self.observation_space, "spaces", {}):
            obs_space["image"] = self.observation_space["image"]
        self.observation_space = spaces.Dict(obs_space)

        print(f"[VservoMultiballEnv] source={self.vision_source} "
              f"switch={self.switch_mode} ghost={self.N_GHOST} "
              f"pc_channels={self.pc_channels} agent_pos={n_agent}")

    # ------------------------------------------------------------------
    # episode bookkeeping
    # ------------------------------------------------------------------
    def seed(self, seed=None):
        self._episode_seed = seed
        return super().seed(seed)

    def _reset_vision(self):
        self._ball_centre: Optional[np.ndarray] = None
        self._ball_centre_raw: Optional[np.ndarray] = None
        self._ball_rest_z: Optional[float] = None
        self._ball_valid = 0.0
        self._ball_valid_streak = 0
        self._ball_fit: dict = {}
        self._box_est: Optional[dict] = None
        self._pre_drop: Optional[np.ndarray] = None
        self._stage = 0
        self._stage_locked = False
        self._servo_err = np.zeros(self.N_ERR, dtype=np.float32)
        self._servo_valid = 0.0
        self._servo_target = np.zeros(3, dtype=np.float64)
        self._phi_prev: Optional[float] = None
        self._centered_stage = False
        self._vision_frames = 0
        self._vision_valid_frames = 0
        self._guard_grip_closed = False
        self._guard_grip_open = False
        self._recovery_count = 0
        self._post_recovery = False
        self._grip_check_step = 0
        self._grip_check_z0: Optional[float] = None
        self._grip_check_ee0: Optional[float] = None
        self._grip_check_done = False
        self._recover_phase = 0
        self._drop_streak = 0
        self._drop_invalid = 0
        self._lift_armed = False
        self._recovery_blocked = False
        self._grasp_wait = 0
        self._drop_hold = 0
        self._grasp_ramp = 0
        self.vision_info = {}

    def reset(self):
        self._reset_vision()
        # Reseed per episode so the same seed reproduces the same fit stream.
        base = 7 if self._episode_seed is None else int(self._episode_seed) + 7
        self._vision_rng = np.random.default_rng(base)
        return super().reset()

    # ------------------------------------------------------------------
    # stage / target control
    # ------------------------------------------------------------------
    def set_stage(self, stage: int, lock: bool = True):
        """Force the virtual target stage (used by the demo collector)."""
        stage = int(stage)
        if stage == self._stage:
            return
        self._stage = stage
        self._stage_locked = bool(lock)
        self._phi_prev = None
        self._centered_stage = False
        self._update_servo()

    @property
    def stage(self) -> int:
        return int(self._stage)

    def _box_rim_z(self) -> float:
        if self._box_est is not None:
            return float(self._box_est["rim_z"])
        return float(BOX_RIM_Z)

    def _box_centre_xy(self) -> np.ndarray:
        # The box is a STATIC fixture: its xy is known and must not be
        # replaced by the brown-wall estimate.  In this scene the wall
        # visibility is asymmetric and biases the estimated centre ~13 mm in
        # -y, shifting the release toward the south wall; balls then hit that
        # wall on the drop and the stiff contact solver ejects them south
        # (the observed x=14-22 m, y~-13 m failures).  The observed box is
        # still used for the rim HEIGHT (cloud z offset), not the xy.
        return fixture_box()["centre_xy"]

    def _pre_drop_target(self) -> np.ndarray:
        """Ball-centre pose directly above the box opening before release."""
        xy = self._box_centre_xy()
        z = (self._box_rim_z() + self.pre_drop_clearance - EE_TO_BALL_Z)
        return np.array([xy[0], xy[1], z], dtype=np.float64)

    def _maybe_switch_stage(self):
        if self._stage != 0 or self._stage_locked:
            return
        if self.switch_mode == "gt":
            lifted = bool(self.is_lifted())
        else:
            lifted = (self._ball_valid > 0
                      and self._ball_rest_z is not None
                      and self._ball_centre is not None
                      and float(self._ball_centre[2])
                      > self._ball_rest_z + self.lift_eps
                      and self._ball_valid_streak >= self.LIFT_CONFIRM_FRAMES)
        if lifted:
            self._stage = 1
            self._phi_prev = None
            self._centered_stage = False

    def _update_servo(self):
        """Servo error = target ball pose - implied ball pose at the EE."""
        ee = self.data.site(self._finger_site_id).xpos.copy()
        ee_ball = ee - np.array([0.0, 0.0, EE_TO_BALL_Z])
        if self._stage == 1:
            # fixture fallback always provides the pre-drop target
            target = self._pre_drop_target()
            valid = 1.0
        else:
            if self._ball_centre is not None:
                target = self._ball_centre
            else:
                target = ee_ball
            valid = float(self._ball_valid > 0)
        self._servo_target = np.asarray(target, dtype=np.float64)
        self._servo_err = (self._servo_target - ee_ball).astype(np.float32)
        self._servo_valid = valid

    # ------------------------------------------------------------------
    # optional deployment grasp guard (training data is untouched)
    # ------------------------------------------------------------------
    def step(self, action):
        if self.grasp_guard:
            action = self._guard_action(np.asarray(action, dtype=np.float64))
        return super().step(action)

    def _guard_action(self, a: np.ndarray) -> np.ndarray:
        """Gripper phase machine + descent clamp around the learned policy.

        stage 0: force the gripper open until the EE-implied ball is inside
        the grasp window; then close and latch, and clamp the commanded z so
        the finger site cannot sink below the grasp height (ball + 11 mm).
        stage 1: keep the gripper closed until the held-ball pose is inside
        the pre-drop window, then open and latch.

        Only the gripper and the near-grasp z command are touched; the xyz
        policy remains in control everywhere else.  ``grasp_guard=False``
        (the training default) leaves the policy action bit-identical.
        """
        a = a.copy()
        # ---- recovery re-approach: scripted stand-off above the ball ----
        if self._stage == 0 and self._recover_phase == 1:
            a[3] = -1.0
            a[0] = a[1] = a[2] = 0.0
            if self._servo_valid > 0:
                e = self._servo_err.astype(np.float64)
                scale = float(self.mw_action_scale)
                ez_stand = float(e[2]) + self.RECOVER_STANDOFF
                a[0] = float(np.clip(self.RECOVER_GAIN * e[0] / scale,
                                     -1.0, 1.0))
                a[1] = float(np.clip(self.RECOVER_GAIN * e[1] / scale,
                                     -1.0, 1.0))
                a[2] = float(np.clip(self.RECOVER_GAIN * ez_stand / scale,
                                     -1.0, 1.0))
                if (float(np.hypot(e[0], e[1])) <= self.RECOVER_XY_TOL
                        and abs(ez_stand) <= self.RECOVER_Z_TOL):
                    self._recover_phase = 2
            return a.astype(np.float64)
        if self._servo_valid <= 0:
            return a
        e = self._servo_err.astype(np.float64)
        d_xy = float(np.hypot(e[0], e[1]))
        dz = float(e[2])
        scale = float(self.mw_action_scale)
        if self._stage == 0:
            if not self._guard_grip_closed:
                a[3] = -1.0
                # lateral centring (xy only; the z stays with the policy and
                # the descent clamp) so the tighter grasp window is reachable
                if d_xy <= self.GRASP_CENTER_RANGE:
                    a[0] = float(np.clip(
                        a[0] + self.GRASP_LATERAL_GAIN * e[0] / scale,
                        -1.0, 1.0))
                    a[1] = float(np.clip(
                        a[1] + self.GRASP_LATERAL_GAIN * e[1] / scale,
                        -1.0, 1.0))
                if d_xy <= self.GRASP_Z_CLAMP_XY:
                    # Inside the grasp cone the BC policy's z command swings
                    # between -0.28 and +0.27 from one 4-action chunk to the
                    # next (near-OOD grasp state), which is the visible
                    # up/down jerk right before the close.  Above the grasp
                    # height: allow the descent but never an upward policy
                    # move.  Below it: the servo owns the upward correction
                    # (otherwise the policy's small down bias cancels it and
                    # the arm hovers under the close window for hundreds of
                    # steps).
                    _zservo = self.GRASP_Z_GAIN * dz / scale
                    if dz >= 0.0:
                        a[2] = float(np.clip(max(_zservo, 0.0),
                                             -1.0, 1.0))
                    else:
                        a[2] = float(np.clip(min(a[2], 0.0) + _zservo,
                                             -1.0, 1.0))
                # descent gate: an open gripper descending while laterally
                # off the ball hits it with the fingertip pads and knocks it
                # across the table.  Centre first, descend only inside the
                # gate (0 disables this; guard5 default).  Recovery
                # approaches use their own gate because the ball has rolled
                # to a spot the stage-0 policy did not see in the demos.
                _gate = self.GRASP_DESCENT_XY_TOL
                if self._post_recovery:
                    _gate = max(_gate, self.RECOVERY_DESCENT_XY_TOL)
                if _gate > 0.0 and d_xy > _gate:
                    a[2] = float(np.clip(max(a[2], 0.0), -1.0, 1.0))
                if (d_xy <= self.GRASP_XY_TOL
                        and abs(dz) <= self.GRASP_Z_TOL):
                    self._guard_grip_closed = True
                    self._grasp_wait = 0
                    self._grasp_ramp = 0
                    # The demo collector switches the virtual target at the
                    # expert LIFT phase (right after the gripper closes); the
                    # stage-1 skill owns the lift actions.  Switch here too,
                    # otherwise the stage-0 policy (trained only on the
                    # descend) must invent the lift and the ball never rises
                    # enough for the vision threshold.
                    self.set_stage(1, lock=True)
        else:
            # release settle: near the pre-drop pose, damp the lateral and
            # vertical motion so the ball is released with little horizontal
            # velocity (a fast release hits the box wall and the stiff
            # contact solver can eject it)
            if (not self._guard_grip_open
                    and d_xy <= self.DROP_SETTLE_XY
                    and abs(dz) <= self.DROP_SETTLE_Z):
                a[0] *= self.DROP_SETTLE_LATERAL
                a[1] *= self.DROP_SETTLE_LATERAL
                a[2] = float(np.clip(a[2], -self.DROP_SETTLE_MAX_Z,
                                     self.DROP_SETTLE_MAX_Z))
            # lift-first: no lateral translation while the held ball is still
            # below the box wall clearance (prevents the carry from dragging
            # the ball into the rim)
            ee = self.data.site(self._finger_site_id).xpos
            ee_ball_z = float(ee[2]) - EE_TO_BALL_Z
            if ee_ball_z < (self._box_rim_z()
                            + self.SAFE_CARRY_CLEARANCE):
                a[0] *= self.LIFT_FIRST_LATERAL_SCALE
                a[1] *= self.LIFT_FIRST_LATERAL_SCALE
                a[2] = float(np.clip(max(a[2], self.LIFT_FIRST_MIN_Z),
                                     -1.0, 1.0))
        # ---- carry speed cap ----
        _carry_scale = self.CARRY_ACTION_SCALE
        if self._post_recovery:
            _carry_scale = min(_carry_scale, self.RECOVERY_CARRY_SCALE)
        if (self._stage != 0 and not self._guard_grip_open
                and self._grasp_ramp >= (self.GRASP_RAMP_STEPS
                                         + self.GRASP_SETTLE_STEPS)
                and _carry_scale < 1.0):
            a[0] *= _carry_scale
            a[1] *= _carry_scale
            a[2] *= _carry_scale
        # ---- closed gripper: gradual, force-limited close ----
        # The pads are closed over GRASP_RAMP_STEPS while the arm is held
        # still, then held at GRASP_CLOSE_CMD (a calibrated pad gap just
        # below the ball diameter) instead of the crushing full-close target.
        if self._guard_grip_closed and not self._guard_grip_open:
            _hold = self.GRASP_RAMP_STEPS + self.GRASP_SETTLE_STEPS
            if self._grasp_ramp < _hold:
                self._grasp_ramp += 1
                if self._grasp_ramp <= self.GRASP_RAMP_STEPS:
                    frac = self._grasp_ramp / float(self.GRASP_RAMP_STEPS)
                    a[3] = -1.0 + frac * (self.GRASP_CLOSE_CMD + 1.0)
                else:
                    a[3] = self.GRASP_CLOSE_CMD
                a[0] = a[1] = a[2] = 0.0
            else:
                a[3] = self.GRASP_CLOSE_CMD
        # ---- test lift: did the ball actually come up with the hand? ----
        if (self._stage != 0 and self._guard_grip_closed
                and not self._guard_grip_open
                and self._grasp_ramp >= (self.GRASP_RAMP_STEPS
                                         + self.GRASP_SETTLE_STEPS)
                and not self._grip_check_done):
            ee_z = float(self.data.site(self._finger_site_id).xpos[2])
            ball_ok = (self._ball_valid > 0
                       and self._ball_centre is not None)
            if self._grip_check_z0 is None:
                if ball_ok:
                    self._grip_check_z0 = float(self._ball_centre[2])
                    self._grip_check_ee0 = ee_z
            else:
                if ball_ok:
                    d_ee = ee_z - float(self._grip_check_ee0)
                    d_ball = float(self._ball_centre[2]) - self._grip_check_z0
                    if (d_ee > 0.005
                            and d_ball < (self.GRIP_CHECK_RISE_FRAC * d_ee
                                          - self.GRIP_CHECK_RISE_SLACK)):
                        # the hand rose but the ball did not: the pinch is
                        # slipping -> re-grasp now, while the ball is still
                        # on the table (no reset, same episode)
                        self._grip_check_done = True
                        print("[VservoMultiballEnv] test lift: ball did not "
                              "rise with the hand -> re-grasp", flush=True)
                        self._start_recovery()
                        return a.astype(np.float64)
            self._grip_check_step += 1
            if self._grip_check_step >= self.GRIP_CHECK_STEPS:
                self._grip_check_done = True
        # ---- release window (stage 1, after the pads have settled) ----
        if (self._stage != 0 and not self._guard_grip_open
                and self._grasp_ramp >= (self.GRASP_RAMP_STEPS
                                         + self.GRASP_SETTLE_STEPS)):
            if self._drop_hold > 0:
                # hold the arm still for the last steps before release
                a[0] = a[1] = a[2] = 0.0
                self._drop_hold -= 1
                if self._drop_hold == 0:
                    self._guard_grip_open = True
            else:
                lat = float(np.hypot(a[0], a[1]))
                if (d_xy <= self.DROP_XY_TOL
                        and abs(dz) <= self.DROP_Z_TOL
                        and lat <= self.DROP_LATERAL_SPEED_TOL):
                    self._drop_hold = self.DROP_HOLD_STEPS
        if self._guard_grip_open:
            a[3] = -1.0
        return a.astype(np.float64)

    # ------------------------------------------------------------------
    # deployment drop detection -> in-episode re-grasp (no env reset)
    # ------------------------------------------------------------------
    def _recovery_ball_reachable(self) -> bool:
        """Only retry when the fallen ball is on the table in one of the
        spawn bands, where the stage-0 policy is in distribution.  Retrying
        a ball behind the box or knocked far away caused the re-approach to
        push it into a wall/constraint (physics blow-ups), so those are
        skipped and the episode simply times out."""
        if self._ball_valid > 0 and self._ball_centre is not None:
            x, y, z = (float(v) for v in self._ball_centre)
        else:
            # No vision fix -> the ball pose is unknown.  There is no
            # ground-truth object pose on a real robot, and re-approaching a
            # blind guess is unsafe: do not retry.
            return False
        if not (0.16 < x < 0.44 and 0.19 < z < 0.30):
            return False
        # the spawn zones are y <~ 0 (front) and y >~ 0.32 (rear); the gap
        # behind the box wall is not trainable/reachable for the re-grasp
        return (y < 0.02) or (y > 0.30)

    def _maybe_drop_recovery(self):
        if not self.grasp_guard or self._stage != 1:
            return
        if self._recovery_blocked:
            return
        if self._recovery_count >= self.RECOVERY_MAX:
            return
        # Arm only after the ball has been lifted well clear of the table
        # (table + radius + 7 cm).  The guard forces the stage at the grasp
        # while the ball is still on the table, and a ball hovering around
        # the 4 cm lift predicate is not a drop.  Use the vision estimate
        # (the deployment signal); if it is momentarily invalid, fall back
        # to the EE-implied held-ball height (forward kinematics), never to
        # the simulator object pose.
        if self._ball_valid > 0 and self._ball_centre is not None:
            held_z = float(self._ball_centre[2])
        else:
            held_z = float(
                self.data.site(self._finger_site_id).xpos[2]) - EE_TO_BALL_Z
        if held_z > (TABLE_SURFACE_Z + self.ball_radius
                     + self.LIFT_ARM_CLEARANCE):
            self._lift_armed = True
        if not self._lift_armed:
            # a closed gripper that does not lift the ball within the timeout
            # is a weak/popped grasp -> open and re-approach (same episode)
            if self._guard_grip_closed:
                self._grasp_wait += 1
                if self._grasp_wait >= self.GRASP_TIMEOUT_STEPS:
                    self._start_recovery()
            return
        ee_z = float(self.data.site(self._finger_site_id).xpos[2])
        ee_ball_z = ee_z - EE_TO_BALL_Z
        if self._ball_valid > 0 and self._ball_centre is not None:
            self._drop_invalid = 0
            ball_z = float(self._ball_centre[2])
            lost = ball_z < ee_ball_z - self.DROP_DZ
            # the drop is confirmed when the vision estimate puts the ball
            # back on the (known-height) table
            on_table = ball_z < (TABLE_SURFACE_Z + self.ball_radius
                                 + self.BALL_ON_TABLE_CLEARANCE)
        else:
            self._drop_invalid += 1
            lost = self._drop_invalid >= self.DROP_INVALID_STREAK
            on_table = False
        lost = bool(lost) or bool(on_table)
        self._drop_streak = self._drop_streak + 1 if lost else 0
        if self._drop_streak >= self.DROP_STREAK:
            self._start_recovery()

    def _start_recovery(self):
        """Revert to the grasp stage and re-approach the fallen ball in the
        SAME episode.  No environment reset; capped by RECOVERY_MAX."""
        if not self._recovery_ball_reachable():
            self._drop_streak = 0
            self._recovery_blocked = True
            print("[VservoMultiballEnv] drop detected; ball outside the "
                  "reachable re-grasp band -> no retry", flush=True)
            return
        self._recovery_count += 1
        self._post_recovery = True
        self._grip_check_step = 0
        self._grip_check_z0 = None
        self._grip_check_ee0 = None
        self._grip_check_done = False
        self._recover_phase = 1
        self._drop_streak = 0
        self._drop_invalid = 0
        self._lift_armed = False
        self._grasp_wait = 0
        self._drop_hold = 0
        self._grasp_ramp = 0
        self._stage = 0
        self._stage_locked = False
        self._guard_grip_closed = False
        self._guard_grip_open = False
        self._phi_prev = None
        self._centered_stage = False
        # the fallen ball can be >8 cm from the last (held) estimate, which
        # the jump gate would reject forever; forget the held estimate so the
        # next valid vision detection is accepted as a fresh acquisition
        self._ball_centre = None
        self._ball_centre_raw = None
        self._ball_valid = 0.0
        self._ball_valid_streak = 0
        self._update_servo()
        print(f"[VservoMultiballEnv] drop detected -> in-episode re-grasp "
              f"#{self._recovery_count}", flush=True)

    # ------------------------------------------------------------------
    # vision update (runs on the dense cloud before FPS)
    # ------------------------------------------------------------------
    def _update_vision(self, dense: np.ndarray):
        self._vision_frames += 1

        if self.vision_source == "gt":
            # debugging / ablation only: simulator object pose as the target
            self._ball_centre_raw = self.sphere_pos().astype(np.float64)
            self._ball_centre = self._ball_centre_raw.copy()
            self._ball_valid = 1.0
            self._ball_valid_streak += 1
            if self._ball_rest_z is None:
                self._ball_rest_z = float(self._ball_centre[2])
            if self._box_est is None:
                self._box_est = fixture_box()
                self._box_est["rim_z"] = float(self.box_top_z)
            self._maybe_switch_stage()
            self._maybe_drop_recovery()
            self._update_servo()
            self._vision_valid_frames += 1
            self.vision_info = {"valid": True, "source": "gt",
                                "stage": self._stage}
            return

        # ---- GREEN target ball ----
        cam_pos = self.data.cam_xpos[self._cam_pc_id].astype(np.float64)
        ball = estimate_green_ball(dense, radius=self.ball_radius,
                                   cam_pos=cam_pos)
        if ball is None:
            self._ball_valid = 0.0
            self._ball_valid_streak = 0
        else:
            c = np.asarray(ball["centre"], dtype=np.float64)
            jump = (np.linalg.norm(c - self._ball_centre)
                    if self._ball_centre is not None else None)
            if jump is not None and jump > 0.080:
                # a >8 cm single-frame jump is a segmentation/fit failure
                self._ball_valid = 0.0
                self._ball_valid_streak = 0
            else:
                self._ball_centre_raw = c
                if self._ball_centre is None:
                    self._ball_centre = c
                else:
                    a = self.ALPHA_BALL
                    self._ball_centre = (1 - a) * self._ball_centre + a * c
                self._ball_valid = 1.0
                self._ball_valid_streak += 1
                self._vision_valid_frames += 1
                if self._ball_rest_z is None:
                    self._ball_rest_z = float(self._ball_centre[2])
            self._ball_fit = ball

        # ---- static box opening (freeze the first plausible fit) ----
        if self._box_est is None:
            box = estimate_box(dense)
            if box is not None:
                box["source"] = "vision"
                self._box_est = box

        self._maybe_switch_stage()
        self._maybe_drop_recovery()
        self._update_servo()

        self.vision_info = {
            "valid": bool(self._servo_valid > 0),
            "stage": self._stage,
            "ball_valid": bool(self._ball_valid > 0),
            "ball_centre": (None if self._ball_centre is None
                            else self._ball_centre.copy()),
            "ball_rest_z": self._ball_rest_z,
            "ball_fit": {k: v for k, v in self._ball_fit.items()
                         if k in ("rms", "n_inliers", "n_color_points",
                                  "cost")},
            "box": (None if self._box_est is None else {
                "centre_xy": self._box_est["centre_xy"].copy(),
                "rim_z": self._box_est["rim_z"],
                "source": self._box_est.get("source", "vision")}),
            "target": self._servo_target.copy(),
            "err": self._servo_err.copy(),
            "n_frames": self._vision_frames,
            "n_valid": self._vision_valid_frames,
        }

    # ------------------------------------------------------------------
    # point cloud: dense render -> vision -> ghost points -> FPS
    # ------------------------------------------------------------------
    def _ghost_points(self) -> np.ndarray:
        if self._stage == 1:
            if self._box_est is None and self._ball_centre is None:
                return np.zeros((self.N_GHOST, 3), dtype=np.float32)
            return (self._ghost_ball_local
                    + self._pre_drop_target()).astype(np.float32)
        if self._ball_centre is None:
            return np.zeros((self.N_GHOST, 3), dtype=np.float32)
        return (self._ghost_halo_local + self._ball_centre).astype(np.float32)

    def _render_point_cloud(self):
        """Render depth+RGB+segmentation, update vision, append ghosts.

        Mirrors the parent ``_render_point_cloud`` exactly for the real
        points; the only changes are the vision update on the dense cloud and
        the reserved ghost slots at the end (1944 real + 64 ghost at 2048,
        960 + 64 at 1024, ...).
        """
        if self._pc_renderer is None:
            return np.zeros((self.n_points, self.pc_channels),
                            dtype=np.float32)

        cam_id = self._cam_pc_id

        self._pc_renderer.enable_depth_rendering()
        self._pc_renderer.update_scene(self.data, camera=cam_id)
        depth = self._pc_renderer.render().astype(np.float32)
        self._pc_renderer.disable_depth_rendering()
        self._pc_renderer.update_scene(self.data, camera=cam_id)
        rgb = self._pc_renderer.render()

        self._pc_renderer.enable_segmentation_rendering()
        self._pc_renderer.update_scene(self.data, camera=cam_id)
        seg = self._pc_renderer.render().astype(np.int32)
        self._pc_renderer.disable_segmentation_rendering()
        seg_keep = (seg[..., 0] >= 0).astype(np.float32)
        for gid in self._pc_exclude_geom_ids:
            seg_keep = seg_keep * (seg[..., 0] != gid).astype(np.float32)
        is_table = (seg[..., 0] == self.model.geom("table_top").id)

        H, W = depth.shape
        f = H / (2.0 * np.tan(np.radians(self.model.cam_fovy[cam_id]) / 2.0))
        cx, cy = W / 2.0, H / 2.0

        v, u = np.mgrid[0:H, 0:W]
        x = (u - cx) * depth / f
        y = (v - cy) * depth / f
        pts_cam = np.stack([x, y, depth], axis=-1).reshape(-1, 3)

        mask = (depth > 0.001) & (depth < self._pc_far)
        mask = mask * (seg_keep > 0.5)
        mask = mask.reshape(-1)
        if not np.any(mask):
            self._update_vision(np.zeros((0, 6), dtype=np.float32))
            return np.zeros((self.n_points, self.pc_channels),
                            dtype=np.float32)
        pts_cam = pts_cam[mask]

        R_x180 = np.array([[1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0],
                           [0.0, 0.0, -1.0]], dtype=np.float32)
        R_cw = self.data.cam_xmat[cam_id].reshape(3, 3).astype(np.float32) @ R_x180
        t_cw = self.data.cam_xpos[cam_id].astype(np.float32)
        pts_world = pts_cam @ R_cw.T + t_cw

        xmin, ymin, zmin, xmax, ymax, zmax = self.pc_crop_bounds
        keep = (pts_world[:, 0] >= xmin) & (pts_world[:, 0] <= xmax) & \
               (pts_world[:, 1] >= ymin) & (pts_world[:, 1] <= ymax) & \
               (pts_world[:, 2] >= zmin) & (pts_world[:, 2] <= zmax)
        pts_world = pts_world[keep]
        is_table_crop = is_table.reshape(-1)[mask][keep]
        colors = rgb.reshape(-1, 3)[mask][keep].astype(np.float32)

        if len(pts_world) == 0:
            self._update_vision(np.zeros((0, 6), dtype=np.float32))
            return np.zeros((self.n_points, self.pc_channels),
                            dtype=np.float32)

        dense = np.concatenate([pts_world, colors], axis=-1)
        self._update_vision(dense)

        # reserve ghost slots; FPS only the real scene
        n_real = self.n_points - self.N_GHOST
        pts = np.concatenate([pts_world, colors], axis=-1)
        table_pts = pts[is_table_crop]
        obj_pts = pts[~is_table_crop]
        n_table = int(round(0.10 * n_real))
        if len(table_pts) > n_table:
            table_pts = _fps_downsample(table_pts, n_table, seed=1)
        n_obj = n_real - len(table_pts)
        if len(obj_pts) > n_obj:
            obj_pts = _fps_downsample(obj_pts, n_obj, seed=0)
        real = np.concatenate([obj_pts, table_pts], axis=0)
        if len(real) < n_real:
            pad = np.zeros((n_real - len(real), 6), dtype=np.float32)
            real = np.concatenate([real, pad], axis=0)
        real = real[:n_real].astype(np.float32)

        out = np.zeros((self.n_points, self.pc_channels), dtype=np.float32)
        out[:n_real, :6] = real
        out[n_real:, :3] = self._ghost_points()
        out[n_real:, 3:6] = GHOST_RGB
        out[n_real:, 6] = 1.0
        return out

    # ------------------------------------------------------------------
    # observation: append the servo error + validity + stage
    # ------------------------------------------------------------------
    def _get_obs(self):
        obs = super()._get_obs()
        obs["agent_pos"] = np.concatenate([
            np.asarray(obs["agent_pos"], dtype=np.float32),
            self._servo_err.astype(np.float32),
            np.array([self._servo_valid, float(self._stage)],
                     dtype=np.float32),
        ]).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # dense virtual-target reward
    # ------------------------------------------------------------------
    def _vservo_phi(self) -> Optional[float]:
        if self._servo_valid <= 0:
            return None
        e = self._servo_err.astype(np.float64)
        d_xy = float(np.hypot(e[0], e[1]))
        dz = abs(float(e[2]))
        # Always use the full 3-D target potential.  Gating the z term on
        # lateral proximity (the Hanoi insert shape) would make the potential
        # MORE negative as the agent centres/ lifts when the target is far
        # laterally -- which is the case at the start of both multiball
        # stages.  The approach/ lift ORDER is instead encoded by the smooth
        # penalties in _compute_reward, so the potential itself stays
        # continuous and monotone.
        return -(self.W_XY * d_xy + self.W_Z * dz)

    def _compute_reward(self):
        sphere = self.sphere_pos()
        ee = self.data.site(self._finger_site_id).xpos.copy()
        in_box = self.in_box()
        lifted = self.is_lifted()

        reward = 0.0
        done = False
        info = {}
        if in_box:
            reward += self.SUCCESS_REWARD
            done = True
            self._grasp_success = True
        elif self.step_count >= self.max_episode_steps:
            done = True
        else:
            phi = self._vservo_phi()
            if phi is not None:
                if self._phi_prev is None:
                    self._phi_prev = phi
                r = phi - self._phi_prev
                self._phi_prev = phi
                reward += r
                info["reward_vservo"] = float(r)

                e = self._servo_err.astype(np.float64)
                d_xy = float(np.hypot(e[0], e[1]))
                dz = abs(float(e[2]))
                if (not self._centered_stage
                        and d_xy <= self.CENTER_XY_TOL
                        and dz <= self.CENTER_Z_TOL):
                    self._centered_stage = True
                    reward += self.CENTER_BONUS
                    info["centered"] = True

                ee_ball_z = float(ee[2]) - EE_TO_BALL_Z
                if self._stage == 0 and d_xy > self.SAFE_XY_TOL:
                    # premature-descent penalty in the grasp stage: moving the
                    # implied ball below the approach guard while off-centre
                    if (self._ball_rest_z is not None
                            and ee_ball_z
                            < self._ball_rest_z + self.APPROACH_GUARD):
                        reward -= self.DESCENT_PENALTY
                        info["descent_penalty"] = self.DESCENT_PENALTY
                elif self._stage == 1:
                    # low-carry penalty: dragging the held ball across the
                    # table/box wall instead of lifting clear of the rim first
                    safe_z = (self._box_rim_z()
                              + self.SAFE_CARRY_CLEARANCE)
                    low = float(np.clip(
                        (safe_z + self.SOFT_Z - ee_ball_z) / self.SOFT_Z,
                        0.0, 1.0))
                    far = float(np.clip(
                        (d_xy - self.SAFE_XY_TOL) / self.SOFT_XY, 0.0, 1.0))
                    if low > 0.0 and far > 0.0:
                        pen = self.LOW_CARRY_PENALTY * low * far
                        reward -= pen
                        info["low_carry_penalty"] = float(pen)
            else:
                info["reward_vservo"] = 0.0

        if lifted and not self._grasped_ever:
            reward += self.GRASP_REWARD
            self._grasped_ever = True

        info.update({
            "success": bool(self._grasp_success),
            "grasp_success": bool(self._grasp_success),
            "placed": self.placed(),
            "in_box": in_box,
            "released": self.is_released(),
            "lifted": lifted,
            "dist_to_box": float(np.hypot(sphere[0] - self.box_center[0],
                                          sphere[1] - self.box_center[1])),
            "object_pos": sphere,
            "current_object": "sphere",
            "vision_valid": bool(self._servo_valid > 0),
            "vision_err": self._servo_err.tolist(),
            "vision_target": self._servo_target.tolist(),
            "vision_stage": int(self._stage),
            "vision_source": self.vision_source,
            "vision_ball": (None if self._ball_centre is None
                            else self._ball_centre.tolist()),
            "vision_rest_z": self._ball_rest_z,
            "vision_box_source": (None if self._box_est is None
                                  else self._box_est.get("source")),
            "vision_frames": self._vision_frames,
            "vision_valid_frames": self._vision_valid_frames,
            "grasp_guard": bool(self.grasp_guard),
            "guard_grip_closed": bool(self._guard_grip_closed),
            "guard_grip_open": bool(self._guard_grip_open),
            "recoveries": int(self._recovery_count),
        })
        return reward, done, info
