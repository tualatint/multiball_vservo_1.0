#!/usr/bin/env python
"""Generate MULTIBALL sphere-in-box demos (MetaWorld EE-space) -> zarr.

Task: THREE spheres on the table -- one GREEN (the target) + two WHITE
distractors. Pick the GREEN sphere and place it inside the FIXED target box
(the distractors are never touched). The demo collector drives the SAME env
used at eval (``SphereBoxMultiballEnv``, action_mode='metaworld'): 4-dim
actions ``[dx, dy, dz, gripper]`` (+1 = close / -1 = open) realized by the
differential IK + PD control, and per-step RGB point cloud + 9-dim agent
state. The scripted expert always targets ``env.sphere_pos()`` = the GREEN
ball, so the demonstrated action distribution is green-ball-conditioned.

Reward structure (written to the dataset, used at eval; BC ignores it):
  * +1.0 sparse terminal when the sphere is INSIDE the box (``env.in_box()``
    -- centre within the box footprint and below the rim; a ball that drops
    or rolls into the box counts, not just clean gripper releases);
  * +0.5 one-time bonus the first time the sphere is lifted off the table
    (separates the approach from the carry in the recorded return);
  * +0.1 * (phi_prev - phi) potential-based shaping per step where phi is
    the EE->sphere xy distance before the grasp and the sphere->box-center
    xy distance once lifted (sign-safe: only decreases on genuine progress).

Scripted expert (per-step EE controller, phases):
  approach -> descend -> close -> lift -> carry -> open (drop release)
The phases are designed to be IDLE-FREE:
  * every phase servos the EE / gripper continuously (no zero-action holds);
  * the release happens at the CARRY height: the gripper opens over the box
    centre and the ball visibly drops into the box (no slow lower-and-rest
    phase). The episode ends the instant the sphere is placed (env done),
    so there is no post-success idle tail;
  * failures are truncated at ``--max-steps`` (and optionally saved to a
    separate zarr as terminal timeouts).

Per env step it records (format identical to the working arx5 grasp zarr so
``Arx5GraspDataset`` loads it unchanged):
  state       (T+1, 9)  [ee(finger_site)_xyz, left_touch_site_xyz, right_touch_site_xyz]
  action      (T, 4)    [dx, dy, dz, gripper] in [-1, 1]
  point_cloud (T+1, N, 6) xyz+rgb from the fixed cam_pc
  object_pos  (T+1, 3)  sphere centre
  object_quat (T+1, 4)  sphere orientation (visualization aid)
plus next_state/next_action/next_point_cloud/reward/done/timeout/return.

Run (from the repo root):
  source scripts/env.sh
  python third_party/multiball/collect_demos_multiball.py \
    --out data/multiball_demos.zarr --n-demos 200

--tune N: run N episodes, print per-phase step counts and total lengths,
save NOTHING. Useful to sanity-check the fast natural pace.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiball_env import SphereBoxMultiballEnv
from rl_100.common.replay_buffer import ReplayBuffer

GAMMA = 0.99

# ---------------------------------------------------------------------------
# scripted expert constants (geometry validated by scripts/sanity_sphere_box.py:
# a full grasp -> carry -> drop -> release trajectory ends with placed=True)
# ---------------------------------------------------------------------------
# EE (finger_site) hover height above the sphere centre during approach.
APPROACH_Z = 0.12
# EE descend target above the sphere centre. The finger pads sit ~1.1 cm
# BELOW the finger_site (world vertical), so a site at ~1.1 cm above the
# centre puts the pad CENTRE level with the sphere centre: the pads then
# pinch the WIDEST 7 cm equator band, the gripper stops at a ~7 cm gap and
# the pads never cut into the ball's silhouette (no "fingers inside the
# ball"). Closing any higher wedges the pads into the narrower upper band
# (sphere width there ~5.3 cm) and the fingers visibly sink into the sphere.
GRASP_Z = 0.011
# LIFT / CARRY: finger_site z while carrying. The sphere centre hangs ~1.1 cm
# below the site; the box rim is at 0.27, so a site z >= 0.35 keeps the sphere
# bottom (site - 0.046) >= 0.30 -> >= 3 cm rim clearance during the carry.
LIFT_Z_ABS = 0.365
CARRY_Z = 0.365
# There is NO separate CLOSE phase: the descend reaches the grasp height with
# the gripper OPEN (nothing blocks the descent), and the LIFT phase commands
# gripper=+1 from its very first step. The gripper closes almost instantly
# (measured ~0.019 m within one env step), so the pads nip the ball at/near
# the equator exactly as the site starts rising -- the clamp happens while the
# arm is ALREADY moving, so there is no stationary "hold-and-grasp" pause.
# The LIFT's continuous gripper=+1 then keeps the squeeze firm for the carry.
# OPEN is only a short safety hold after the arm has stopped at the box
# centre: the ball is released during the DECELERATING final carry approach
# (see CARRY below), and the env ends the episode the instant placed()
# fires, so the stationary hold is normally 0-1 steps.
OPEN_STEPS = 4             # safety cap on the open hold (env usually ends first)

APPROACH_MAX_STEP = 0.045
APPROACH_TOL = 0.006
APPROACH_BUDGET = 400
DESCEND_MAX_STEP = 0.030
DESCEND_TOL = 0.003
DESCEND_BUDGET = 120
LIFT_MAX_STEP = 0.028
LIFT_BUDGET = 120
CARRY_MAX_STEP = 0.042
CARRY_TOL = 0.008
CARRY_BUDGET = 120
# CARRY release window: once the site is within ``CARRY_OPEN_DIST`` of the box
# centre the gripper starts opening while the arm decelerates, so the ball is
# dropped with (almost) no forward speed straight into the box.
CARRY_OPEN_DIST = 0.030

# Phase ids (for bookkeeping / per-phase step reports).
# The gripper close is folded into DESCEND and the drop release is folded into
# the CARRY approach, so the phases are: approach -> descend(+close) ->
# lift -> carry(+open) -> open(hold, usually 0-1 steps) -> done.
APPROACH, DESCEND, LIFT, CARRY, OPEN, DONE = range(6)
PHASE_NAMES = ["approach", "descend", "lift", "carry", "open", "done"]


# Per-episode FAILURE MODES. The dataset must contain diverse negatives for
# IQL (offline RL learns the succ/fail boundary from these), so instead of a
# single slip pattern we perturb the GRASP/PLACEMENT pose in a few physically
# distinct, REALISTIC ways (small deviations that fumble the ball, not gross
# scripted misses). Modes (see sample_expert_params for the noise ranges):
#   clean       nominal grasp (no extra perturbation) -> usually succeeds
#   offcenter   small lateral grasp miss (8-14 mm) -> off-centre pinch, the
#               ball squirts/slips out during lift/carry
#   toohigh     pads pinch just above the equator -> shallow hold, ball pops
#               out of the jaws mid-carry
#   nudge       descend lands 18-26 mm beside the ball, a pad bumps it, the
#               close misses or half-catches it, the robot gives up after a
#               short lift effort (no long hover)
#   misplace    correct grasp + lift + carry, but the drop target is
#               ~7-11 cm beside the box centre -> ball lands next to the box
FAIL_MODES = ["clean", "offcenter", "toohigh", "nudge", "misplace"]
FAIL_MODE_PROBS = [0.72, 0.10, 0.06, 0.06, 0.06]


def sample_expert_params(rng, mode=None):
    """Per-episode jitter so the demonstrated action distribution is
    multimodal instead of one exact trajectory. Speeds are the FAST natural
    robot pace (approach ~0.04 m/step, carry ~0.06 m/step) exploiting the
    raised PD gains + doubled IK clip (tracking up to ~0.09 m/step);
    episodes complete in ~30 env steps (~30 frames at 1 frame/step) with NO
    idle stalls. The carry/lift heights keep the sphere bottom at least
    3 cm above the box rim (rim at z = 0.27).

    A FAIL MODE is drawn per episode (FAIL_MODES / FAIL_MODE_PROBS) and
    shapes the grasp-pose noise -- so the collection contains a diverse mix
    of negative examples for IQL (grasp failures of several kinds), not just
    one slip pattern. ``mode`` can be passed explicitly to FORCE a specific
    case (used by the gallery's per-case sample recorder)."""
    p = {
        "approach_z": float(rng.uniform(0.08, 0.13)),
        "approach_step": float(rng.uniform(0.035, 0.050)),
        "descend_step": float(rng.uniform(0.022, 0.030)),
        "lift_z_abs": float(rng.uniform(0.355, 0.385)),
        "lift_step": float(rng.uniform(0.024, 0.030)),
        "carry_z": float(rng.uniform(0.352, 0.382)),
        "carry_step": float(rng.uniform(0.040, 0.055)),
        # open_steps: safety cap on the drop-hold. The ball is normally
        # already released during the CARRY approach, so placed() fires
        # before this budget is consumed.
        "open_steps": int(rng.integers(3, 5)),
    }
    if mode is None:
        mode = str(rng.choice(FAIL_MODES, p=FAIL_MODE_PROBS))
    p["fail_mode"] = mode

    # grasp_z: finger_site height above the sphere centre when the pads
    # close. The pad centre is ~1.1 cm BELOW the site, so 0.009-0.013 keeps
    # the pinch band within +-2 mm of the equator (widest 7 cm).
    if mode == "clean":
        p["grasp_z"] = float(rng.uniform(0.009, 0.013))
        p["grasp_xy_noise_x"] = float(rng.uniform(-0.006, 0.006))
        p["grasp_xy_noise_y"] = float(rng.uniform(-0.006, 0.006))
    elif mode == "offcenter":
        # 8-14 mm lateral miss in a random direction: the pads pinch the
        # sphere off-centre -> the hold is weak and the ball squirts/slips
        # out during the lift or carry (a natural fumbled grasp).
        ang = rng.uniform(0.0, 2.0 * np.pi)
        r = rng.uniform(0.008, 0.014)
        p["grasp_z"] = float(rng.uniform(0.009, 0.013))
        p["grasp_xy_noise_x"] = float(r * np.cos(ang))
        p["grasp_xy_noise_y"] = float(r * np.sin(ang))
    elif mode == "toohigh":
        # Pads pinch just ABOVE the equator (pad centre ~1-8 mm up the ball,
        # site 0.012-0.019 above the centre): the shallow hold lets the ball
        # pop/slip out of the jaws during the carry.
        p["grasp_z"] = float(rng.uniform(0.012, 0.019))
        p["grasp_xy_noise_x"] = float(rng.uniform(-0.004, 0.004))
        p["grasp_xy_noise_y"] = float(rng.uniform(-0.004, 0.004))
    elif mode == "nudge":
        # Descend lands 18-26 mm beside the ball: a pad bumps the ball during
        # the descent, the close misses or half-catches it, and the robot
        # GIVES UP after a short lift effort (see ScriptedPolicy._lift_done)
        # instead of hovering over an empty grasp.
        ang = rng.uniform(0.0, 2.0 * np.pi)
        r = rng.uniform(0.018, 0.026)
        p["grasp_z"] = float(rng.uniform(0.008, 0.012))
        p["grasp_xy_noise_x"] = float(r * np.cos(ang))
        p["grasp_xy_noise_y"] = float(r * np.sin(ang))
        p["descend_giveup_steps"] = int(rng.integers(8, 11))
        p["lift_giveup_steps"] = int(rng.integers(8, 12))
    elif mode == "misplace":
        # Correct grasp + lift + carry, but the drop target sits 8-14 cm
        # beside the box centre (guaranteed outside the 5.5 cm success
        # half-extent and clamped to the reachable table area): the ball is
        # placed cleanly NEXT TO the box -- a natural placement error.
        ox = float(rng.uniform(-0.070, 0.070))
        oy = float(rng.uniform(-0.090, 0.090))
        mag = np.hypot(ox, oy)
        if mag < 0.080:  # ensure the drop lands OUTSIDE the success region
            s = 0.080 / max(mag, 1e-6)
            ox *= s
            oy *= s
        p["carry_xy_offset"] = (float(ox), float(oy))
        p["grasp_z"] = float(rng.uniform(0.009, 0.013))
        p["grasp_xy_noise_x"] = float(rng.uniform(-0.006, 0.006))
        p["grasp_xy_noise_y"] = float(rng.uniform(-0.006, 0.006))
    return p


def _move_action(env, goal, gripper, max_step, tol):
    """4-dim action [dx, dy, dz, gripper] moving the EE (finger_site) toward
    ``goal`` at at most ``max_step`` m/step. Returns (action, reached)."""
    p = env.data.site(env._finger_site_id).xpos.copy()
    dv = goal - p
    n = float(np.linalg.norm(dv))
    if n < tol:
        return np.array([0.0, 0.0, 0.0, gripper], dtype=np.float32), True
    step = dv / n * min(n, max_step)
    return np.array([step[0] / env.mw_action_scale,
                     step[1] / env.mw_action_scale,
                     step[2] / env.mw_action_scale,
                     gripper], dtype=np.float32), False


class ScriptedPolicy:
    """Per-step EE-space expert: approach -> descend(+close) -> lift ->
    carry(+drop release) -> open(hold). Stall-free by construction:

      * phase ADVANCE happens BEFORE the step's action is emitted, so a
        completed phase never leaves a zero-action stall at the boundary;
      * the gripper close is folded into the DESCEND tail (the site slows to
        ``CLOSE_DESCEND_STEP`` while the pads close), so the arm never stands
        still to grasp;
      * the drop release is folded into the CARRY tail (the gripper opens
        while the arm decelerates onto the box centre), so the arm never
        stands still to release. OPEN only holds the (already-open) gripper
        as a safety until the env flags placed().
    """

    def __init__(self, params=None):
        self.p = dict(sample_expert_params(np.random.default_rng(0)))
        self.p.update(params or {})
        self.reset()

    def reset(self):
        self.phase = APPROACH
        self.open_remain = int(self.p["open_steps"])
        self.steps = 0
        self.phase_steps = []
        # absolute targets captured when a phase starts (avoids the
        # goal-chasing bug where the lifted sphere keeps raising the goal)
        self.descend_target = None
        self.lift_target = None
        self.carry_target = None

    # ---- per-phase completion predicates (re-evaluated every act()) ----
    def _descend_goal(self, sphere_pos):
        """Descend target = sphere xy (+ the fixed per-episode lateral grasp
        noise) at the grasp height. Captured once so the noisy offset stays
        constant while the site converges -- an off-centre grasp makes some
        episodes slip and fail (genuine negatives for IQL)."""
        if self.descend_target is None:
            self.descend_target = np.array([
                sphere_pos[0] + self.p.get("grasp_xy_noise_x", 0.0),
                sphere_pos[1] + self.p.get("grasp_xy_noise_y", 0.0),
                sphere_pos[2] + self.p["grasp_z"]])
        return self.descend_target

    def _approach_done(self, env, sphere_pos):
        goal = sphere_pos + np.array([0.0, 0.0, self.p["approach_z"]])
        p = env.data.site(env._finger_site_id).xpos
        return bool(np.linalg.norm(goal - p) < APPROACH_TOL or
                    self.steps >= APPROACH_BUDGET)

    def _descend_done(self, env, sphere_pos):
        target = self._descend_goal(sphere_pos)
        p = env.data.site(env._finger_site_id).xpos
        dv = target - p
        # If a bumped ball blocks the descent, give up after a short effort
        # and let the LIFT phase try the close anyway -- a natural fumble,
        # not a 120-step grind against the ball. Only the ``nudge`` mode
        # sets a short ``descend_giveup_steps``; other modes keep the full
        # budget.
        if self.steps >= self.p.get("descend_giveup_steps", DESCEND_BUDGET):
            return True
        # done when the site reaches the grasp height. The gripper is OPEN
        # during the whole descent (nothing blocks it), so this is a clean
        # geometric arrival; the clamp happens at the START of the LIFT phase.
        return bool(np.linalg.norm(dv) < DESCEND_TOL or
                    self.steps >= DESCEND_BUDGET)

    def _lift_done(self, env, sphere_pos):
        if self.lift_target is None:
            self.lift_target = np.array(
                [sphere_pos[0], sphere_pos[1], self.p["lift_z_abs"]])
        lifted = env.is_lifted()
        z_ok = (env.data.site(env._finger_site_id).xpos[2] >
                self.lift_target[2] - 0.01)
        # Realistic fumble: if the gripper never lifts the ball after a short
        # effort, GIVE UP instead of hovering 120 steps -- the robot carries
        # on to the box with an empty gripper and opens, which reads as a
        # natural failed attempt rather than a scripted hold.
        gave_up = (not lifted and
                   self.steps >= self.p.get("lift_giveup_steps", 12))
        return bool(((lifted and z_ok) or gave_up) or
                    self.steps >= LIFT_BUDGET)

    def _carry_goal(self, env):
        """Carry/drop target: the box centre plus the per-episode placement
        offset (``carry_xy_offset``, the ``misplace`` failure mode) at the
        carry height -- an empty gripper (after a give-up) still completes
        this trajectory, which reads as a natural failed attempt."""
        ox, oy = self.p.get("carry_xy_offset", (0.0, 0.0))
        return np.array([env.box_center[0] + ox, env.box_center[1] + oy,
                         self.p["carry_z"]])

    def _carry_done(self, env, sphere_pos):
        if self.carry_target is None:
            self.carry_target = self._carry_goal(env)
        p = env.data.site(env._finger_site_id).xpos
        return bool(np.linalg.norm(self.carry_target - p) < CARRY_TOL or
                    self.steps >= CARRY_BUDGET)

    def _phase_done(self, env, sphere_pos):
        if self.phase == APPROACH:
            return self._approach_done(env, sphere_pos)
        if self.phase == DESCEND:
            return self._descend_done(env, sphere_pos)
        if self.phase == LIFT:
            return self._lift_done(env, sphere_pos)
        if self.phase == CARRY:
            return self._carry_done(env, sphere_pos)
        if self.phase == OPEN:
            return self.open_remain <= 0
        return False

    def _advance(self):
        self.phase += 1
        self.phase_steps.append(self.steps)
        self.steps = 0
        if self.phase == OPEN:
            self.open_remain = int(self.p["open_steps"])

    # ---- per-phase action (emitted only AFTER all completed phases advance) --
    def _action(self, env, sphere_pos):
        if self.phase == APPROACH:
            action, _ = _move_action(
                env, sphere_pos + np.array([0.0, 0.0, self.p["approach_z"]]),
                -1.0, self.p["approach_step"], APPROACH_TOL)
        elif self.phase == DESCEND:
            action = self._descend_action(env, sphere_pos)
        elif self.phase == LIFT:
            if self.lift_target is None:
                self.lift_target = np.array(
                    [sphere_pos[0], sphere_pos[1], self.p["lift_z_abs"]])
            action, _ = _move_action(env, self.lift_target, 1.0,
                                     self.p["lift_step"], 1e-9)
        elif self.phase == CARRY:
            action = self._carry_action(env, sphere_pos)
        elif self.phase == OPEN:
            # safety hold: the ball is already released (opened during CARRY);
            # the env ends the episode on placed().
            action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            self.open_remain -= 1
        else:  # DONE
            action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        self.steps += 1
        return action

    def _descend_action(self, env, sphere_pos):
        target = self._descend_goal(sphere_pos)
        p = env.data.site(env._finger_site_id).xpos
        dv = target - p
        n = float(np.linalg.norm(dv))
        # Gripper stays OPEN for the whole descent (a contact would block the
        # descent: the ball+table form a rigid stop). The clamp happens on the
        # first LIFT step. Cap the z travel so the site NEVER undershoots
        # below the grasp height.
        if n < DESCEND_TOL:
            return np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        step = dv / n * min(n, self.p["descend_step"])
        if abs(step[2]) > abs(dv[2]) > 1e-6:
            step = step * (abs(dv[2]) / abs(step[2]))
        return np.array([step[0] / env.mw_action_scale,
                         step[1] / env.mw_action_scale,
                         step[2] / env.mw_action_scale,
                         -1.0], dtype=np.float32)

    def _carry_action(self, env, sphere_pos):
        if self.carry_target is None:
            self.carry_target = self._carry_goal(env)
        p = env.data.site(env._finger_site_id).xpos
        dv = self.carry_target - p
        n = float(np.linalg.norm(dv))
        # Drop release: only when BOTH the finger_site AND the sphere are
        # within the window of the box centre. During the decelerating fast
        # carry the sphere lags behind the site (inertia pushes it against
        # the front pad); releasing on the site distance alone drops the
        # ball short of the box (seen in seeds 2025/2029). The window can be
        # shifted per-episode (``carry_xy_offset``, the ``misplace`` failure
        # mode) so the ball is dropped cleanly next to the box.
        open_dist = self.p.get("carry_open_dist", CARRY_OPEN_DIST)
        ball_off = float(np.linalg.norm(sphere_pos[:2] - self.carry_target[:2]))
        opening = bool(n < open_dist and ball_off < open_dist)
        gripper = -1.0 if opening else 1.0
        if n < CARRY_TOL:
            return np.array([0.0, 0.0, 0.0, gripper], dtype=np.float32)
        max_step = min(self.p["carry_step"], max(0.010, n * 0.35))
        step = dv / n * min(n, max_step)
        return np.array([step[0] / env.mw_action_scale,
                         step[1] / env.mw_action_scale,
                         step[2] / env.mw_action_scale,
                         gripper], dtype=np.float32)

    def act(self, env, sphere_pos):
        # Advance through every completed phase BEFORE emitting the step's
        # action, so a phase boundary never costs a zero-action stall step.
        while self.phase != DONE and self._phase_done(env, sphere_pos):
            self._advance()
        if self.phase == DONE:
            return np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        return self._action(env, sphere_pos)


def run_episode(env, seed, params=None, max_steps=300):
    """Run one scripted episode; returns (ep, success, phase_steps, length).
    The episode ends at the first env done (= sphere placed, the natural
    no-idle termination) or when the scripted policy reaches DONE."""
    env.seed(seed)
    obs = env.reset()
    policy = ScriptedPolicy(params)
    state = obs["agent_pos"].copy()
    point_cloud = obs["point_cloud"].astype(np.float32).copy()
    obj_pos = env.sphere_pos().copy()
    obj_quat = env.data.body("sphere").xquat.copy()

    ep = {
        "state": [state],
        "action": [],
        "point_cloud": [point_cloud],
        "object_pos": [obj_pos],
        "object_quat": [obj_quat],
        "reward": [],
        "done": [],
        "timeout": [],
    }

    while True:
        sphere = env.sphere_pos()
        action = policy.act(env, sphere)
        obs, reward, done, info = env.step(action)
        ep["state"].append(obs["agent_pos"].copy())
        ep["action"].append(action.astype(np.float32))
        ep["point_cloud"].append(obs["point_cloud"].astype(np.float32).copy())
        ep["object_pos"].append(env.sphere_pos().copy())
        ep["object_quat"].append(env.data.body("sphere").xquat.copy())
        placed = bool(info["in_box"])
        ep["reward"].append(np.float32(1.0 if placed else reward))
        ep["done"].append(np.bool_(done))
        ep["timeout"].append(np.bool_(done and not placed))
        # idle-free termination: end the instant the sphere is inside the box
        # (env done fires on the same step), or when the scripted expert
        # finishes.
        if done or policy.phase == DONE or len(ep["action"]) >= max_steps:
            break
    success = bool(info["in_box"])
    return ep, success, policy.phase_steps, len(ep["action"])


def to_episode_data(ep, T):
    state = np.stack(ep["state"]).astype(np.float32)          # (T+1, 9)
    action = np.stack(ep["action"]).astype(np.float32)        # (T, 4)
    point_cloud = np.stack(ep["point_cloud"]).astype(np.float32)  # (T+1, N, 6)
    object_pos = np.stack(ep["object_pos"]).astype(np.float32)
    object_quat = np.stack(ep["object_quat"]).astype(np.float32)
    reward = np.stack(ep["reward"]).astype(np.float32)
    done = np.stack(ep["done"])
    timeout = np.stack(ep["timeout"])

    returns = np.zeros(T, dtype=np.float32)
    for i in reversed(range(T)):
        returns[i] = reward[i] + GAMMA * (returns[i + 1] if i + 1 < T else 0.0)

    return {
        "state": state[:-1], "next_state": state[1:],
        "action": action, "next_action": action,
        "point_cloud": point_cloud[:-1], "next_point_cloud": point_cloud[1:],
        "object_pos": object_pos[:-1],
        "object_quat": object_quat[:-1],
        "reward": reward,
        "done": done.astype(np.uint8),
        "timeout": timeout.astype(np.uint8),
        "return": returns,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",
                        default="data/sphere_box_demos_multiball_400.zarr")
    parser.add_argument("--n-demos", type=int, default=200)
    parser.add_argument("--n-points", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--target-steps", type=int, default=200,
                        help="~episode length to aim for (calibration aid)")
    parser.add_argument("--tune", type=int, default=0,
                        help="run this many episodes and print per-phase step "
                             "counts WITHOUT saving anything")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="hard per-episode step cap (failures cut short)")
    parser.add_argument("--save-failures", action="store_true",
                        help="save failed episodes (truncated, terminal "
                             "done=1/timeout=1) to <out>_failures.zarr")
    parser.add_argument("--max-failures", type=int, default=50)
    args = parser.parse_args()

    env = SphereBoxMultiballEnv(use_image=False, n_points=args.n_points,
                                seed=args.seed, verbose=False)
    param_rng = np.random.default_rng(args.seed + 99991)

    if args.tune > 0:
        print(f"[tune] running {args.tune} episodes (no save)")
        lens = []
        for i in range(args.tune):
            seed = args.seed + i
            params = sample_expert_params(param_rng)
            ep, success, psteps, length = run_episode(
                env, seed, params=params, max_steps=args.max_steps)
            lens.append(length)
            names = [f"{PHASE_NAMES[k]}:{v}" for k, v in enumerate(psteps)
                     if k < len(PHASE_NAMES) - 1]
            print(f"  [tune {i + 1}/{args.tune}] seed={seed} len={length} "
                  f"mode={params.get('fail_mode', 'clean')} "
                  f"success={success} | " + " ".join(names))
        print(f"[tune] mean length {np.mean(lens):.1f} "
              f"(target ~{args.target_steps})")
        return

    buffer = ReplayBuffer.create_empty_numpy()
    fail_buffer = (ReplayBuffer.create_empty_numpy()
                   if args.save_failures else None)

    n_success, n_fail, attempts = 0, 0, 0
    t0 = time.time()
    while n_success < args.n_demos:
        seed = args.seed + attempts
        attempts += 1
        params = sample_expert_params(param_rng)
        print(f"[demo {n_success + 1}/{args.n_demos}] attempt {attempts} "
              f"(seed {seed})")
        ep, success, psteps, length = run_episode(
            env, seed, params=params, max_steps=args.max_steps)
        if success:
            episode_data = to_episode_data(ep, len(ep["action"]))
            buffer.add_episode(episode_data)
            n_success += 1
            print(f"    success (len {length}, "
                  f"{time.time() - t0:.1f}s elapsed)")
        else:
            n_fail += 1
            print(f"    FAILED (len {length}), retrying with a new seed")
            if fail_buffer is not None and n_fail <= args.max_failures:
                ep_t = {k: list(v[:args.max_steps + 1]) if len(v) == length + 1
                        else list(v[:args.max_steps])
                        for k, v in ep.items()}
                ep_t["done"][len(ep_t["action"]) - 1] = np.bool_(True)
                ep_t["timeout"][len(ep_t["action"]) - 1] = np.bool_(True)
                fail_data = to_episode_data(ep_t, len(ep_t["action"]))
                fail_buffer.add_episode(fail_data)
                print(f"    FAILURE #{n_fail} SAVED "
                      f"({len(ep_t['action'])} steps)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    print(f"saved {n_success} demos, {buffer.n_steps} steps -> {args.out} "
          f"({attempts} attempts, failure rate {n_fail / attempts:.1%}, "
          f"{time.time() - t0:.1f}s)")
    buffer.save_to_path(args.out)
    if fail_buffer is not None and fail_buffer.n_episodes > 0:
        root, ext = os.path.splitext(args.out)
        fail_path = f"{root}_failures{ext}"
        fail_buffer.save_to_path(fail_path)
        print(f"saved {fail_buffer.n_episodes} failures -> {fail_path}")


if __name__ == "__main__":
    main()
