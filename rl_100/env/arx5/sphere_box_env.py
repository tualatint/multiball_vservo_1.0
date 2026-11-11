"""
SphereBoxEnv -- fresh single-object pick-and-place task.

Scene (sphere_box.xml): exactly ONE movable object -- a 7 cm sphere on the
table -- and a FIXED open-top box. Task: pick the sphere and place it INSIDE
the box.

Inherits the validated MetaWorld EE-space interface of ``Arx5GraspEnv``
(action_mode='metaworld': 4-dim [dx, dy, dz, gripper] realized by
differential IK + PD control), so the demos / eval / policy interface are
identical to the working DP3 pick-place experiments.

Reward structure (documented in the demo collector):
  * sparse terminal +1.0 when the sphere is INSIDE the box (``in_box``:
    centre within the box footprint and below the rim -- dropped/rolled-in
    balls count, not just clean gripper releases);
  * one-time +0.5 grasp bonus the first time the sphere is lifted off the
    table (separates approach from carry in the recorded return);
  * SPEED-NEUTRAL quantized potential-cell shaping: phi = EE->sphere xy
    distance while un-grasped, phi = sphere->box-center xy distance once
    lifted; a cell of ``PHI_CELL`` is granted ONLY when phi improves past
    the running anchor (at most ONE cell per step). A slow approach earns
    exactly the same cells as a fast one, so per-chunk reward no longer
    scales with chunk speed -- this removes the reward misalignment that
    made the IQL critic rank fast/aggressive chunks higher and BPPO knock
    the sphere off the table.
"""
import numpy as np

import mujoco

from rl_100.env.arx5.arx5_env import Arx5GraspEnv


class SphereBoxEnv(Arx5GraspEnv):
    # ---- fixed box geometry (must match sphere_box.xml) ----
    BOX_CENTER = (0.30, 0.15)          # box center xy on the table
    BOX_INNER_HALF = 0.09              # inner half-extent (walls inner face)
    BOX_HEIGHT = 0.08                  # wall height; rim at 0.19 + 0.08
    TABLE_SURFACE_Z = 0.19
    SPHERE_RADIUS = 0.035
    # Success region (half-extent around the box centre, xy). Set tighter
    # than the physical box so a slip grasp whose ball rolls into the box
    # and rests against a wall FAILS: the ball must land near the centre
    # (max possible centre offset inside the 0.09 box is 0.055 = 0.09-0.035
    # ball radius, so 0.055 only counts near-centre landings).
    SUCCESS_HALF = 0.055
    # Sphere spawn ZONES: two disjoint, fully reachable+liftable table regions
    # (verified by a grid probe across x in [0.20, 0.38]): the front/south
    # zone behind the arm (y in [-0.16, 0.00]) and the rear/north zone behind
    # the box (y in [0.32, 0.36]). Both keep the sphere OUTSIDE the box
    # footprint (box outer y in ~[0.05, 0.25]) and clear of the unusable band
    # y in [0.04, 0.28] (box footprint / unreachable). The env picks a zone
    # uniformly (50/50) then a uniform position inside it, so the demos span
    # ~2x the x-range and ~3x the y-range of the old single-box spawn.
    SPAWN_ZONES = [
        (0.20, 0.38, -0.16, 0.00),   # front/south zone (near the camera)
        (0.20, 0.38, 0.32, 0.36),    # rear/north zone (behind the box)
    ]
    # Union bounding box fed to the parent (unused for the sphere spawn
    # itself, which samples SPAWN_ZONES; kept for API compatibility).
    SPAWN_X = (0.20, 0.38)
    SPAWN_Y = (-0.16, 0.36)
    # Point-cloud workspace crop: widened in +y (0.32 -> 0.42) so the rear
    # spawn zone (y up to 0.36 + sphere radius 0.035) stays fully visible to
    # the policy at episode start (the parent default clips it at y = 0.32).
    PC_CROP_BOUNDS = (-0.08, -0.32, 0.10, 0.48, 0.42, 0.55)

    # ---- reward hyper-parameters ----
    SHAPING_GAIN = 0.1
    GRASP_REWARD = 0.5
    SUCCESS_REWARD = 1.0
    # Quantized potential-cell size (m). >= the max per-step phi improvement
    # so at most ONE cell is crossed per step and chunk reward is decoupled
    # from chunk speed (see module docstring).
    PHI_CELL = 0.03

    def __init__(self, render_size=84, max_episode_steps=500, n_points=512,
                 image_size=84, verbose=False, seed=42,
                 use_point_cloud=True, use_image=False, pc_resolution=160,
                 mw_action_scale=None, mw_max_step=None, mw_n_substeps=None,
                 mw_spawn_x=None, mw_spawn_y=None):
        if mw_spawn_x is None:
            mw_spawn_x = self.SPAWN_X
        if mw_spawn_y is None:
            mw_spawn_y = self.SPAWN_Y
        super().__init__(
            render_size=render_size,
            max_episode_steps=max_episode_steps,
            n_points=n_points,
            image_size=image_size,
            verbose=verbose,
            seed=seed,
            use_point_cloud=use_point_cloud,
            use_image=use_image,
            pc_resolution=pc_resolution,
            pc_crop_bounds=self.PC_CROP_BOUNDS,
            action_mode='metaworld',
            physics='sphere_box',
            mw_action_scale=mw_action_scale,
            mw_max_step=mw_max_step,
            mw_n_substeps=mw_n_substeps,
            mw_spawn_x=tuple(mw_spawn_x),
            mw_spawn_y=tuple(mw_spawn_y),
        )
        self.box_center = np.array(self.BOX_CENTER, dtype=np.float64)
        self.box_top_z = self.TABLE_SURFACE_Z + self.BOX_HEIGHT
        self._obj_body_id = self._object_body_ids['sphere']
        self._grasped_ever = False
        self._phi_anchor = None
        self._lifted_prev = False
        # Sanity: EVERY spawn zone must stay outside the box footprint (with
        # the sphere radius as margin), or demos could start pre-placed.
        bx, by = self.box_center
        for x0, x1, y0, y1 in self.SPAWN_ZONES:
            if (x0 < bx + self.BOX_INNER_HALF + self.SPHERE_RADIUS and
                    x1 > bx - self.BOX_INNER_HALF - self.SPHERE_RADIUS and
                    y0 < by + self.BOX_INNER_HALF + self.SPHERE_RADIUS and
                    y1 > by - self.BOX_INNER_HALF - self.SPHERE_RADIUS):
                raise ValueError(
                    f"sphere spawn zone ({x0},{y0})-({x1},{y1}) overlaps the "
                    f"target box footprint around ({bx},{by})")

    # ------------------------------------------------------------------
    # task predicates
    # ------------------------------------------------------------------
    def sphere_pos(self):
        return self.data.body(self._obj_body_id).xpos.copy()

    def in_box(self):
        """True iff the sphere center is inside the box (footprint + below
        the rim + above the table). The SAME predicate is used by the demo
        collector, the trainer eval and the final eval."""
        s = self.sphere_pos()
        bx, by = self.box_center
        inside_xy = (abs(s[0] - bx) < self.SUCCESS_HALF and
                     abs(s[1] - by) < self.SUCCESS_HALF)
        below_rim = s[2] < self.box_top_z - 0.01
        above_table = s[2] > self.TABLE_SURFACE_Z + 0.01
        return bool(inside_xy and below_rim and above_table)

    def is_lifted(self):
        """True iff the sphere is held above its rest height (grasped/carried)."""
        s = self.sphere_pos()
        return bool(s[2] > self.TABLE_SURFACE_Z + self.SPHERE_RADIUS + 0.04)

    def is_released(self):
        """True iff the gripper has (at least partly) opened -- distinguishes a
        RELEASED sphere from one still clamped between the fingers.

        Threshold 0.075 sits between the clamped width on the 7 cm sphere
        (~0.069 -- the fingers stop at the full 7 cm diameter) and the
        fully-open width (0.088)."""
        return bool(self._get_gripper_width() > 0.075)

    def placed(self):
        """Legacy predicate (info only): the sphere is inside the box AND no
        longer held by the gripper. Task SUCCESS is now just ``in_box()`` --
        the ball being inside the box -- so a ball that drops/rolls into the
        box also counts (see the collector + eval, which read info['in_box'])."""
        return bool(self.in_box() and self.is_released())

    def reset(self):
        obs = super().reset()
        self._grasped_ever = False
        self._phi_anchor = None
        self._lifted_prev = False
        return obs

    # ------------------------------------------------------------------
    # spawn
    # ------------------------------------------------------------------
    def _randomize_sphere_pose(self):
        """Diverse spawn: pick one of the two verified zones uniformly
        (50/50), then a uniform position inside it. The rest (rest on the
        analytic table plane, motion killed) is identical to the parent."""
        obj_name = 'sphere'
        body_id = self._object_body_ids[obj_name]
        body = self.model.body(body_id)

        zone = self.SPAWN_ZONES[int(self.np_random.randint(len(self.SPAWN_ZONES)))]
        x = self.np_random.uniform(zone[0], zone[1])
        y = self.np_random.uniform(zone[2], zone[3])

        TABLE_SURFACE_Z = 0.19
        sphere_r = self._get_object_surface_offset(obj_name)  # 0.035
        qpos_start = self.model.jnt_qposadr[body.jntadr[0]]
        dof_start = body.dofadr[0]
        self.data.qpos[qpos_start:qpos_start + 3] = [
            x, y, TABLE_SURFACE_Z + sphere_r + 0.0002]
        self.data.qpos[qpos_start + 3:qpos_start + 7] = [0.0, 0.0, 0.0, 1.0]
        self.data.qvel[dof_start:dof_start + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        for _ in range(8):
            for _ in range(50):
                mujoco.mj_step(self.model, self.data)
            self.data.qvel[dof_start:dof_start + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # reward
    # ------------------------------------------------------------------
    def _compute_reward(self):
        sphere = self.sphere_pos()
        ee = self.data.site(self._finger_site_id).xpos.copy()
        # SUCCESS = the sphere is inside the box (footprint + below the rim);
        # whether it was released in the box or dropped/rolled in does not
        # matter. The env ends + gives the terminal reward on this predicate.
        in_box = self.in_box()
        lifted = self.is_lifted()

        reward = 0.0
        done = False
        if in_box:
            reward += self.SUCCESS_REWARD
            done = True
            self._grasp_success = True
        elif self.step_count >= self.max_episode_steps:
            done = True
        else:
            # SPEED-NEUTRAL quantized potential-cell reward: phi is the
            # approach potential (EE->sphere xy distance) while un-grasped,
            # and the carry potential (sphere->box xy distance) once lifted.
            # A cell is granted ONLY when phi improves past the running
            # anchor, at most ONE cell per step -- so a fast chunk and a slow
            # chunk covering the same range earn the same cells, and chunk
            # reward no longer scales with chunk speed (the misalignment that
            # made the IQL critic rank fast/aggressive chunks higher and BPPO
            # knock the sphere off the table). No penalty for regress: a
            # knocked-away sphere simply earns nothing until re-approached.
            if lifted:
                phi = float(np.hypot(sphere[0] - self.box_center[0],
                                     sphere[1] - self.box_center[1]))
            else:
                phi = float(np.hypot(ee[0] - sphere[0],
                                     ee[1] - sphere[1]))
            if lifted != self._lifted_prev:
                # regime switch (lift/drop): re-anchor so the carry phase is
                # rewarded against its own start, not the approach minimum
                self._phi_anchor = phi
                self._lifted_prev = lifted
            if self._phi_anchor is None:
                self._phi_anchor = phi
            if self._phi_anchor - phi >= self.PHI_CELL:
                reward += self.SHAPING_GAIN * self.PHI_CELL
                self._phi_anchor -= self.PHI_CELL

        if lifted and not self._grasped_ever:
            reward += self.GRASP_REWARD
            self._grasped_ever = True

        info = {
            'success': bool(self._grasp_success),
            'grasp_success': bool(self._grasp_success),
            'placed': self.placed(),
            'in_box': in_box,
            'released': self.is_released(),
            'lifted': lifted,
            'dist_to_box': float(np.hypot(sphere[0] - self.box_center[0],
                                          sphere[1] - self.box_center[1])),
            'object_pos': sphere,
            'current_object': 'sphere',
        }
        return reward, done, info
