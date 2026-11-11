"""SphereBoxMultiballEnv -- pick the GREEN sphere into the box among distractors.

Scene (multiball/sphere_box_multiball.xml): THREE movable spheres on the
table -- one GREEN (body 'sphere', the target) and two WHITE distractors
('sphere_d1', 'sphere_d2') -- plus the fixed open-top box. Task is otherwise
identical to ``SphereBoxEnv``: pick the GREEN sphere and place it inside the
box. Success / reward / expert all act on the green sphere only; the white
balls are physical clutter the policy must learn to ignore using the RGB
point cloud (cam_pc already renders 6-ch xyz+rgb).

Implementation: subclasses ``SphereBoxEnv`` and, after the parent caches the
single-sphere model, SWAPS in the multiball scene and re-caches every
model-derived handle. Body 'sphere' is aliased to the GREEN target so the
entire parent stack (reset -> _current_object='sphere', _get_obs object_pos,
_step_metaworld grasp detection) keeps working unchanged. No shared file in
rl_100/env/arx5/ is modified.
"""
import os

import numpy as np
import mujoco

from rl_100.env.arx5.sphere_box_env import SphereBoxEnv
from rl_100.env.arx5.ik import IKSolver

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sphere_box_multiball.xml")

# Bodies the parent caches in _object_body_ids. 'sphere' = GREEN target
# (alias keeps _current_object='sphere' working); the two distractors are
# the extra white balls.
MULTIBALL_BODIES = ("sphere", "sphere_d1", "sphere_d2")

# Table-top geoms excluded from the point cloud (same names/ids as the base
# scene; recomputed after the model swap for safety).
PC_EXCLUDE_NAMES = (
    "floor", "bg_ceiling", "bg_back", "bg_left", "bg_right",
    "table_leg1", "table_leg2", "table_leg3", "table_leg4",
    "table_top_objplane",
)


class SphereBoxMultiballEnv(SphereBoxEnv):
    # Min centre-to-centre separation between any two spawned balls (m).
    MIN_SEP = 0.10
    # How many distractor balls (beyond the green target) the scene holds.
    N_DISTRACTORS = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._swap_model(XML_PATH)

    # ------------------------------------------------------------------
    # model swap
    # ------------------------------------------------------------------
    def _swap_model(self, xml_path):
        """Replace the single-sphere model loaded by the parent init with the
        multiball scene and re-cache every model-derived handle. Only the
        object body set differs; all robot / site / actuator / joint names
        are identical, so the parent's caches are rebuilt with the same code
        the parent used."""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._pc_far = float(self.model.vis.map.zfar * self.model.stat.extent)

        self._object_body_ids = {
            name: self.model.body(name).id for name in MULTIBALL_BODIES}
        self._obj_body_id = self._object_body_ids["sphere"]  # GREEN target

        self._ee_site_id = self.model.site("ee_site").id
        self._grasp_site_id = self.model.site("grasp_site").id
        self._link6_body_id = self.model.body("link6").id
        try:
            self._finger_site_id = self.model.site("finger_site").id
        except KeyError:
            self._finger_site_id = self._grasp_site_id

        self._joint_actuator_ids = [
            self.model.actuator(f"joint{i}").id for i in range(1, 7)]
        self._gripper_actuator_id = self.model.actuator("gripper").id
        try:
            self._gripper2_actuator_id = self.model.actuator("gripper2").id
        except KeyError:
            self._gripper2_actuator_id = None

        self._joint_qpos_adrs = [
            self.model.jnt_qposadr[self.model.joint(f"joint{i}").id]
            for i in range(1, 7)]
        self._gripper_qpos_adr = self.model.jnt_qposadr[
            self.model.joint("joint7").id]
        self._gripper2_qpos_adr = self.model.jnt_qposadr[
            self.model.joint("joint8").id]
        self._joint_dof_adrs = [
            self.model.jnt_dofadr[self.model.joint(f"joint{i}").id]
            for i in range(1, 7)]
        self._joint_lows = np.array([r[0] for r in self.JOINT_RANGES])
        self._joint_highs = np.array([r[1] for r in self.JOINT_RANGES])
        self.ik = IKSolver(self) if IKSolver is not None else None

        self._cam_pc_id = self.model.camera("cam_pc").id
        self._pc_exclude_geom_ids = set()
        for i in range(self.model.ngeom):
            if self.model.geom(i).name in PC_EXCLUDE_NAMES:
                self._pc_exclude_geom_ids.add(i)
        if self.use_point_cloud:
            self._pc_renderer = mujoco.Renderer(
                self.model, height=self.pc_resolution,
                width=self.pc_resolution)

        # Let the balls settle onto the analytic table plane at rest.
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------
    # spawn: green target + white distractors, all separated + outside box
    # ------------------------------------------------------------------
    def _randomize_sphere_pose(self):
        """Place all three spheres: each sampled uniformly from the two
        reachable spawn zones, kept outside the box footprint (guaranteed by
        the zones) and >= ``MIN_SEP`` from every already-placed ball. The
        green 'sphere' is placed FIRST so it always gets a clean spot."""
        TABLE_SURFACE_Z = 0.19
        placed = []
        for name in MULTIBALL_BODIES:
            body = self.model.body(self._object_body_ids[name])
            sphere_r = self._get_object_surface_offset(name)
            qpos_start = self.model.jnt_qposadr[body.jntadr[0]]
            dof_start = body.dofadr[0]

            x = y = None
            for _ in range(1000):
                zone = self.SPAWN_ZONES[
                    int(self.np_random.randint(len(self.SPAWN_ZONES)))]
                cx = self.np_random.uniform(zone[0], zone[1])
                cy = self.np_random.uniform(zone[2], zone[3])
                if any(np.hypot(cx - p[0], cy - p[1]) < self.MIN_SEP
                       for p in placed):
                    continue
                x, y = cx, cy
                break
            if x is None:
                raise RuntimeError(
                    "multiball spawn rejection sampling failed "
                    f"(could not place {name})")

            self.data.qpos[qpos_start:qpos_start + 3] = [
                x, y, TABLE_SURFACE_Z + sphere_r + 0.0002]
            self.data.qpos[qpos_start + 3:qpos_start + 7] = [0.0, 0.0, 0.0, 1.0]
            self.data.qvel[dof_start:dof_start + 6] = 0.0
            placed.append((x, y))

        mujoco.mj_forward(self.model, self.data)
        for _ in range(8):
            for _ in range(50):
                mujoco.mj_step(self.model, self.data)
            for name in MULTIBALL_BODIES:
                dof_start = self.model.body(
                    self._object_body_ids[name]).dofadr[0]
                self.data.qvel[dof_start:dof_start + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ---- helper for gallery / diagnostics: positions of every ball ----
    def ball_positions(self):
        """dict name -> world xyz for the green target and all distractors."""
        return {name: self.data.body(self._object_body_ids[name]).xpos.copy()
                for name in MULTIBALL_BODIES}
