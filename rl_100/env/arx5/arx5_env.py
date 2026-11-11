"""
ARX L5 Robot Grasping Environment

A MuJoCo-based simulation environment for the ARX L5 6-DOF robot arm
with a parallel gripper, tasked with grasping geometric objects (cube, sphere, cylinder).
Uses the official MJCF model from deepmind/mujoco_menagerie.
Designed to integrate with the RL-100 framework.
"""

import os
import numpy as np
import gym
from gym import spaces

# Use osmesa for headless rendering (EGL not available on this server)
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'osmesa'
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

try:
    import mujoco
    from mujoco import MjModel, MjData
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: mujoco not available. Install with: pip install mujoco")

try:
    from rl_100.env.arx5.ik import IKSolver, rotvec_to_matrix, fk_site_pose, matrix_to_quat_wxyz
except ImportError:
    IKSolver = None
    rotvec_to_matrix = None
    fk_site_pose = None
    matrix_to_quat_wxyz = None


def _fps_downsample(points, num_points, seed=0):
    """Farthest-point sampling (numpy, CPU) preserving spatial coverage.

    points: (N, C) array with C >= 3 (the first 3 columns are used as xyz).
    Returns: (min(num_points, N), C) array.
    """
    N = points.shape[0]
    if N <= num_points:
        return points
    rng = np.random.RandomState(seed)
    # Coarse uniform pre-sample when N is large so the O(N*K) FPS loop stays fast.
    if N > num_points * 4:
        idx = rng.choice(N, num_points * 4, replace=False)
        points = points[idx]
        N = points.shape[0]
    xyz = points[..., :3].astype(np.float32)
    d = np.sum((xyz - xyz[rng.randint(N)]) ** 2, axis=1).astype(np.float32)
    sel = np.empty(num_points, dtype=np.int64)
    for i in range(num_points):
        sel[i] = int(np.argmax(d))
        dd = np.sum((xyz - xyz[sel[i]]) ** 2, axis=1)
        np.minimum(d, dd, out=d)
    return points[sel]


class Arx5GraspEnv(gym.Env):
    """
    ARX L5 Robot Grasping Environment.

    Action Space, controlled by `action_mode`:
      * 'joint' (legacy): (7,) [joint1..joint6, gripper] normalized to [-1, 1].
        Joint positions are mapped to joint ranges; gripper: -1 close, 1 open.
      * 'ee_delta': (7,) [dx, dy, dz, rotvec_x, rotvec_y, rotvec_z, gripper]
        as ACTUAL world-frame end-effector deltas (meters / radians) computed
        from the grasp_site pose:
            pos_target  = pos_cur + dpos
            R_target    = exp(drot) @ R_cur     (rotvec -> matrix)
        The joint targets that realize the desired EE delta are obtained with an
        explicit numerical IK solver (MuJoCo has no built-in IK), then tracked
        by the PD position controllers. The delta is clamped to
        (+-ee_pos_clip m, +-ee_rot_clip rad) to keep motions stable.
        Gripper: -1 = close, 1 = open (unchanged).
      * 'metaworld' (NEW): (4,) [dx, dy, dz, gripper] -- EXACTLY the MetaWorld
        Sawyer interface used by the working DP3 pick-place experiment:
          - dx/dy/dz are normalized in [-1, 1], scaled by ``mw_action_scale``
            (meters) and clipped to +- ``mw_max_step``;
          - the deltas are realized by RESOLVED-RATE (differential) IK on the
            6-DOF arm using MuJoCo's analytic site Jacobian, holding the
            end-effector (finger_site) at a fixed grasp orientation;
          - gripper follows the MetaWorld convention: +1 = close, -1 = open;
          - both fingers are driven symmetrically by two position actuators
            (no equality constraint), on the grav-compensated soft-gain robot
            model (arx5_robot_mw.xml);
          - the scene (arx5_mw.xml) holds a single 4 cm UPRIGHT cylinder on the
            table (vertical 3DoF grasping: the policy only commands xyz +
            gripper, orientation is held by the IK).

    Observation Space (Dict):
        image: (3, 84, 84) RGB image from camera
        point_cloud: (1024, 6) point cloud with xyz-rgb
        agent_pos:
          'joint'     -> (10,)  joint positions + ee_site position + gripper width
          'ee_delta'  -> (14,)  joint positions + grasp_site pos (3) +
                               grasp_site quat (4, wxyz) + gripper width
          'metaworld' -> (9,)   [ee_xyz (finger_site), left_fingertip_xyz,
                                 right_fingertip_xyz] (MetaWorld-style)
    """

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 30}

    # Joint limits (radians) from official MJCF
    JOINT_RANGES = [
        (-3.14, 3.14),   # joint1: base rotation (Z)
        (0.0, 3.14),     # joint2: shoulder (Y)
        (0.0, 3.14),     # joint3: upper arm (Y)
        (-1.7, 1.7),     # joint4: elbow (Y)
        (-1.7, 1.7),     # joint5: wrist roll (Z)
        (-3.14, 3.14),   # joint6: wrist pitch (X)
    ]

    # Gripper: joint7 is slide, 0 = closed, 0.044 = open
    # Equality constraint makes joint8 = -joint7
    GRIPPER_OPEN = 0.044
    GRIPPER_CLOSE = 0.0

    # Home position from keyframe
    HOME_JOINTS = [0.0, 0.251, 0.314, 0.0, 0.0, 0.0]

    # ---- MetaWorld-style interface ('metaworld' action mode) ----
    # Home configuration of the VALIDATED working pick-place experiment: a
    # "level_down" grasp orientation (site y-axis / jaw opening horizontal,
    # fingertip approach pointing down-forward). Top-down is NOT reachable at
    # the object area (x~0.3) for this robot.
    MW_HOME_JOINTS = [-0.0261, 1.0204, 1.1488, -0.8765, -0.0446, -0.0394]
    # Fixed grasp orientation held by the differential IK (columns = site axes).
    MW_GRASP_ROTATION = np.array([
        [0.733158, -0.033406, 0.679237],
        [0.025473, 0.999441, 0.021658],
        [-0.679581, 0.001424, 0.733599],
    ], dtype=float)
    # Normalized-action -> meters scaling and per-step clip (MetaWorld style).
    MW_ACTION_SCALE = 0.10
    MW_MAX_STEP = 0.12
    # Physics substeps per env step for the MW robot model (matches the working
    # experiment; the legacy modes use 120 on the stiff robot).
    MW_N_SUBSTEPS = 40
    # Vertical-cylinder spawn region (verified reachable + liftable with the
    # level_down orientation across x in [0.22, 0.38], z in [0.21, 0.42]).
    MW_OBJ_SPAWN_X = (0.25, 0.35)
    MW_OBJ_SPAWN_Y = (-0.12, 0.12)
    # Single upright cylinder: 4 cm diameter / height (matches the working
    # experiment's MetaWorld pick-place object).
    MW_OBJ_HALF_HEIGHT = 0.02
    MW_OBJ_RADIUS = 0.02

    # Object types (the MW scene has only 'cylinder'; the legacy XML has
    # cube/sphere/cylinder; the mw_multi scene carries all four incl. cuboid).
    OBJECT_TYPES = ['cube', 'cuboid', 'sphere', 'cylinder']
    # Cuboid half-extents along its local x/y/z axes (5x7x5 cm box).
    # The 9 cm tall variant was dropped: resting on its 5 cm face left the
    # 9 cm axis horizontal, and a 7x9 cm slab is wider than the open jaw gap
    # (9.2 cm minus pad thickness) -> the fingertip plates grazed the top
    # corners on a top-down vertical grasp and the object got shoved aside.
    # 5x7x5 keeps the max horizontal extent (7 cm) equal to the cube, which
    # is verified graspable for every rest pose.
    CUBOID_HALF_EXTENTS = (0.025, 0.035, 0.025)

    def __init__(self,
                 render_size=84,
                 max_episode_steps=500,
                 n_points=1024,
                 image_size=84,
                 verbose=False,
                 seed=42,
                 use_point_cloud=True,
                 use_image=True,
                 pc_resolution=160,
                 pc_crop_bounds=None,
                 action_mode='ee_delta',
                 physics=None,
                 ee_pos_clip=0.04,
                 ee_rot_clip=0.5,
                 mw_action_scale=None,
                 mw_max_step=None,
                 mw_n_substeps=None,
                 mw_spawn_x=None,
                 mw_spawn_y=None):
        super().__init__()

        self.render_size = render_size
        self.max_episode_steps = max_episode_steps
        self.n_points = n_points
        self.image_size = image_size
        self.verbose = verbose
        self.use_point_cloud = use_point_cloud
        # Whether to also render the RGB `image` observation. The DP3 point
        # cloud pipeline (backbone=None) does NOT use the RGB image, so it can
        # be disabled to skip the expensive RGB render and keep obs/obs_space
        # free of an unused key.
        self.use_image = use_image
        self.seed_val = seed
        self.np_random = np.random.RandomState(seed)
        # Resolution of the depth render used for the point cloud. Higher gives
        # a denser cloud (more raw points -> better object coverage after FPS).
        self.pc_resolution = int(pc_resolution)
        # Workspace box (xmin, ymin, zmin, xmax, ymax, zmax) kept after depth
        # back-projection. Captures the arm + table top + object while dropping
        # the floor / walls / background that would otherwise dominate the cloud.
        if pc_crop_bounds is None:
            # Workspace box: covers the whole object spawn region + lift, and
            # is wide enough to keep a good part of the arm visible (matching
            # the cam_pc view), while the table surface stays inside.
            pc_crop_bounds = (-0.08, -0.32, 0.10, 0.48, 0.32, 0.55)
        self.pc_crop_bounds = tuple(float(v) for v in pc_crop_bounds)

        # Action mode: 'ee_delta' (world-frame EE deltas resolved by numerical
        # IK), the legacy 'joint' (normalized joint-position targets), or
        # 'metaworld' (MetaWorld-style 4-dim EE deltas + differential IK).
        assert action_mode in ('ee_delta', 'joint', 'metaworld'), \
            f"Unknown action_mode {action_mode}"
        self.action_mode = action_mode
        self.is_metaworld = (action_mode == 'metaworld')
        # Physics/scene selection, DECOUPLED from the action interface so the
        # ablations can mix them (joint interface on MW physics = A01, MW EE
        # interface on the legacy XML = A02, MW EE + legacy object set = A03).
        assert physics in (None, 'mw', 'legacy', 'mw_multi', 'sphere_box'), \
            f"Unknown physics {physics}"
        self.physics = ('mw' if self.is_metaworld else 'legacy') if physics is None else physics
        # The v1-style TASK (single upright cylinder vertical grasp, MW home
        # pose + MW spawn box) is used by the MW interface and by any run on
        # the MW scene; the legacy multi-object task otherwise. 'mw_multi'
        # keeps the MW interface/home/task semantics but swaps in the legacy
        # 3-object set + per-object random poses (A03).
        self.mw_task = self.is_metaworld or (self.physics in ('mw', 'mw_multi'))
        # Half-height of the task cylinder in the loaded scene (MW XML: 0.02,
        # legacy XML cylinder: 0.035) -- used for spawning + lift thresholds.
        self._cylinder_half_height = self.MW_OBJ_HALF_HEIGHT if self.physics == 'mw' else 0.035
        # Per-step clamps applied to the EE delta (m / rad) at the env boundary.
        self.ee_pos_clip = float(ee_pos_clip)
        self.ee_rot_clip = float(ee_rot_clip)
        # MetaWorld-style action scaling / clipping / substeps (overridable).
        self.mw_action_scale = self.MW_ACTION_SCALE if mw_action_scale is None else float(mw_action_scale)
        self.mw_max_step = self.MW_MAX_STEP if mw_max_step is None else float(mw_max_step)
        self.mw_n_substeps = self.MW_N_SUBSTEPS if mw_n_substeps is None else int(mw_n_substeps)
        # Vertical-cylinder spawn region override (v2 diverse datasets use a
        # wider box than MW_OBJ_SPAWN_X/Y; defaults keep the v1 behaviour).
        self.mw_spawn_x = (self.MW_OBJ_SPAWN_X if mw_spawn_x is None
                           else tuple(float(v) for v in mw_spawn_x))
        self.mw_spawn_y = (self.MW_OBJ_SPAWN_Y if mw_spawn_y is None
                           else tuple(float(v) for v in mw_spawn_y))
        self.mw_grasp_rot = self.MW_GRASP_ROTATION.copy()
        # Optional per-episode orientation HOLD for the differential IK
        # (None -> the fixed MW_GRASP_ROTATION). Set by external scripts
        # (e.g. the shake probe) so the scripted phase CONTINUES the exact
        # orientation the policy handed off with instead of forcing the fixed
        # MW grasp orientation onto an arbitrary policy-final pose, which
        # would inject a large orientation-correction joint command at the
        # handoff instant and yank the gripped object loose.
        self._mw_hold_rot = None
        # Optional per-physics-substep callback (set by external tools, e.g. a
        # video renderer that wants frames between control steps). Signature:
        # hook() is called right after each mujoco.mj_step inside _step_metaworld.
        self._mw_step_hook = None
        # NOTE: self.ik is created after the joint qpos addresses are cached
        # (IKSolver reads env._joint_qpos_adrs / env.JOINT_RANGES).
        self.ik = None

        self.step_count = 0
        self._current_object = None
        self._grasp_success = False
        self._prev_object_pos = None

        # Load MuJoCo model. The 'mw' physics uses its own scene (single
        # upright cylinder) and physics-tuned robot (gravcomp + soft gains +
        # symmetric dual gripper actuators); 'legacy' uses the original.
        # Selected by the `physics` kwarg, independent of the action mode.
        if self.physics == 'sphere_box':
            xml_name = 'sphere_box.xml'
        elif self.physics == 'mw':
            xml_name = 'arx5_mw.xml'
        elif self.physics == 'mw_multi':
            xml_name = 'arx5_mw_multi.xml'
        else:
            xml_name = 'arx5.xml'
        xml_path = os.path.join(os.path.dirname(__file__), xml_name)
        self.model = MjModel.from_xml_path(xml_path)
        self.data = MjData(self.model)

        # MuJoCo far plane (zfar*extent): background pixels render at this value
        # exactly, so filtering depth < far drops the background.
        self._pc_far = float(self.model.vis.map.zfar * self.model.stat.extent)

        # Action space:
        #   legacy modes: 6 joint positions + 1 gripper = 7, normalized [-1, 1]
        #   metaworld:    [dx, dy, dz, gripper] = 4, normalized [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(4,) if self.is_metaworld else (7,),
            dtype=np.float32
        )

        # Observation space
        if self.is_metaworld:
            agent_pos_dim = 9
        else:
            agent_pos_dim = 14 if action_mode == 'ee_delta' else 10
        obs_space = {
            'agent_pos': spaces.Box(low=-np.inf, high=np.inf, shape=(agent_pos_dim,), dtype=np.float32),
        }
        if use_image:
            obs_space['image'] = spaces.Box(
                low=0, high=1, shape=(3, image_size, image_size), dtype=np.float32)
        if use_point_cloud:
            obs_space['point_cloud'] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_points, 6), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_space)

        # Cache body IDs (guard each name: the MW scene only has the
        # single cylinder body; the legacy XML has the original triple).
        if self.physics == 'sphere_box':
            # Fresh sphere-in-box task: the scene holds exactly one movable
            # object (the sphere) + the fixed target box.
            self.OBJECT_TYPES = ['sphere']
        self._object_body_ids = {}
        for name in self.OBJECT_TYPES:
            try:
                self._object_body_ids[name] = self.model.body(name).id
            except KeyError:
                if self.physics == 'mw' and name != 'cylinder':
                    continue  # MW scene has only the upright cylinder
                if name == 'cuboid' and self.physics == 'legacy':
                    continue  # legacy XML has no cuboid
                raise
        self._ee_site_id = self.model.site('ee_site').id
        self._grasp_site_id = self.model.site('grasp_site').id
        self._link6_body_id = self.model.body('link6').id
        # MetaWorld-style EE reference: the fingertip midpoint site (driving it
        # to the object centre puts the fingertips exactly on the object).
        try:
            self._finger_site_id = self.model.site('finger_site').id
        except KeyError:
            self._finger_site_id = self._grasp_site_id

        # Cache actuator IDs
        self._joint_actuator_ids = [
            self.model.actuator(f'joint{i}').id for i in range(1, 7)
        ]
        self._gripper_actuator_id = self.model.actuator('gripper').id
        # The metaworld robot drives BOTH fingers with their own actuators.
        try:
            self._gripper2_actuator_id = self.model.actuator('gripper2').id
        except KeyError:
            self._gripper2_actuator_id = None

        # Cache joint qpos addresses
        self._joint_qpos_adrs = []
        for i in range(1, 7):
            jnt_id = self.model.joint(f'joint{i}').id
            self._joint_qpos_adrs.append(self.model.jnt_qposadr[jnt_id])
        self._gripper_qpos_adr = self.model.jnt_qposadr[self.model.joint('joint7').id]
        self._gripper2_qpos_adr = self.model.jnt_qposadr[self.model.joint('joint8').id]

        # Numerical IK solver for EE-delta control (after joint caches exist).
        self.ik = IKSolver(self) if IKSolver is not None else None

        # Cache joint dof addresses (for velocity control)
        self._joint_dof_adrs = []
        for i in range(1, 7):
            jnt_id = self.model.joint(f'joint{i}').id
            self._joint_dof_adrs.append(self.model.jnt_dofadr[jnt_id])
        # Joint limit arrays (metaworld differential IK).
        self._joint_lows = np.array([r[0] for r in self.JOINT_RANGES])
        self._joint_highs = np.array([r[1] for r in self.JOINT_RANGES])

        # Initialize renderers for offscreen rendering.
        # The Renderer may not be available in all environments (e.g., headless server).
        self._renderer = None
        self._pc_renderer = None
        self._cam_pc_id = self.model.camera('cam_pc').id
        # Geoms to EXCLUDE from the point cloud (DP3-style segmentation mask):
        # drop the table / floor / background room so the cloud contains only
        # the robot arm + graspable objects (raises object+robot SNR, matches
        # the real DP3 preprocessing).
        self._pc_exclude_geom_ids = set()
        for i in range(self.model.ngeom):
            if self.model.geom(i).name in (
                    'floor', 'bg_ceiling', 'bg_back', 'bg_left', 'bg_right',
                    'table_leg1', 'table_leg2', 'table_leg3', 'table_leg4',
                    'table_top_objplane'):
                self._pc_exclude_geom_ids.add(i)
        try:
            if use_image:
                self._renderer = mujoco.Renderer(self.model, height=image_size, width=image_size)
                self._render_camera = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(self.model, self._render_camera)
                # Pose picked with scripts/pick_camera_pose.py (data/picked_cam.json),
                # slightly raised so the arm is more visible.
                self._render_camera.azimuth = 129.4
                self._render_camera.elevation = -48.1
                self._render_camera.distance = 0.794
                self._render_camera.lookat = np.array([0.2278, -0.0307, 0.2763])
            # Separate renderer for the depth-based point cloud (fixed cam_pc).
            self._pc_renderer = mujoco.Renderer(
                self.model, height=self.pc_resolution, width=self.pc_resolution)
        except Exception as e:
            if verbose:
                print(f"Warning: Renderer not available ({e}). Using dummy image.")
            self._renderer = None
            self._pc_renderer = None

        # Do initial simulation steps to stabilize
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

    def seed(self, seed=None):
        if seed is not None:
            self.seed_val = seed
            self.np_random = np.random.RandomState(seed)
        return [self.seed_val]

    def _get_ee_pos(self):
        """Get end-effector position in world coordinates."""
        return self.data.site(self._ee_site_id).xpos.copy()

    def _get_gripper_width(self):
        """Get current gripper opening width (distance between fingers)."""
        # joint7 range is [0, 0.044], joint8 = -joint7
        # When joint7 = 0.044, joint8 = -0.044 => width = 0.088
        left_pos = self.data.joint('joint7').qpos[0]
        right_pos = self.data.joint('joint8').qpos[0]
        return left_pos - right_pos  # positive = open

    def _get_obs(self):
        """Get observation dictionary."""
        obs = {}

        if self.use_image:
            # Render image
            rgb = self._render()
            obs['image'] = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

        # Get joint positions
        joint_pos = np.array([self.data.joint(f'joint{i}').qpos[0] for i in range(1, 7)])

        # Get gripper width
        gripper_width = self._get_gripper_width()

        if self.is_metaworld:
            # MetaWorld-style 9-dim agent state: EE (finger_site) position +
            # both fingertip positions. No joints, no gripper width -- exactly
            # what the working DP3 metaworld/arx5 pick-place policy consumed.
            ee = self.data.site(self._finger_site_id).xpos.copy()
            lf = self.data.site('left_touch_site').xpos.copy()
            rf = self.data.site('right_touch_site').xpos.copy()
            agent_pos = np.concatenate([ee, lf, rf]).astype(np.float32)
        elif self.action_mode == 'ee_delta':
            # EE pose observed at the grasp_site (the point the policy commands
            # EE deltas around). Quaternion in (w,x,y,z) convention.
            g_pos = self.data.site(self._grasp_site_id).xpos.copy()
            g_xmat = self.data.site(self._grasp_site_id).xmat.reshape(3, 3)
            g_quat = matrix_to_quat_wxyz(g_xmat)
            agent_pos = np.concatenate([
                joint_pos, g_pos, g_quat, [gripper_width],
            ]).astype(np.float32)
        else:
            # Get end-effector position (legacy ee_site)
            ee_pos = self._get_ee_pos()
            agent_pos = np.concatenate([
                joint_pos,
                ee_pos,
                [gripper_width],
            ]).astype(np.float32)

        obs['agent_pos'] = agent_pos

        if self.use_point_cloud:
            point_cloud = self._render_point_cloud()
            obs['point_cloud'] = point_cloud

        return obs

    def _render(self):
        """Render RGB image from camera using MuJoCo Renderer."""
        if self._renderer is not None:
            # The camera is FIXED (az/el/dist/lookat set in __init__) so the
            # viewpoint never moves between or within episodes.
            self._renderer.update_scene(self.data, self._render_camera)
            rgb = self._renderer.render()
            return rgb.copy()
        # Fallback: return a dummy gray image when renderer is unavailable
        h, w = self.image_size, self.image_size
        return np.full((h, w, 3), 128, dtype=np.uint8)

    def _render_point_cloud(self):
        """Render a REAL depth-based point cloud from the fixed cam_pc camera.

        Mirrors the DP3 pipeline: render depth -> back-project with the camera
        intrinsics/extrinsics -> crop to the workspace box -> FPS downsample to
        n_points. Returns (n_points, 6) float32 [xyz + rgb(0-255)].
        """
        if self._pc_renderer is None:
            return np.zeros((self.n_points, 6), dtype=np.float32)

        cam_id = self._cam_pc_id

        # 1) Render depth (linear meters in mujoco 3.x) + RGB for colors.
        self._pc_renderer.enable_depth_rendering()
        self._pc_renderer.update_scene(self.data, camera=cam_id)
        depth = self._pc_renderer.render().astype(np.float32)
        self._pc_renderer.disable_depth_rendering()
        self._pc_renderer.update_scene(self.data, camera=cam_id)
        rgb = self._pc_renderer.render()

        # 1b) Segmentation mask: keep robot + object + table top, drop the
        #     floor / walls / table legs / invisible contact plane.
        self._pc_renderer.enable_segmentation_rendering()
        self._pc_renderer.update_scene(self.data, camera=cam_id)
        seg = self._pc_renderer.render().astype(np.int32)  # [...,0]=geom id, -1=bg
        self._pc_renderer.disable_segmentation_rendering()
        seg_keep = (seg[..., 0] >= 0).astype(np.float32)
        for gid in self._pc_exclude_geom_ids:
            seg_keep = seg_keep * (seg[..., 0] != gid).astype(np.float32)
        # Mark pixels on the table top so we can keep a sparse table subset.
        is_table = (seg[..., 0] == self.model.geom('table_top').id)

        H, W = depth.shape
        f = H / (2.0 * np.tan(np.radians(self.model.cam_fovy[cam_id]) / 2.0))
        cx, cy = W / 2.0, H / 2.0

        # 2) Back-project pixels to camera-frame points (z = depth, meters).
        v, u = np.mgrid[0:H, 0:W]
        x = (u - cx) * depth / f
        y = (v - cy) * depth / f
        pts_cam = np.stack([x, y, depth], axis=-1).reshape(-1, 3)

        # Drop background (far plane) / invalid pixels / excluded geoms.
        mask = (depth > 0.001) & (depth < self._pc_far)
        mask = mask * (seg_keep > 0.5)
        mask = mask.reshape(-1)
        if not np.any(mask):
            return np.zeros((self.n_points, 6), dtype=np.float32)
        pts_cam = pts_cam[mask]

        # 3) Camera -> world. MuJoCo cameras look along local -z; the 180 deg
        #    x-rotation aligns the (x-right, y-down, z-forward) back-projected
        #    frame to the MuJoCo camera frame before applying cam_xmat.
        R_x180 = np.array([[1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0],
                           [0.0, 0.0, -1.0]], dtype=np.float32)
        R_cw = self.data.cam_xmat[cam_id].reshape(3, 3).astype(np.float32) @ R_x180
        t_cw = self.data.cam_xpos[cam_id].astype(np.float32)
        pts_world = pts_cam @ R_cw.T + t_cw

        # 4) Crop to the workspace box (keeps arm + table + object, drops
        #    floor/walls/background).
        xmin, ymin, zmin, xmax, ymax, zmax = self.pc_crop_bounds
        keep = (pts_world[:, 0] >= xmin) & (pts_world[:, 0] <= xmax) & \
               (pts_world[:, 1] >= ymin) & (pts_world[:, 1] <= ymax) & \
               (pts_world[:, 2] >= zmin) & (pts_world[:, 2] <= zmax)
        pts_world = pts_world[keep]
        is_table_crop = is_table.reshape(-1)[mask][keep]

        colors = rgb.reshape(-1, 3)[mask][keep].astype(np.float32)

        if len(pts_world) == 0:
            return np.zeros((self.n_points, 6), dtype=np.float32)

        # 5) Stratified FPS downsample to exactly n_points: keep a sparse
        #    table subset (~10%) and fill the rest with robot + object points
        #    (deterministic-ish seeds so the sample is stable across identical
        #    observations).
        pts = np.concatenate([pts_world, colors], axis=-1)
        table_pts = pts[is_table_crop]
        obj_pts = pts[~is_table_crop]
        n_table = int(round(0.10 * self.n_points))
        if len(table_pts) > n_table:
            table_pts = _fps_downsample(table_pts, n_table, seed=1)
        n_obj = self.n_points - len(table_pts)
        if len(obj_pts) > n_obj:
            obj_pts = _fps_downsample(obj_pts, n_obj, seed=0)
        pts = np.concatenate([obj_pts, table_pts], axis=0)

        if len(pts) < self.n_points:
            pad = np.zeros((self.n_points - len(pts), 6), dtype=np.float32)
            pts = np.concatenate([pts, pad], axis=0)
        return pts.astype(np.float32)

    def _get_object_surface_offset(self, obj_name):
        """Get the height offset from table surface to object center."""
        TABLE_SURFACE_Z = 0.19  # table body at z=0.15, top geom half-size 0.04
        if self.physics == 'mw':
            # single upright 4 cm cylinder (half-height 0.02)
            return self.MW_OBJ_HALF_HEIGHT
        if obj_name == 'cube':
            return 0.035  # half-size of cube
        elif obj_name == 'sphere':
            return 0.035  # radius of sphere
        elif obj_name == 'cylinder':
            return 0.035  # half-height of cylinder
        elif obj_name == 'cuboid':
            # A settled cuboid rests on one face: the vertical half-extent is
            # the half-extent along whichever local axis is vertical. Read the
            # current orientation (set during spawn/settle) to get it exactly.
            quat = self.data.body(self._object_body_ids['cuboid']).xquat.copy()
            w, x, y, z = quat
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ])
            ax = int(np.argmax(np.abs(R[:, 2])))
            return float(self.CUBOID_HALF_EXTENTS[ax])
        return 0.035

    def _randomize_object_pose_mw(self):
        """MetaWorld-mode object spawn: single UPRIGHT cylinder.

        Only the xy position is randomized (vertical 3DoF task - the cylinder
        is rotationally symmetric and always stands on its end cap), inside the
        region verified reachable + liftable with the fixed level_down grasp
        orientation. The object is placed exactly touching the table and any
        residual motion is killed so episodes start at rest.
        """
        obj_name = 'cylinder'
        body_id = self._object_body_ids[obj_name]
        body = self.model.body(body_id)

        TABLE_SURFACE_Z = 0.19
        x = self.np_random.uniform(*self.mw_spawn_x)
        y = self.np_random.uniform(*self.mw_spawn_y)

        qpos_start = self.model.jnt_qposadr[body.jntadr[0]]
        dof_start = body.dofadr[0]

        # upright (rotationally symmetric -> identity orientation is a rest pose)
        self.data.qpos[qpos_start:qpos_start + 3] = [
            x, y, TABLE_SURFACE_Z + self._cylinder_half_height + 0.0002]
        self.data.qpos[qpos_start + 3:qpos_start + 7] = [0.0, 0.0, 0.0, 1.0]
        self.data.qvel[dof_start:dof_start + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        for _ in range(8):
            for _ in range(50):
                mujoco.mj_step(self.model, self.data)
            self.data.qvel[dof_start:dof_start + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _randomize_sphere_pose(self):
        """Sphere-in-box task: random xy spawn on the table (NEVER inside the
        fixed box footprint), sphere at rest, motion killed.

        The sphere is rotationally symmetric -> identity orientation is a rest
        pose. The spawn box ``mw_spawn_x/mw_spawn_y`` must be disjoint from the
        box footprint (checked in the demo collector / sanity test).
        """
        obj_name = 'sphere'
        body_id = self._object_body_ids[obj_name]
        body = self.model.body(body_id)

        TABLE_SURFACE_Z = 0.19
        sphere_r = self._get_object_surface_offset(obj_name)  # 0.035
        x = self.np_random.uniform(*self.mw_spawn_x)
        y = self.np_random.uniform(*self.mw_spawn_y)

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

    def _randomize_object_pose(self, obj_name):
        """Randomize the pose of a graspable object on the table."""
        body_id = self._object_body_ids[obj_name]
        body = self.model.body(body_id)

        TABLE_SURFACE_Z = 0.19  # table top surface (body z=0.15 + top half-size 0.04)
        obj_offset = self._get_object_surface_offset(obj_name)
        object_radius = obj_offset  # approximate radius for reachability

        # Random position in the verified reachable + liftable region.
        # The robot base is at (0,0). With the pedestal mount (base z=0.15) and
        # a vertical (top-down) approach, the arm reliably reaches AND lifts
        # objects across x in [0.24, 0.36], y in [-0.16, 0.16].
        #
        # x is capped at 0.36 (not 0.40): beyond ~0.375 the lift reachability
        # is marginal (max grasp-site z leaves < 4cm of object-center margin
        # above the 0.305 success threshold), so the demo controller only
        # succeeds ~13%/attempt there and slips. Keeping the spawn region
        # inside the reliably-liftable area keeps the demonstration pose
        # distribution equal to the eval-env pose distribution (both draw from
        # this same function) and lets a BC policy actually learn the task.
        # MW-scene experiments (mw / mw_multi) always draw from the
        # mw_spawn_x/y override (default: the verified vertical-grasp
        # region); only the legacy physics keeps its original wider box.
        if self.physics == 'legacy':
            x = self.np_random.uniform(0.24, 0.36)
            y = self.np_random.uniform(-0.16, 0.16)
        else:
            x = self.np_random.uniform(*self.mw_spawn_x)
            y = self.np_random.uniform(*self.mw_spawn_y)

        qpos_start = self.model.jnt_qposadr[body.jntadr[0]]

        # Random orientation.
        #  * sphere: rotationally symmetric -> stays upright.
        #  * cylinder: EXPLICIT diversity - ~50% VERTICAL (standing on its end
        #    cap, random yaw) and ~50% HORIZONTAL (lying on its side, random
        #    yaw). This guarantees the dataset contains BOTH grasp types.
        #  * cube: FULLY RANDOM 3D orientation (uniform over SO(3), Shoemake's
        #    method) so it can rest on any face with any twist.
        # The object then physically settles onto its stable rest pose.
        if obj_name == 'sphere':
            quat = np.array([0.0, 0.0, 0.0, 1.0])
        elif obj_name == 'cylinder':
            yaw = self.np_random.uniform(0.0, 2 * np.pi)
            if self.np_random.random() < 0.5:
                # vertical: long axis (local +z) stays vertical (yaw about
                # world +z -- proper z-rotation quat, scalar first)
                quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            else:
                # horizontal: tip the long axis to horizontal (90 deg about
                # world +y) then yaw about world +z
                s2 = np.sqrt(0.5)
                q_yaw = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
                q_tilt = np.array([s2, 0.0, s2, 0.0])  # 90 deg about +y
                w1, x1, y1, z1 = q_yaw
                w2, x2, y2, z2 = q_tilt
                quat = np.array([
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
                    w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
                ])
        else:
            u1, u2, u3 = self.np_random.uniform(0.0, 1.0, 3)
            quat = np.array([
                np.sqrt(1.0 - u1) * np.sin(2 * np.pi * u2),
                np.sqrt(1.0 - u1) * np.cos(2 * np.pi * u2),
                np.sqrt(u1) * np.sin(2 * np.pi * u3),
                np.sqrt(u1) * np.cos(2 * np.pi * u3),
            ])
        # Normalize (numerical safety) and enforce quat w>=0 so the
        # recorded orientation is unambiguous.
        quat /= np.linalg.norm(quat)
        if quat[3] < 0:
            quat = -quat

        R = np.array([
            [1 - 2 * (quat[2]**2 + quat[3]**2), 2 * (quat[1]*quat[2] - quat[0]*quat[3]), 2 * (quat[1]*quat[3] + quat[0]*quat[2])],
            [2 * (quat[1]*quat[2] + quat[0]*quat[3]), 1 - 2 * (quat[1]**2 + quat[3]**2), 2 * (quat[2]*quat[3] - quat[0]*quat[1])],
            [2 * (quat[1]*quat[3] - quat[0]*quat[2]), 2 * (quat[2]*quat[3] + quat[0]*quat[1]), 1 - 2 * (quat[1]**2 + quat[2]**2)],
        ])

        if obj_name in ('cube', 'cuboid'):
            # Snap the random SO(3) orientation to its natural REST pose:
            # rotate minimally so the lowest face lies flat on the table
            # (yaw preserved). A settled box always rests on some face; a
            # corner/edge-down spawn instead topples and bounces — visible
            # as object oscillation with no external contact. All faces and
            # yaw/twist angles remain covered -> diversity preserved.
            # lowest-pointing local axis = column with min world-z
            half = (np.array([0.035, 0.035, 0.035]) if obj_name == 'cube'
                    else np.array(self.CUBOID_HALF_EXTENTS, dtype=float))
            j = int(np.argmin([R[0, 2], R[1, 2], R[2, 2]]))
            u = R[:, j].copy()
            s = -1.0 if u[2] > 0 else 1.0  # make it point DOWN
            u *= s / np.linalg.norm(u)
            t = np.array([0.0, 0.0, -1.0])
            v_axis = np.cross(u, t)
            c = float(np.clip(np.dot(u, t), -1.0, 1.0))
            if np.linalg.norm(v_axis) < 1e-9:
                Q = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
            else:
                v_axis /= np.linalg.norm(v_axis)
                K = np.array([[0, -v_axis[2], v_axis[1]],
                              [v_axis[2], 0, -v_axis[0]],
                              [-v_axis[1], v_axis[0], 0]])
                Q = np.eye(3) + K + K @ K * ((1 - c) / max(1e-12, np.linalg.norm(v_axis) ** 2))
            R = Q @ R
            # re-normalize orientation via SVD (Q is orthonormal, R stays SO(3))
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            q2 = np.empty(4)
            mujoco.mju_mat2Quat(q2, R.reshape(-1))
            quat = q2
            if quat[3] < 0:
                quat = -quat
            snap_axis = j

        self.data.qpos[qpos_start + 3:qpos_start + 7] = quat

        if obj_name in ('cube', 'cuboid'):
            z_half_extent = half[snap_axis]  # half-extent along the resting axis
        elif obj_name == 'cylinder':
            hh, r = 0.035, 0.028  # half-height, radius (matches arx5.xml)
            az = np.abs(R[2, 2])  # world-z component of the cylinder axis
            z_half_extent = hh * az + r * np.sqrt(max(0.0, 1.0 - az * az))
        else:  # sphere: rotationally symmetric
            z_half_extent = 0.035

        # Spawn clearance: every spawn pose is now a stable REST pose
        # (sphere exact; cylinder vertical or side-lying exact; cube face-
        # down after the snap). Place them just touching the table — any
        # drop lets the contact impulse inject bounce/spin that condim=3
        # contacts cannot dissipate (objects were seen oscillating with no
        # external contact).
        CLEARANCE = 0.0002

        # Spawn clearance: cylinder (vertical OR side-lying) and sphere are
        # constructed directly IN a stable rest pose, so place them exactly
        # touching the table — any drop lets the contact impulse inject spin,
        # and with condim=3 there is no rolling friction to ever dissipate it
        # (objects were seen rolling/oscillating with no external contact).
        # A fully-random-SO(3) box is NOT at rest and must topple onto a
        # face, so give it a small drop; the topple dissipates the energy.
        if obj_name in ('cube', 'cuboid'):
            CLEARANCE = 0.002
        else:
            CLEARANCE = 0.0002
        z = TABLE_SURFACE_Z + z_half_extent + CLEARANCE

        # Set position in MuJoCo
        self.data.qpos[qpos_start:qpos_start + 3] = [x, y, z]

        # Reset velocity (body.dofadr gives the starting DOF index for free joints)
        dof_start = body.dofadr[0]
        self.data.qvel[dof_start:dof_start + 6] = 0

        # Settle, then repeatedly kill residual motion so episodes never
        # start with a drifting / micro-bouncing / slowly-rolling object.
        mujoco.mj_forward(self.model, self.data)
        for _ in range(8):
            for _ in range(50):
                mujoco.mj_step(self.model, self.data)
            self.data.qvel[dof_start:dof_start + 6] = 0
        # Final forward to get accurate positions
        mujoco.mj_forward(self.model, self.data)

    def _activate_object(self, obj_name):
        """Activate one object and deactivate others (contact AND visibility).

        Inactive objects are made invisible (rgba alpha=0) as well as
        contact-disabled, so exactly one object is ever present in the scene
        regardless of camera angle or parked position.
        """
        for name in self.OBJECT_TYPES:
            body_id = self._object_body_ids[name]
            for geom_id in range(self.model.ngeom):
                geom_body = self.model.geom_bodyid[geom_id]
                if geom_body == body_id:
                    if name == obj_name:
                        self.model.geom_contype[geom_id] = 1
                        self.model.geom_conaffinity[geom_id] = 1
                        self.model.geom_rgba[geom_id][3] = 1.0
                    else:
                        self.model.geom_contype[geom_id] = 0
                        self.model.geom_conaffinity[geom_id] = 0
                        self.model.geom_rgba[geom_id][3] = 0.0

    def _move_to_joint_pos(self, target_joints):
        """Move joints to target positions using position control."""
        for i in range(6):
            self.data.ctrl[self._joint_actuator_ids[i]] = target_joints[i]

    def _set_gripper(self, open_gripper):
        """Set gripper position. 0 = closed, 0.044 = open."""
        target = self.GRIPPER_OPEN if open_gripper else self.GRIPPER_CLOSE
        # Use the actuator control to close the gripper with strong force
        self.data.ctrl[self._gripper_actuator_id] = target

    def _is_grasping(self):
        """Check if the gripper is grasping an object."""
        width = self._get_gripper_width()
        gripper_cmd = self.data.ctrl[self._gripper_actuator_id]
        # If commanded to close significantly (target < 50% of max open) but fingers are apart
        if gripper_cmd < self.GRIPPER_OPEN * 0.5:  # trying to close
            if width > 0.01:  # but fingers are apart (object in between)
                return True
        return False

    def _check_grasp_success(self, obj_name):
        """Check if the object has been lifted above the table."""
        body_id = self._object_body_ids[obj_name]
        obj_pos = self.data.body(body_id).xpos.copy()
        # Object lifted above table surface + lift threshold (0.10m)
        # Table surface is at z=0.19, object center offset is ~0.035
        # So object center is normally at ~0.225. Lifted means > 0.33
        TABLE_SURFACE_Z = 0.19
        obj_offset = self._get_object_surface_offset(obj_name)
        lift_threshold = TABLE_SURFACE_Z + obj_offset + 0.08
        return obj_pos[2] > lift_threshold

    def reset(self):
        """Reset the environment."""
        self.step_count = 0

        # Reset all qpos to zero
        self.data.qpos[:] = 0.0

        if self.mw_task:
            # Set home joint positions (level_down grasp orientation home)
            for i in range(6):
                self.data.qpos[self._joint_qpos_adrs[i]] = self.MW_HOME_JOINTS[i]
            # Set gripper to open (both fingers symmetric)
            self.data.qpos[self._gripper_qpos_adr] = self.GRIPPER_OPEN
            self.data.qpos[self._gripper2_qpos_adr] = -self.GRIPPER_OPEN
            self.data.qvel[:] = 0.0

            if self.physics == 'mw_multi':
                # A03: MW interface + legacy 3-object set. Hold the arm at
                # home during the (long) object settle, park all objects,
                # pick one at random, then reuse the legacy per-object spawn
                # (cube full SO(3) + snap-to-face, cylinder 50/50 vertical/
                # horizontal, sphere upright).
                for i in range(6):
                    self.data.ctrl[self._joint_actuator_ids[i]] = self.MW_HOME_JOINTS[i]
                self.data.ctrl[self._gripper_actuator_id] = self.GRIPPER_OPEN
                if self._gripper2_actuator_id is not None:
                    self.data.ctrl[self._gripper2_actuator_id] = -self.GRIPPER_OPEN
                for obj_name in self.OBJECT_TYPES:
                    body_id = self._object_body_ids[obj_name]
                    qpos_start = self.model.jnt_qposadr[self.model.body(body_id).jntadr[0]]
                    self.data.qpos[qpos_start:qpos_start + 3] = [10.0, 10.0, -10.0]
                    self.data.qpos[qpos_start + 3:qpos_start + 7] = [0, 0, 0, 1]
                mujoco.mj_forward(self.model, self.data)
                # _object_allowlist (set by demo generators for targeted
                # top-up shards) restricts the CHOICE only; parking above
                # still covers every OBJECT_TYPES entry.
                pool = getattr(self, '_object_allowlist', None) or self.OBJECT_TYPES
                self._current_object = self.np_random.choice(pool)
                self._activate_object(self._current_object)
                self._randomize_object_pose(self._current_object)
            else:
                # Single object: the upright cylinder (always active). On the
                # legacy XML the cube/sphere bodies also exist -> park them far
                # away and disable their contacts/visibility (A02 runs the MW
                # task on legacy physics).
                self._current_object = 'cylinder'
                if self.physics == 'sphere_box':
                    # Fresh sphere-in-box task: only the sphere exists; spawn
                    # it at a random reachable table position (never inside
                    # the fixed box footprint).
                    self._current_object = 'sphere'
                elif self.physics == 'legacy':
                    for obj_name in self.OBJECT_TYPES:
                        if obj_name == 'cylinder':
                            continue
                        body_id = self._object_body_ids[obj_name]
                        qpos_start = self.model.jnt_qposadr[self.model.body(body_id).jntadr[0]]
                        self.data.qpos[qpos_start:qpos_start + 3] = [10.0, 10.0, -10.0]
                        self.data.qpos[qpos_start + 3:qpos_start + 7] = [0, 0, 0, 1]
                    self._activate_object('cylinder')
                mujoco.mj_forward(self.model, self.data)
                if self.physics == 'sphere_box':
                    self._randomize_sphere_pose()
                else:
                    self._randomize_object_pose_mw()

            # Hold the arm at home (gripper open) during settling so it does
            # not collapse when ctrl defaults to 0.
            for i in range(6):
                self.data.ctrl[self._joint_actuator_ids[i]] = self.MW_HOME_JOINTS[i]
            self.data.ctrl[self._gripper_actuator_id] = self.GRIPPER_OPEN
            if self._gripper2_actuator_id is not None:
                self.data.ctrl[self._gripper2_actuator_id] = -self.GRIPPER_OPEN

            # Step to stabilize
            for _ in range(100):
                mujoco.mj_step(self.model, self.data)

            self._grasp_success = False
            self._prev_object_pos = self.data.body(
                self._object_body_ids[self._current_object]).xpos.copy()
            return self._get_obs()

        # Set home joint positions
        for i in range(6):
            self.data.qpos[self._joint_qpos_adrs[i]] = self.HOME_JOINTS[i]

        # Set gripper to open
        self.data.qpos[self._gripper_qpos_adr] = self.GRIPPER_OPEN
        self.data.qpos[self._gripper2_qpos_adr] = -self.GRIPPER_OPEN

        # Reset all object positions (far away/off-screen initially)
        for obj_name in self.OBJECT_TYPES:
            body_id = self._object_body_ids[obj_name]
            qpos_start = self.model.jnt_qposadr[self.model.body(body_id).jntadr[0]]
            # Place far below the floor so they don't interfere
            self.data.qpos[qpos_start:qpos_start + 3] = [10.0, 10.0, -10.0]
            self.data.qpos[qpos_start + 3:qpos_start + 7] = [0, 0, 0, 1]

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Select and activate a random object
        pool = getattr(self, '_object_allowlist', None) or self.OBJECT_TYPES
        self._current_object = self.np_random.choice(pool)
        self._activate_object(self._current_object)
        self._randomize_object_pose(self._current_object)

        # Apply control for home position
        self._move_to_joint_pos(self.HOME_JOINTS)
        self._set_gripper(True)

        # Step to stabilize
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        self._grasp_success = False
        self._prev_object_pos = self.data.body(self._object_body_ids[self._current_object]).xpos.copy()

        return self._get_obs()

    def step(self, action):
        """
        Execute action using PD control.

        The PD controller tracks the smooth trajectory for physics-based
        gripper-object interaction. During the approach phase, the controller
        sets qpos directly (bypassing step()) to ensure perfect trajectory
        tracking. During the close and lift phases, step() uses PD control
        so that gripper-object physics works correctly.

        action: (7,)
          'joint' mode: normalized to [-1, 1]
              [0:6]: joint positions (mapped to joint ranges)
              [6]: gripper (-1 close, 1 open)
          'ee_delta' mode: ACTUAL world-frame EE deltas (already de-normalized)
              [0:3]: pos delta (m), [3:6]: rotvec delta (rad), [6]: gripper
        """
        if self.action_mode == 'ee_delta':
            return self._step_ee_delta(action)
        if self.action_mode == 'metaworld':
            return self._step_metaworld(action)
        return self._step_joint(action)

    def _step_joint(self, action):
        self.step_count += 1

        # Map normalized actions to joint positions
        target_joints = np.zeros(6)
        for i in range(6):
            low, high = self.JOINT_RANGES[i]
            target_joints[i] = low + (action[i] + 1.0) / 2.0 * (high - low)
            target_joints[i] = np.clip(target_joints[i], low, high)

        # Send position targets to the PD controllers
        for i in range(6):
            self.data.ctrl[self._joint_actuator_ids[i]] = target_joints[i]

        # Gripper: PD control
        gripper_target = (action[6] + 1.0) / 2.0 * (self.GRIPPER_OPEN - self.GRIPPER_CLOSE) + self.GRIPPER_CLOSE
        gripper_target = np.clip(gripper_target, self.GRIPPER_CLOSE, self.GRIPPER_OPEN)
        self.data.ctrl[self._gripper_actuator_id] = gripper_target
        # MW robot: both fingers have their own actuator (no equality
        # coupling) -> drive the second finger symmetrically.
        if self._gripper2_actuator_id is not None:
            self.data.ctrl[self._gripper2_actuator_id] = -gripper_target

        # Step simulation — PD controller converges to target
        # High substep count ensures smooth physics and accurate tracking
        for _ in range(120):
            mujoco.mj_step(self.model, self.data)
            if self._mw_step_hook is not None:
                self._mw_step_hook()

        # Update derived quantities (sensor, site xpos, etc.)
        mujoco.mj_forward(self.model, self.data)

        # Get observation
        obs = self._get_obs()

        # Compute reward
        reward, done, info = self._compute_reward()

        return obs, reward, done, info

    def _step_ee_delta(self, action):
        """
        Execute an end-effector delta command via FK + numerical IK + PD.

        action (7,) contains ACTUAL (de-normalized) world-frame deltas:
            [0:3] pos delta in meters, [3:6] rotvec delta in radians,
            [6]   gripper command in [-1, 1] (-1 close, +1 open).
        The delta is clamped to (+-ee_pos_clip, +-ee_rot_clip), the desired
        grasp_site pose is FK-computed, and the joint targets are obtained with
        the numerical IKSolver (MuJoCo exposes no built-in IK), then tracked by
        the PD position controllers over 120 substeps.
        """
        assert self.ik is not None, "IKSolver unavailable; cannot run ee_delta mode"
        self.step_count += 1

        a = np.asarray(action, dtype=np.float64).reshape(7)
        dpos = np.clip(a[0:3], -self.ee_pos_clip, self.ee_pos_clip)
        drot = np.clip(a[3:6], -self.ee_rot_clip, self.ee_rot_clip)

        current_joints = np.array([
            self.data.joint(f'joint{i}').qpos[0] for i in range(1, 7)
        ], dtype=np.float64)

        # Current grasp_site pose (saved/restored by fk_site_pose).
        pos_cur, rot_cur = fk_site_pose(self, current_joints, self._grasp_site_id)

        # Desired EE pose = current + delta (world frame).
        pos_target = pos_cur + dpos
        rot_target = rotvec_to_matrix(drot) @ rot_cur

        # Solve IK for joint targets, seeding from the current configuration so
        # the solution stays on the continuous motion branch.
        joint_targets = self.ik.solve(
            pos_target, target_mat=rot_target,
            guess=current_joints, use_grasp_site=True,
        )
        if joint_targets is None:
            # IK failed (target out of reach / singularity): hold position.
            joint_targets = current_joints
            if self.verbose:
                print("[Arx5GraspEnv] ee_delta IK failed; holding joint position")

        # Track the target with the PD position controllers.
        for i in range(6):
            self.data.ctrl[self._joint_actuator_ids[i]] = joint_targets[i]
        gripper_target = (a[6] + 1.0) / 2.0 * (self.GRIPPER_OPEN - self.GRIPPER_CLOSE) + self.GRIPPER_CLOSE
        gripper_target = np.clip(gripper_target, self.GRIPPER_CLOSE, self.GRIPPER_OPEN)
        self.data.ctrl[self._gripper_actuator_id] = gripper_target

        for _ in range(120):
            mujoco.mj_step(self.model, self.data)

        # Update derived quantities (sensor, site xpos, etc.)
        mujoco.mj_forward(self.model, self.data)

        # Get observation
        obs = self._get_obs()

        # Compute reward
        reward, done, info = self._compute_reward()

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # MetaWorld-style interface ('metaworld' action mode)
    # ------------------------------------------------------------------

    def _mw_get_arm_joints(self):
        return np.array([
            self.data.qpos[adr] for adr in self._joint_qpos_adrs
        ], dtype=np.float64)

    def _mw_site_pose(self, q=None):
        """finger_site world position + rotation matrix for given arm joints.

        The sim state is saved/restored so repeated FK queries never corrupt
        the live environment."""
        d, m = self.data, self.model
        if q is None:
            pos = d.site(self._finger_site_id).xpos.copy()
            mat = d.site(self._finger_site_id).xmat.reshape(3, 3).copy()
            return pos, mat
        saved = d.qpos.copy()
        for i in range(6):
            d.qpos[self._joint_qpos_adrs[i]] = q[i]
        mujoco.mj_forward(m, d)
        pos = d.site(self._finger_site_id).xpos.copy()
        mat = d.site(self._finger_site_id).xmat.reshape(3, 3).copy()
        d.qpos[:] = saved
        mujoco.mj_forward(m, d)
        return pos, mat

    @staticmethod
    def _mw_rotvec(R):
        """World-frame axis-angle of rotation matrix R."""
        cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(cos_t)
        if theta < 1e-6:
            return np.zeros(3)
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        n = np.linalg.norm(axis)
        if n < 1e-9:
            return np.zeros(3)
        return axis / n * theta

    def _mw_analytic_pose_jacobian(self, q):
        """6x6 analytic Jacobian of the finger_site pose (pos + rotvec) using
        MuJoCo's exact mj_jacSite (much faster and more accurate than the
        finite-difference Jacobian used by the legacy ee_delta mode)."""
        d, m = self.data, self.model
        saved = d.qpos.copy()
        for i in range(6):
            d.qpos[self._joint_qpos_adrs[i]] = q[i]
        mujoco.mj_forward(m, d)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self._finger_site_id)
        dofs = np.array(self._joint_dof_adrs)
        J = np.vstack([jacp[:, dofs], jacr[:, dofs]])
        d.qpos[:] = saved
        mujoco.mj_forward(m, d)
        return J

    def _mw_diff_ik(self, d_pos):
        """Resolved-rate (differential) IK: joint targets for an EE position
        delta while HOLDING the finger_site at the grasp orientation.

        The held orientation is ``_mw_hold_rot`` when set (per-episode, so a
        scripted probe can continue the policy's final orientation), and the
        fixed ``mw_grasp_rot`` otherwise. Exactly the working pick-place
        formulation:
            err = [d_pos, rotvec(R_grasp @ R_cur.T)]
            dq  = pinv(J) @ err  (clipped to +-0.3 rad)
        """
        grasp_rot = (self.mw_grasp_rot if self._mw_hold_rot is None
                     else self._mw_hold_rot)
        q = self._mw_get_arm_joints()
        cur_pos, cur_mat = self._mw_site_pose(q)
        err = np.concatenate([d_pos, self._mw_rotvec(grasp_rot @ cur_mat.T)])
        J = self._mw_analytic_pose_jacobian(q)
        dq = np.linalg.pinv(J) @ err
        # Per-step joint-delta clip (rad): 0.6 (doubled from 0.3) so the arm
        # can track the faster EE commands (mw_max_step 0.12 -> ~0.09-0.12 m
        # of EE travel per env step with the raised PD gains). Still bounded
        # to keep the PD stable near singularities / during contact.
        dq = np.clip(dq, -0.6, 0.6)
        q_new = np.clip(q + dq, self._joint_lows, self._joint_highs)
        return q_new

    def _mw_gripper_target(self, gripper_cmd):
        """MetaWorld convention: +1 = close, -1 = open.
        ARX5 gripper: joint7 = 0.044 -> open, joint7 = 0.0 -> closed."""
        t = self.GRIPPER_OPEN + (gripper_cmd + 1.0) / 2.0 * \
            (self.GRIPPER_CLOSE - self.GRIPPER_OPEN)
        return float(np.clip(t, self.GRIPPER_CLOSE, self.GRIPPER_OPEN))

    def _step_metaworld(self, action):
        """Execute a MetaWorld-style 4-dim action: [dx, dy, dz, gripper].

        action (4,) normalized to [-1, 1]:
            [0:3] EE (finger_site) position deltas, scaled by
                  ``mw_action_scale`` (m) and clipped to +- ``mw_max_step``;
            [3]   gripper command: +1 = close, -1 = open (MetaWorld).

        The deltas are realized by resolved-rate differential IK (holding the
        fixed grasp orientation) and tracked by the PD position controllers,
        exactly like the validated working pick-place environment. Both
        gripper fingers are driven symmetrically.
        """
        self.step_count += 1

        a = np.asarray(action, dtype=np.float64).reshape(4)
        d_pos = np.clip(a[0:3], -1.0, 1.0) * self.mw_action_scale
        d_pos = np.clip(d_pos, -self.mw_max_step, self.mw_max_step)
        gripper_cmd = float(np.clip(a[3], -1.0, 1.0))

        # Differential IK -> joint targets, then PD tracking.
        target_q = self._mw_diff_ik(d_pos)
        for i in range(6):
            self.data.ctrl[self._joint_actuator_ids[i]] = target_q[i]

        gripper_target = self._mw_gripper_target(gripper_cmd)
        self.data.ctrl[self._gripper_actuator_id] = gripper_target
        if self._gripper2_actuator_id is not None:
            self.data.ctrl[self._gripper2_actuator_id] = -gripper_target

        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.mw_n_substeps):
            mujoco.mj_step(self.model, self.data)
            if self._mw_step_hook is not None:
                self._mw_step_hook()
        mujoco.mj_forward(self.model, self.data)

        # Get observation
        obs = self._get_obs()

        # Compute reward
        reward, done, info = self._compute_reward()

        return obs, reward, done, info

    def _compute_reward(self):
        """Compute reward for the grasping task.

        The robot approaches the object from above, closes the gripper,
        and lifts it. Sparse reward: +1.0 when the object is lifted
        above the table (z > table surface + offset + 0.08m).
        """
        obj_name = self._current_object
        body_id = self._object_body_ids[obj_name]
        obj_pos = self.data.body(body_id).xpos.copy()
        ee_pos = self._get_ee_pos()

        # Distance from end-effector to object (xy plane)
        dist_to_obj = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
        # Height difference
        height_diff = ee_pos[2] - obj_pos[2]

        # Check if object has been lifted
        object_lifted = self._check_grasp_success(obj_name)

        reward = 0.0
        done = False
        info = {}

        # Main reward: object lifted above table
        if object_lifted:
            reward = 1.0
            done = True
            self._grasp_success = True
        elif self.step_count >= self.max_episode_steps:
            done = True
        elif self.step_count > 10:
            # Shaping reward: encourage being near the object
            if dist_to_obj < 0.05 and height_diff < 0.05:
                reward += 0.1

        info['success'] = self._grasp_success
        info['grasp_success'] = self._grasp_success
        info['object_lifted'] = object_lifted
        info['dist_to_object'] = dist_to_obj
        info['current_object'] = obj_name
        info['object_pos'] = obj_pos

        return reward, done, info

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self._render()
        return None

    def close(self):
        if hasattr(self, '_renderer') and self._renderer is not None:
            self._renderer.close()
        if hasattr(self, '_pc_renderer') and self._pc_renderer is not None:
            self._pc_renderer.close()

    def get_state(self):
        """Get state for replay buffer."""
        return self.data.qpos.copy()

    def set_state(self, state):
        """Set state from replay buffer."""
        self.data.qpos[:] = state
        mujoco.mj_forward(self.model, self.data)

    def get_contact_info(self):
        """Get contact information between gripper and objects."""
        contacts = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            try:
                g1 = self.model.geom(contact.geom1).name if contact.geom1 >= 0 else 'world'
                g2 = self.model.geom(contact.geom2).name if contact.geom2 >= 0 else 'world'
            except:
                g1 = f'geom{contact.geom1}'
                g2 = f'geom{contact.geom2}'
            if g1 and g2:
                contacts.append((g1, g2))
        return contacts


if __name__ == '__main__':
    # Quick test
    env = Arx5GraspEnv(verbose=True)
    obs = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"Image shape: {obs['image'].shape}")
    print(f"Agent pos shape: {obs['agent_pos'].shape}")
    print(f"Agent pos: {obs['agent_pos']}")
    if 'point_cloud' in obs:
        print(f"Point cloud shape: {obs['point_cloud'].shape}")

    done = False
    total_reward = 0
    step = 0
    while not done and step < 50:
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        print(f"Step {step}: reward={reward:.3f}, done={done}, info={info}")

    print(f"Total reward: {total_reward:.3f}")
    print("Environment test passed!")