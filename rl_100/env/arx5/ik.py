"""Shared forward/inverse kinematics utilities for the ARX L5 arm.

MuJoCo does NOT provide a built-in IK solver (mj_IK does not exist in the
python bindings), so we implement an explicit numerical-Jakobian damped
least-squares IK here. The demo generator, the closed-loop env (EE-delta
action execution) and the dataset conversion tool all use this module.

Conventions
-----------
* The controlled point is the `grasp_site` (finger midpoint) - the same
  point the scripted demo controller targets.
* Forward kinematics returns (position, rotation-matrix) for a given joint
  vector, WITHOUT permanently disturbing the sim state (state is saved and
  restored around every FK evaluation).
* Rotation deltas are expressed as world-frame axis-angle vectors (rotvec):
        R_next = exp(delta) @ R_cur   (delta = rotvec(R_next @ R_cur.T))
"""

import os
import numpy as np

if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'osmesa'
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import mujoco

try:
    from scipy.spatial.transform import Rotation
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ----------------------------------------------------------------------------
# Small rotation helpers (scipy if available, else manual Rodrigues)
# ----------------------------------------------------------------------------

def rotvec_to_matrix(v):
    """3-vector axis-angle -> 3x3 rotation matrix (world frame)."""
    v = np.asarray(v, dtype=float)
    if _HAVE_SCIPY:
        return Rotation.from_rotvec(v).as_matrix()
    # manual Rodrigues
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def matrix_to_rotvec(R):
    """3x3 rotation matrix -> world-frame axis-angle vector."""
    R = np.asarray(R, dtype=float)
    if _HAVE_SCIPY:
        return Rotation.from_matrix(R).as_rotvec()
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = np.arccos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3)
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz])
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return np.zeros(3)
    return axis / axis_norm * theta


def ee_delta_between(R_cur, R_next):
    """World-frame axis-angle delta taking R_cur to R_next.

    delta = rotvec(R_next @ R_cur.T)  such that R_next = exp(delta) @ R_cur.
    """
    return matrix_to_rotvec(R_next @ R_cur.T)


def matrix_to_quat_wxyz(R):
    """3x3 rotation matrix -> unit quaternion (w, x, y, z) [MuJoCo convention].

    MuJoCo's python `site` view exposes only xmat/xpos (no xquat), so convert
    explicitly with scipy (returns xyzw, rolled to wxyz).
    """
    R = np.asarray(R, dtype=float)
    if _HAVE_SCIPY:
        q = Rotation.from_matrix(R).as_quat()  # (x, y, z, w)
        return np.roll(q, 1)
    # fallback: trace-based Shepperd's method
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


# ----------------------------------------------------------------------------
# Forward kinematics
# ----------------------------------------------------------------------------

def _save_state(env):
    return (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy())


def _restore_state(env, saved):
    env.data.qpos[:] = saved[0]
    env.data.qvel[:] = saved[1]
    env.data.ctrl[:] = saved[2]
    mujoco.mj_forward(env.model, env.data)


def fk_site_pose(env, joints, site_id, restore=True):
    """Forward kinematics: grasp-site pose for a joint vector.

    Returns (pos(3), rot(3,3)). The sim state is saved before and (optionally)
    restored after, so repeated FK queries never corrupt the live env.
    """
    saved = _save_state(env)
    for i in range(6):
        env.data.qpos[env._joint_qpos_adrs[i]] = joints[i]
    mujoco.mj_forward(env.model, env.data)
    pos = env.data.site(site_id).xpos.copy()
    rot = env.data.site(site_id).xmat.reshape(3, 3).copy()
    if restore:
        _restore_state(env, saved)
    return pos, rot


def fk_site_quat(env, joints, site_id, restore=True):
    """Forward kinematics returning the site quaternion (wxyz) as well."""
    saved = _save_state(env)
    for i in range(6):
        env.data.qpos[env._joint_qpos_adrs[i]] = joints[i]
    mujoco.mj_forward(env.model, env.data)
    pos = env.data.site(site_id).xpos.copy()
    rot = env.data.site(site_id).xmat.reshape(3, 3).copy()
    quat = matrix_to_quat_wxyz(rot)
    if restore:
        _restore_state(env, saved)
    return pos, rot, quat


# ----------------------------------------------------------------------------
# Inverse kinematics (numerical Jacobian, damped least squares)
# ----------------------------------------------------------------------------

class IKSolver:
    """Inverse kinematics for the ARX L5 arm (6-DOF), targeting the grasp_site.

    Uses a numerical Jacobian with damped least squares. Returns the solution
    closest to the initial guess to prevent joint-space jumps between steps.
    """

    def __init__(self, env, damping=0.01, max_iters=800, pos_tol=0.002, rot_tol=0.08):
        self.env = env
        self.damping = damping
        self.max_iters = max_iters
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self._joint_qpos_adrs = env._joint_qpos_adrs
        self._joint_ranges = env.JOINT_RANGES

    def _get_site_pos(self, q, site_id):
        for i in range(6):
            self.env.data.qpos[self._joint_qpos_adrs[i]] = q[i]
        mujoco.mj_forward(self.env.model, self.env.data)
        return self.env.data.site(site_id).xpos.copy()

    def _get_site_rot(self, q, site_id):
        for i in range(6):
            self.env.data.qpos[self._joint_qpos_adrs[i]] = q[i]
        mujoco.mj_forward(self.env.model, self.env.data)
        return self.env.data.site(site_id).xmat.reshape(3, 3).copy()

    def _numerical_jacobian(self, q, site_id, eps=1e-4, with_rot=False):
        """Numerical Jacobian of the grasp_site pose (position [+ rotation])."""
        if not with_rot:
            J = np.zeros((3, 6))
            pos0 = self._get_site_pos(q, site_id)
            for i in range(6):
                q_plus = q.copy()
                q_plus[i] += eps
                low, high = self._joint_ranges[i]
                q_plus[i] = np.clip(q_plus[i], low, high)
                pos_plus = self._get_site_pos(q_plus, site_id)
                J[:, i] = (pos_plus - pos0) / eps
            return J

        J = np.zeros((6, 6))
        pos0 = self._get_site_pos(q, site_id)
        R0 = self._get_site_rot(q, site_id)
        for i in range(6):
            q_plus = q.copy()
            q_plus[i] += eps
            low, high = self._joint_ranges[i]
            q_plus[i] = np.clip(q_plus[i], low, high)
            pos_plus = self._get_site_pos(q_plus, site_id)
            J[:3, i] = (pos_plus - pos0) / eps
            R_plus = self._get_site_rot(q_plus, site_id)
            J[3:, i] = self._rotation_error(R0, R_plus) / eps
        return J

    @staticmethod
    def _rotation_error(R_desired, R_actual):
        R_err = R_actual @ R_desired.T
        cos_theta = (np.trace(R_err) - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        if theta < 1e-6:
            return np.zeros(3)
        rx = R_err[2, 1] - R_err[1, 2]
        ry = R_err[0, 2] - R_err[2, 0]
        rz = R_err[1, 0] - R_err[0, 1]
        axis = np.array([rx, ry, rz])
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            return np.zeros(3)
        return axis / axis_norm * theta

    def solve(self, target_pos, target_mat=None, guess=None,
              use_grasp_site=True, max_iters=None, max_restarts=5):
        """Solve IK. Targets the grasp_site (finger midpoint) by default.

        Returns the solution closest to the initial guess to prevent
        joint-space jumps. The sim state is fully saved/restored.
        """
        m = self.env.model
        d = self.env.data

        if max_iters is None:
            max_iters = self.max_iters

        saved_qpos = d.qpos.copy()
        saved_qvel = d.qvel.copy()
        saved_ctrl = d.ctrl.copy()

        site_id = self.env._grasp_site_id if use_grasp_site else self.env._ee_site_id

        if guess is not None:
            initial_guess = np.array(guess, dtype=float)
        else:
            initial_guess = np.array([d.joint(f'joint{i+1}').qpos[0] for i in range(6)])

        q = initial_guess.copy()
        best_q = q.copy()
        best_error = np.inf
        best_dist_to_guess = np.inf

        for restart in range(max_restarts):
            if restart > 0:
                for i in range(6):
                    low, high = self.env.JOINT_RANGES[i]
                    margin = (high - low) * 0.1
                    q[i] = np.random.uniform(low + margin, high - margin)

            for iteration in range(max_iters):
                site_pos = self._get_site_pos(q, site_id)
                pos_error = target_pos - site_pos
                pos_err_norm = np.linalg.norm(pos_error)

                if target_mat is not None:
                    site_rot = self._get_site_rot(q, site_id)
                    rot_error = self._rotation_error(site_rot, target_mat)
                    rot_err_norm = np.linalg.norm(rot_error)
                    error = np.concatenate([pos_error, rot_error])
                    err_norm = np.linalg.norm(error)
                    J = self._numerical_jacobian(q, site_id, with_rot=True)
                else:
                    rot_err_norm = 0.0
                    error = pos_error
                    err_norm = pos_err_norm
                    J = self._numerical_jacobian(q, site_id, with_rot=False)

                dist_to_guess = np.linalg.norm(q - initial_guess)
                if err_norm < best_error - 0.001:
                    best_error = err_norm
                    best_q = q.copy()
                    best_dist_to_guess = dist_to_guess
                elif err_norm < best_error + 0.001 and dist_to_guess < best_dist_to_guess:
                    best_error = err_norm
                    best_q = q.copy()
                    best_dist_to_guess = dist_to_guess

                if pos_err_norm < self.pos_tol and rot_err_norm < self.rot_tol:
                    _restore_state(self.env, (saved_qpos, saved_qvel, saved_ctrl))
                    return q

                n_dims = J.shape[0]
                JJt = J @ J.T
                lambda_sq = self.damping ** 2
                try:
                    dq = J.T @ np.linalg.solve(JJt + lambda_sq * np.eye(n_dims), error)
                except np.linalg.LinAlgError:
                    dq = np.linalg.lstsq(J, error, rcond=1e-4)[0]

                dq_norm = np.linalg.norm(dq)
                if dq_norm > 0.2:
                    dq = dq * 0.2 / dq_norm
                q += dq
                for i in range(6):
                    low, high = self.env.JOINT_RANGES[i]
                    q[i] = np.clip(q[i], low, high)

        _restore_state(self.env, (saved_qpos, saved_qvel, saved_ctrl))
        return best_q
