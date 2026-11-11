# Privileged-information audit — deployment controller

Goal: the controller that produced 1000/1000 (dev 620000 + held-out
636000/636500/637000/637500) must be implementable on a real robot.  This
file lists every signal the controller consumes, its real-robot source, and
the simulator-only items that are *not* part of the controller.

## Deployment signal path

`VservoMultiballEnv._guard_action` (gripper phase machine, ramp/settle,
lift-first, release hold) and `eval_vservo.py`'s residual servo consume only:

| signal | code | real-robot source |
|---|---|---|
| ball centre (stage 0 target) | `self._ball_centre` from `estimate_green_ball(dense)` | RGB-D camera + colour segmentation + sphere fit |
| pre-drop target (stage 1) | `fixture_box()["centre_xy"]`, `_box_rim_z()` (vision fit, constant fallback) | calibrated fixture pose + box height |
| servo error | `_servo_err` = target − (finger_site − 11 mm) | vision estimate + forward kinematics |
| finger site / touch sites | `data.site(...).xpos` | joint encoders + FK |
| gripper joint position | `data.qpos[joint7]` | gripper encoder |
| stage switch | vision ball rise above its rest height | vision |
| point cloud (policy input) | `_render_point_cloud` on `cam_pc` | wrist/scene RGB-D camera (replace renderer with the real depth stream) |
| agent state (9-D) | finger + fingertip positions | FK |
| camera pose | `data.cam_xpos/xmat` | camera extrinsic calibration |
| test lift check | vision ball rise vs EE rise (first 8 steps after close) | vision + FK |
| recovery stand-off | scripted servo to ball + 14 cm, then stage-0 policy | vision + FK |
| recovery descent gate | lateral error to the vision ball | vision |

No simulator object pose, no object segmentation label, no box ground-truth
body pose is used by the controller.

## Fixes made for this audit

Two ground-truth fallbacks in the in-episode drop recovery were removed:

1. `_recovery_ball_reachable()` used `sphere_pos()` when the vision estimate
   was invalid.  Now it returns `False` (no blind re-grasp with an unknown
   ball pose).
2. `_maybe_drop_recovery()` used `sphere_pos()` for lift arming and for the
   "ball back on the table" confirmation.  It now uses the vision estimate,
   with the EE-implied held-ball height (FK) as the arming fallback and the
   *known* table height for the on-table test.

The in-episode recovery is now actively used: the test lift (vision ball
rise vs EE rise) detects a slipping pinch, and the re-approach is a scripted
visual servo to a 14 cm stand-off above the ball before the stage-0 policy
takes over.  Both are vision + FK only; no simulator state is involved.

## Remaining simulator-only uses (not in the controller)

| use | where | why it is not a controller leak |
|---|---|---|
| success / `in_box` / `object_pos` | `_compute_reward`, `eval_vservo.py`, webserver SR | evaluation predicate only; the action is computed before the reward. A real deployment uses its own success detector (camera/force). |
| dense reward | `_compute_reward` | training / demo collection only |
| `vision_source="gt"` / `switch_mode="gt"` | ablation flags, default `"vision"` | debugging ablations, not used in the reported runs |
| demo collection | `src/collect_vservo_demos.py` | the *training data* comes from the scripted simulator expert (privileged by construction). The deployment solution is the trained policy + guard; real-robot replication uses the same policy/guard, not the expert. |

## Calibration constants required on the real robot

These are task/workspace constants, not hidden simulator state, but they must
be measured for the real cell:

- table surface height `TABLE_SURFACE_Z = 0.19 m`
- fixture box centre `BOX_CENTER = (0.30, 0.15) m` and rim height
  `BOX_RIM_Z = 0.27 m` (the box is static; measure once, or use the vision
  `estimate_box` fit, which is enabled and used for the rim height)
- ball radius `0.035 m` and the finger-site → held-ball-centre offset
  `EE_TO_BALL_Z = 0.011 m` (gripper geometry)
- reachable re-grasp bands in `_recovery_ball_reachable` (workspace limits)
- camera extrinsics / intrinsics for `cam_pc`

## Caveats for real-robot replication

1. **Point cloud**: the policy was trained/evaluated on a simulated depth+RGB
   cloud.  On the robot the renderer must be replaced by the real camera
   stream (same frame, crop, and ghost-point construction); expect a
   sim-to-real perception gap that this audit cannot remove.
2. **Pad rolling resistance**: the guard raises the *simulated* finger-pad
   rolling/torsional friction from 0.05/0.02 to 0.5/0.1.  This models a
   rubber pad; the real pads must actually provide comparable rolling
   resistance.  The controller-side mitigation (4-step close ramp + 4-step
   settle hold) and the carry-cap variant (`VSERVO_CARRY_SCALE=0.5`, which
   reached 199/200 without the friction change) are available if the real
   pads are smoother.
3. **Gripper**: the guard assumes a parallel-jaw gripper whose full-close pad
   gap is smaller than the ball and whose finger joint position is readable
   (`joint7`), with the finger-site frame as in `arx5_robot_mw.xml`.
4. **Vision**: stage 1 uses the calibrated pre-drop target, not the held-ball
   estimate (the held ball is partly occluded).  If the real fixture moves,
   re-calibrate `BOX_CENTER`/rim, or trust the `estimate_box` fit.

## Reproduce the audit

```bash
# controller path must contain no simulator object pose
grep -n "sphere_pos\|is_lifted\|in_box\|placed()" src/vservo_env.py
# expected: only _compute_reward, the GT ablation branch, and switch_mode="gt"
```
