#!/usr/bin/env python
"""Smoke-test VservoMultiballEnv: virtual targets, ghost points, dense reward.

Runs the incumbent scripted expert for closed-loop episodes and compares the
vision-estimated targets with the simulator's ground truth (GT is used ONLY
for this log; the env itself never reads it in the default "vision" mode).

Run:
  cd multiball_vservo_2026
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python src/check_vservo_env.py \
      --n 8 --seed-base 500000
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INC = os.environ.get("MULTIBALL_INC",
                       os.path.join(ROOT, "third_party", "multiball"))
sys.path.insert(0, INC)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from collect_demos_multiball import (  # noqa: E402
    LIFT, ScriptedPolicy, sample_expert_params)
from vservo_env import VservoMultiballEnv  # noqa: E402
from vision_pose import BOX_CENTER, BOX_RIM_Z, CARRY_SITE_Z, EE_TO_BALL_Z  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=500000)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--pc-resolution", type=int, default=160)
    ap.add_argument("--mode", default="clean",
                    help="expert failure mode (clean/offcenter/...)")
    ap.add_argument("--vision-source", default="vision",
                    choices=["vision", "gt"])
    ap.add_argument("--switch-mode", default="vision",
                    choices=["vision", "gt"])
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    env = VservoMultiballEnv(
        max_episode_steps=args.max_steps, n_points=args.n_points,
        pc_resolution=args.pc_resolution, seed=args.seed_base,
        vision_source=args.vision_source, switch_mode=args.switch_mode)
    rng = np.random.default_rng(args.seed_base + 991)

    ball_xy = {0: [], 1: []}
    ball_z = {0: [], 1: []}
    box_xy, rim_z = [], []
    switch_delay, rows = [], []
    for i in range(args.n):
        seed = int(args.seed_base + i)
        env.seed(seed)
        obs = env.reset()
        params = sample_expert_params(rng, mode=(args.mode or None))
        policy = ScriptedPolicy(params)
        info = {}
        forced = False
        gt_lift_step = None
        switch_step = None
        step = 0
        valid = 0
        while step < args.max_steps:
            action = policy.act(env, env.sphere_pos())
            if (not forced) and policy.phase >= LIFT and env.stage == 0:
                env.set_stage(1)
                obs = env._get_obs()
                forced = True
                switch_step = step
            obs, _r, done, info = env.step(action)
            step += 1
            if info.get("vision_valid"):
                valid += 1
            if gt_lift_step is None and info.get("lifted"):
                gt_lift_step = step
            # vision-vs-GT target accuracy (vision mode only).  Compare only
            # frames where the estimator is valid and the GT ball is inside
            # the table workspace; a knocked-off ball falls metres below the
            # table and is not a meaningful accuracy sample.
            gt = env.sphere_pos()
            ball_valid = bool(env.vision_info.get("ball_valid"))
            if (env._ball_centre is not None and ball_valid
                    and 0.18 < gt[2] < 0.50):
                st = int(env.stage)
                ball_xy[st].append(float(np.hypot(
                    env._ball_centre[0] - gt[0],
                    env._ball_centre[1] - gt[1])))
                ball_z[st].append(float(env._ball_centre[2] - gt[2]))
            if env._box_est is not None:
                box_xy.append(float(np.linalg.norm(
                    env._box_est["centre_xy"] - BOX_CENTER)))
                rim_z.append(float(env._box_est["rim_z"] - BOX_RIM_Z))
            if args.debug:
                print(f"    step {step:3d} phase={policy.phase} "
                      f"stage={env.stage} gt={np.round(gt, 4)} "
                      f"est={None if env._ball_centre is None else np.round(env._ball_centre, 4)} "
                      f"valid={ball_valid} err={np.round(np.asarray(info.get('vision_err', [0,0,0])) * 1e3, 1)}mm "
                      f"r={float(_r):+.4f}")
            if done or policy.phase == 5:  # env done or expert DONE
                break
        if switch_step is not None and gt_lift_step is not None:
            switch_delay.append(switch_step - gt_lift_step)
        rows.append({
            "seed": seed, "mode": params.get("fail_mode"),
            "success": bool(info.get("in_box")), "steps": step,
            "stage": env.stage, "vision_valid": valid,
            "vision_frames": info.get("vision_frames"),
            "gt_lift_step": gt_lift_step, "switch_step": switch_step,
            "final_err": info.get("vision_err"),
        })
        print(f"[{i + 1}/{args.n}] seed={seed} mode={params.get('fail_mode')} "
              f"steps={step} success={rows[-1]['success']} "
              f"stage={env.stage} switch={switch_step} gt_lift={gt_lift_step} "
              f"valid={valid}/{info.get('vision_frames')} "
              f"err={np.round(info.get('vision_err', [0, 0, 0]), 4)}")

    n = len(rows)
    k = sum(r["success"] for r in rows)
    print("\n=== summary ===")
    print(f"success {k}/{n} = {k / max(n, 1):.3f}")
    for st in (0, 1):
        if ball_xy[st]:
            print(f"ball vision-vs-GT stage {st} xy  "
                  f"RMS {np.sqrt(np.mean(np.square(ball_xy[st]))) * 1e3:.2f} mm  "
                  f"max {np.max(ball_xy[st]) * 1e3:.2f} mm  "
                  f"(n={len(ball_xy[st])})")
            print(f"ball vision-vs-GT stage {st} z   "
                  f"mean {np.mean(ball_z[st]) * 1e3:+.2f} mm  "
                  f"RMS {np.sqrt(np.mean(np.square(ball_z[st]))) * 1e3:.2f} mm")
    if box_xy:
        print(f"box vision-vs-GT centre {np.mean(box_xy) * 1e3:.2f} mm  "
              f"rim {np.mean(rim_z) * 1e3:+.2f} mm  (n={len(box_xy)})")
    if switch_delay:
        print(f"stage switch vs GT lift delay: mean {np.mean(switch_delay):+.1f} "
              f"steps  (negative = earlier)")
    print(f"pre-drop ball target (fixture): "
          f"{CARRY_SITE_Z - EE_TO_BALL_Z:.3f} m")
    env.close()


if __name__ == "__main__":
    main()
