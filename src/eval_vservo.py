#!/usr/bin/env python
"""Closed-loop evaluation of a virtual-target multiball BC policy.

The policy sees only the point cloud (N, 7: xyz + rgb + ghost flag) and the
14-D agent state (9 proprio + 3-D servo error + validity + stage); success is
the env predicate (green ball inside the box).  The virtual targets are
vision-estimated by default; ``--vision-source gt`` is an ablation.

Run:
  cd multiball_vservo_2026
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python src/eval_vservo.py \
      --ckpt runs/bc_vservo_multiball/best.ckpt --n 50 --seed-base 620000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque

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
import torch  # noqa: E402

from rl_100.common.ckpt_loader import build_policy_from_ckpt  # noqa: E402
from rl_100.common.pytorch_util import dict_apply  # noqa: E402
from vservo_env import VservoMultiballEnv  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def _obs_tensor(obs_hist, device):
    stack = {
        "agent_pos": np.stack([o["agent_pos"] for o in obs_hist])[None],
        "point_cloud": np.stack([o["point_cloud"] for o in obs_hist])[None],
    }
    return dict_apply(stack, lambda x: torch.from_numpy(x).to(
        device=device).float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=620000)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--pc-resolution", type=int, default=160)
    ap.add_argument("--vision-source", default="vision",
                    choices=["vision", "gt"])
    ap.add_argument("--switch-mode", default="vision",
                    choices=["vision", "gt"])
    ap.add_argument("--step-replan", action="store_true")
    ap.add_argument("--residual-gain", type=float, default=0.0,
                    help="explicit residual servo gain K; executed action = "
                         "clip(a_res_pred + K*e/scale, -1, 1). 0 disables.")
    ap.add_argument("--residual-scale", type=float, default=0.10,
                    help="env metres per unit action (MW_ACTION_SCALE)")
    ap.add_argument("--residual-stage", default="predrop",
                    choices=["predrop", "all"],
                    help="where the explicit servo term is applied")
    ap.add_argument("--lift-eps", type=float, default=0.030,
                    help="vision ball-rise threshold (m) for the stage switch; "
                         "the demo labels switch at the expert lift, so 0.010-"
                         "0.015 matches the training distribution")
    ap.add_argument("--grasp-guard", action="store_true",
                    help="deployment gripper phase machine + near-grasp z "
                         "clamp (fixed grasp, latched carry, drop at target)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    policy, _cfg = build_policy_from_ckpt(args.ckpt, device, use_ema=True)
    policy.eval()
    env = VservoMultiballEnv(
        max_episode_steps=args.max_steps, n_points=args.n_points,
        pc_resolution=args.pc_resolution, seed=args.seed_base,
        vision_source=args.vision_source, switch_mode=args.switch_mode,
        lift_eps=args.lift_eps, grasp_guard=args.grasp_guard)
    n_obs = int(policy.n_obs_steps)

    rows = []
    for i in range(args.n):
        seed = int(args.seed_base + i)
        env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        obs = env.reset()
        hist = deque([obs] * n_obs, maxlen=n_obs)
        step = 0
        done = False
        info = {}
        while step < args.max_steps and not done:
            od = _obs_tensor(hist, device)
            with torch.no_grad():
                act_seq = policy.predict_action(od)["action"][0].cpu().numpy()
                if args.step_replan:
                    act_seq = act_seq[:1]
            stage_chunk = int(env.stage)
            for a in act_seq:
                a = np.asarray(a, dtype=np.float32).copy()
                if (args.residual_gain > 0.0
                        and (args.residual_stage == "all"
                             or float(obs["agent_pos"][13]) >= 0.5)):
                    # Explicit residual servo in the PRE-DROP stage only: the
                    # policy predicts a_res, the executed action adds the
                    # current error feedback.  Use the freshest observation
                    # (per-step feedback), not the chunk-start one.  Stage 0
                    # keeps the raw BC action because the expert approaches
                    # high above the ball.
                    e = np.asarray(obs["agent_pos"][9:12], dtype=np.float32)
                    servo = ((args.residual_gain / args.residual_scale) * e)
                    if float(obs["agent_pos"][13]) >= 0.5:
                        # lift-first: do not pull laterally while the held
                        # ball is still below the box rim + 4 cm
                        ee_ball_z = float(obs["agent_pos"][2]) - 0.011
                        if ee_ball_z < env._box_rim_z() + 0.04:
                            servo[0] = 0.0
                            servo[1] = 0.0
                    a[:3] = np.clip(a[:3] + servo, -1.0, 1.0)
                obs, _r, done, info = env.step(a)
                hist.append(obs)
                step += 1
                if int(env.stage) != stage_chunk:
                    # a stage switch (including a drop-recovery re-grasp)
                    # invalidates the rest of the chunk; replan with a clean
                    # stage-consistent history
                    hist = deque([obs] * n_obs, maxlen=n_obs)
                    break
                if done:
                    break
        verr = np.asarray(info.get("vision_err", [np.nan] * 3))
        rows.append({
            "seed": seed, "success": bool(info.get("in_box", False)),
            "steps": step, "stage": int(info.get("vision_stage", -1)),
            "vision_valid": bool(info.get("vision_valid", False)),
            "vision_err": verr.tolist(),
            "vision_xy_mm": float(np.hypot(verr[0], verr[1]) * 1e3),
            "vision_z_mm": float(verr[2] * 1e3),
            "lifted": info.get("lifted"),
            "placed": info.get("placed"),
            "dist_to_box": info.get("dist_to_box"),
            "object_pos": (np.asarray(info.get("object_pos")).tolist()
                           if info.get("object_pos") is not None else None),
            "vision_frames": info.get("vision_frames"),
            "vision_valid_frames": info.get("vision_valid_frames"),
            "recoveries": int(info.get("recoveries", 0)),
        })
        print(f"[vservo {i + 1}/{args.n}] seed={seed} steps={step} "
              f"success={rows[-1]['success']} stage={rows[-1]['stage']} "
              f"verr=({rows[-1]['vision_xy_mm']:.2f},"
              f"{rows[-1]['vision_z_mm']:.2f})mm", flush=True)

    k = sum(r["success"] for r in rows)
    n = len(rows)
    ci = wilson(k, n)
    succ = [r for r in rows if r["success"]]
    xy = np.array([r["vision_xy_mm"] for r in succ]) if succ else np.zeros(1)
    out = {
        "ckpt": os.path.abspath(args.ckpt),
        "vision_source": args.vision_source,
        "switch_mode": args.switch_mode,
        "residual_gain": args.residual_gain,
        "residual_scale": args.residual_scale,
        "residual_stage": args.residual_stage,
        "lift_eps": args.lift_eps,
        "grasp_guard": args.grasp_guard,
        "n": n, "success": k,
        "success_rate": k / n if n else 0.0, "wilson_ci": list(ci),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "success_xy_mm_mean": float(xy.mean()),
        "success_xy_mm_p95": float(np.percentile(xy, 95)),
        "stage1_reached": int(sum(r["stage"] >= 1 for r in rows)),
        "recoveries_total": int(sum(r.get("recoveries", 0) for r in rows)),
        "rows": rows,
    }
    print(f"[vservo] {k}/{n} = {out['success_rate']:.4f} CI={ci} "
          f"steps={out['mean_steps']:.1f} "
          f"success_xy={out['success_xy_mm_mean']:.2f}mm "
          f"p95={out['success_xy_mm_p95']:.2f}mm", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"[vservo] wrote {args.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
