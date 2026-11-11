#!/usr/bin/env python
"""Collect virtual-target multiball demos with the scripted expert.

The expert actions are the BC labels; the virtual-target points, the servo
error and the dense reward are observation/reward-side only, exactly as they
will be at deployment.

Examples
--------
  cd multiball_vservo_2026
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python src/collect_vservo_demos.py \
      --tune 3

  ... src/collect_vservo_demos.py --out data/vservo_multiball.zarr \
      --n-demos 400
"""
from __future__ import annotations

import argparse
import os
import sys
import time

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

# Import our env FIRST so `rl_100` resolves to this folder's copy (the
# multiball_env module inserts its own directory into sys.path when imported).
from vservo_env import VservoMultiballEnv  # noqa: E402
from collect_demos_multiball import (  # noqa: E402
    DONE, LIFT, ScriptedPolicy, sample_expert_params)
from rl_100.common.replay_buffer import ReplayBuffer  # noqa: E402

GAMMA = 0.99


def run_episode(env, seed, params=None, max_steps=300, action_noise=0.0):
    env.seed(seed)
    obs = env.reset()
    policy = ScriptedPolicy(params)
    rng = np.random.default_rng(seed + 12345)
    ep = {
        "state": [obs["agent_pos"].copy()],
        "action": [],
        "point_cloud": [obs["point_cloud"].copy()],
        "object_pos": [env.sphere_pos().copy()],
        "object_quat": [env.data.body("sphere").xquat.copy()],
        "reward": [],
        "done": [],
        "timeout": [],
    }
    forced = False
    info = {}
    while True:
        action = policy.act(env, env.sphere_pos())
        # The expert's LIFT transition defines the virtual-target switch for
        # clean demo labels.  Refresh the pre-step observation so the action
        # that starts the lift is conditioned on the pre-drop target.
        if not forced and policy.phase >= LIFT and env.stage == 0:
            env.set_stage(1)
            obs = env._get_obs()
            ep["state"][-1] = obs["agent_pos"].copy()
            ep["point_cloud"][-1] = obs["point_cloud"].copy()
            forced = True
        applied = action
        if action_noise > 0.0:
            applied = action.copy()
            applied[:3] = np.clip(
                applied[:3] + rng.normal(0.0, action_noise, size=3),
                -1.0, 1.0)
        obs, reward, done, info = env.step(applied)
        ep["state"].append(obs["agent_pos"].copy())
        ep["action"].append(action.astype(np.float32))
        ep["point_cloud"].append(obs["point_cloud"].astype(np.float32).copy())
        ep["object_pos"].append(env.sphere_pos().copy())
        ep["object_quat"].append(env.data.body("sphere").xquat.copy())
        placed = bool(info["in_box"])
        ep["reward"].append(np.float32(1.0 if placed else reward))
        ep["done"].append(np.bool_(done))
        ep["timeout"].append(np.bool_(done and not placed))
        if done or policy.phase == DONE or len(ep["action"]) >= max_steps:
            break
    success = bool(info.get("in_box", False))
    return ep, success, policy.phase, len(ep["action"]), info


def to_episode_data(ep, T):
    state = np.stack(ep["state"]).astype(np.float32)            # (T+1, 14)
    action = np.stack(ep["action"]).astype(np.float32)          # (T, 4)
    point_cloud = np.stack(ep["point_cloud"]).astype(np.float32)  # (T+1,N,7)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/vservo_multiball.zarr")
    ap.add_argument("--n-demos", type=int, default=400)
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--pc-resolution", type=int, default=160)
    ap.add_argument("--seed", type=int, default=600000)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--action-noise", type=float, default=0.0,
                    help="DAgger-lite xyz noise added to the EXECUTED action "
                         "(the recorded label stays the expert action)")
    ap.add_argument("--mode", default="",
                    help="force one expert fail mode (clean/offcenter/...) "
                         "instead of sampling the training mixture")
    ap.add_argument("--tune", type=int, default=0,
                    help="run this many episodes and print phase lengths, "
                         "save nothing")
    ap.add_argument("--save-failures", action="store_true")
    ap.add_argument("--max-failures", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=0,
                    help="write <out>.partial every N successful demos")
    args = ap.parse_args()

    env = VservoMultiballEnv(
        max_episode_steps=args.max_steps, n_points=args.n_points,
        pc_resolution=args.pc_resolution, seed=args.seed)
    param_rng = np.random.default_rng(args.seed + 99991)

    if args.tune > 0:
        print(f"[tune] running {args.tune} episodes (no save)")
        for i in range(args.tune):
            seed = args.seed + i
            params = sample_expert_params(param_rng,
                                          mode=(args.mode or None))
            ep, success, phase, length, info = run_episode(
                env, seed, params=params, max_steps=args.max_steps,
                action_noise=args.action_noise)
            vf = info.get("vision_valid_frames", 0)
            frames = info.get("vision_frames", 0)
            print(f"  [tune {i + 1}/{args.tune}] seed={seed} len={length} "
                  f"mode={params.get('fail_mode')} success={success} "
                  f"stage={env.stage} vision={vf}/{frames}")
        env.close()
        return

    buffer = ReplayBuffer.create_empty_numpy()
    fail_buffer = (ReplayBuffer.create_empty_numpy()
                   if args.save_failures else None)
    n_success = n_fail = attempts = 0
    n_valid = n_frames = 0
    t0 = time.time()
    while n_success < args.n_demos:
        seed = args.seed + attempts
        attempts += 1
        params = sample_expert_params(param_rng, mode=(args.mode or None))
        ep, success, phase, length, info = run_episode(
            env, seed, params=params, max_steps=args.max_steps,
            action_noise=args.action_noise)
        n_valid += int(info.get("vision_valid_frames", 0))
        n_frames += int(info.get("vision_frames", 0))
        if success:
            buffer.add_episode(to_episode_data(ep, len(ep["action"])))
            n_success += 1
            print(f"[demo {n_success}/{args.n_demos}] attempt {attempts} "
                  f"seed={seed} mode={params.get('fail_mode')} len={length} "
                  f"vision={info.get('vision_valid_frames')}/"
                  f"{info.get('vision_frames')} "
                  f"({time.time() - t0:.1f}s)", flush=True)
            if args.save_every and n_success % args.save_every == 0:
                os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                            exist_ok=True)
                buffer.save_to_path(args.out + ".partial")
                print(f"    checkpointed -> {args.out}.partial", flush=True)
        else:
            n_fail += 1
            print(f"    FAILED attempt {attempts} seed={seed} "
                  f"mode={params.get('fail_mode')} len={length}", flush=True)
            if fail_buffer is not None and n_fail <= args.max_failures:
                ep_t = {k: list(v) for k, v in ep.items()}
                ep_t["done"][-1] = np.bool_(True)
                ep_t["timeout"][-1] = np.bool_(True)
                fail_buffer.add_episode(
                    to_episode_data(ep_t, len(ep_t["action"])))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    print(f"saved {n_success} demos, {buffer.n_steps} steps -> {args.out} "
          f"({attempts} attempts, failure rate {n_fail / attempts:.1%}, "
          f"vision valid {n_valid}/{n_frames}, {time.time() - t0:.1f}s)")
    buffer.save_to_path(args.out)
    if fail_buffer is not None and fail_buffer.n_episodes > 0:
        root, ext = os.path.splitext(args.out)
        fail_path = f"{root}_failures{ext}"
        fail_buffer.save_to_path(fail_path)
        print(f"saved {fail_buffer.n_episodes} failures -> {fail_path}")
    env.close()


if __name__ == "__main__":
    main()
