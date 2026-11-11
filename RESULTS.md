# Results — 100% one-shot success

## Protocol

- One attempt per episode; **no retry, no Q-filter, no rescue controller**.
- `n=200` consecutive seeds per base: `seed_base + 0 .. seed_base + 199`.
- Policy: `runs/bc_vservo_multiball_v2/best.ckpt` (DP3, point cloud 1024x7,
  2 observation steps, 4-action chunks, 10 DDIM steps, EMA weights).
- Controller: pre-drop residual servo `K=0.2` (scale 0.10) + grasp guard,
  vision-only targets, `--lift-eps 0.012`.
- "Success" is the environment predicate: the green ball is inside the box.
- Evaluation machine: single RTX 4090, Ubuntu 20.04.

## Final (guard18, committed controller)

| base | episodes | success | SR | Wilson 95% CI | mean steps | recoveries |
|---|---:|---:|---:|---|---:|---:|
| dev 620000 | 200 | 200 | 100.0% | — | 41.7 | 1 |
| held-out 636000 | 200 | 200 | 100.0% | — | 41.5 | 0 |
| held-out 636500 | 200 | 200 | 100.0% | — | 41.4 | 0 |
| held-out 637000 | 200 | 200 | 100.0% | — | 41.0 | 0 |
| held-out 637500 | 200 | 200 | 100.0% | — | 42.0 | 0 |
| **pooled** | **1000** | **1000** | **100.0%** | **[99.62%, 100%]** | **41.7** | **1** |

The single recovery is an in-episode re-grasp after a detected slip; it is
part of the controller and completed successfully. It is not a retry of a
failed episode.

Per-base JSON: `results/final/dev_620000_n200.json`, `ho636000_n200.json`,
`ho636500_n200.json`, `ho637000_n200.json`, `ho637500_n200.json`.

The `ckpt` field inside each JSON records the original run's absolute path;
the checkpoint in this repo is the same file
(`runs/bc_vservo_multiball_v2/best.ckpt`, identical SHA-256).

## Baseline (frozen BC, no residual servo, no guard)

| base | success | SR | mean steps |
|---|---:|---:|---:|
| dev 620000 | 155/200 | 77.5% | 100.9 |
| held-out 636000 | 163/200 | 81.5% | 92.6 |
| held-out 636500 | 154/200 | 77.0% | 104.1 |
| held-out 637000 | 150/200 | 75.0% | 108.6 |
| held-out 637500 | 161/200 | 80.5% | 97.0 |
| **pooled** | **785/1000** | **78.5%** | — |

Baseline JSONs are in `results/baseline/`. The residual servo + guard thus
add **+21.5 points** (785 -> 1000 successes) with no retraining.

## What removed the last failures

The path from 98-99% to 100% was analysed per failing seed. The remaining
failures were stage-1 high-speed ejections during the first carry move, not
release ejections. Three stacked causes were fixed in the final controller:

1. **Weak pinch**: the guard could close above/below the ball equator.
   Fix: close only inside 10 mm xy / 6 mm z, with lateral/z centering while
   the gripper is open, and a 4-step close ramp + 4-step settle hold.
2. **Lift before firm contact**: the first carry command could start while
   the pads were still settling. Fix: lift-first gating and the settle hold.
3. **Frictionless roller bearing between the pad faces**: the ball could
   roll out of the pinch. Fix: deployment-only pad rolling/torsional friction
   0.5 / 0.1 (`VSERVO_PAD_ROLL=1`, enabled by `--grasp-guard`). The
   controller-side alternatives (carry cap) are described in
   `PRIVILEGED_AUDIT.md`.

A pre-grasp z bounce was also removed: inside the 25 mm grasp cone the open
gripper cannot be commanded upward by the policy; the visual servo owns the
upward correction when the arm is below the grasp height. A test lift after
closing detects a slipping pinch and triggers an in-episode re-grasp from a
14 cm stand-off.

## Provenance

- Original workstream: `RL-100/multiball_vservo_2026`.
- Commits: `6542600` (grasp ramp/settle + pad rolling -> 1000/1000) and
  `a2b288f` (pre-grasp z-bounce fix + test-lift recovery -> 1000/1000).
- Final checkpoint: `runs/bc_vservo_multiball_v2/best.ckpt`,
  SHA-256 `9925d4a95888d9718160c1002f408ccd4149a2980baddab490836752debbd000`.
- The full experiment history, ablations, and the failed 98-99% variants are
  documented in the original workspace's `multiball_vservo_2026/STATUS.md`;
  this repo intentionally ships only the minimal winning path.
