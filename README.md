# multiball_vservo_1.0 — 100% one-shot SR on multiball

Clean, self-contained version of the **multiball virtual-servo experiment**
that reached **1000/1000 = 100.0% one-shot success** on the sphere-in-box
task (MuJoCo, DP3 point-cloud diffusion policy, 4-action chunks).

The controller is a frozen BC policy plus two **deployment-side** mechanisms:

1. **Explicit residual servo** on the pre-drop stage:
   `a_exec = clip(a_bc + 0.2 * e / 0.10, -1, 1)` on xyz, recomputed every
   step from the freshest vision error `e`.
2. **Grasp guard**: a vision/FK-only gripper phase machine (4-step close
   ramp + 4-step settle, lift-first carry, release hold, test-lift slip
   detection, in-episode re-grasp).

No retry after an episode ends, no Q-filter, no rescue controller, and no
simulator object pose in the deployment signal path. See
`PRIVILEGED_AUDIT.md`.

## Result

Frozen protocol: one attempt per episode, chunk size 4, n=200 per base,
vision-only targets.

| base | episodes | success | SR | mean steps |
|---|---:|---:|---:|---:|
| dev 620000 | 200 | 200 | 100.0% | 41.7 |
| held-out 636000 | 200 | 200 | 100.0% | 41.5 |
| held-out 636500 | 200 | 200 | 100.0% | 41.4 |
| held-out 637000 | 200 | 200 | 100.0% | 41.0 |
| held-out 637500 | 200 | 200 | 100.0% | 42.0 |
| **pooled** | **1000** | **1000** | **100.0%** | **41.7** |

Wilson 95% CI on 1000/1000: **[99.62%, 100%]**. The frozen-BC baseline was
785/1000 = 78.5%. Raw per-episode results are in `results/final/`.

## Layout

```text
src/eval_vservo.py             one-shot evaluator (this is the reproduction entry point)
src/vservo_env.py              env wrapper: virtual targets, servo error, reward, grasp guard
src/vision_pose.py             vision estimators (ball, box) for the two virtual targets
src/check_vservo_env.py        sanity check: vision targets vs GT (GT is log-only)
src/collect_vservo_demos.py    scripted-expert demo collection (training only)
train_dp3_bc.py                DP3 BC training
config/                        training config used for the included checkpoint
rl_100/                        local DP3 stack (policy, encoder, dataset, env, IK)
third_party/multiball/         multiball scene + scripted expert + MuJoCo XML
runs/bc_vservo_multiball_v2/best.ckpt   the 1000/1000 BC checkpoint (Git LFS)
results/final/                 the five 200-episode result JSONs
results/baseline/              frozen-BC baseline JSONs for comparison
```

Everything is resolved relative to the repo root; there are no host-specific
absolute paths in the runtime code.

## Setup

Tested on Ubuntu 20.04, NVIDIA RTX 4090, driver supporting CUDA 12.1.

```bash
git lfs install          # required before cloning / pulling the checkpoint
git clone <repo-url>
cd multiball_vservo_1.0

bash setup_env.sh        # creates conda env "vservo" and installs requirements
conda activate vservo
source scripts/env.sh    # EGL + PYTHONPATH + CUDA libs
```

Manual path:

```bash
conda env create -f environment.yml
conda activate vservo
python -m pip install -r requirements.txt
```

System packages for headless MuJoCo rendering:

```bash
sudo apt-get install -y libgl1-mesa-glx libegl1-mesa libglew-dev libglfw3 \
                        libosmesa6-dev patchelf
```

`MUJOCO_GL=egl` is set by `scripts/env.sh`. If EGL is unavailable, run with
`MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`, but EGL is what the reported
results used.

## Reproduce

Quick smoke test (5 episodes per base, ~1 minute on a 4090):

```bash
N=5 ./reproduce.sh
```

Full protocol (5 bases x 200 episodes; ~30-60 minutes on a 4090):

```bash
N=200 ./reproduce.sh
```

The script runs exactly the promoted controller:

```bash
python src/eval_vservo.py --ckpt runs/bc_vservo_multiball_v2/best.ckpt \
  --n 200 --seed-base <base> \
  --residual-gain 0.2 --residual-stage predrop \
  --grasp-guard --lift-eps 0.012
```

and exits non-zero unless the pooled result is 100%. Results are written to
`results/reproduced/`. Set `STRICT=0` to allow a non-100% summary, or
`DEVICE=cpu` to run without a GPU (much slower).

Seeds are exhaustive per base (`seed_base + 0 .. n-1`); each episode is
seeded deterministically (`env.seed`, `np.random.seed`, `torch.manual_seed`),
so the full run is repeatable on the same software/GPU stack.

## Checkpoint

`runs/bc_vservo_multiball_v2/best.ckpt` — 621 MB, Git LFS.

- SHA-256: `9925d4a95888d9718160c1002f408ccd4149a2980baddab490836752debbd000`
- Original LFS pointer: `9925d4a9` (same object used by the source experiment)

If you download a ZIP instead of cloning, fetch the real file with:

```bash
git lfs install && git lfs pull
```

If you publish this repo, make sure the LFS object is uploaded:

```bash
git lfs push --all origin main
```

GitHub clones fetch LFS automatically. If you clone from a local `file://`
path with an old git-lfs and the checkpoint is still a 134-byte pointer,
run `git lfs pull` inside the clone.

`CHECKSUMS.sha256` has the same hash for verification.

## Training from scratch (optional)

The included checkpoint is the artifact needed to reproduce the 1.0 SR
evaluation. The training recipe is:

```bash
source scripts/env.sh

# 300 clean scripted-expert demos
python src/collect_vservo_demos.py --n-demos 300 --seed 630000 \
    --out data/vservo_multiball.zarr --save-every 50

# 300 DAgger-lite demos: same expert with xyz action noise 0.003
python src/collect_vservo_demos.py --n-demos 300 --seed 640000 \
    --action-noise 0.003 --out data/vservo_multiball_dagger.zarr \
    --save-every 50

# 600-demo DP3 BC (config/bc_vservo_multiball_v2.yaml)
python train_dp3_bc.py --config config/bc_vservo_multiball_v2.yaml
```

The 1000/1000 controller additionally applies the deployment-side residual
servo and grasp guard above; they require no retraining. Training is not
bit-reproducible across GPU/driver versions, so use the included checkpoint
for the reported numbers.

## Provenance

This folder is a minimal copy of
`RL-100/multiball_vservo_2026` from the original workspace:

- `6542600` — grasp ramp/settle + pad rolling friction -> 1000/1000 SR
- `a2b288f` — remove pre-grasp z bounce + test-lift recovery -> 1000/1000 SR

The final result set is the guard18 evaluation in `results/final/` (dev
620000 + held-out 636000/636500/637000/637500). The same 1000/1000 was also
measured with guard11 and the GT-free guard12; guard18 is the final controller
in `src/vservo_env.py`.

## Real-robot note

The deployment path consumes only RGB-D point clouds, forward kinematics,
the gripper encoder, the vision ball/box estimates, and calibrated static
fixture geometry. Simulator state is used only for evaluation predicates
(success, object pose) and explicit ablation flags. Details and the remaining
sim-to-real caveats are in `PRIVILEGED_AUDIT.md`.

## License

Apache-2.0, inherited from RL-100. See `LICENSE` and `NOTICE`.
