#!/usr/bin/env bash
# Reproduce the 1.0 SR evaluation.
#
#   N=5   ./reproduce.sh     # fast smoke test (5 episodes per base)
#   N=200 ./reproduce.sh     # full protocol: 5 x 200 = 1000 episodes
#
# Default controller (the one that scored 1000/1000):
#   v2 BC checkpoint + pre-drop residual servo K=0.2 + grasp guard,
#   lift-eps 0.012, one attempt, no retry, no Q-filter.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

CKPT="${CKPT:-runs/bc_vservo_multiball_v2/best.ckpt}"
N="${N:-200}"
BASES="${BASES:-620000 636000 636500 637000 637500}"
OUTDIR="${OUTDIR:-results/reproduced}"
DEVICE="${DEVICE:-cuda:0}"
STRICT="${STRICT:-1}"

mkdir -p "$OUTDIR"

if [ ! -f "$CKPT" ]; then
    echo "[ERROR] checkpoint not found: $CKPT"
    echo "        git lfs install && git lfs pull"
    exit 1
fi

for base in $BASES; do
    echo "=== base ${base} : ${N} episodes ==="
    "$PY" src/eval_vservo.py \
        --ckpt "$CKPT" \
        --n "$N" \
        --seed-base "$base" \
        --residual-gain 0.2 \
        --residual-stage predrop \
        --grasp-guard \
        --lift-eps 0.012 \
        --device "$DEVICE" \
        --out "${OUTDIR}/base_${base}.json"
done

echo "=== aggregate ==="
"$PY" - "$OUTDIR" "$STRICT" <<'PY'
import glob
import json
import os
import sys

outdir, strict = sys.argv[1], sys.argv[2] == "1"
files = sorted(glob.glob(os.path.join(outdir, "base_*.json")))
if not files:
    raise SystemExit("no result files found in " + outdir)

tot_n = tot_s = 0
for f in files:
    d = json.load(open(f))
    n, s = int(d["n"]), int(d["success"])
    tot_n += n
    tot_s += s
    print(f"{os.path.basename(f):24s} {s:4d}/{n:<4d} = {s / n:.1%}  "
          f"mean_steps={d['mean_steps']:.1f}  recoveries={d.get('recoveries_total', 0)}")

print(f"{'POOLED':24s} {tot_s:4d}/{tot_n:<4d} = {tot_s / tot_n:.2%}")
if strict and tot_s != tot_n:
    raise SystemExit(f"[FAIL] SR is not 100% ({tot_s}/{tot_n})")
print("[OK] 100% one-shot success on every base")
PY
