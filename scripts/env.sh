#!/usr/bin/env bash
# Source this file (do not execute it):
#   source scripts/env.sh
#
# It makes the repo self-contained: PYTHONPATH, MuJoCo EGL, the vendored
# multiball env, and the CUDA shared libraries shipped by the torch wheel.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT

if [ -z "${PY:-}" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        export PY="${CONDA_PREFIX}/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        export PY="python3"
    else
        export PY="python"
    fi
fi

# The cu121 torch wheel keeps cudnn/cublas/etc. under site-packages/nvidia.
# Add every nvidia/*/lib directory we can find (conda env, venv, user site).
_lib_paths=""
for _base in "${CONDA_PREFIX:-}" "${VIRTUAL_ENV:-}" "${HOME}/.local"; do
    [ -n "$_base" ] || continue
    for _d in "$_base"/lib/python*/site-packages/nvidia/*/lib; do
        [ -d "$_d" ] && _lib_paths="${_lib_paths}:${_d}"
    done
done
if [ -n "$_lib_paths" ]; then
    export LD_LIBRARY_PATH="${_lib_paths#:}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MULTIBALL_INC="${MULTIBALL_INC:-${ROOT}/third_party/multiball}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$ROOT/logs" "$ROOT/data" "$ROOT/runs" "$ROOT/results"
