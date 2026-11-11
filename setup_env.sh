#!/usr/bin/env bash
# Create the conda environment and install the Python dependencies.
#
#   bash setup_env.sh              # env name: vservo
#   bash setup_env.sh my_env       # custom env name
#
# Tested with Python 3.8.20 + torch 2.4.0+cu121 on Ubuntu 20.04 / RTX 4090.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-vservo}"

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found. Install Miniconda first:"
    echo "        https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "[1/4] system libraries (MuJoCo EGL / GL)"
missing=""
for lib in libEGL.so.1 libGL.so.1 libglew.so libglfw.so.3; do
    if ! ldconfig -p 2>/dev/null | grep -q "$lib"; then
        missing="${missing} ${lib}"
    fi
done
if [ -n "$missing" ]; then
    echo "      missing:${missing}"
    echo "      install with:"
    echo "      sudo apt-get install -y libgl1-mesa-glx libegl1-mesa libglew-dev libglfw3 libosmesa6-dev patchelf"
else
    echo "      all present"
fi

echo "[2/4] conda environment '${ENV_NAME}'"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "      ${ENV_NAME} already exists (reusing it)"
else
    conda create -n "${ENV_NAME}" python=3.8.20 pip=24.3.1 -y
fi
conda activate "${ENV_NAME}"

echo "[3/4] pip dependencies"
python -m pip install --upgrade "pip==24.3.1" "setuptools==59.5.0" wheel
python -m pip install -r "${HERE}/requirements.txt"

# Some conda libffi builds only ship libffi.so.8.x while CPython asks for
# libffi.so.8.  Create the soname symlink if missing.
LIBFFI="$(ls "${CONDA_PREFIX}"/lib/libffi.so.8.* 2>/dev/null | head -n 1 || true)"
if [ -n "${LIBFFI}" ] && [ ! -e "${CONDA_PREFIX}/lib/libffi.so.8" ]; then
    ln -s "${LIBFFI}" "${CONDA_PREFIX}/lib/libffi.so.8"
    echo "      created libffi.so.8 -> $(basename "${LIBFFI}")"
fi

echo "[4/4] import check"
python - <<'PY'
import torch, mujoco, gym, diffusers, zarr, numpy
print("python  ", __import__("sys").version.split()[0])
print("torch   ", torch.__version__, "cuda", torch.cuda.is_available())
print("mujoco  ", mujoco.__version__)
print("gym     ", gym.__version__)
print("diffusers", diffusers.__version__)
print("zarr    ", zarr.__version__)
print("numpy   ", numpy.__version__)
PY

echo
echo "Done.  Next:"
echo "  conda activate ${ENV_NAME}"
echo "  source scripts/env.sh"
echo "  N=5 ./reproduce.sh          # quick smoke test"
echo "  N=200 ./reproduce.sh        # full 1000-episode reproduction"
