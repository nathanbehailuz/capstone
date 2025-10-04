#!/usr/bin/env bash
# create_conda_env.sh
# Purpose: Create a user Conda environment on the CRC cluster (Step 2 from your setup).
# Usage examples:
#   bash create_conda_env.sh -n pytorch-gpu -p 3.9 --gpu
#   bash create_conda_env.sh -n myenv -p 3.10
#   bash create_conda_env.sh -n myenv -p 3.9 -r /path/to/requirements.txt
#
# Notes:
# - Run on a login or compute node *after* loading CRC miniconda module:
#       module purge
#       module load miniconda/3-4.11.0
# - If you use --gpu, this installs PyTorch + CUDA (pytorch, torchvision, torchaudio, pytorch-cuda=11.8)
#   from the official pytorch + nvidia channels.
# - You can safely re-run; the script will refuse to overwrite an existing env name.

set -euo pipefail

ENV_NAME="pytorch-gpu"
PY_VER="3.9"
GPU="true"
REQ_FILE="requirements.txt"

print_help() {
  sed -n '2,25p' "$0"
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--name)
      ENV_NAME="$2"; shift 2;;
    -p|--python)
      PY_VER="$2"; shift 2;;
    --gpu)
      GPU="true"; shift 1;;
    -r|--requirements)
      REQ_FILE="$2"; shift 2;;
    -h|--help)
      print_help; exit 0;;
    *)
      echo "Unknown argument: $1"; print_help; exit 1;;
  esac
done

# Basic checks
if ! command -v conda >/dev/null 2>&1; then
  echo "[Error] 'conda' not found on PATH."
  echo "Load the module first:  module purge && module load miniconda/3-4.11.0"
  exit 1
fi

echo "----------------------------------------"
echo "Conda env creation parameters:"
echo "  Name:           ${ENV_NAME}"
echo "  Python version: ${PY_VER}"
echo "  GPU stack:      ${GPU}"
if [[ -n "${REQ_FILE}" ]]; then
  echo "  Requirements:   ${REQ_FILE}"
fi
echo "----------------------------------------"

# Ensure env does not already exist
if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[Error] Conda env '${ENV_NAME}' already exists. Choose a different name with -n."
  exit 1
fi

# Create the env
if [[ "${GPU}" == "true" ]]; then
  echo "[Info] Creating GPU-enabled PyTorch environment '${ENV_NAME}' ..."
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}" pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
else
  echo "[Info] Creating base environment '${ENV_NAME}' ..."
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

# Optionally install requirements
if [[ -n "${REQ_FILE}" ]]; then
  if [[ ! -f "${REQ_FILE}" ]]; then
    echo "[Warn] Requirements file '${REQ_FILE}' not found; skipping."
  else
    echo "[Info] Installing requirements from ${REQ_FILE} ..."
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
    pip install -r "${REQ_FILE}"
  fi
fi

echo "----------------------------------------"
echo "[Done] Environment '${ENV_NAME}' created."
echo "To activate it in your shell:"
echo "  source \"$(conda info --base)/etc/profile.d/conda.sh\""
echo "  conda activate ${ENV_NAME}"
echo
echo "Housekeeping (optional to save space):"
echo "  conda clean -a  &&  pip cache purge"
echo "----------------------------------------"
