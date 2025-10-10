#!/bin/bash
# File: syn3_ppo.slurm

## ===================== SLURM =====================
#SBATCH -J rgx_syn3_ppo
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 7:00:00
#SBATCH --array=0-0
#SBATCH -o slurm_out/%x_%A_%a.out
#SBATCH -e slurm_out/%x_%A_%a.err
## =================================================

# set -euo pipefail

# Always run from the submit dir
cd "${SLURM_SUBMIT_DIR}"

# Folders for logs & outputs
mkdir -p slurm_out results

# echo "[$(date)] Node: $(hostname)"
# echo "[$(date)] JOBID=$SLURM_JOB_ID  ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
# echo "[$(date)] CWD: $PWD"

## ================== Environment ==================
module purge
module load miniconda/3-4.11.0

# Initialize Conda for non-interactive shells (try both styles)
if ! source "/share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh" 2>/dev/null; then
  eval "$(/share/apps/NYUAD5/miniconda/3-4.11.0/bin/conda shell.bash hook)"
fi

# Activate your env by absolute path
conda activate /home/nz2212/.conda/envs/rlgn

# Optional: prefer env libs first (harmless even if libstdc++ isn't present)
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Sanity: use the env's Python explicitly
PYBIN="${CONDA_PREFIX}/bin/python"

# Helpful runtime flags while debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTHONNOUSERSITE=1


# "$PYBIN" - <<'PY'
# import sys, numpy, scipy, torch
# print("python:", sys.executable)
# print("numpy:", numpy.__version__)
# print("scipy:", scipy.__version__)
# print("torch:", torch.__version__)
# PY

## ==================== Params =====================
SEED="${SLURM_ARRAY_TASK_ID}"
DATASET="syn3"
POLICY="PPO"
N_EPOCHS=30

# Regularization - MINIMAL to focus on fidelity
SIZE_REG=0.0001      # Reduced 100x: was 0.01 (too aggressive)
SIM_REG=0.001        # Reduced 1000x: was 1.0 (causing high value loss)
RADIUS_PENALTY=0.01  # Reduced 10x: was 0.1 (too high)

# Learning rate - Higher for value function learning
LR=0.0003            # Increased 3x: was 0.0001 default (value function needs faster learning)

# Causal mechanism - Reduced weight and faster warmup
CAUSAL_WEIGHT=0.2    # Reduced from 0.5 (too much noise)
CAUSAL_WARMUP=10     # Increased from 5 (slower warmup)

# Size targets - Remove conflicting penalties
LAMBDA_SIZE=0.0      # Disable hinge loss (conflicts with SIZE_REG)
LAMBDA_RADIUS=0.0    # Disable adaptive radius (use RADIUS_PENALTY only)
STEP_COST=0.0        # Disable step cost (conflicts with SIZE_REG)

# Fidelity threshold - DISABLE artificial bonuses
FIDELITY_THRESH=0.0  # Disable EXTREME bonuses (causing value explosion)

RESULT_FILE="results/${DATASET}_${POLICY,,}_seed_${SEED}.txt"

# Clean the result file before starting
> "$RESULT_FILE"

echo "[$(date)] Starting seed=$SEED" | tee -a "$RESULT_FILE"

## ============== GPU & Torch Probes ===============
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" | tee -a "$RESULT_FILE"

# Confirm GPU is allocated to THIS step
srun --gres=gpu:1 nvidia-smi | tee -a "$RESULT_FILE"

# Probe torch CUDA inside the allocation
# srun --gres=gpu:1 bash -lc " \
#   ${PYBIN} - <<'PY'
# import os, torch
# print('torch:', torch.__version__)
# print('torch.version.cuda:', torch.version.cuda)
# print('built_with_cuda:', torch.backends.cuda.is_built())
# print('is_available:', torch.cuda.is_available())
# print('device_count:', torch.cuda.device_count())
# if torch.cuda.is_available():
#     print('current_device:', torch.cuda.current_device())
#     print('name0:', torch.cuda.get_device_name(0))
# PY
# " | tee -a "$RESULT_FILE"

## ================== TRAINING =====================
# Launch your script INSIDE the GPU step, using the env's python
srun --export=ALL --gres=gpu:1 -c "${SLURM_CPUS_PER_TASK:-4}" \
  "${PYBIN}" run_RG_explainer.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --policy "${POLICY}" \
    --n_epochs "${N_EPOCHS}" \
    --lr "${LR}" \
    --size_reg "${SIZE_REG}" \
    --sim_reg "${SIM_REG}" \
    --radius_penalty "${RADIUS_PENALTY}" \
    --causal_weight "${CAUSAL_WEIGHT}" \
    --causal_warmup_epochs "${CAUSAL_WARMUP}" \
    --lambda_size "${LAMBDA_SIZE}" \
    --lambda_radius "${LAMBDA_RADIUS}" \
    --step_cost "${STEP_COST}" \
    --fidelity_thresh "${FIDELITY_THRESH}" \
  >> "$RESULT_FILE" 2>&1

echo "[$(date)] Finished seed=$SEED" | tee -a "$RESULT_FILE"
