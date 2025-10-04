#!/bin/bash
# File: syn2_ppo.slurm
# Purpose: PPO RGExplainer with higher GPU utilization

## ===================== SLURM =====================
#SBATCH -J rgx_syn2_ppo
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8                     # more CPU to feed the GPU
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --array=0-9
#SBATCH -o slurm_out/%x_%A_%a.out
#SBATCH -e slurm_out/%x_%A_%a.err
## =================================================

# set -euo pipefail    # Uncomment if your script is robust to -e

# Always run from the submit dir
cd "${SLURM_SUBMIT_DIR}"

# Folders for logs & outputs
mkdir -p slurm_out results

## ================== Environment ==================
module purge
module load miniconda/3-4.11.0

# Initialize Conda for non-interactive shells
if ! source "/share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh" 2>/dev/null; then
  eval "$(/share/apps/NYUAD5/miniconda/3-4.11.0/bin/conda shell.bash hook)"
fi

# Activate env
conda activate /home/nz2212/.conda/envs/rlgn

# Prefer env libs
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Explicit python
PYBIN="${CONDA_PREFIX}/bin/python"

# Threads & perf-friendly defaults
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

# CUDA placement: Slurm sets CUDA_VISIBLE_DEVICES; honor it.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# PyTorch memory allocator: reduces fragmentation in long runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# (Optional) Dataloader workers hint your code can read; harmless if unused
export DATALOADER_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} > 1 ? ${SLURM_CPUS_PER_TASK:-8} - 1 : 1 ))

# Minimal device/logging header
echo "[$(date)] Node: $(hostname)"
echo "JobID=$SLURM_JOB_ID  ArrayTaskID=$SLURM_ARRAY_TASK_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

## ==================== Params =====================
SEED="${SLURM_ARRAY_TASK_ID}"
DATASET="syn2"
POLICY="PPO"
N_EPOCHS=50
SIZE_REG=0.01
SIM_REG=1.0
RADIUS_PENALTY=0.1

RESULT_FILE="results/${DATASET}_${POLICY,,}_seed_${SEED}.txt"

echo "[$(date)] Starting seed=$SEED" | tee -a "$RESULT_FILE"

## ================== TRAINING =====================
# Launch directly (avoid extra srun layer that can add overhead)
"${PYBIN}" run_RG_explainer.py \
  --dataset "${DATASET}" \
  --seed "${SEED}" \
  --policy "${POLICY}" \
  --n_epochs "${N_EPOCHS}" \
  --size_reg "${SIZE_REG}" \
  --sim_reg "${SIM_REG}" \
  --radius_penalty "${RADIUS_PENALTY}" \
  >> "$RESULT_FILE" 2>&1

echo "[$(date)] Finished seed=$SEED" | tee -a "$RESULT_FILE"

