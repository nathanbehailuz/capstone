#!/bin/bash
# File: test_mutag_fixed.slurm
# Purpose: Test fixed hyperparameters on mutag dataset to resolve AUC issues

## ===================== SLURM =====================
#SBATCH -J test_mutag_fixed
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8                     # More CPU for stable training
#SBATCH --mem=24G
#SBATCH -t 12:00:00              # Extended time for more epochs
#SBATCH -o slurm_out/%x_%j.out
#SBATCH -e slurm_out/%x_%j.err
## =================================================

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

# Threading optimization
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

# CUDA optimization
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Result file
RESULT_FILE="results/mutag_test_seed_0.txt"

# Clear/create result file
echo "========================================" > "$RESULT_FILE"
echo "MUTAG FIXED PARAMETERS TEST" >> "$RESULT_FILE"
echo "Date: $(date)" >> "$RESULT_FILE"
echo "Seed: 0" >> "$RESULT_FILE"
echo "Job ID: $SLURM_JOB_ID" >> "$RESULT_FILE"
echo "Node: $(hostname)" >> "$RESULT_FILE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}" >> "$RESULT_FILE"
echo "========================================" >> "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

# Test GPU availability
echo "[$(date)] Testing GPU availability..." | tee -a "$RESULT_FILE"
nvidia-smi -L | tee -a "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

## ==================== FIXED PARAMETERS =====================
# Dataset-specific parameters for mutag
DATASET="mutag"
SEED=0
POLICY="PPO"

# FIXED HYPERPARAMETERS TO RESOLVE AUC ISSUES:
N_EPOCHS=50              # Increased from 10 for better convergence
SIZE_REG=0.001           # Decreased from 0.01 to reduce size penalty
SIM_REG=0.1              # Increased from 0.01 to emphasize similarity
RADIUS_PENALTY=0.0       # Disabled to remove conflicting penalties
CAUSAL_WEIGHT=0.5        # Increased from 0.2 for stronger causal signal

# PPO stabilization parameters
CLIP_EPS=0.1             # Decreased from 0.2 for more stable updates
TARGET_KL=0.01           # Decreased from 0.03 to prevent early stopping
MINIBATCH_SIZE=512       # Increased from 256 for more stable gradients

echo "========================================" | tee -a "$RESULT_FILE"
echo "[$(date)] Starting mutag dataset with FIXED parameters" | tee -a "$RESULT_FILE"
echo "Dataset: $DATASET" | tee -a "$RESULT_FILE"
echo "Seed: $SEED" | tee -a "$RESULT_FILE"
echo "Policy: $POLICY" | tee -a "$RESULT_FILE"
echo "Epochs: $N_EPOCHS (increased from 10)" | tee -a "$RESULT_FILE"
echo "Size Reg: $SIZE_REG (decreased from 0.01)" | tee -a "$RESULT_FILE"
echo "Sim Reg: $SIM_REG (increased from 0.01)" | tee -a "$RESULT_FILE"
echo "Radius Penalty: $RADIUS_PENALTY (disabled)" | tee -a "$RESULT_FILE"
echo "Causal Weight: $CAUSAL_WEIGHT (increased from 0.2)" | tee -a "$RESULT_FILE"
echo "Clip Eps: $CLIP_EPS (decreased from 0.2)" | tee -a "$RESULT_FILE"
echo "Target KL: $TARGET_KL (decreased from 0.03)" | tee -a "$RESULT_FILE"
echo "Minibatch Size: $MINIBATCH_SIZE (increased from 256)" | tee -a "$RESULT_FILE"
echo "========================================" | tee -a "$RESULT_FILE"

# Run the algorithm with fixed parameters
"${PYBIN}" run_RG_explainer.py \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --policy "$POLICY" \
    --n_epochs "$N_EPOCHS" \
    --size_reg "$SIZE_REG" \
    --sim_reg "$SIM_REG" \
    --radius_penalty "$RADIUS_PENALTY" \
    --use_causal \
    --causal_weight "$CAUSAL_WEIGHT" \
    --clip_eps "$CLIP_EPS" \
    --target_kl "$TARGET_KL" \
    --minibatch_size "$MINIBATCH_SIZE" \
    >> "$RESULT_FILE" 2>&1

echo "[$(date)] Completed mutag dataset with fixed parameters" | tee -a "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

echo "========================================" | tee -a "$RESULT_FILE"
echo "[$(date)] TEST COMPLETED" | tee -a "$RESULT_FILE"
echo "Results saved to: $RESULT_FILE" | tee -a "$RESULT_FILE"
echo "Expected improvements:" | tee -a "$RESULT_FILE"
echo "- AUC should be >0.8 (was 0.316)" | tee -a "$RESULT_FILE"
echo "- Rewards should be positive (was -0.84)" | tee -a "$RESULT_FILE"
echo "- Training should be more stable" | tee -a "$RESULT_FILE"
echo "========================================" | tee -a "$RESULT_FILE"
