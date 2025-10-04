#!/bin/bash
# File: test_causal_unified.slurm
# Purpose: Test updated causal algorithm on all datasets with seed=0

## ===================== SLURM =====================
#SBATCH -J test_causal_unified
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8                     # More CPU for sequential runs
#SBATCH --mem=24G
#SBATCH -t 12:00:00              # Extended time for all datasets
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

# Threading optimization (reduced to avoid OpenBLAS issues)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# CUDA optimization
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Unified result file
UNIFIED_RESULT_FILE="mutag_ppo_seed_test_0.txt"

# Clear/create unified result file
echo "========================================" > "$UNIFIED_RESULT_FILE"
echo "CAUSAL ALGORITHM TEST - MUTAG DATASET" >> "$UNIFIED_RESULT_FILE"
echo "Date: $(date)" >> "$UNIFIED_RESULT_FILE"
echo "Seed: 0" >> "$UNIFIED_RESULT_FILE"
echo "Job ID: $SLURM_JOB_ID" >> "$UNIFIED_RESULT_FILE"
echo "Node: $(hostname)" >> "$UNIFIED_RESULT_FILE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}" >> "$UNIFIED_RESULT_FILE"
echo "========================================" >> "$UNIFIED_RESULT_FILE"
echo "" >> "$UNIFIED_RESULT_FILE"

# Test GPU availability
echo "[$(date)] Testing GPU availability..." | tee -a "$UNIFIED_RESULT_FILE"
nvidia-smi -L | tee -a "$UNIFIED_RESULT_FILE"
echo "" >> "$UNIFIED_RESULT_FILE"

## ==================== MUTAG DATASET CONFIGURATION =====================
# Fixed parameters for mutag dataset
DATASET="mutag"
SEED=0
POLICY="PPO"
N_EPOCHS=10
SIZE_REG=0.01
SIM_REG=0.01
RADIUS_PENALTY=0.01

## ================== RUN MUTAG DATASET =====================
echo "========================================" | tee -a "$UNIFIED_RESULT_FILE"
echo "[$(date)] Starting dataset: $DATASET" | tee -a "$UNIFIED_RESULT_FILE"
echo "Parameters: epochs=$N_EPOCHS, size_reg=$SIZE_REG, sim_reg=$SIM_REG, radius_penalty=$RADIUS_PENALTY" | tee -a "$UNIFIED_RESULT_FILE"
echo "========================================" | tee -a "$UNIFIED_RESULT_FILE"

# Run the algorithm
"${PYBIN}" run_RG_explainer.py \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --policy "$POLICY" \
    --n_epochs "$N_EPOCHS" \
    --size_reg "$SIZE_REG" \
    --sim_reg "$SIM_REG" \
    --use_causal \
    --causal_weight 0.2 \
    --radius_penalty "$RADIUS_PENALTY" \
    >> "$UNIFIED_RESULT_FILE" 2>&1

echo "[$(date)] Completed dataset: $DATASET" | tee -a "$UNIFIED_RESULT_FILE"
echo "" >> "$UNIFIED_RESULT_FILE"

echo "========================================" | tee -a "$UNIFIED_RESULT_FILE"
echo "[$(date)] MUTAG DATASET COMPLETED" | tee -a "$UNIFIED_RESULT_FILE"
echo "Results saved to: $UNIFIED_RESULT_FILE" | tee -a "$UNIFIED_RESULT_FILE"
echo "========================================" | tee -a "$UNIFIED_RESULT_FILE"
