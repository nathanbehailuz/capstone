#!/bin/bash
# File: test_edge_dropout.slurm
# Purpose: Test edge dropout implementation

## ===================== SLURM =====================
#SBATCH -J test_edge_dropout
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH -o slurm_out/%x_%j.out
#SBATCH -e slurm_out/%x_%j.err
## =================================================

set -euo pipefail

# Always run from the submit dir
cd "${SLURM_SUBMIT_DIR}"

# Folders for logs & outputs
mkdir -p slurm_out results

echo "[$(date)] Node: $(hostname)"
echo "[$(date)] JOBID=$SLURM_JOB_ID"
echo "[$(date)] CWD: $PWD"

## ================== Environment ==================
module purge
module load miniconda/3-4.11.0

# Initialize Conda for non-interactive shells
if ! source "/share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh" 2>/dev/null; then
  eval "$(/share/apps/NYUAD5/miniconda/3-4.11.0/bin/conda shell.bash hook)"
fi

# Activate environment
conda activate /home/nz2212/.conda/envs/rlgn

# Use the env's Python explicitly
PYBIN="${CONDA_PREFIX}/bin/python"

# Environment variables
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONNOUSERSITE=1

echo ""
echo "=========================================="
echo "EDGE DROPOUT IMPLEMENTATION TESTS"
echo "=========================================="
echo ""

## ==================== Test 1: Unit Tests =====================
echo "[$(date)] Running unit tests..."
srun --gres=gpu:1 "${PYBIN}" test_edge_dropout.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ UNIT TESTS PASSED"
    echo "=========================================="
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ UNIT TESTS FAILED"
    echo "=========================================="
    echo ""
    exit $EXIT_CODE
fi

## ==================== Test 2: Baseline (No Dropout) =====================
echo ""
echo "=========================================="
echo "TEST 2: Baseline (dropout=0.0)"
echo "=========================================="
echo ""

RESULT_FILE="results/test_edge_dropout_baseline.txt"
> "$RESULT_FILE"

echo "[$(date)] Running baseline test with dropout=0.0" | tee -a "$RESULT_FILE"

srun --gres=gpu:1 "${PYBIN}" run_RG_explainer.py \
    --dataset syn1 \
    --seed 42 \
    --policy PPO \
    --n_epochs 2 \
    --ppo_epochs 2 \
    --n_rollouts 8 \
    --lr 0.0001 \
    --max_size 15 \
    --use_causal True \
    --random_edge_drop 0.0 \
    --eval_every 1 \
    --dry_run_debug >> "$RESULT_FILE" 2>&1

echo "[$(date)] Baseline test completed" | tee -a "$RESULT_FILE"

## ==================== Test 3: With Dropout (No Preserve) =====================
echo ""
echo "=========================================="
echo "TEST 3: Edge dropout (p=0.2, no preserve)"
echo "=========================================="
echo ""

RESULT_FILE="results/test_edge_dropout_simple.txt"
> "$RESULT_FILE"

echo "[$(date)] Running test with dropout=0.2, no connectivity preservation" | tee -a "$RESULT_FILE"

srun --gres=gpu:1 "${PYBIN}" run_RG_explainer.py \
    --dataset syn1 \
    --seed 42 \
    --policy PPO \
    --n_epochs 2 \
    --ppo_epochs 2 \
    --n_rollouts 8 \
    --lr 0.0001 \
    --max_size 15 \
    --use_causal True \
    --random_edge_drop 0.2 \
    --eval_every 1 \
    --dry_run_debug >> "$RESULT_FILE" 2>&1

echo "[$(date)] Simple dropout test completed" | tee -a "$RESULT_FILE"

## ==================== Test 4: With Dropout (Preserve Connectivity) =====================
echo ""
echo "=========================================="
echo "TEST 4: Edge dropout (p=0.2, preserve)"
echo "=========================================="
echo ""

RESULT_FILE="results/test_edge_dropout_preserve.txt"
> "$RESULT_FILE"

echo "[$(date)] Running test with dropout=0.2, WITH connectivity preservation" | tee -a "$RESULT_FILE"

srun --gres=gpu:1 "${PYBIN}" run_RG_explainer.py \
    --dataset syn1 \
    --seed 42 \
    --policy PPO \
    --n_epochs 2 \
    --ppo_epochs 2 \
    --n_rollouts 8 \
    --lr 0.0001 \
    --max_size 15 \
    --use_causal True \
    --random_edge_drop 0.2 \
    --preserve_connectivity \
    --eval_every 1 \
    --dry_run_debug >> "$RESULT_FILE" 2>&1

echo "[$(date)] Preserve connectivity test completed" | tee -a "$RESULT_FILE"

## ==================== Summary =====================
echo ""
echo "=========================================="
echo "ALL TESTS COMPLETED"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - results/test_edge_dropout_baseline.txt"
echo "  - results/test_edge_dropout_simple.txt"
echo "  - results/test_edge_dropout_preserve.txt"
echo ""
echo "Check for:"
echo "  1. [EdgeDrop] log messages in training output"
echo "  2. [Reward] log showing original graph usage"
echo "  3. [ExpansionEnv] logs showing edge counts"
echo ""

echo "[$(date)] Test suite finished successfully"

