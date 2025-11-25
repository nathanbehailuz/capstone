#!/bin/bash
# File: analyze_results.slurm

## ===================== SLURM =====================
#SBATCH -J analyze_results
#SBATCH -p serial
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 0:30:00
#SBATCH -o slurm_out/analyze_%A_%a.out
#SBATCH -e slurm_out/analyze_%A_%a.err
## =================================================

set -euo pipefail

# Usage:
#   sbatch --array=0-9 slurm/analyze_results.slurm <dataset> [casual]

# Always run from the submit dir
cd "${SLURM_SUBMIT_DIR}"

# Create output directory
mkdir -p slurm_out

echo "[$(date)] Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "[$(date)] Node: $(hostname)"
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

## ================= Parse Arguments ================
DATASET=${1:-syn2}
CASUAL=${2:-0}
SEED=${SLURM_ARRAY_TASK_ID}

echo "[$(date)] Dataset: $DATASET"
echo "[$(date)] Seed: $SEED"
echo "[$(date)] Casual: $CASUAL"

## ==================== Run Analysis =================
CMD="${PYBIN} analysis/extract_ppo_performance.py ${DATASET} ${SEED}"

# Add casual flag if set to 1
if [ "$CASUAL" = "1" ]; then
  CMD="${CMD} --casual 1"
fi

echo "[$(date)] Running: $CMD"
$CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "[$(date)] ✓ Analysis completed successfully"
else
  echo "[$(date)] ✗ Analysis failed with exit code $EXIT_CODE"
  exit $EXIT_CODE
fi

echo "[$(date)] Job finished"


