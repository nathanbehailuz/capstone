#!/bin/bash
# File: extract.slurm

## ===================== SLURM =====================
#SBATCH -J extract_results
#SBATCH -p serial
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0:30:00
#SBATCH -o slurm_out/extract_%j.out
#SBATCH -e slurm_out/extract_%j.err
## =================================================

set -euo pipefail

# Usage:
#   sbatch slurm/extract.slurm <dataset> [seed] [epoch] [casual]
# Examples:
#   sbatch slurm/extract.slurm syn1              # Extract all seeds and epochs
#   sbatch slurm/extract.slurm syn1 0            # Extract all epochs for seed 0
#   sbatch slurm/extract.slurm syn1 0 5          # Extract epoch 5 for seed 0
#   sbatch slurm/extract.slurm syn1 0 all 1      # Extract seed 0 with casual pattern
#   sbatch slurm/extract.slurm syn2 all all 1    # Extract all with casual pattern

# Always run from the submit dir
cd "${SLURM_SUBMIT_DIR}"

# Create output directory
mkdir -p slurm_out

echo "[$(date)] Extract Job ID: $SLURM_JOB_ID"
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
SEED=${2:-0}
EPOCH=${3:-all}
CASUAL=${4:-0}

echo "[$(date)] Dataset: $DATASET"
echo "[$(date)] Seed: $SEED"
echo "[$(date)] Epoch: $EPOCH"
echo "[$(date)] Casual: $CASUAL"

## ==================== Run Extract =================
EXTRACT_CMD="${PYBIN} analysis/extract.py ${DATASET}"

# Add seed if not 'all'
if [ "$SEED" != "all" ]; then
  EXTRACT_CMD="${EXTRACT_CMD} ${SEED}"
  
  # Add epoch if not 'all' (only valid if seed is specified)
  if [ "$EPOCH" != "all" ]; then
    EXTRACT_CMD="${EXTRACT_CMD} ${EPOCH}"
  fi
fi

# Add casual flag if set to 1
if [ "$CASUAL" = "1" ]; then
  EXTRACT_CMD="${EXTRACT_CMD} --casual 1"
fi

echo "[$(date)] Running: $EXTRACT_CMD"
$EXTRACT_CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "[$(date)] ✓ Extraction completed successfully"
else
  echo "[$(date)] ✗ Extraction failed with exit code $EXIT_CODE"
  exit $EXIT_CODE
fi

echo "[$(date)] Job finished"

