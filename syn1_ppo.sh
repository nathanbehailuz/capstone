#!/bin/bash
# File: rgx_gpu_array.slurm

########################
# SLURM resources
########################
#SBATCH -J rgx_syn1_ppo
#SBATCH -p nvidia
#SBATCH --gres=gpu:1                 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G                    # bump if needed (e.g., 64G/128G)
#SBATCH -t 12:00:00                  # bump if needed (e.g., 40:00:00)
#SBATCH --array=0-9
#SBATCH -o slurm_out/%x_%A_%a_%j.out
#SBATCH -e slurm_out/%x_%A_%a_%j.err
# Optional notifications:
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=<you@nyu.edu>

########################
# Housekeeping
########################
set -euo pipefail
mkdir -p slurm_out

echo "[$(date)] Node: $(hostname)"
echo "[$(date)] JOBID=$SLURM_JOB_ID  ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

# Prefer scratch for I/O
WORKDIR="${SCRATCH:-$PWD}/rgx_runs/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WORKDIR"

# Define your project root ON SHARED FS (e.g., your home repo clone)
PROJECT_ROOT="$HOME/capstone"   # <<< CHANGE THIS
cd "$WORKDIR"

########################
# Environment
########################
module purge
module load miniconda/3-4.11.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv                    

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

########################
# Parameters
########################
SEED="${SLURM_ARRAY_TASK_ID}"
DATASET="syn1"
POLICY="PPO"
N_EPOCHS=10
SIZE_REG=0.01
SIM_REG=1.0
RADIUS_PENALTY=0.1

RUN_DIR="$WORKDIR/${DATASET}/seed_${SEED}"
mkdir -p "$RUN_DIR"
LOGFILE="$RUN_DIR/run.log"

########################
# Stage code (optional but good for reproducibility)
########################
# Keeps each task self-contained under scratch; speeds up random small-file I/O.
rsync -a --delete --exclude '.git' "$PROJECT_ROOT/" "$RUN_DIR/code/"
cd "$RUN_DIR/code/"

########################
# Launch (use srun)
########################
echo "[$(date)] Starting seed=$SEED" | tee -a "$LOGFILE"

srun -c "${SLURM_CPUS_PER_TASK:-4}" python run_RG_explainer.py \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --policy "$POLICY" \
  --n_epochs "$N_EPOCHS" \
  --size_reg "$SIZE_REG" \
  --sim_reg "$SIM_REG" \
  --radius_penalty "$RADIUS_PENALTY" \
  > "$RUN_DIR/stdout.txt" 2> "$RUN_DIR/stderr.txt"

echo "[$(date)] Finished seed=$SEED" | tee -a "$LOGFILE"

########################
# (Optional) Stage-out results
########################
# RESULTS_DIR="$HOME/rgx_results/${SLURM_JOB_ID}/seed_${SEED}"
# mkdir -p "$RESULTS_DIR"
# rsync -a "$RUN_DIR/" "$RESULTS_DIR/"

