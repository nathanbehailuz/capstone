#!/bin/bash
# File: syn1_ppo.slurm

## SLURM
#SBATCH -J rgx_syn1_ppo
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --array=0-9
#SBATCH -o slurm_out/%x_%A_%a_%j.out
#SBATCH -e slurm_out/%x_%A_%a_%j.err
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=nz2212@nyu.edu

set -euo pipefail

# Work from the directory where you ran `sbatch`
cd "$SLURM_SUBMIT_DIR"

# Folders for logs & outputs
mkdir -p slurm_out results

echo "[$(date)] Node: $(hostname)"
echo "[$(date)] JOBID=$SLURM_JOB_ID  ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "[$(date)] CWD: $PWD"

## Env
module purge
module load miniconda/3-4.11.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv        # <-- use your actual env name

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

## Params
SEED="${SLURM_ARRAY_TASK_ID}"
DATASET="syn1"
POLICY="PPO"
N_EPOCHS=10
SIZE_REG=0.01
SIM_REG=1.0
RADIUS_PENALTY=0.1

# results/syn1_ppo_seed0.txt ... results/syn1_ppo_seed9.txt
RESULT_FILE="results/${DATASET}_${POLICY,,}_seed${SEED}.txt"

echo "[$(date)] Starting seed=$SEED" | tee -a "$RESULT_FILE"

srun -c "${SLURM_CPUS_PER_TASK:-4}" python run_RG_explainer.py \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --policy "$POLICY" \
  --n_epochs "$N_EPOCHS" \
  --size_reg "$SIZE_REG" \
  --sim_reg "$SIM_REG" \
  --radius_penalty "$RADIUS_PENALTY" \
  > "$RESULT_FILE" 2>&1

echo "[$(date)] Finished seed=$SEED" | tee -a "$RESULT_FILE"
