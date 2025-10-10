#!/bin/bash
# File: ba2_ppo_bugcheck.slurm
# Bug check version with extreme parameters to test clip fraction

## ===================== SLURM =====================
#SBATCH -J rgx_ba2_bugcheck
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 1:00:00
#SBATCH --array=0-0
#SBATCH -o slurm_out/%x_%A_%a.out
#SBATCH -e slurm_out/%x_%A_%a.err
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

# Activate your env by absolute path
conda activate /home/nz2212/.conda/envs/rlgn

# Sanity: use the env's Python explicitly
PYBIN="${CONDA_PREFIX}/bin/python"

# Helpful runtime flags while debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTHONNOUSERSITE=1

## ==================== Params =====================
# Core Parameters
SEED="${SLURM_ARRAY_TASK_ID}"
DATASET="ba2"
POLICY="PPO"
MODEL="GNN"

# Training Parameters - SHORT TEST
N_EPOCHS=3
N_ROLLOUTS=4
PRETRAIN_G_BATCH_SIZE=32
G_BATCH_SIZE=128
N_G_UPDATES=1
N_L_UPDATES=10
EVAL_EVERY=1

# Learning Rates
L_LR=0.01
G_LR=0.01
LR=0.001  # LARGE LR FOR BUG CHECK

# Regularization Parameters
SIZE_REG=0.1
SIM_REG=0.01
RADIUS_PENALTY=0.1

# PPO Parameters - BUG CHECK SETTINGS
PPO_EPOCHS=4
MINIBATCH_SIZE=64
GRAD_CLIP=1.0
PPO_CLIP=0.00001  # EXTREMELY SMALL CLIP FOR BUG CHECK
KL_TARGET=0.0075
GAMMA=0.99
GAE_LAMBDA=0.95
MAX_GRAD_NORM=1.0
VF_CLIP=0.2

# Pretraining Parameters
PRETRAIN_L_SAMPLE_RATE=1.0
UPDATE_L_SAMPLE_RATE=0.2
PRETRAIN_L_ITER=200
PRETRAIN_LIST=10
PRETRAIN_SET=25

# Model Architecture
HIDDEN_SIZE=64
N_HOP=3
MAX_SIZE=12

# Causal Parameters
CAUSAL_WEIGHT=0.3
CAUSAL_WARMUP_EPOCHS=2

# Reward Parameters
TARGET_SIZE_K=10
LAMBDA_SIZE=0.5
LAMBDA_RADIUS=2.0
LAMBDA_RADIUS_MIN=1.0
STEP_COST=0.01
STOP_BIAS=0.25
FIDELITY_THRESH=0.8

# Entropy Decay
ENTROPY_DECAY_START=0.001
ENTROPY_DECAY_END=0.0001
ENTROPY_DECAY_FINAL=0.0
ENTROPY_COEFF_START=0.01
ENTROPY_COEFF_END=0.001

# Other Parameters
FRONTIER_SHAPING_BETA=0.3
EDGE_SCORE_AGG="max"
DEBUG_EVAL_GRAPH=-1

RESULT_FILE="results/${DATASET}_${POLICY,,}_bugcheck_seed_${SEED}.txt"

# Clean the result file before starting
> "$RESULT_FILE"

echo "[$(date)] Starting seed=$SEED (BUG CHECK - PPO_CLIP=$PPO_CLIP, LR=$LR)" | tee -a "$RESULT_FILE"

## ================== TRAINING =====================
# Launch your script INSIDE the GPU step, using the env's python
srun --export=ALL --gres=gpu:1 -c "${SLURM_CPUS_PER_TASK:-4}" \
  "${PYBIN}" run_RG_explainer.py \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --seed "${SEED}" \
    --with_attr \
    --n_hop "${N_HOP}" \
    --max_size "${MAX_SIZE}" \
    --pretrain_l_sample_rate "${PRETRAIN_L_SAMPLE_RATE}" \
    --update_l_sample_rate "${UPDATE_L_SAMPLE_RATE}" \
    --l_lr "${L_LR}" \
    --pretrain_l_iter "${PRETRAIN_L_ITER}" \
    --hidden_size "${HIDDEN_SIZE}" \
    --g_lr "${G_LR}" \
    --n_rollouts "${N_ROLLOUTS}" \
    --pretrain_g_batch_size "${PRETRAIN_G_BATCH_SIZE}" \
    --g_batch_size "${G_BATCH_SIZE}" \
    --size_reg "${SIZE_REG}" \
    --sim_reg "${SIM_REG}" \
    --radius_penalty "${RADIUS_PENALTY}" \
    --pretrain_list "${PRETRAIN_LIST}" \
    --pretrain_set "${PRETRAIN_SET}" \
    --n_epochs "${N_EPOCHS}" \
    --n_g_updates "${N_G_UPDATES}" \
    --n_l_updates "${N_L_UPDATES}" \
    --eval_every "${EVAL_EVERY}" \
    --policy "${POLICY}" \
    --ppo_epochs "${PPO_EPOCHS}" \
    --minibatch_size "${MINIBATCH_SIZE}" \
    --grad_clip "${GRAD_CLIP}" \
    --frontier_shaping_beta "${FRONTIER_SHAPING_BETA}" \
    --entropy_decay_start "${ENTROPY_DECAY_START}" \
    --entropy_decay_end "${ENTROPY_DECAY_END}" \
    --entropy_decay_final "${ENTROPY_DECAY_FINAL}" \
    --use_causal \
    --causal_weight "${CAUSAL_WEIGHT}" \
    --causal_warmup_epochs "${CAUSAL_WARMUP_EPOCHS}" \
    --target_size_k "${TARGET_SIZE_K}" \
    --lambda_size "${LAMBDA_SIZE}" \
    --lambda_radius "${LAMBDA_RADIUS}" \
    --lambda_radius_min "${LAMBDA_RADIUS_MIN}" \
    --step_cost "${STEP_COST}" \
    --stop_bias "${STOP_BIAS}" \
    --fidelity_thresh "${FIDELITY_THRESH}" \
    --ppo_clip "${PPO_CLIP}" \
    --kl_target "${KL_TARGET}" \
    --lr "${LR}" \
    --gamma "${GAMMA}" \
    --gae_lambda "${GAE_LAMBDA}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --use_value_clip \
    --entropy_coeff_start "${ENTROPY_COEFF_START}" \
    --entropy_coeff_end "${ENTROPY_COEFF_END}" \
    --edge_score_agg "${EDGE_SCORE_AGG}" \
    --debug_eval_graph "${DEBUG_EVAL_GRAPH}" \
    --vf_clip "${VF_CLIP}" \
    --use_gae \
  >> "$RESULT_FILE" 2>&1

echo "[$(date)] Finished seed=$SEED (BUG CHECK)" | tee -a "$RESULT_FILE"
