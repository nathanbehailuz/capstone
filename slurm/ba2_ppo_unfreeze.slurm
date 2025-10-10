#!/bin/bash
# File: ba2_ppo_unfreeze.slurm
# Un-freeze PPO and give it gradient signal

## ===================== SLURM =====================
#SBATCH -J rgx_ba2_unfreeze
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
# Core Parameters
SEED="${SLURM_ARRAY_TASK_ID}"
DATASET="ba2"
POLICY="PPO"
MODEL="GNN"

# Training Parameters
N_EPOCHS=10
N_ROLLOUTS=5
PRETRAIN_G_BATCH_SIZE=32
G_BATCH_SIZE=512
N_G_UPDATES=1
N_L_UPDATES=10
EVAL_EVERY=1

# Learning Rates
L_LR=0.01
G_LR=0.01
LR=0.0003

# Regularization Parameters
SIZE_REG=0.01
SIM_REG=0.01
RADIUS_PENALTY=0.1
# ENTROPY_COEF removed - using entropy schedule instead

# PPO Parameters - UNFROZEN PPO
PPO_EPOCHS=8
MINIBATCH_SIZE=128
# CLIP_EPS removed - keeping only PPO_CLIP
GRAD_CLIP=1.0
PPO_CLIP=0.2
KL_TARGET=0.03
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
MAX_SIZE=20

# Causal Parameters
CAUSAL_WEIGHT=0.5
CAUSAL_WARMUP_EPOCHS=5

# Reward Parameters
TARGET_SIZE_K=10
LAMBDA_SIZE=0.5
LAMBDA_RADIUS=1.0
LAMBDA_RADIUS_MIN=0.2
STEP_COST=0.001
STOP_BIAS=0.5
FIDELITY_THRESH=0.9

# Entropy Decay - USING ENTROPY SCHEDULE
ENTROPY_DECAY_START=0.001
ENTROPY_DECAY_END=0.0001
ENTROPY_DECAY_FINAL=0.0
ENTROPY_COEFF_START=0.01
ENTROPY_COEFF_END=0.001

# Other Parameters
FRONTIER_SHAPING_BETA=0.0
EDGE_SCORE_AGG="max"
DEBUG_EVAL_GRAPH=-1

RESULT_FILE="results/${DATASET}_${POLICY,,}_unfreeze_seed_${SEED}.txt"

# Clean the result file before starting
> "$RESULT_FILE"

echo "[$(date)] Starting seed=$SEED (UNFROZEN PPO)" | tee -a "$RESULT_FILE"

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

echo "[$(date)] Finished seed=$SEED (UNFROZEN PPO)" | tee -a "$RESULT_FILE"
