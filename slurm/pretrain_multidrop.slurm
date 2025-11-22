#!/bin/bash
# File: pretrain_multidrop.slurm
# Multi-Drop Ratio Pretraining Experiment
# Tests pretraining at edge drop ratios: [0, 5, 10, 15, 20, 25, 30]%

## ===================== SLURM =====================
#SBATCH -J pretrain_multidrop
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --array=0
#SBATCH -o slurm_out/%x_%A_%a.out
#SBATCH -e slurm_out/%x_%A_%a.err
## =================================================

set -euo pipefail

# Always run from the submit dir
cd "${SLURM_SUBMIT_DIR}"

# Folders for logs & outputs
mkdir -p slurm_out results/pretraining_tests

echo "[$(date)] Node: $(hostname)"
echo "[$(date)] JOBID=$SLURM_JOB_ID  ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "[$(date)] CWD: $PWD"

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

# Threads & perf-friendly defaults
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

# CUDA placement: Slurm sets CUDA_VISIBLE_DEVICES; honor it.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# PyTorch memory allocator: reduces fragmentation in long runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# (Optional) Dataloader workers hint your code can read; harmless if unused
export DATALOADER_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-4} > 1 ? ${SLURM_CPUS_PER_TASK:-4} - 1 : 1 ))

# Helpful runtime flags while debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTHONNOUSERSITE=1

## ==================== Params =====================
SEED="${SLURM_ARRAY_TASK_ID}"
DATASET="syn1"  # Can be: syn1, syn2, syn3, syn4, ba2, mutag

# Note: Edge drop ratios [0, 5, 10, 15, 20, 25, 30]% are hardcoded in pretrain_test_runner.py
# This script will pretrain at each ratio and then train RL models

# Pretraining parameters
PRETRAIN_LIST=50         # List-based pretraining iterations
PRETRAIN_SET=100         # Set-based pretraining iterations
PRETRAIN_BATCH_SIZE=32   # Pretraining batch size
PRETRAIN_L_ITER=50       # Locator pretraining iterations
PRETRAIN_L_SAMPLE=1.0    # Locator sample rate

# Architecture
MAX_SIZE=15
N_HOP=3
HIDDEN_SIZE=64
WITH_ATTR=1

# RL Training parameters (after pretraining)
N_EPOCHS=20              # Full RL training epochs per pretrained model
PPO_EPOCHS=4             # PPO optimization epochs
N_ROLLOUTS=32            # Number of rollouts
G_BATCH_SIZE=128         # Generator batch size
MINIBATCH_SIZE=256       # PPO minibatch size

# Learning rates
LR=0.0003                # Main learning rate
G_LR=0.01                # Generator learning rate during pretraining
L_LR=0.01                # Locator learning rate during pretraining

# Regularization - BALANCED for proper size control
SIZE_REG=0.01            # Size regularization
SIM_REG=1.0              # Similarity regularization
RADIUS_PENALTY=0.1       # Radius penalty

# Causal mechanism
USE_CAUSAL=True
CAUSAL_WEIGHT=0.3        # Causal weight
CAUSAL_WARMUP=0          # Causal warmup epochs

# Size targets - Enable proper size control
TARGET_SIZE_K=10         # Target size closer to ground truth
LAMBDA_SIZE=0.5          # Enable hinge loss for size control
LAMBDA_RADIUS=0.5        # Enable adaptive radius penalty
STEP_COST=0.01           # Enable step cost to encourage efficiency

# Fidelity threshold
FIDELITY_THRESH=0.0      # Disable EXTREME bonuses

# STOP action parameters
DELTA_REWARD=1
STOP_BONUS_OK=0.10       # Reduced from 0.5 to avoid over-rewarding
STOP_PENALTY_BAD=0.30    # Increased from 0.2 to discourage bad STOPs
REWARD_SCALE=5.0         # Global reward scaling factor
STOP_BIAS=0.5            # STOP action bias

# λ Schedules - More aggressive warmup and scheduling
WARMUP_EPOCHS=10         # Increased for better stability
SCHEDULE_EPOCHS=20       # Increased for gradual ramp-up
LAMBDA_SIZE_FINAL=0.02   # Increased for stronger size control
LAMBDA_RADIUS_FINAL=0.10
LAMBDA_RADIUS_MIN=0.2    # Minimum radius penalty
LAMBDA_SIM_FINAL=0.005   # Reduced similarity pressure
LAMBDA_SIM_START=0.0
LAMBDA_CONF=0.1          # Confidence lambda

# Normalization
REWARD_NORM=1
TRAJ_REWARD_NORM=1
STD_ADVANTAGE=1

# Entropy scheduling
ENTROPY_COEFF_START=0.003  # Increased for better exploration
ENTROPY_COEFF_END=0.0005   # Increased for sustained exploration
ENTROPY_DECAY_FRACTION=0.7
ENTROPY_DECAY_START=0.001
ENTROPY_DECAY_END=0.0001
ENTROPY_DECAY_FINAL=0.0

# PPO clipping and optimization
CLIP_EPS=0.2
CLIP_RANGE_POLICY=0.2
CLIP_RANGE_VALUE=0.25
PPO_CLIP=0.1
VF_CLIP=0.5
GRAD_CLIP=1.0
MAX_GRAD_NORM=0.5

# Value function
VALUE_LOSS_COEF=0.3
VALUE_LOSS_SCALE=0.5
USE_VALUE_CLIP=1
GAMMA=0.99
USE_GAE=1
GAE_LAMBDA=0.95

# KL divergence adaptive penalty
ADAPTIVE_KL_PENALTY=0
KL_TARGET=0.0075
KL_ADAPT_HIGH_MULTIPLIER=2.0
KL_ADAPT_LOW_MULTIPLIER=0.5
KL_ADAPT_HIGH_THRESHOLD=2.0
KL_ADAPT_LOW_THRESHOLD=0.5

# Causal-specific parameters
CONFIDENCE_THRESHOLD=0.5
CONFIDENCE_TOLERANCE=0.05
CAUSAL_ACTION_WEIGHT=0.2
CAUSAL_GAMMA=0.5
ICE_SCORE_DEFAULT="-5.0"

# STOP masking
STOP_MASK_VALUE="-1e9"
MIN_STEPS_BEFORE_STOP=8

# Locator updates
UPDATE_L_SAMPLE_RATE=0.2
N_G_UPDATES=1
N_L_UPDATES=10

# Evaluation
EVAL_EVERY=1

# Debug
DEBUG_MODE=0
PPO_DEBUG=0
DRY_RUN_DEBUG=0

# Output directory
OUTPUT_DIR="results/pretraining_tests"

# Policy
POLICY="PPO"
FRONTIER_SHAPING_BETA=0.0

# Edge score aggregation
EDGE_SCORE_AGG="max"
DEBUG_EVAL_GRAPH=-1

# Log file for this job
RESULT_FILE="${OUTPUT_DIR}/${DATASET}_multidrop_seed_${SEED}_slurm_${SLURM_JOB_ID}.txt"

# Clean the result file before starting
> "$RESULT_FILE"

echo "======================================" | tee -a "$RESULT_FILE"
echo "Multi-Drop Ratio Pretraining Experiment" | tee -a "$RESULT_FILE"
echo "======================================" | tee -a "$RESULT_FILE"
echo "[$(date)] Starting seed=$SEED" | tee -a "$RESULT_FILE"
echo "Dataset: $DATASET" | tee -a "$RESULT_FILE"
echo "Edge Drop Ratios: [0, 5, 10, 15, 20, 25, 30]%" | tee -a "$RESULT_FILE"
echo "Pretrain Iterations: List=$PRETRAIN_LIST, Set=$PRETRAIN_SET" | tee -a "$RESULT_FILE"
echo "RL Training Epochs: $N_EPOCHS" | tee -a "$RESULT_FILE"
echo "======================================" | tee -a "$RESULT_FILE"

## ============== GPU & Torch Probes ===============
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" | tee -a "$RESULT_FILE"

# Confirm GPU is allocated to THIS step
srun --gres=gpu:1 nvidia-smi | tee -a "$RESULT_FILE"

## ================== TRAINING =====================
echo "[$(date)] Starting multi-drop pretraining experiment..." | tee -a "$RESULT_FILE"

# Launch pretrain_test_runner.py with all parameters
srun --export=ALL --gres=gpu:1 -c "${SLURM_CPUS_PER_TASK:-4}" \
  "${PYBIN}" pretrain_test_runner.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --pretrain_list "${PRETRAIN_LIST}" \
    --pretrain_set "${PRETRAIN_SET}" \
    --pretrain_g_batch_size "${PRETRAIN_BATCH_SIZE}" \
    --pretrain_l_iter "${PRETRAIN_L_ITER}" \
    --pretrain_l_sample_rate "${PRETRAIN_L_SAMPLE}" \
    --g_lr "${G_LR}" \
    --l_lr "${L_LR}" \
    --max_size "${MAX_SIZE}" \
    --n_hop "${N_HOP}" \
    --hidden_size "${HIDDEN_SIZE}" \
    --with_attr \
    --n_epochs "${N_EPOCHS}" \
    --ppo_epochs "${PPO_EPOCHS}" \
    --n_rollouts "${N_ROLLOUTS}" \
    --g_batch_size "${G_BATCH_SIZE}" \
    --minibatch_size "${MINIBATCH_SIZE}" \
    --lr "${LR}" \
    --size_reg "${SIZE_REG}" \
    --sim_reg "${SIM_REG}" \
    --radius_penalty "${RADIUS_PENALTY}" \
    --use_causal "${USE_CAUSAL}" \
    --causal_weight "${CAUSAL_WEIGHT}" \
    --causal_warmup_epochs "${CAUSAL_WARMUP}" \
    --target_size_k "${TARGET_SIZE_K}" \
    --lambda_size "${LAMBDA_SIZE}" \
    --lambda_radius "${LAMBDA_RADIUS}" \
    --lambda_radius_min "${LAMBDA_RADIUS_MIN}" \
    --step_cost "${STEP_COST}" \
    --fidelity_thresh "${FIDELITY_THRESH}" \
    --stop_bias "${STOP_BIAS}" \
    --delta_reward "${DELTA_REWARD}" \
    --stop_bonus_ok "${STOP_BONUS_OK}" \
    --stop_penalty_bad "${STOP_PENALTY_BAD}" \
    --reward_scale "${REWARD_SCALE}" \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --schedule_epochs "${SCHEDULE_EPOCHS}" \
    --lambda_size_final "${LAMBDA_SIZE_FINAL}" \
    --lambda_radius_final "${LAMBDA_RADIUS_FINAL}" \
    --lambda_sim_start "${LAMBDA_SIM_START}" \
    --lambda_sim_final "${LAMBDA_SIM_FINAL}" \
    --lambda_conf "${LAMBDA_CONF}" \
    --reward_norm \
    --traj_reward_norm "${TRAJ_REWARD_NORM}" \
    --std_advantage "${STD_ADVANTAGE}" \
    --entropy_coeff_start "${ENTROPY_COEFF_START}" \
    --entropy_coeff_end "${ENTROPY_COEFF_END}" \
    --entropy_decay_fraction "${ENTROPY_DECAY_FRACTION}" \
    --entropy_decay_start "${ENTROPY_DECAY_START}" \
    --entropy_decay_end "${ENTROPY_DECAY_END}" \
    --entropy_decay_final "${ENTROPY_DECAY_FINAL}" \
    --clip_eps "${CLIP_EPS}" \
    --clip_range_policy "${CLIP_RANGE_POLICY}" \
    --clip_range_value "${CLIP_RANGE_VALUE}" \
    --ppo_clip "${PPO_CLIP}" \
    --vf_clip "${VF_CLIP}" \
    --grad_clip "${GRAD_CLIP}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --value_loss_coef "${VALUE_LOSS_COEF}" \
    --value_loss_scale "${VALUE_LOSS_SCALE}" \
    --use_value_clip \
    --gamma "${GAMMA}" \
    --use_gae \
    --gae_lambda "${GAE_LAMBDA}" \
    --kl_target "${KL_TARGET}" \
    --kl_adapt_high_multiplier "${KL_ADAPT_HIGH_MULTIPLIER}" \
    --kl_adapt_low_multiplier "${KL_ADAPT_LOW_MULTIPLIER}" \
    --kl_adapt_high_threshold "${KL_ADAPT_HIGH_THRESHOLD}" \
    --kl_adapt_low_threshold "${KL_ADAPT_LOW_THRESHOLD}" \
    --confidence_threshold "${CONFIDENCE_THRESHOLD}" \
    --confidence_tolerance "${CONFIDENCE_TOLERANCE}" \
    --causal_action_weight "${CAUSAL_ACTION_WEIGHT}" \
    --causal_gamma "${CAUSAL_GAMMA}" \
    --ice_score_default="${ICE_SCORE_DEFAULT}" \
    --stop_mask_value="${STOP_MASK_VALUE}" \
    --min_steps_before_stop "${MIN_STEPS_BEFORE_STOP}" \
    --update_l_sample_rate "${UPDATE_L_SAMPLE_RATE}" \
    --n_g_updates "${N_G_UPDATES}" \
    --n_l_updates "${N_L_UPDATES}" \
    --eval_every "${EVAL_EVERY}" \
    --debug_mode "${DEBUG_MODE}" \
    --policy "${POLICY}" \
    --frontier_shaping_beta "${FRONTIER_SHAPING_BETA}" \
    --edge_score_agg "${EDGE_SCORE_AGG}" \
    --debug_eval_graph "${DEBUG_EVAL_GRAPH}" \
    --output_dir "${OUTPUT_DIR}" \
    >> "$RESULT_FILE" 2>&1

EXIT_CODE=$?

echo "" | tee -a "$RESULT_FILE"
echo "======================================" | tee -a "$RESULT_FILE"
echo "[$(date)] Finished seed=$SEED" | tee -a "$RESULT_FILE"
echo "Exit Code: $EXIT_CODE" | tee -a "$RESULT_FILE"
echo "======================================" | tee -a "$RESULT_FILE"

# List output files
echo "" | tee -a "$RESULT_FILE"
echo "Output files generated:" | tee -a "$RESULT_FILE"
ls -lh "${OUTPUT_DIR}/${DATASET}_multidrop_seed${SEED}"* 2>/dev/null | tee -a "$RESULT_FILE" || echo "No output files found" | tee -a "$RESULT_FILE"

exit $EXIT_CODE

