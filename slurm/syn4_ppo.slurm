#!/bin/bash
# File: syn4_ppo.slurm

## ===================== SLURM =====================
#SBATCH -J rgx_syn4_ppo
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 7:00:00
#SBATCH --array=0
#SBATCH -o slurm_out/%x_%A_%a.out
#SBATCH -e slurm_out/%x_%A_%a.err
## =================================================

set -euo pipefail

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


# "$PYBIN" - <<'PY'
# import sys, numpy, scipy, torch
# print("python:", sys.executable)
# print("numpy:", numpy.__version__)
# print("scipy:", scipy.__version__)
# print("torch:", torch.__version__)
# PY

## ==================== Params =====================
SEED="${SLURM_ARRAY_TASK_ID}"
DATASET="syn4"
POLICY="PPO"
N_EPOCHS=30
PPO_EPOCHS=6
N_ROLLOUTS=10
USE_CAUSAL=False

# Regularization - BALANCED for proper size control
# SIZE_REG=0.01        # Increased 100x: was 0.0001 (too weak)
# SIM_REG=1.0          # Increased 10x: was 0.001 (need stronger similarity)
# RADIUS_PENALTY=0.0   # Increased 10x: was 0.01 (need radius control)

# Learning rate - Higher for value function learning
LR=0.0003            # Increased 3x: was 0.0001 default (value function needs faster learning)

# Causal mechanism - FULL IMPLEMENTATION with CE-difference
CAUSAL_WEIGHT=0.3   # Moderate weight with warmup (0.05-0.2 range)
CAUSAL_WARMUP=0     # Longer warmup for stability (15-25 epochs)

# Size targets - Enable proper size control
TARGET_SIZE_K=10     # Target size closer to ground truth (11-13 nodes)
LAMBDA_SIZE=0.5      # Enable hinge loss for size control
LAMBDA_RADIUS=0.5    # Enable adaptive radius penalty
STEP_COST=0.01       # Enable step cost to encourage efficiency

# Fidelity threshold - DISABLE artificial bonuses
FIDELITY_THRESH=0.0  # Disable EXTREME bonuses (causing value explosion)

# Debug and size parameters
DEBUG_MODE=0
MAX_SIZE=15
REWARD_NORM=1

# === NEW PPO REFACTORING PARAMETERS ===
# Delta rewards and STOP bonuses
DELTA_REWARD=1
STOP_BONUS_OK=0.5
STOP_PENALTY_BAD=0.2
TARGET_SIZE_K_NEW=8  # Dataset-specific target size

# λ Schedules
WARMUP_EPOCHS=5
SCHEDULE_EPOCHS=10
LAMBDA_SIZE_FINAL=0.01
LAMBDA_RADIUS_FINAL=0.1
LAMBDA_SIM_FINAL=0.01
LAMBDA_SIM_START=0.0

# Normalization
TRAJ_REWARD_NORM=1
STD_ADVANTAGE=1

# Entropy scheduling
ENTROPY_COEFF_START=1e-3
ENTROPY_COEFF_END=1e-4

# Debug
PPO_DEBUG=1

SIZE_REG=0.01
SIM_REG=1.0
RADIUS_PENALTY=0.0

# === Additional Hyperparameters (previously hard-coded) ===
VALUE_LOSS_COEF=0.3
VALUE_LOSS_SCALE=0.5
CONFIDENCE_THRESHOLD=0.5
CONFIDENCE_TOLERANCE=0.05
CAUSAL_ACTION_WEIGHT=0.2
CAUSAL_GAMMA=0.5
ICE_SCORE_DEFAULT=-5.0
STOP_MASK_VALUE=-1e9
MIN_STEPS_BEFORE_STOP=8
KL_ADAPT_HIGH_MULTIPLIER=2.0
KL_ADAPT_LOW_MULTIPLIER=0.5
KL_ADAPT_HIGH_THRESHOLD=2.0
KL_ADAPT_LOW_THRESHOLD=0.5
ENTROPY_DECAY_FRACTION=0.7

# Set result file name based on causal flag
if [ "${USE_CAUSAL}" = "True" ] || [ "${USE_CAUSAL}" = "true" ]; then
    RESULT_FILE="results/${DATASET}_ppo_causal_seed_${SEED}.txt"
else
    RESULT_FILE="results/${DATASET}_ppo_seed_${SEED}.txt"
fi

# Clean the result file before starting
: > "$RESULT_FILE"

# Mirror all subsequent stdout/stderr to both the results file and SLURM log
exec > >(tee -a "$RESULT_FILE")
exec 2>&1

echo "[$(date)] Starting seed=$SEED"

## ============== GPU & Torch Probes ===============
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# Confirm GPU is allocated to THIS step
srun --gres=gpu:1 nvidia-smi

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
    --seed "${SEED}" \
    --policy "${POLICY}" \
    --n_epochs "${N_EPOCHS}" \
    --ppo_epochs "${PPO_EPOCHS}" \
    --n_rollouts "${N_ROLLOUTS}" \
    --lr "${LR}" \
    --size_reg "${SIZE_REG}" \
    --sim_reg "${SIM_REG}" \
    --radius_penalty "${RADIUS_PENALTY}" \
    --causal_weight "${CAUSAL_WEIGHT}" \
    --causal_warmup_epochs "${CAUSAL_WARMUP}" \
    --lambda_size "${LAMBDA_SIZE}" \
    --lambda_radius "${LAMBDA_RADIUS}" \
    --step_cost "${STEP_COST}" \
    --use_causal "${USE_CAUSAL}" \
    --clip_range_policy 0.2 \
    --clip_range_value 0.5 \
    --kl_target 0.025 \
    --adaptive_kl_penalty \
    --entropy_decay_start 0.001 \
    --entropy_decay_end 0.0005 \
    --fidelity_thresh "${FIDELITY_THRESH}" \
    --max_size "${MAX_SIZE}" \
    --debug_mode "${DEBUG_MODE}" \
    --target_size_k "${TARGET_SIZE_K}" \
    --reward_norm \
    --delta_reward "${DELTA_REWARD}" \
    --stop_bonus_ok "${STOP_BONUS_OK}" \
    --stop_penalty_bad "${STOP_PENALTY_BAD}" \
    --target_size_k "${TARGET_SIZE_K_NEW}" \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --schedule_epochs "${SCHEDULE_EPOCHS}" \
    --lambda_size_final "${LAMBDA_SIZE_FINAL}" \
    --lambda_radius_final "${LAMBDA_RADIUS_FINAL}" \
    --lambda_sim_final "${LAMBDA_SIM_FINAL}" \
    --lambda_sim_start "${LAMBDA_SIM_START}" \
    --traj_reward_norm "${TRAJ_REWARD_NORM}" \
    --std_advantage "${STD_ADVANTAGE}" \
    --entropy_coeff_start "${ENTROPY_COEFF_START}" \
    --entropy_coeff_end "${ENTROPY_COEFF_END}" \
    --ppo_debug \
    --value_loss_coef "${VALUE_LOSS_COEF}" \
    --value_loss_scale "${VALUE_LOSS_SCALE}" \
    --confidence_threshold "${CONFIDENCE_THRESHOLD}" \
    --confidence_tolerance "${CONFIDENCE_TOLERANCE}" \
    --causal_action_weight "${CAUSAL_ACTION_WEIGHT}" \
    --causal_gamma "${CAUSAL_GAMMA}" \
    --ice_score_default "${ICE_SCORE_DEFAULT}" \
    --stop_mask_value="${STOP_MASK_VALUE}" \
    --min_steps_before_stop "${MIN_STEPS_BEFORE_STOP}" \
    --kl_adapt_high_multiplier "${KL_ADAPT_HIGH_MULTIPLIER}" \
    --kl_adapt_low_multiplier "${KL_ADAPT_LOW_MULTIPLIER}" \
    --kl_adapt_high_threshold "${KL_ADAPT_HIGH_THRESHOLD}" \
    --kl_adapt_low_threshold "${KL_ADAPT_LOW_THRESHOLD}" \
    --entropy_decay_fraction "${ENTROPY_DECAY_FRACTION}"

echo "[$(date)] Finished seed=$SEED"
