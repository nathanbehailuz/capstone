#!/bin/bash
# File: ppo_hyperparam_sweep.slurm
# =========================
# ⚙️ PPO Hyperparameter Sweep for syn1
# Tests learning rate, entropy schedule, clipping, lambda size, and reward scale
# Saves all results in: results/ppo_tuning_summary.txt
# Total combinations: 3 × 3 × 3 × 3 × 4 = 108 runs
# =========================

## ===================== SLURM =====================
#SBATCH -J ppo_hyperparam_sweep
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 72:00:00
#SBATCH -o slurm_out/%x_%j.out
#SBATCH -e slurm_out/%x_%j.err
## =================================================

set -euo pipefail

# Always run from the submit dir
cd "${SLURM_SUBMIT_DIR}"

# Folders for logs & outputs
mkdir -p slurm_out results

echo "[$(date)] Starting PPO Hyperparameter Sweep"
echo "[$(date)] Node: $(hostname)"
echo "[$(date)] JOBID=$SLURM_JOB_ID"
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

## ============== GPU & Torch Probes ===============
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# Confirm GPU is allocated to THIS step
srun --gres=gpu:1 nvidia-smi

## ============== Hyperparameter Sweep =============
SUMMARY_FILE="results/ppo_tuning_summary.txt"
> "$SUMMARY_FILE"

echo "==========================" | tee -a "$SUMMARY_FILE"
echo "⚙️ PPO Hyperparameter Sweep" | tee -a "$SUMMARY_FILE"
echo "Dataset: syn1" | tee -a "$SUMMARY_FILE"
echo "Started: $(date)" | tee -a "$SUMMARY_FILE"
echo "==========================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Define sweep values (Phase 1–3 combined)
LRS=(0.0001 0.0003 0.001)
ENTROPY_STARTS=(0.003 0.005 0.01)
CLIP_RANGES=(0.15 0.2 0.25)
LAM_SIZE_FINALS=(0.01 0.02 0.05)
REWARD_SCALES=(1 3 5 10)

# Total combinations = 3 × 3 × 3 × 3 × 4 = 108
echo "Total combinations to test: ${#LRS[@]} × ${#ENTROPY_STARTS[@]} × ${#CLIP_RANGES[@]} × ${#LAM_SIZE_FINALS[@]} × ${#REWARD_SCALES[@]} = 108" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

COUNTER=1
for LR in "${LRS[@]}"; do
  for ENT in "${ENTROPY_STARTS[@]}"; do
    for CLIP in "${CLIP_RANGES[@]}"; do
      for LSIZE in "${LAM_SIZE_FINALS[@]}"; do
        for SCALE in "${REWARD_SCALES[@]}"; do
          echo "=====================================" | tee -a "$SUMMARY_FILE"
          echo "=== RUN #$COUNTER / 108 ===" | tee -a "$SUMMARY_FILE"
          echo "=====================================" | tee -a "$SUMMARY_FILE"
          echo "Configuration:" | tee -a "$SUMMARY_FILE"
          echo "  LR               = $LR" | tee -a "$SUMMARY_FILE"
          echo "  ENTROPY_START    = $ENT" | tee -a "$SUMMARY_FILE"
          echo "  CLIP_RANGE       = $CLIP" | tee -a "$SUMMARY_FILE"
          echo "  LAMBDA_SIZE_FINAL = $LSIZE" | tee -a "$SUMMARY_FILE"
          echo "  REWARD_SCALE     = $SCALE" | tee -a "$SUMMARY_FILE"
          echo "Started: $(date)" | tee -a "$SUMMARY_FILE"
          echo "" | tee -a "$SUMMARY_FILE"

          # Run training with current hyperparameter combination
          srun --export=ALL --gres=gpu:1 -c "${SLURM_CPUS_PER_TASK:-4}" \
            "${PYBIN}" run_RG_explainer.py \
              --dataset syn1 \
              --policy PPO \
              --n_epochs 3 \
              --ppo_epochs 4 \
              --n_rollouts 32 \
              --lr "$LR" \
              --clip_range_policy "$CLIP" \
              --clip_range_value 0.25 \
              --entropy_coeff_start "$ENT" \
              --entropy_coeff_end 0.0005 \
              --lambda_size_final "$LSIZE" \
              --lambda_radius_final 0.10 \
              --reward_scale "$SCALE" \
              --reward_norm \
              --std_advantage 1 \
              --value_loss_coef 0.3 \
              --value_loss_scale 0.5 \
              --seed 0 \
              --use_causal True \
              --debug_mode 0 >> "$SUMMARY_FILE" 2>&1

          echo "" | tee -a "$SUMMARY_FILE"
          echo "Finished: $(date)" | tee -a "$SUMMARY_FILE"
          echo "--- End of run #$COUNTER ---" | tee -a "$SUMMARY_FILE"
          echo "" | tee -a "$SUMMARY_FILE"
          echo "" | tee -a "$SUMMARY_FILE"
          
          COUNTER=$((COUNTER + 1))
        done
      done
    done
  done
done

echo "=====================================" | tee -a "$SUMMARY_FILE"
echo "✅ All 108 runs complete!" | tee -a "$SUMMARY_FILE"
echo "Completed: $(date)" | tee -a "$SUMMARY_FILE"
echo "Results saved to $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "=====================================" | tee -a "$SUMMARY_FILE"

echo "[$(date)] Hyperparameter sweep finished"

