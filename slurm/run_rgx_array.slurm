#!/bin/bash
# File: run_rgx_array.slurm

## ===================== SLURM =====================
#SBATCH -J rgx_array
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 08:00:00
#SBATCH --array=0-4
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

## ==================== Params =====================
SEED="${SLURM_ARRAY_TASK_ID}"
DATASET="mutag"
RESULT_FILE="results/${DATASET}_seed_${SEED}.txt"

echo "[$(date)] Starting seed=$SEED" | tee -a "$RESULT_FILE"

## ============== GPU & Torch Probes ===============
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" | tee -a "$RESULT_FILE"

# Confirm GPU is allocated to THIS step
srun --gres=gpu:1 nvidia-smi | tee -a "$RESULT_FILE"

## ================== TRAINING =====================
# Launch your script INSIDE the GPU step, using the env's python
srun --export=ALL --gres=gpu:1 -c "${SLURM_CPUS_PER_TASK:-4}" \
  "${PYBIN}" run_RG_explainer.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --n_epochs 10 \
    --ppo_clip 0.1 \
    --kl_target 0.0075 \
    --lr 1e-4 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --max_grad_norm 0.5 \
    --use_value_clip \
    --entropy_coeff_start 1e-3 \
    --entropy_coeff_end 1e-4 \
    --target_size_k 6 \
    --lambda_size 0.5 \
    --lambda_radius 1.0 \
    --lambda_radius_min 0.2 \
    --step_cost 0.001 \
    --stop_bias 0.5 \
    --fidelity_thresh 0.9 \
    --causal_w 0.5 \
    --causal_warmup_epochs 5 \
    --edge_score_agg max \
    --debug_eval_graph -1 \
  >> "$RESULT_FILE" 2>&1

echo "[$(date)] Finished seed=$SEED" | tee -a "$RESULT_FILE"
