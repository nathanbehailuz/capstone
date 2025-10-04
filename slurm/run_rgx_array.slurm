#!/bin/bash
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

PYTHON=/anaconda/envs/py37_pytorch/bin/python3
SEED=${SLURM_ARRAY_TASK_ID}

# Create results directory
mkdir -p results

# Run with seed-specific output files
$PYTHON run_RG_explainer.py \
  --dataset mutag \
  --seed $SEED \
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
  > results/mutag_seed_$SEED.txt 2>&1
