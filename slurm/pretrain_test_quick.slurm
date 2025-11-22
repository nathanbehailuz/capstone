#!/bin/bash
#SBATCH --job-name=pretrain-quick
#SBATCH --output=slurm_out/pretrain_quick_%j.out
#SBATCH --error=slurm_out/pretrain_quick_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Quick pretraining test (minimal iterations, skip PPO)
# Usage: sbatch slurm/pretrain_test_quick.slurm [dataset] [seed] [edge_drop]

echo "Quick Pretraining Test - Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

# Load environment
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null
source ~/.bashrc
conda activate rlgn

# Create directories
mkdir -p slurm_out results/pretraining_tests logs

# Configuration
DATASET=${1:-syn2}
SEED=${2:-0}
EDGE_DROP=${3:-0.0}

echo "Dataset: $DATASET | Seed: $SEED | Edge Drop: $EDGE_DROP"

# Run quick test (minimal iterations, skip PPO)
python pretrain_test_runner.py \
    --dataset $DATASET \
    --seed $SEED \
    --random_edge_drop $EDGE_DROP \
    --pretrain_list 3 \
    --pretrain_set 5 \
    --max_size 15 \
    --skip_ppo_test \
    --output_dir results/pretraining_tests

echo "End Time: $(date)"

