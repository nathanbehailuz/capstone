#!/bin/bash
#SBATCH --job-name=pretrain-compare
#SBATCH --output=slurm_out/pretrain_compare_%j.out
#SBATCH --error=slurm_out/pretrain_compare_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Compare pretraining with and without edge dropping
# Usage: sbatch slurm/pretrain_test_compare.slurm [dataset] [seed]

echo "=========================================="
echo "Pretraining Comparison Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "=========================================="

# Load environment
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null
source ~/.bashrc
conda activate rlgn

# Create directories
mkdir -p slurm_out results/pretraining_tests logs

# Configuration
DATASET=${1:-syn2}
SEED=${2:-0}

echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo "Running 3 tests: no drop, 10% drop, 20% drop"
echo ""

# Test 1: No edge dropping (baseline)
echo "=========================================="
echo "Test 1: Baseline (no edge dropping)"
echo "=========================================="
python pretrain_test_runner.py \
    --dataset $DATASET \
    --seed $SEED \
    --random_edge_drop 0.0 \
    --pretrain_list 5 \
    --pretrain_set 10 \
    --ppo_test_epochs 2 \
    --output_dir results/pretraining_tests

echo ""
echo "Test 1 completed at $(date)"
echo ""

# Test 2: 10% edge dropping
echo "=========================================="
echo "Test 2: 10% edge dropping"
echo "=========================================="
python pretrain_test_runner.py \
    --dataset $DATASET \
    --seed $SEED \
    --random_edge_drop 0.1 \
    --pretrain_list 5 \
    --pretrain_set 10 \
    --ppo_test_epochs 2 \
    --output_dir results/pretraining_tests

echo ""
echo "Test 2 completed at $(date)"
echo ""

# Test 3: 20% edge dropping
echo "=========================================="
echo "Test 3: 20% edge dropping"
echo "=========================================="
python pretrain_test_runner.py \
    --dataset $DATASET \
    --seed $SEED \
    --random_edge_drop 0.2 \
    --pretrain_list 5 \
    --pretrain_set 10 \
    --ppo_test_epochs 2 \
    --output_dir results/pretraining_tests

echo ""
echo "Test 3 completed at $(date)"
echo ""

echo "=========================================="
echo "All Tests Complete"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "Results saved in: results/pretraining_tests/"
echo "Check JSON files for detailed metrics comparison"

