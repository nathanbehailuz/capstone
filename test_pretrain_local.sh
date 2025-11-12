#!/bin/bash
# Quick local testing script for pretrain_test_runner.py
# Usage: ./test_pretrain_local.sh [dataset] [seed] [edge_drop]

set -e  # Exit on error

DATASET=${1:-syn2}
SEED=${2:-0}
EDGE_DROP=${3:-0.1}

echo "=========================================="
echo "Local Pretraining Test"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo "Edge Drop: $EDGE_DROP"
echo "=========================================="
echo ""

# Create output directory
mkdir -p results/pretraining_tests

# Run with minimal iterations for quick testing
python pretrain_test_runner.py \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --random_edge_drop "$EDGE_DROP" \
    --pretrain_list 3 \
    --pretrain_set 5 \
    --max_size 15 \
    --n_hop 3 \
    --ppo_test_epochs 2 \
    --hidden_size 64 \
    --with_attr \
    --output_dir results/pretraining_tests

echo ""
echo "=========================================="
echo "Test Complete!"
echo "Check results/pretraining_tests/ for outputs"
echo "=========================================="

