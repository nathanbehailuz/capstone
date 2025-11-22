#!/bin/bash
#SBATCH --job-name=pretrain-test
#SBATCH --output=slurm_out/pretrain_test_%j.out
#SBATCH --error=slurm_out/pretrain_test_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@example.com

# This is a template SLURM script for running pretraining tests
# Usage: sbatch slurm/pretrain_test.slurm

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Load modules (adjust based on your cluster)
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || echo "Warning: CUDA module not loaded"

# Activate conda environment
source ~/.bashrc
conda activate rlgn || echo "Warning: Could not activate rlgn environment"

# Verify environment
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not found')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
echo ""

# Create output directories
mkdir -p slurm_out
mkdir -p results/pretraining_tests
mkdir -p logs

# Configuration
DATASET=${1:-syn2}
SEED=${2:-0}
EDGE_DROP=${3:-0.1}

echo "=========================================="
echo "Test Configuration"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo "Edge Drop Probability: $EDGE_DROP"
echo "=========================================="
echo ""

# Generate log filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DROP_STR=$(printf "drop%02d" $(echo "$EDGE_DROP * 100" | bc | cut -d. -f1))
LOG_FILE="logs/pretrain_test_${DATASET}_${DROP_STR}_seed${SEED}_${TIMESTAMP}.log"

echo "Output will be saved to: $LOG_FILE"
echo ""

# Run the pretraining test
echo "Starting pretraining test..."
echo ""

python pretrain_test_runner.py \
    --dataset $DATASET \
    --model GNN \
    --seed $SEED \
    --random_edge_drop $EDGE_DROP \
    --pretrain_list 5 \
    --pretrain_set 10 \
    --pretrain_g_batch_size 32 \
    --max_size 15 \
    --n_hop 3 \
    --ppo_test_epochs 2 \
    --n_epochs 5 \
    --g_batch_size 128 \
    --hidden_size 64 \
    --with_attr \
    --output_dir results/pretraining_tests \
    2>&1 | tee $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo "Job Completion"
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "Log File: $LOG_FILE"
echo "=========================================="

exit $EXIT_CODE

