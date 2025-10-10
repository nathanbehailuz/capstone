#!/bin/bash

# Usage: ./run_analysis.sh <dataset> [start_seed] [end_seed]
# Example: ./run_analysis.sh ba2 0 1

DATASET=${1:-"ba2"}
START_SEED=${2:-0}
END_SEED=${3:-1}

echo "Submitting analysis jobs for dataset: $DATASET"
echo "Seed range: $START_SEED to $END_SEED"

# Calculate array range
ARRAY_RANGE="$START_SEED-$END_SEED"

# Submit SLURM job
sbatch --array=$ARRAY_RANGE slurm/analyze_results.slurm $DATASET

echo "Analysis jobs submitted!"
echo "Check status with: squeue -u \$USER"
echo "Monitor logs with: tail -f slurm_out/analyze_results_*.out"
