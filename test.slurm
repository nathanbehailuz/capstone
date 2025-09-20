#!/bin/bash
# test.slurm

#SBATCH -J hello_test
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:05:00
#SBATCH --array=0-2
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.err

set -euo pipefail

# Run from the directory where you did `sbatch`
cd "$SLURM_SUBMIT_DIR"
echo "[$(date)] PWD=$PWD  JOBID=$SLURM_JOB_ID  TASK=$SLURM_ARRAY_TASK_ID"

# Use srun so Slurm tracks the task
srun -c "${SLURM_CPUS_PER_TASK:-1}" /usr/bin/env python3 test.py

