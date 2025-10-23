# SLURM Files Updated for PPO Refactoring

## Overview
All SLURM files have been updated to include the new PPO refactoring parameters for delta rewards, λ schedules, STOP bonuses/penalties, and normalization features.

## Updated Files
- `slurm/syn1_ppo.slurm`
- `slurm/syn2_ppo.slurm` 
- `slurm/syn3_ppo.slurm`
- `slurm/syn4_ppo.slurm`
- `slurm/mutag_ppo.slurm`
- `slurm/ba2_ppo.slurm`

## New Parameters Added

### Delta Rewards and STOP Bonuses
```bash
DELTA_REWARD=1                    # Enable incremental causal rewards
STOP_BONUS_OK=0.5                 # Bonus when STOP preserves class and size is small
STOP_PENALTY_BAD=0.2              # Penalty for premature or incorrect STOP
TARGET_SIZE_K_NEW=8               # Dataset-specific target size (varies by dataset)
```

### λ Schedules
```bash
WARMUP_EPOCHS=5                   # Epochs for preservation-first warmup
SCHEDULE_EPOCHS=10                # Epochs to ramp regularizers to full strength
LAMBDA_SIZE_FINAL=0.01            # Final size penalty weight
LAMBDA_RADIUS_FINAL=0.1           # Final radius penalty weight
LAMBDA_SIM_FINAL=0.01             # Final similarity penalty weight
LAMBDA_SIM_START=0.0              # Starting similarity penalty weight
```

### Normalization
```bash
TRAJ_REWARD_NORM=1                # Enable per-trajectory reward normalization
STD_ADVANTAGE=1                   # Enable batch advantage standardization
```

### Entropy Scheduling
```bash
ENTROPY_COEF_START=1e-3           # Starting entropy coefficient
ENTROPY_COEF_END=1e-4             # Ending entropy coefficient
```

### Debug
```bash
PPO_DEBUG=1                       # Enable PPO debug logging
```

## Dataset-Specific Target Sizes
- **syn1, syn2, syn3, syn4**: `TARGET_SIZE_K_NEW=8` (synthetic datasets typically have ~8 edge motifs)
- **mutag**: `TARGET_SIZE_K_NEW=6` (smaller motifs for molecular data)
- **ba2**: `TARGET_SIZE_K_NEW=10` (larger motifs for BA-Shapes dataset)

## Training Command Updates
All training commands now include the new parameters:

```bash
srun --export=ALL --gres=gpu:1 -c "${SLURM_CPUS_PER_TASK:-4}" \
  "${PYBIN}" run_RG_explainer.py \
    # ... existing parameters ... \
    --reward_norm \
    \
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
    --entropy_coef_start "${ENTROPY_COEF_START}" \
    --entropy_coef_end "${ENTROPY_COEF_END}" \
    --ppo_debug \
  >> "$RESULT_FILE" 2>&1
```

## Key Features Enabled

### 1. Incremental Delta Rewards
- Rewards reflect causal screening improvement at each step
- Positive rewards for reducing composite loss J(G)
- Baseline is empty subgraph J(∅)

### 2. STOP-Aware Training
- Bonus for good STOP: class preserved + size ≤ target_size_k
- Penalty for bad STOP: premature or class not preserved
- Encourages appropriate stopping behavior

### 3. λ Scheduling
- **Phase 1 (warmup)**: Focus on CE prediction, minimal regularization
- **Phase 2 (ramp)**: Gradually increase size/radius penalties  
- **Phase 3 (full)**: Full regularization strength
- Entropy coefficient decays over training

### 4. Normalization
- Per-trajectory reward normalization stabilizes training
- Batch advantage standardization improves PPO stability
- Both are optional and controlled by flags

### 5. Enhanced Debugging
- Debug logging for λ schedule values
- Delta reward statistics (mean, std, avg subgraph size, stop rate)
- Controlled by `PPO_DEBUG` flag

## Usage
To submit jobs with the new PPO refactoring:

```bash
# Submit individual datasets
sbatch slurm/syn1_ppo.slurm
sbatch slurm/syn3_ppo.slurm
sbatch slurm/mutag_ppo.slurm

# Or submit all datasets
for dataset in syn1 syn2 syn3 syn4 mutag ba2; do
    sbatch slurm/${dataset}_ppo.slurm
done
```

## Expected Benefits
1. **Better Reward Shaping**: Delta rewards provide more informative feedback
2. **Improved STOP Behavior**: Bonuses/penalties encourage appropriate stopping
3. **Stable Training**: λ schedules prevent early over-regularization
4. **Enhanced Normalization**: Per-trajectory and batch normalization improve stability
5. **Better Debugging**: Comprehensive logging for monitoring training progress

The updated SLURM files maintain backward compatibility while enabling all new PPO refactoring features for improved causal subgraph explanation training.
