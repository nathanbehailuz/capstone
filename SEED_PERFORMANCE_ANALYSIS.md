# Seed Performance Analysis: Why Seed 0 Outperforms Seed 1

## Executive Summary

**Seed 0** achieves **0.986 AUC** (epoch 19) vs **Seed 1** achieving **0.821 AUC** (epoch 20) - a **16.5% gap**.

## Key Differences Identified

### 1. **Pretraining Status** (CRITICAL FACTOR)

| Seed | Pretraining Status | CODE Hash |
|------|-------------------|-----------|
| 0 | ✅ **Loaded pretrained model** | `0BFE1B3B7884F400819A8FFA4E240859` |
| 1 | ❌ **Pretrained from scratch** | `6E12DC4340EDDF4087F18665EC7C513E` |

**Impact**: Seed 0 started with a better initialization. The pretrained model was already optimized for this configuration, giving it a head start.

### 2. **Training Dynamics**

#### Reward Signal Stability
- **Seed 0**: Consistent negative rewards (-0.003 to -0.010), indicating stable optimization
- **Seed 1**: Mixed rewards (-0.005 to +0.001), less stable signal

#### Early Performance
- **Seed 0**: Starts at 0.772 AUC (epoch 1)
- **Seed 1**: Starts at 0.799 AUC (epoch 1) - **actually higher initially!**

#### Learning Trajectory
- **Seed 0**: **Monotonic improvement** → 0.772 → 0.986 (28% improvement)
- **Seed 1**: **Plateaus early** → 0.799 → 0.821 (only 3% improvement)

### 3. **Edge Dropout Stochasticity**

Both seeds use identical edge dropout configuration:
```
[EdgeDrop] Enabled: p=0.100, preserve_connectivity=False
```

However, **edge dropout is resampled per episode**, meaning:
- Each rollout sees a different dropped graph structure
- Even with same seed, edge dropping introduces stochasticity
- This creates different exploration paths

### 4. **Configuration Identical**

Both runs used the same hyperparameters:
- `random_edge_drop=0.1`
- `causal_weight=0.3`
- `preserve_connectivity=False`
- `n_rollouts=32`
- All other PPO parameters identical

## Root Causes

### Primary Cause: Pretraining Initialization

**Why this matters:**
1. Seed 0's pretrained model (`0BFE1B3B7884F400819A8FFA4E240859.pth`) already existed
   - Created: **Nov 28 14:30** (before seed 0's main run at 23:34)
   - This model was already optimized from a previous run
   - Seed 0 **loaded** this pretrained model → better initialization
   
2. Seed 1's pretrained model (`6E12DC4340EDDF4087F18665EC7C513E.pth`) didn't exist at start
   - Created: **Nov 29 01:28** (DURING seed 1's main run)
   - Seed 1 had to **pretrain from scratch** during the run
   - Pretraining uses MLE (supervised learning), not RL
   - Less optimal initialization for PPO training

**Verification:**
```bash
$ ls -lh pretrained/
-rw-r--r-- 0BFE1B3B7884F400819A8FFA4E240859.pth  # Created Nov 28 14:30 (before run)
-rw-r--r-- 6E12DC4340EDDF4087F18665EC7C513E.pth  # Created Nov 29 01:28 (during run)
```

**The pretraining CODE hash includes:**
- `seed` (so seed 0 and seed 1 have different pretrained models)
- `pretrain_list`, `pretrain_set`, `hidden_size`, `dataset`, `max_size`, etc.
- `random_edge_drop` (so edge dropout config affects pretraining too)

### Secondary Cause: Stochastic Edge Dropout

**Even with deterministic seeds:**
- Edge dropout is applied **per-episode** during training
- Each episode samples a new dropped graph
- This creates different exploration landscapes
- Small differences early on compound over epochs

**Example**: 
- Seed 0 might explore paths where edges are dropped that lead to better solutions
- Seed 1 might explore paths where dropped edges block optimal paths

### Tertiary Cause: Reward Signal Quality

**Seed 0**: More consistent negative rewards → better gradient signal
- Negative rewards in this context likely mean good performance
- Consistent signal → stable optimization

**Seed 1**: Mixed rewards → less stable learning
- Positive rewards might indicate over-exploration
- Less stable signal → plateaus

## Recommendations

### 1. **Ensure Consistent Pretraining**
```bash
# Before running main training, pretrain all seeds:
for SEED in 0 1 2 3 4 5 6 7 8 9; do
    # Run pretraining only (you may need a separate script)
    python run_RG_explainer.py --seed $SEED --pretrain_only
done

# Then run main training - all will load pretrained models
```

### 2. **Report Mean ± Std Across Seeds**
- Single seed results are misleading
- Should report: `Mean AUC: 0.90 ± 0.05` across 10 seeds

### 3. **Check Pretrained Model Availability**
```python
# In pretrain() method, log if model exists:
pth_fname = self.pretrain_dir / f'{code}.pth'
if pth_fname.exists():
    print(f'✅ Loaded pretrained: {pth_fname}')
else:
    print(f'⚠️  No pretrained model found: {pth_fname}')
    print('   This seed will start from scratch - expect variance!')
```

### 4. **Fix Pretraining Inconsistency**
The pretraining hash includes `seed`, so each seed needs its own pretrained model. Consider:
- **Option A**: Pretrain all seeds upfront in parallel
- **Option B**: Remove `seed` from pretraining hash (share pretrained model)
- **Option C**: Always pretrain from scratch (slower but consistent)

### 5. **Analyze Edge Dropout Impact**
Run ablation study:
- No dropout vs 0.1 dropout vs 0.2 dropout
- Measure variance across seeds for each setting
- Edge dropout might amplify seed variance

## Verification Steps

1. **Check pretrained model files:**
   ```bash
   ls -lh pretrained/0BFE1B3B7884F400819A8FFA4E240859.pth  # Seed 0
   ls -lh pretrained/6E12DC4340EDDF4087F18665EC7C513E.pth  # Seed 1 (should not exist)
   ```

2. **Re-run seed 1 after pretraining:**
   - Pretrain seed 1 separately
   - Then run main training - should see improvement

3. **Compare final trajectories:**
   - Check if seed 0 finds better explanation patterns
   - Analyze subgraph sizes and structures

## Conclusion

**Seed 0's superior performance is primarily due to:**
1. ✅ **Better initialization** (loaded pretrained model)
2. ✅ **More stable reward signal** (consistent optimization)
3. ⚠️  **Stochastic edge dropout** (luck of the draw)

**To ensure fair comparisons:**
- Always pretrain all seeds consistently
- Report mean ± std across multiple seeds
- Consider the impact of edge dropout on variance

