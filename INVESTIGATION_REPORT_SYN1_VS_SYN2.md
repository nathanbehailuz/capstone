# Investigation Report: syn1 vs syn2 AUC Performance Discrepancy

**Date:** November 20, 2025  
**Analyst:** AI Assistant  
**Task:** Investigate why syn2 AUC performance (~0.69) is significantly worse than syn1 (~0.97) despite structural similarities

---

## Executive Summary

**ROOT CAUSE IDENTIFIED:** The primary cause of poor syn2 explainer performance is the **low-quality pretrained GNN classifier model**. The syn2 GNN has only **72.14% test accuracy** compared to syn1's **100% test accuracy**. An explainer can only explain what a model has learned - if the base model makes 28% errors, the explanations will be inconsistent and unreliable.

---

## Key Findings

### 1. Pretrained GNN Model Quality (PRIMARY ISSUE)

| Metric | syn1 | syn2 | Δ |
|--------|------|------|---|
| **Train Accuracy** | 96.96% | 90.18% | **-6.78%** |
| **Val Accuracy** | 100.00% | 75.71% | **-24.29%** |
| **Test Accuracy** | 100.00% | 72.14% | **-27.86%** |
| **Eval Seed Error Rate** | 1.67% | 46.67% | **+45.00%** |

**Impact:** The syn2 GNN classifier makes **27.86% errors** on test data. This makes it nearly impossible for the explainer to generate consistent, high-quality explanations because:
- The model's predictions are unreliable
- Ground truth explanations don't align with what the model actually learned
- The explainer learns to explain incorrect predictions ~28% of the time

### 2. Explainer Training Results

#### syn1 (with high-quality GNN):
```
Epoch  1: AUC = 0.8039  ✅
Epoch  5: AUC = 0.7266
Epoch 10: AUC = 0.8256
Epoch 15: AUC = 0.8693
Epoch 20: AUC = 0.9713  ✅ (EXCELLENT!)
```
- **Steady improvement over 20 epochs**
- Achieves **97.13% AUC** - near-perfect explanation quality

#### syn2 (with low-quality GNN):
```
seed=0 (5 epochs):
Epoch 1: AUC = 0.7091
Epoch 5: AUC = 0.6930  ❌ (DEGRADING!)

seed=2 (50 epochs):
Epoch  1: AUC = 0.6778
Epoch 10: AUC = 0.6840
Epoch 20: AUC = 0.6854
Epoch 30: AUC = 0.6780
Epoch 40: AUC = 0.7003
Epoch 50: AUC = 0.6865  ❌ (STAGNANT!)
```
- **Fluctuates around 0.67-0.72, never improves**
- Even with 50 epochs, cannot exceed 0.73 AUC
- Performance ceiling imposed by poor base model

### 3. Configuration Discrepancies

#### seed=0 Configuration Mismatch
- **SLURM file** specifies: `N_EPOCHS=50, N_ROLLOUTS=32`
- **Actual run** used: `n_epochs=5, n_rollouts=10`
- **Conclusion:** seed=0 was run with an older/incorrect configuration

#### seed=2 Configuration (Correct)
- **SLURM file** matches actual run
- Ran for full 50 epochs with 32 rollouts
- **But performance still poor due to GNN quality**

### 4. Hyperparameter Differences

| Parameter | syn1 | syn2 (SLURM) | Impact |
|-----------|------|--------------|--------|
| n_epochs | 20 | 50 | syn2 trains longer but doesn't help |
| n_rollouts | 32 | 32 | Same |
| lr | 0.0003 | 0.0001 | syn2 slower learning |
| ppo_epochs | 4 | 5 | syn2 more PPO updates |
| entropy_coeff_start | 0.003 | 0.04 | syn2 more exploration |
| minibatch_size | 256 | 512 | syn2 larger batches |
| stop_bonus_ok | 0.10 | 1.0 | syn2 10x higher reward |
| reward_scale | 5.0 | 5.0 | Same |

**Note:** These hyperparameter differences are **secondary** - they cannot compensate for a 28% error rate in the base GNN.

---

## Root Cause Analysis

### Why Does syn2 Have a Poor GNN Model?

The syn2 dataset has **8 classes** (compared to syn1's 4 classes) and uses Gaussian-distributed features to distinguish between two BA-shapes. Possible reasons for poor GNN performance:

1. **More challenging classification task**: 8 classes vs 4 classes
2. **Feature distribution overlap**: Gaussian features may have insufficient separation
3. **Training issues**: The GNN may not have been trained long enough or with appropriate hyperparameters
4. **Dataset characteristics**: The two BA-shapes may be too similar in structure

### Verification
Check the pretrained models:
```bash
/scratch/nz2212/capstone/models/pretrained/GNN/syn1/best_model  # 100% test accuracy
/scratch/nz2212/capstone/models/pretrained/GNN/syn2/best_model  # 72% test accuracy
```

---

## Recommended Solutions

### Option 1: Retrain the syn2 GNN Classifier (RECOMMENDED)

**Goal:** Achieve >95% test accuracy on syn2 dataset

**Steps:**
1. **Check dataset quality**
   ```python
   # Verify syn2 dataset has good class separation
   python -c "
   from datasets.dataset_loaders import load_graph_dataset
   data = load_graph_dataset('syn2')
   # Analyze feature distributions, class balance, etc.
   "
   ```

2. **Retrain GNN with better hyperparameters**
   - Increase training epochs (try 500-1000)
   - Use learning rate scheduling
   - Add dropout for regularization
   - Try different architectures (more layers, attention mechanisms)
   - Use cross-validation to tune hyperparameters

3. **Script to retrain** (check `pretrain_test_runner.py`):
   ```bash
   # Modify and run the GNN training script
   python pretrain_test_runner.py --dataset syn2 --epochs 1000 --lr 0.001
   ```

4. **Replace the pretrained model**
   ```bash
   # Backup old model
   mv models/pretrained/GNN/syn2/best_model models/pretrained/GNN/syn2/best_model_old_72pct
   
   # Copy new trained model
   cp <new_model_path> models/pretrained/GNN/syn2/best_model
   ```

### Option 2: Adjust Explainer Hyperparameters (PARTIAL FIX)

If retraining the GNN is not feasible, adjust explainer hyperparameters to be more tolerant of errors:

**Changes to `slurm/syn2_ppo.slurm`:**
```bash
# Reduce confidence requirements
CONFIDENCE_THRESHOLD=0.6  # Was 0.5, increase to be less strict

# Adjust reward scaling to be less sensitive to prediction errors
REWARD_SCALE=3.0  # Was 5.0, reduce to dampen error impact

# Increase exploration to find robust patterns despite errors  
ENTROPY_COEFF_START=0.08  # Was 0.04, increase exploration
N_EPOCHS=100  # Was 50, train longer to find stable patterns

# Adjust stop bonuses to be less strict
STOP_BONUS_OK=0.5  # Was 1.0, reduce to avoid over-rewarding uncertain stops
```

**Expected improvement:** May reach 0.75-0.78 AUC, but unlikely to exceed 0.80 with a 72% accuracy GNN.

### Option 3: Use a Different Dataset Structure (ALTERNATIVE)

If syn2's 8-class structure is fundamentally difficult:
- Consider reducing to 2-4 classes
- Increase feature separation between classes
- Use one-hot encoding instead of Gaussian features

---

## Immediate Action Items

### Priority 1: Verify GNN Training Process
```bash
# Check if there's a GNN training script
ls -la pretrain_test_runner.py

# Check what parameters were used to train the original syn2 GNN
# Look for training logs or configuration files
find . -name "*syn2*train*" -o -name "*syn2*log*"
```

### Priority 2: Quick Test - Retrain Small GNN
```bash
# Quick test with better hyperparameters (30 min test)
python pretrain_test_runner.py \
  --dataset syn2 \
  --epochs 500 \
  --lr 0.001 \
  --hidden_size 128 \
  --dropout 0.3
```

### Priority 3: Re-run Explainer with New GNN
```bash
# Once GNN is retrained and achieves >95% accuracy:
sbatch slurm/syn2_ppo.slurm  # Run with array=0 for seed 0
```

---

## Expected Outcomes After Fix

If the syn2 GNN is retrained to achieve >95% test accuracy:

**Expected AUC progression:**
```
Epoch  5: AUC ~ 0.75-0.78
Epoch 10: AUC ~ 0.80-0.85
Epoch 20: AUC ~ 0.87-0.92
Epoch 50: AUC ~ 0.93-0.96
```

This would bring syn2 performance in line with syn1, validating that the issue was indeed the base GNN quality.

---

## Conclusion

**The performance gap between syn1 and syn2 is NOT due to:**
- ❌ Explainer hyperparameters (though seed=0 used wrong config)
- ❌ Dataset structural differences (both are BA-based graphs)
- ❌ Training instability (syn2 seed=2 ran 50 epochs stably)

**The performance gap IS due to:**
- ✅ **Poor pretrained GNN classifier quality** (72% vs 100% accuracy)
- ✅ **Fundamental limitation: cannot explain an unreliable model**

**Recommended action:** Retrain the syn2 GNN classifier to achieve >95% test accuracy, then re-run the explainer experiments.

---

## Files to Check

1. **Pretrained Models:**
   - `/scratch/nz2212/capstone/models/pretrained/GNN/syn1/best_model`
   - `/scratch/nz2212/capstone/models/pretrained/GNN/syn2/best_model`

2. **Training Scripts:**
   - `/scratch/nz2212/capstone/pretrain_test_runner.py`
   - Look for GNN training scripts in the codebase

3. **Configuration Files:**
   - `/scratch/nz2212/capstone/slurm/syn1_ppo.slurm`
   - `/scratch/nz2212/capstone/slurm/syn2_ppo.slurm`

4. **Result Files:**
   - `/scratch/nz2212/capstone/results/syn1_ppo_seed_0.txt`
   - `/scratch/nz2212/capstone/results/syn2_ppo_seed_0.txt`
   - `/scratch/nz2212/capstone/results/syn2_ppo_seed_2.txt`

---

**Investigation Complete**  
**Status:** Root cause identified, solutions proposed  
**Next Steps:** Await user decision on retraining GNN or adjusting hyperparameters





