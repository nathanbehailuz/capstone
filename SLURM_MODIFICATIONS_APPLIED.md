# SLURM Modifications Applied to syn2_ppo.slurm

**Date:** November 20, 2025  
**Purpose:** Adjust explainer hyperparameters to be more tolerant of 72% GNN accuracy  
**Based on:** INVESTIGATION_REPORT_SYN1_VS_SYN2.md - Option 2: Adjust Explainer Hyperparameters

---

## Changes Applied

### 1. **N_EPOCHS: 50 → 100**
```bash
N_EPOCHS=100  # MODIFIED: Increased from 50
```
**Rationale:** Train longer to find stable patterns despite GNN prediction errors. More epochs allow the explainer to learn more robust patterns that work even when the base model is unreliable.

### 2. **STOP_BONUS_OK: 1.0 → 0.5**
```bash
STOP_BONUS_OK=0.5  # MODIFIED: Reduced from 1.0
```
**Rationale:** Reduce bonus for stopping to avoid over-rewarding uncertain stops. With a 72% accuracy GNN, the explainer shouldn't be overly confident about when to stop.

### 3. **REWARD_SCALE: 5.0 → 3.0**
```bash
REWARD_SCALE=3.0  # MODIFIED: Reduced from 5.0
```
**Rationale:** Dampen the impact of prediction errors. Lower reward scaling makes the learning process less sensitive to the GNN's 28% error rate.

### 4. **ENTROPY_COEFF_START: 0.04 → 0.08**
```bash
ENTROPY_COEFF_START=0.08  # MODIFIED: Increased from 0.04
```
**Rationale:** Increase exploration to find robust patterns. More exploration helps discover explanations that work despite model unreliability.

### 5. **CONFIDENCE_THRESHOLD: 0.5 → 0.6**
```bash
CONFIDENCE_THRESHOLD=0.6  # MODIFIED: Increased from 0.5
```
**Rationale:** Be less strict about confidence requirements. With an unreliable base model, we need to be more tolerant of uncertainty.

---

## Expected Outcomes

### Conservative Estimate:
- **Current AUC:** 0.69 (stagnant across 50 epochs)
- **Expected AUC:** 0.75-0.78 (modest improvement)
- **Improvement:** +6-9 percentage points

### Why Not Higher?
The fundamental limitation remains: **you cannot explain an unreliable model better than the model itself**. With 72% GNN accuracy, there's a ceiling on explanation quality.

### Performance Ceiling Analysis:
```
Best Case Scenario (if all goes well):
- AUC ≈ 0.78-0.80 (approaching GNN accuracy ceiling)

Realistic Expectation:
- AUC ≈ 0.75-0.77 (stable, less fluctuation)

Minimum Improvement:
- AUC ≈ 0.72-0.74 (at least more stable than 0.67-0.72 fluctuation)
```

---

## How to Run

### Option 1: Run a single seed for quick testing
```bash
cd /scratch/nz2212/capstone

# Test with seed 0
sbatch --array=0 slurm/syn2_ppo.slurm

# Monitor progress
tail -f results/syn2_ppo_seed_0.txt
```

### Option 2: Run multiple seeds for statistical significance
```bash
# Edit the array parameter in syn2_ppo.slurm first:
# Change: #SBATCH --array=2
# To:     #SBATCH --array=0-4

sbatch slurm/syn2_ppo.slurm

# Monitor all runs
ls -lh results/syn2_ppo_seed_*.txt
```

### Option 3: Compare with original configuration
```bash
# Keep a backup of the original results
mkdir -p results/syn2_original_config
cp results/syn2_ppo_seed_*.txt results/syn2_original_config/

# Run with new configuration
sbatch --array=0 slurm/syn2_ppo.slurm

# Compare after completion
python3 << 'EOF'
import re

# Read old result (seed 2, 50 epochs)
with open('results/syn2_original_config/syn2_ppo_seed_2.txt') as f:
    old_text = f.read()
    old_aucs = [float(x) for x in re.findall(r'AUC=([0-9.]+)', old_text)]
    print(f"OLD (50 epochs): Final AUC = {old_aucs[-1]:.4f}, Max = {max(old_aucs):.4f}")

# Read new result (seed 0, 100 epochs) - after it completes
with open('results/syn2_ppo_seed_0.txt') as f:
    new_text = f.read()
    new_aucs = [float(x) for x in re.findall(r'AUC=([0-9.]+)', new_text)]
    print(f"NEW (100 epochs): Final AUC = {new_aucs[-1]:.4f}, Max = {max(new_aucs):.4f}")
    print(f"Improvement: {new_aucs[-1] - old_aucs[-1]:+.4f}")
EOF
```

---

## Monitoring Progress

### Check AUC trends during training:
```bash
# Real-time AUC monitoring
watch -n 30 "grep 'EVAL-Epoch.*AUC' results/syn2_ppo_seed_0.txt | tail -20"

# Or manually:
grep "EVAL-Epoch.*AUC" results/syn2_ppo_seed_0.txt
```

### Expected AUC progression with new hyperparameters:
```
Epoch   1-10: AUC ~ 0.68-0.72 (similar to before, but more stable)
Epoch  11-30: AUC ~ 0.72-0.75 (gradual improvement)
Epoch  31-60: AUC ~ 0.74-0.77 (reaching plateau)
Epoch  61-100: AUC ~ 0.75-0.78 (stable at new ceiling)
```

### Warning Signs (if AUC still degrades):
If AUC drops below 0.67 or shows high variance (±0.05 per epoch), it indicates:
1. The 72% GNN accuracy is too limiting
2. **Option 1 (Retrain GNN)** becomes necessary
3. No amount of hyperparameter tuning can fix a fundamentally unreliable model

---

## Next Steps Based on Results

### If AUC reaches 0.75-0.78: ✅ Success (as good as possible with 72% GNN)
- **Action:** Document the limitation in your paper/thesis
- **Note:** Mention that explainer performance is bounded by base model quality
- **Compare:** Show syn1 (97% AUC with 100% GNN) vs syn2 (76% AUC with 72% GNN)

### If AUC reaches 0.72-0.75: ⚠️ Marginal Improvement
- **Action:** Consider Option 1 - Retrain the GNN classifier
- **Reason:** Small gain doesn't justify keeping a poor base model
- **ROI:** Retraining GNN (1 day effort) → 95% accuracy → 93% AUC is better

### If AUC stays below 0.72: ❌ No Improvement
- **Action:** MUST retrain the GNN (Option 1)
- **Reason:** Hyperparameter adjustments insufficient
- **Conclusion:** The 72% GNN accuracy is fundamentally incompatible with high-quality explanations

---

## Rollback Instructions

If you need to revert to the original configuration:

```bash
cd /scratch/nz2212/capstone

# Restore original values
sed -i 's/N_EPOCHS=100/N_EPOCHS=50/' slurm/syn2_ppo.slurm
sed -i 's/STOP_BONUS_OK=0.5/STOP_BONUS_OK=1.0/' slurm/syn2_ppo.slurm
sed -i 's/REWARD_SCALE=3.0/REWARD_SCALE=5.0/' slurm/syn2_ppo.slurm
sed -i 's/ENTROPY_COEFF_START=0.08/ENTROPY_COEFF_START=0.04/' slurm/syn2_ppo.slurm
sed -i 's/CONFIDENCE_THRESHOLD=0.6/CONFIDENCE_THRESHOLD=0.5/' slurm/syn2_ppo.slurm

# Or use git to revert (if tracked)
git checkout slurm/syn2_ppo.slurm
```

---

## File Locations

- **Modified SLURM script:** `/scratch/nz2212/capstone/slurm/syn2_ppo.slurm`
- **Results will be saved to:** `/scratch/nz2212/capstone/results/syn2_ppo_seed_<N>.txt`
- **Investigation report:** `/scratch/nz2212/capstone/INVESTIGATION_REPORT_SYN1_VS_SYN2.md`
- **This file:** `/scratch/nz2212/capstone/SLURM_MODIFICATIONS_APPLIED.md`

---

## Summary

✅ **5 key parameters modified** to make the explainer more tolerant of GNN errors  
✅ **100 epochs** instead of 50 for longer training  
✅ **Reduced reward sensitivity** to dampen impact of prediction errors  
✅ **Increased exploration** to find robust patterns  
✅ **Less strict confidence** requirements for uncertain model  

**Status:** Ready to run  
**Expected Runtime:** ~20-24 hours for 100 epochs (2x original)  
**Expected Outcome:** AUC improvement from 0.69 to 0.75-0.78 (6-9 point gain)  

**Note:** This is a **partial fix**. For best results, retrain the syn2 GNN to >95% accuracy (see INVESTIGATION_REPORT_SYN1_VS_SYN2.md Option 1).





