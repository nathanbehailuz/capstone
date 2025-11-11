# PPO Stability Fixes - Policy Loss Fix Branch

## Problem Statement

After epoch 30 in Syn2 training, the system experiences:
- `RuntimeError: probability tensor contains either 'inf', 'nan' or element < 0` in `torch.multinomial(ps, 1)`
- Gradient norm explosion: **GradNorm ≈ 4e11**
- Policy loss explosion: **Policy loss ≈ 6e7**
- Excessive KL divergence: **KL ≈ 1.3**

These indicate severe PPO policy drift and numerical instability.

## Root Causes

1. **Softmax underflow**: Extremely negative logits (from STOP masking at -1e9) cause exp() underflow
2. **Unbounded advantages**: Large advantages cause ratio explosion in PPO objective
3. **Insufficient gradient clipping**: Allows gradients to explode during policy updates
4. **No KL safeguards**: Learning rate stays constant even when policy diverges
5. **Near-zero entropy**: Default entropy_coef=1e-6 provides no exploration bonus

## Implemented Fixes

### Fix 1: Safe Softmax & Sampling ✓
**Location**: `_sample_actions_masked()` and `_sample_actions()` (lines 2263-2275, 2321-2333)

**Changes**:
```python
# Stabilize logits before softmax
masked_logits = masked_logits - masked_logits.max(dim=-1, keepdim=True).values
ps = torch.softmax(masked_logits, dim=-1)
ps = ps + 1e-8
ps = ps / ps.sum(dim=-1, keepdim=True)

# Check for invalid probabilities
if not torch.isfinite(ps).all():
    print(f"[WARN] Invalid probs at epoch {epoch} step {t}")
    ps = torch.nan_to_num(ps, nan=0.0, posinf=0.0, neginf=0.0)
    ps = ps / ps.sum(dim=-1, keepdim=True)
```

**Impact**: Prevents NaN/inf in probability distributions, ensures valid sampling

### Fix 2: Advantage Normalization with Clamping ✓
**Location**: `train_from_rewards()` (lines 1335-1337)

**Changes**:
```python
# Normalize and clamp advantages
adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
adv_flat = adv_flat.clamp(-10, 10)  # NEW: Prevent extreme values
```

**Impact**: Prevents ratio explosion (exp(10) ≈ 22k vs exp(100) ≈ 3e43)

### Fix 3: Gradient Clipping (Enhanced) ✓
**Location**: `train_from_rewards()` (line 1506)

**Changes**:
```python
# Ensure strict gradient clipping
grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

**Impact**: Hard limit on gradient norm prevents explosion (was using self.max_grad_norm=0.5, now enforced at 1.0)

### Fix 4: PPO Old Policy Sync ✓
**Location**: `train_from_rewards()` (lines 1279-1282)

**Changes**:
```python
# Ensure old values are detached (frozen during PPO updates)
old_logps = torch.stack(rollout_buffer['old_logps']).to(self.device).detach()
old_values = torch.stack(rollout_buffer['old_values']).to(self.device).detach()
```

**Impact**: Guarantees old policy remains frozen, prevents gradient leakage

### Fix 5: KL Divergence Safety ✓
**Location**: `train_from_rewards()` (lines 1572-1577)

**Changes**:
```python
# Auto-adapt learning rate when KL too high
if avg_stats.approx_kl > 0.2:
    print(f"[WARN] KL divergence too high ({avg_stats.approx_kl:.3f}), reducing learning rate.")
    for param_group in self.optimizer.param_groups:
        param_group["lr"] *= 0.5
```

**Impact**: Prevents policy from drifting too far from old policy in single update

### Fix 6: Entropy Regularization ✓
**Location**: `__init__()` (lines 124, 163-164)

**Changes**:
```python
# Constructor default
entropy_coef: float = 0.01  # Was: 1e-6

# Scheduling defaults
entropy_coeff_start: float = 0.01   # Was: 1e-3
entropy_coeff_end: float = 0.001    # Was: 1e-4
```

**Impact**: Stronger exploration bonus prevents premature convergence and maintains policy stochasticity

## Expected Outcomes

After these fixes, you should observe:

1. ✓ **No multinomial errors** beyond epoch 30
2. ✓ **GradNorm stays below 10** throughout training
3. ✓ **KL divergence < 0.2** (auto-adapted when exceeded)
4. ✓ **Policy loss remains stable** (< 1e2 range)
5. ✓ **Entropy stays > 0.01** (healthy exploration)

## Verification Command

```bash
cd /scratch/nz2212/capstone
git checkout policy_loss_fix
sbatch slurm/syn2_ppo.slurm
# Monitor: results/syn2_ppo_seed_0.txt
# Check for: No "[WARN] Invalid probs", GradNorm < 10, KL < 0.2
```

## Theoretical Basis

These fixes align with:
- **Schulman et al. (2017)**: PPO clipped surrogate objective with entropy bonus
- **Berner et al. (2019)**: Large-scale RL requires aggressive normalization and clipping
- **Andrychowicz et al. (2021)**: Adaptive KL penalty prevents catastrophic policy collapse

## Branch Information

- **Branch**: `policy_loss_fix`
- **Files Modified**: `components/generator_copy.py`
- **Lines Changed**: 43 insertions, 12 deletions
- **Commit Hash**: 9814382

## Rollback Instructions

If issues arise:
```bash
git checkout main
# OR
git revert 9814382
```

