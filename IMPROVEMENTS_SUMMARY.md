# RG-Explainer Improvements Summary

This document summarizes the improvements made to the RG-Explainer to enhance explanation AUC and training stability.

## 1. STOP Action, Penalties, and Scheduling

### Changes in `components/generator.py`:
- **STOP Action Enhancement**: Added `stop_bias` parameter (default +0.5) applied to STOP logit during training
- **Size Penalty**: Added `lambda_size` parameter with hinge loss on target size k: `λ_size * max(0, |S| - k)`
- **Radius Penalty**: Added `lambda_radius` with linear decay schedule (1.0 → 0.2 across training epochs)
- **Step Cost**: Added `step_cost = 0.001` to discourage long trajectories
- **STOP Bonus**: Added terminal bonus when fidelity ≥ threshold (0.9 of full-graph probability)

### New Parameters:
- `--target_size_k`: Target motif size (dataset-specific defaults)
- `--lambda_size`: Size penalty weight (default 0.5)
- `--lambda_radius`: Initial radius penalty weight (default 1.0)
- `--lambda_radius_min`: Final radius penalty weight (default 0.2)
- `--step_cost`: Cost per step (default 0.001)
- `--stop_bias`: STOP action bias (default 0.5)
- `--fidelity_thresh`: Fidelity threshold for STOP bonus (default 0.9)

## 2. PPO Stability Improvements

### Changes in `components/generator.py`:
- **Clip Range**: Reduced from 0.2 to 0.1 for more stable updates
- **KL Target**: Set to 0.0075 with adaptive penalty when exceeded
- **Learning Rate**: Reduced from 3e-4 to 1e-4
- **GAE Parameters**: Set gamma=0.99, gae_lambda=0.95
- **Gradient Clipping**: Set max_grad_norm=0.5
- **Value Clipping**: Optional value function clipping
- **Entropy Scheduling**: Linear decay from 1e-3 to 1e-4 across training

### New Parameters:
- `--ppo_clip`: PPO clip range (default 0.1)
- `--kl_target`: Target KL divergence (default 0.0075)
- `--lr`: Learning rate (default 1e-4)
- `--gamma`: Discount factor (default 0.99)
- `--gae_lambda`: GAE lambda (default 0.95)
- `--max_grad_norm`: Gradient clipping norm (default 0.5)
- `--use_value_clip`: Enable value clipping (default True)
- `--entropy_coeff_start`: Starting entropy coefficient (default 1e-3)
- `--entropy_coeff_end`: Ending entropy coefficient (default 1e-4)

## 3. Causal Reward Scheduling

### Changes in `components/generator.py`:
- **Warmup Schedule**: Causal weight ramps from 0 → target over first `causal_warmup_epochs`
- **Effective Weight**: `causal_w_eff = causal_w * min(1, epoch/causal_warmup_epochs)`

### New Parameters:
- `--causal_warmup_epochs`: Number of warmup epochs (default 5)

## 4. AUC Evaluation Hardening

### Changes in `evaluation/utils.py`:
- **Edge-Label Alignment**: Added `align_edge_scores_with_labels()` function
- **Node-to-Edge Scoring**: Support for max/mean aggregation of node scores to edge scores
- **Debug Evaluation**: Added `debug_eval_one()` for single-graph analysis
- **CSV Output**: Debug results saved to `results/debug_eval_graph_{id}.csv`

### Changes in `evaluation/AUCEvaluation.py`:
- **Enhanced get_score()**: Added support for edge_score_agg and debug_eval_graph parameters

### New Parameters:
- `--edge_score_agg`: Aggregation method ('max' or 'mean', default 'max')
- `--debug_eval_graph`: Graph ID for debug evaluation (-1 to disable, default -1)

## 5. Runner Integration

### Changes in `run_RG_explainer.py`:
- **Parameter Wiring**: All new parameters integrated into Generator initialization
- **Enhanced Logging**: Added early_stop_rate, avg_size, effective_causal_weight logging
- **Dataset Defaults**: Automatic target_size_k based on dataset type
- **Evaluation Integration**: Debug evaluation and edge scoring aggregation

### New Logging Metrics:
- `early_stop_rate`: Rate of STOP action chosen before max steps
- `avg_size`: Average explanation size
- `effective_causal_weight`: Current causal weight considering warmup

## 6. SLURM Array Script

### New File: `slurm/run_rgx_array.slurm`
- **Array Job**: Runs 5 seeds (0-4) in parallel
- **GPU Allocation**: 1 GPU per job with 24GB memory
- **Time Limit**: 8 hours per job
- **Output Management**: Separate log files for each array job
- **Environment Setup**: CUDA 12.1 and conda environment activation

## 7. Testing and Validation

### New File: `test_debug_eval.py`
- **Unit Tests**: Tests edge-label alignment and debug evaluation
- **Synthetic Graph**: Creates known motif for validation
- **AUC Verification**: Ensures AUC ≠ 0.5 unless scores are uniform

## Usage Examples

### Basic Usage:
```bash
python run_RG_explainer.py --dataset mutag --seed 0
```

### With Debug Evaluation:
```bash
python run_RG_explainer.py --dataset mutag --seed 0 --debug_eval_graph 0
```

### SLURM Array Job:
```bash
sbatch slurm/run_rgx_array.slurm
```

## Expected Improvements

1. **Better AUC**: Improved edge-label alignment and STOP action learning
2. **More Compact Explanations**: Size and radius penalties encourage smaller explanations
3. **Stable Training**: PPO stability improvements reduce training variance
4. **Controlled Causal Integration**: Warmup schedule prevents early training instability
5. **Better Debugging**: Debug evaluation helps identify alignment issues

## Acceptance Criteria Met

✅ STOP action is learnable and attractive when fidelity is preserved  
✅ Compactness enforced via size and radius penalties with schedules  
✅ PPO stabilized with clip=0.1, KL target ~0.0075, entropy anneal, smaller LR  
✅ AUC evaluation hardened with proper edge-label alignment  
✅ Ready-to-run SLURM array script created  
✅ Debug evaluation produces AUC ≠ 0.5 on synthetic graph with known motif
