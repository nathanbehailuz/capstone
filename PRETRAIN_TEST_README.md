# Pretraining Test Runner

A standalone testing framework for validating generator pretraining with edge dropping functionality.

## Overview

The pretraining test runner allows you to:
- Test pretraining with different edge drop probabilities
- Evaluate model robustness on edge-dropped graphs
- Run short PPO warmup phases to inspect training metrics
- Generate comprehensive reports without full multi-hour training runs

## Quick Start

### Local Testing

**Basic test (no edge dropping):**
```bash
python pretrain_test_runner.py --dataset syn2 --seed 0
```

**Test with edge dropping:**
```bash
python pretrain_test_runner.py --dataset syn2 --seed 0 --random_edge_drop 0.2
```

**Quick test (minimal epochs, skip PPO):**
```bash
python pretrain_test_runner.py --dataset syn2 --pretrain_list 3 --pretrain_set 5 --skip_ppo_test
```

**Skip robustness evaluation:**
```bash
python pretrain_test_runner.py --dataset syn2 --skip_eval_robustness
```

### SLURM Submission

**Standard test:**
```bash
sbatch slurm/pretrain_test.slurm syn2 0 0.1
# Arguments: [dataset] [seed] [edge_drop_prob]
```

**Quick test (30 minutes):**
```bash
sbatch slurm/pretrain_test_quick.slurm syn2 0 0.0
```

**Comparison test (baseline vs 10% vs 20% drop):**
```bash
sbatch slurm/pretrain_test_compare.slurm syn2 0
```

## Command Line Arguments

### Essential Arguments

- `--dataset`: Dataset name (syn1/syn2/syn3/syn4/ba2/mutag)
- `--seed`: Random seed for reproducibility
- `--random_edge_drop`: Edge drop probability (0.0-1.0, default: 0.0)

### Pretraining Parameters

- `--pretrain_list`: List-based pretraining iterations (default: 5)
- `--pretrain_set`: Set-based pretraining iterations (default: 10)
- `--pretrain_g_batch_size`: Batch size for pretraining (default: 32)
- `--max_size`: Maximum subgraph size (default: 15)
- `--n_hop`: Number of hops for subgraph (default: 3)

### Testing Parameters

- `--ppo_test_epochs`: Number of PPO warmup epochs (default: 2)
- `--skip_ppo_test`: Skip PPO warmup phase
- `--skip_eval_robustness`: Skip robustness evaluation
- `--output_dir`: Output directory (default: results/pretraining_tests/)

### Architecture Parameters

- `--hidden_size`: Hidden layer size (default: 64)
- `--with_attr`: Use node attributes (default: True)
- `--model`: Model type (default: GNN)

## Output Files

### Log Files

Location: `results/pretraining_tests/{dataset}_drop{prob}_seed{seed}_{timestamp}.log`

Contains:
- Complete console output
- Pretraining progress
- Robustness evaluation results
- PPO warmup metrics
- Final summary

### JSON Metrics

Location: `results/pretraining_tests/{dataset}_drop{prob}_seed{seed}_{timestamp}.json`

Contains:
- Configuration parameters
- Pretraining metrics (duration, AUC score)
- Robustness test results (loss deltas for different drop rates)
- PPO warmup metrics (rewards, policy loss, entropy, KL)

## Test Phases

### Phase 1: Pretraining

- Initializes Runner with specified parameters
- Runs list-based and set-based pretraining
- Saves checkpoint to `pretrained/test_runs/`
- Evaluates pretrained model (AUC score)

**Output:**
```
===== PHASE 1: PRETRAINING =====
Initializing Runner...
Starting pretraining phase...
[Pretrain Edge Drop] Probability: 0.200
[Pretrain Edge Drop] Edges: 1234 -> 987 (retained: 80.0%)
...
Pretraining completed in 180.0 seconds (3.0 minutes)
AUC Score: 0.8523
```

### Phase 2: Robustness Evaluation

Tests model on graphs with varying edge drop rates:
- 0.0 (baseline)
- 0.1 (10% drop)
- 0.2 (20% drop)
- 0.3 (30% drop)

Computes loss differences (Δ) compared to baseline.

**Output:**
```
===== PHASE 2: ROBUSTNESS EVALUATION =====
Testing with drop probability: 0.00
  [EdgeDrop-Test] Pred Loss: 0.1234 | Total: 0.1456

Testing with drop probability: 0.10
  Edges: 1234 -> 1111 (retained: 90.0%)
  [EdgeDrop-Test] Pred Loss: 0.1289 | Total: 0.1511

ΔLoss (drop 0.1 - baseline): +0.0055
ΔLoss (drop 0.2 - baseline): +0.0112
ΔLoss (drop 0.3 - baseline): +0.0198
```

### Phase 3: PPO Warmup (Optional)

Runs short PPO training to verify:
- Entropy decay
- KL divergence stability
- Policy/value loss trends
- Early stopping behavior

**Output:**
```
===== PHASE 3: PPO WARMUP TEST =====
[PPO-Test Epoch 1/2]
  Reward: 2.345 | Policy Loss: -0.0012 | Length: 8.3
  Early Stop Rate: 0.456 | Avg Size: 7.8
  λ Schedule: size=0.0000, radius=0.0000, entropy=0.0030

[PPO-Test Epoch 2/2]
  Reward: 2.567 | Policy Loss: -0.0015 | Length: 8.1
  ...
```

## Summary Output

After all phases complete, a formatted summary is printed:

```
===== TEST SUMMARY =====
Dataset: syn2 | Seed: 0
Edge Drop Probability: 0.20

Pretraining:
  Duration: 3.0 minutes
  AUC Score: 0.8523

Robustness Evaluation:
  Baseline Loss (no drop): 0.1456
  Drop 0.1: Loss=0.1511, Δ=+0.0055
  Drop 0.2: Loss=0.1568, Δ=+0.0112
  Drop 0.3: Loss=0.1654, Δ=+0.0198

PPO Warmup (2 epochs):
  Final Reward (Epoch 2): 2.567
  Final Policy Loss: -0.0015
  Final Early Stop Rate: 0.456
  Final Avg Size: 7.8

Total Test Duration: 5.2 minutes
========================
```

## Use Cases

### 1. Validate Edge Dropping Implementation

```bash
# Test that edge dropping works without errors
python pretrain_test_runner.py --dataset syn2 --random_edge_drop 0.1 --pretrain_list 3 --pretrain_set 5
```

### 2. Compare Robustness Across Drop Rates

```bash
# Run comparison on SLURM
sbatch slurm/pretrain_test_compare.slurm syn2 0
```

Check the generated JSON files to compare:
- Baseline loss vs dropped loss
- Loss deltas (Δ) for different drop probabilities
- AUC scores

### 3. Quick Sanity Check Before Full Run

```bash
# Quick test with minimal iterations
python pretrain_test_runner.py \
    --dataset syn3 \
    --pretrain_list 2 \
    --pretrain_set 3 \
    --skip_ppo_test \
    --skip_eval_robustness
```

### 4. Debug PPO Training Issues

```bash
# Run with PPO warmup enabled
python pretrain_test_runner.py \
    --dataset syn2 \
    --ppo_test_epochs 5 \
    --skip_eval_robustness
```

Monitor entropy, KL divergence, and policy loss trends.

### 5. Test Multiple Datasets

```bash
# Test all synthetic datasets
for dataset in syn1 syn2 syn3 syn4; do
    python pretrain_test_runner.py --dataset $dataset --random_edge_drop 0.1
done
```

## Tips

1. **Start with quick tests**: Use `--pretrain_list 3 --pretrain_set 5` for rapid iteration
2. **Skip phases you don't need**: Use `--skip_ppo_test` or `--skip_eval_robustness` to save time
3. **Check JSON files**: They contain detailed metrics for analysis
4. **Compare results**: Run with and without edge dropping to see the impact
5. **Monitor SLURM output**: Check `slurm_out/` for job logs

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python pretrain_test_runner.py --pretrain_g_batch_size 16 --g_batch_size 64
```

### Tests Taking Too Long

Use quick mode:
```bash
sbatch slurm/pretrain_test_quick.slurm
```

### CUDA Not Available

The script will run on CPU, but will be slower. Check:
```python
python -c "import torch; print(torch.cuda.is_available())"
```

## File Structure

```
capstone/
├── pretrain_test_runner.py      # Main test script
├── slurm/
│   ├── pretrain_test.slurm      # Standard SLURM template
│   ├── pretrain_test_quick.slurm    # Quick test template
│   └── pretrain_test_compare.slurm  # Comparison template
├── results/
│   └── pretraining_tests/       # Test outputs (logs + JSON)
└── logs/                        # Additional log files
```

## Integration with Main Training

After validating pretraining with this tool, you can run full training:

```bash
# Use same edge drop probability in main runner
python run_RG_explainer.py \
    --dataset syn2 \
    --seed 0 \
    --random_edge_drop 0.1 \
    --n_epochs 30
```

The `--random_edge_drop` parameter works identically in both scripts.

