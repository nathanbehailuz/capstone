# Capstone project: RL based GNN Explainer

A reinforcement learning-based approach for generating interpretable subgraph explanations for Graph Neural Networks (GNNs), enhanced with causal reasoning capabilities.

## Overview

RC-Explainer combines reinforcement learning (specifically PPO - Proximal Policy Optimization) with causal mechanism integration to generate high-quality subgraph explanations for GNN predictions. The system uses counterfactual reasoning to identify causal edges by comparing factual (full graph) and counterfactual (selected edges only) GNN representations.

## Key Features

- **Reinforcement Learning Framework**: Uses PPO for subgraph generation with policy and value networks
- **Causal Mechanism Integration**: Implements counterfactual reasoning to identify causal edges
- **Multi-Dataset Support**: Works with synthetic datasets (syn1-syn4), BA2, and Mutagenicity datasets
- **Comprehensive Analysis**: Built-in log analysis toolkit for training diagnostics and performance evaluation
- **Explains** both node-level and graph-level causal explainers

## Project Structure

```
capstone/
├── components/                 # Core components
│   ├── generator.py           # Main RL generator with PPO
│   ├── causal_explainer.py    # Causal mechanism implementation
│   ├── graph.py              # Graph utilities and operations
│   └── layers.py             # Neural network layers
├── datasets/                  # Dataset loaders and utilities
│   ├── dataset_loaders.py    # Dataset loading functions
│   ├── ground_truth_loaders.py # Ground truth data
│   └── Mutagenicity/         # Mutagenicity dataset
├── models/                    # GNN model implementations
│   ├── GNN_paper.py          # Graph Neural Network models
│   ├── PG_paper.py           # Policy Gradient models
│   └── model_selector.py     # Model selection utilities
├── evaluation/                # Evaluation metrics and tools
│   ├── AUCEvaluation.py      # AUC-based evaluation
│   ├── BaseEvaluation.py     # Base evaluation class
│   └── EfficiencyEvaluation.py # Efficiency metrics
├── analysis/                  # Log analysis toolkit
│   ├── src/                  # Analysis source code
│   ├── configs/              # Configuration files
│   └── tests/                # Test files
├── slurm/                     # SLURM job scripts
├── sh/                       # Shell scripts for batch runs
├── ckpts/                    # Model checkpoints
├── pretrained/               # Pretrained model weights
├── results/                  # Experimental results
└── run_RG_explainer.py       # Main execution script
```

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- PyTorch Geometric
- CUDA (for GPU acceleration)

### Recommended Packages

For improved performance, install the following optional packages:

```bash
# Accelerate scatter operations (recommended)
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

# Replace ${TORCH} with your PyTorch version (e.g., 1.13.0)
# Replace ${CUDA} with your CUDA version (e.g., cu117 for CUDA 11.7, cpu for CPU-only)
```

**Note**: Without `torch-scatter`, you may see a warning about scatter operations, but the code will still work with a fallback implementation.


## Quick Start

### Basic Usage

```bash
# Run on Mutagenicity dataset with causal mechanism
python run_RG_explainer.py \
    --dataset mutag \
    --seed 0 \
    --policy PPO \
    --n_epochs 10 \
    --use_causal \
    --causal_weight 0.2 \
    --size_reg 0.01 \
    --sim_reg 0.01 \
    --radius_penalty 0.01
```

### Available Datasets

- `mutag`: Mutagenicity dataset
- `ba2`: Barabási-Albert synthetic dataset
- `syn1`, `syn2`, `syn3`, `syn4`: Synthetic datasets

### Key Parameters

- `--use_causal`: Enable causal mechanism
- `--causal_weight`: Weight for causal rewards (default: 0.2)
- `--size_reg`: Size regularization coefficient
- `--sim_reg`: Similarity regularization coefficient
- `--radius_penalty`: Radius penalty for subgraph size control

## HPC Usage

### SLURM Jobs

The project includes optimized SLURM scripts for HPC environments:

```bash
# Submit a causal explainer job
sbatch slurm/run_rgx_array.slurm

# Monitor job status
squeue -u $USER
sacct -u $USER --starttime today
```

### Batch Processing

Use the provided shell scripts for batch experiments:

```bash
# Run BA2 dataset with optimized PPO configuration
bash sh/run_ba2_ppo_optimized.sh
```

## Analysis Toolkit

The project includes a comprehensive log analysis toolkit:

```bash
# Navigate to analysis directory
cd analysis

# Run analysis on results
python -m src.analyze analyze ../results/mutag_fixed_test_seed_0.txt --outdir results --save-csv
```

### Analysis Features

- **Robust Log Parsing**: Handles noisy training logs
- **Structured Data Export**: Saves results as Parquet/CSV
- **Comprehensive Metrics**: Training diagnostics and performance metrics
- **Rich Visualizations**: 6 different plot types
- **Hyperparameter Suggestions**: AI-powered analysis with recommendations

## Causal Mechanism

The causal mechanism implements counterfactual reasoning:

1. **Factual**: GNN representations with full graph
2. **Counterfactual**: GNN representations with selected edges only
3. **Causal Effect**: Difference between factual and counterfactual representations

### Implementation Details

- **NodeLevelCausalExplainer**: For node classification tasks
- **GraphLevelCausalExplainer**: For graph classification tasks
- **CausalRewardCalculator**: Integrates causal rewards with RL rewards

See `CAUSAL_MECHANISM_GUIDE.md` for detailed implementation documentation.

## Configuration

### PPO Hyperparameters

```bash
--g_lr 3e-4              # Generator learning rate
--ppo_epochs 3           # PPO update epochs
--minibatch_size 1024    # Minibatch size
--clip_eps 0.1           # PPO clip parameter
--target_kl 0.03         # KL divergence target
--gae_lambda 0.95        # GAE lambda parameter
--entropy_coef 1e-3      # Entropy coefficient
```

### Causal Parameters

```bash
--use_causal             # Enable causal mechanism
--causal_weight 0.2      # Causal reward weight
--causal_temperature 1.0 # Temperature for causal scoring
```

## Results and Evaluation

### Metrics

- **AUC**: Area Under Curve - primary evaluation metric
- **Length Efficiency**: AUC divided by average subgraph length
- **Early Stop Rate**: Frequency of KL-based early stopping
- **Reward-AUC Correlation**: Alignment between RL reward and AUC

### Output Structure

```
results/
├── mutag_ppo_seed_0.txt     # Training logs
├── figures/                 # Generated plots
│   ├── auc.png
│   ├── kl_clipfrac.png
│   └── ...
└── data/                    # Structured data
    ├── run_meta.parquet
    ├── ppo_diags.parquet
    └── ...
```

## Documentation

- `CAUSAL_MECHANISM_GUIDE.md`: Detailed causal mechanism implementation
- `CAUSAL_INTEGRATION_SUMMARY.md`: Integration summary and status
- `analysis/README.md`: Analysis toolkit documentation
- `hpcnotes.md`: HPC usage notes
