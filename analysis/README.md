# PPO + Causal GNN Explainer Log Analysis Toolkit

A robust, reproducible Python toolkit for parsing training logs from PPO + Causal GNN Explainer experiments, extracting structured metrics, computing diagnostics, and generating comprehensive analysis reports.

## Features

- **Robust Log Parsing**: Handles noisy logs with regex-based extraction
- **Structured Data Export**: Saves parsed data as Parquet/CSV files
- **Comprehensive Metrics**: Computes training diagnostics and performance metrics
- **Rich Visualizations**: Generates 6 different plot types for analysis
- **Hyperparameter Suggestions**: AI-powered levers analysis with concrete recommendations
- **Reproducible Analysis**: Timestamped outputs with full manifest tracking

## Installation

```bash
# Install dependencies
pip install pandas numpy scipy matplotlib typer pyyaml regex pyarrow

# Or install from requirements
pip install -r requirements.txt
```

## Quick Start

### Basic Analysis

```bash
# Analyze a log file
python -m src.analyze --log logs/training.log --outdir results

# With custom configuration
python -m src.analyze --log logs/training.log --outdir results --config configs/custom.yaml

# Also save CSV files
python -m src.analyze --log logs/training.log --outdir results --save-csv
```

### Validate Log Format

```bash
# Check if your log file is parseable
python -m src.analyze validate --log logs/training.log
```

## Output Structure

The toolkit creates a timestamped directory with the following structure:

```
results/mutag_20241201-143022/
├── data/
│   ├── run_meta.parquet          # Run metadata (dataset, accuracies, etc.)
│   ├── pretrained_rows.parquet   # Pretrained evaluation results
│   ├── ppo_diags.parquet         # PPO training diagnostics
│   ├── epoch_summaries.parquet   # Per-epoch summaries
│   └── epoch_metrics.parquet     # Computed metrics and aggregations
├── figures/
│   ├── auc.png                   # AUC vs epoch with annotations
│   ├── kl_clipfrac.png          # KL divergence and clip fraction
│   ├── entropy_expvar.png       # Entropy and explained variance
│   ├── loss_decomp.png          # Loss component decomposition
│   ├── length_trends.png        # Length and efficiency trends
│   └── earlystop_heatmap.png    # Early stopping patterns
├── levers.md                    # Hyperparameter suggestions
└── manifest.json               # Complete analysis manifest
```

## Log Format Requirements

The parser expects logs with these key patterns:

### Initialization Block
```
Loading mutag dataset
Train Acc: 0.8204, Val Acc: 0.8249, Test Acc: 0.8134
total nodes 131488, total edges 266894
Prediction Error for Graphs: 0.4433940511874568
[CAUSAL] Initialized GraphLevelCausalExplainer with 2 classes, weight 0.5
```

### Pretrained Evaluation
```
g_id 5 n_id 12 len 12 p_loss=0.54 size_loss=0.02 sim_loss=0.05 r_p=0.00 sgs [...]
...
avg_p_loss=0.36 avg_size_loss=0.04 avg_sim_loss=0.02 r_p=0.00
[EVAL-Pretrained] AUC=0.5
```

### Training Loop (per epoch)
```
====================
[Epoch    1]
Entropy coefficient: 0.001000
[Diagnostics]KL=0.0000, ClipFrac=0.000, Entropy=0.5515, ExplainedVar=0.0000, PolicyLoss=-0.0243, ValueLoss=5.7899
Early stopping: KL divergence 0.1920 > target 0.01
Reward=-0.40 PLoss=-0.02 Length=18.3
Elapsed Time: 267.1s
avg_p_loss=0.36 avg_size_loss=0.04 avg_sim_loss=0.02 r_p=0.00
AUC=0.5
```

## Configuration

Create a `config.yaml` file to customize analysis:

```yaml
# PPO hyperparameters
kl_target: 0.01
clip_range: 0.2

# Warning thresholds
warn:
  clipfrac_hi: 0.4      # Clip fraction warning threshold
  expv_lo: 0.1          # Explained variance warning threshold
  entropy_collapse: 0.2 # Entropy collapse warning threshold

# Plotting settings
plots:
  dpi: 160
  style: default
```

## Key Metrics Explained

### Training Diagnostics
- **KL Divergence**: Measures policy change magnitude (used for early stopping)
- **Clip Fraction**: Percentage of policy updates that were clipped
- **Entropy**: Policy exploration measure (higher = more exploration)
- **Explained Variance**: How well value function explains returns

### Performance Metrics
- **AUC**: Area Under Curve - main evaluation metric
- **Length Efficiency**: AUC divided by average subgraph length
- **Early Stop Rate**: Frequency of KL-based early stopping
- **Reward-AUC Correlation**: Alignment between RL reward and AUC

## Levers Analysis

The toolkit generates hyperparameter suggestions based on training patterns:

- **High KL + Clip Fraction**: Suggests reducing learning rate or clip range
- **Poor Reward-AUC Alignment**: Recommends adjusting reward function weights
- **Value Function Issues**: Suggests value network improvements
- **Length Inefficiency**: Recommends size penalty adjustments

## Example Analysis

```bash
# Run analysis on sample data
python -m src.analyze --log tests/sample_snippet.txt --outdir results --save-csv

# Check results
ls results/mutag_*/figures/  # View generated plots
cat results/mutag_*/levers.md  # Read suggestions
```

## Testing

```bash
# Run validation on sample data
python -m src.analyze validate --log tests/sample_snippet.txt

# Expected output:
# Parsing Results:
# run_meta           :      1 rows
# pretrained_rows    :      5 rows  
# ppo_diags          :     10 rows
# epoch_summaries    :      5 rows
```

## Troubleshooting

### Common Issues

1. **Empty DataFrames**: Check log format matches expected patterns
2. **Missing Plots**: Ensure epoch data is available
3. **Regex Errors**: Verify log contains required diagnostic patterns

### Debug Mode

```bash
# Validate log format first
python -m src.analyze validate --log your_log.txt
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
