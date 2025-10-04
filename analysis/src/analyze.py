"""
CLI interface for PPO + Causal GNN Explainer log analysis.
"""

import typer
import yaml
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import parse_log
from metrics import compute_metrics
from plots import create_plots
from reporter import generate_levers_report
from utils import (
    load_text, ensure_dir, timestamped_dir, dump_json, save_df, create_manifest
)

app = typer.Typer(help="PPO + Causal GNN Explainer Log Analysis Toolkit")


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Return default config
        return {
            'kl_target': 0.01,
            'clip_range': 0.2,
            'ema_window': 5,
            'warn': {
                'clipfrac_hi': 0.4,
                'expv_lo': 0.1,
                'entropy_collapse': 0.2
            },
            'plots': {
                'dpi': 160,
                'style': 'default'
            }
        }


@app.command()
def analyze(
    log: str = typer.Argument(..., help="Path to log file"),
    outdir: str = typer.Option(..., "--outdir", "-o", help="Output directory"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Config YAML file"),
    kl_target: Optional[float] = typer.Option(None, "--kl-target", help="KL target override"),
    clip_range: Optional[float] = typer.Option(None, "--clip-range", help="Clip range override"),
    save_csv: bool = typer.Option(False, "--save-csv", help="Also save CSV files"),
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Dataset name override")
):
    """Analyze PPO + Causal GNN Explainer training log."""
    
    # Load configuration
    config_dict = load_config(config)
    
    # Override config with CLI arguments
    if kl_target is not None:
        config_dict['kl_target'] = kl_target
    if clip_range is not None:
        config_dict['clip_range'] = clip_range
    
    # Load log file
    typer.echo(f"Loading log file: {log}")
    log_text = load_text(log)
    
    # Parse log
    typer.echo("Parsing log file...")
    parsed_data = parse_log(log_text)
    
    # Extract dataset name
    if dataset:
        dataset_name = dataset
    elif not parsed_data['run_meta'].empty and not pd.isna(parsed_data['run_meta']['dataset'].iloc[0]):
        dataset_name = parsed_data['run_meta']['dataset'].iloc[0]
    else:
        dataset_name = "unknown"
    
    # Create output directory
    output_dir = timestamped_dir(outdir, dataset_name)
    ensure_dir(output_dir)
    typer.echo(f"Output directory: {output_dir}")
    
    # Create subdirectories
    data_dir = Path(output_dir) / 'data'
    figures_dir = Path(output_dir) / 'figures'
    ensure_dir(str(data_dir))
    ensure_dir(str(figures_dir))
    
    # Save raw parsed data
    typer.echo("Saving parsed data...")
    file_paths = {}
    
    for name, df in parsed_data.items():
        if not df.empty:
            file_path = data_dir / f"{name}.parquet"
            save_df(df, str(file_path), save_csv=save_csv)
            file_paths[name] = str(file_path)
            typer.echo(f"  Saved {name}: {len(df)} rows")
        else:
            typer.echo(f"  Skipped {name}: empty DataFrame")
    
    # Compute metrics
    typer.echo("Computing metrics...")
    scalar_metrics, epoch_metrics = compute_metrics(
        parsed_data['ppo_diags'], 
        parsed_data['epoch_summaries']
    )
    
    # Save epoch metrics
    if not epoch_metrics.empty:
        epoch_metrics_path = data_dir / "epoch_metrics.parquet"
        save_df(epoch_metrics, str(epoch_metrics_path), save_csv=save_csv)
        file_paths['epoch_metrics'] = str(epoch_metrics_path)
        typer.echo(f"  Saved epoch_metrics: {len(epoch_metrics)} rows")
    
    # Create plots
    typer.echo("Creating plots...")
    if not epoch_metrics.empty:
        create_plots(epoch_metrics, output_dir, config_dict)
        typer.echo("  Created 6 plots in figures/")
    else:
        typer.echo("  Skipped plots: no epoch data")
    
    # Generate levers report
    typer.echo("Generating levers report...")
    levers_path = Path(output_dir) / "levers.md"
    generate_levers_report(epoch_metrics, scalar_metrics, config_dict, str(levers_path))
    file_paths['levers_report'] = str(levers_path)
    typer.echo(f"  Generated levers report: {levers_path}")
    
    # Create manifest
    typer.echo("Creating manifest...")
    manifest = create_manifest(output_dir, config_dict, file_paths)
    manifest_path = Path(output_dir) / "manifest.json"
    dump_json(manifest, str(manifest_path))
    typer.echo(f"  Created manifest: {manifest_path}")
    
    # Print summary
    typer.echo("\n" + "="*50)
    typer.echo("ANALYSIS COMPLETE")
    typer.echo("="*50)
    typer.echo(f"Output directory: {output_dir}")
    typer.echo(f"Dataset: {dataset_name}")
    
    if scalar_metrics:
        final_auc = scalar_metrics.get('final_auc', 'N/A')
        total_epochs = scalar_metrics.get('total_epochs', 'N/A')
        typer.echo(f"Final AUC: {final_auc}")
        typer.echo(f"Total epochs: {total_epochs}")
    
    typer.echo(f"Files created: {len(file_paths)}")
    typer.echo("="*50)


@app.command()
def validate(
    log: str = typer.Argument(..., help="Path to log file")
):
    """Validate log file format and show parsing statistics."""
    
    typer.echo(f"Validating log file: {log}")
    log_text = load_text(log)
    
    # Parse log
    parsed_data = parse_log(log_text)
    
    # Show statistics
    typer.echo("\nParsing Results:")
    typer.echo("-" * 30)
    
    for name, df in parsed_data.items():
        if not df.empty:
            typer.echo(f"{name:20}: {len(df):6} rows")
            if name == 'run_meta' and not df.empty:
                dataset = df['dataset'].iloc[0] if not pd.isna(df['dataset'].iloc[0]) else 'unknown'
                typer.echo(f"{'':20}  Dataset: {dataset}")
        else:
            typer.echo(f"{name:20}: {'EMPTY':6}")
    
    # Check for key patterns
    typer.echo("\nPattern Validation:")
    typer.echo("-" * 30)
    
    patterns = {
        'Epoch headers': r'\[Epoch\s+\d+\]',
        'Diagnostics': r'\[Diagnostics\]',
        'AUC scores': r'AUC=[\d\.]+',
        'Early stopping': r'Early stopping:',
        'Pretrained eval': r'\[EVAL-Pretrained\]'
    }
    
    import re
    for pattern_name, pattern in patterns.items():
        matches = len(re.findall(pattern, log_text))
        typer.echo(f"{pattern_name:20}: {matches:6} matches")
    
    typer.echo("\nValidation complete!")


if __name__ == "__main__":
    app()
