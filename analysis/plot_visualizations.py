#!/usr/bin/env python3
"""
Read JSON summaries and generate visualization plots under visualization/<dataset>/plots/

Generates multiple plots showing AUC, loss metrics, and diagnostics across seeds and epochs.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_seed_summaries(viz_path: Path) -> List[Dict[str, Any]]:
    """
    Load all seed_*_epoch_summary.json files from visualization directory.
    
    Returns list of summary dictionaries.
    """
    summaries = []
    
    for json_file in sorted(viz_path.glob('seed_*_epoch_summary.json')):
        try:
            with open(json_file, 'r') as f:
                summary = json.load(f)
                summaries.append(summary)
        except Exception as e:
            print(f"  Warning: Could not load {json_file.name}: {e}")
    
    return summaries


def load_final_summary(viz_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load final_epoch_summary.json if it exists.
    
    Returns summary dictionary or None.
    """
    final_path = viz_path / 'final_epoch_summary.json'
    
    if not final_path.exists():
        return None
    
    try:
        with open(final_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: Could not load final_epoch_summary.json: {e}")
        return None


def plot_auc_per_epoch(summaries: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Plot AUC per epoch for all seeds (line chart).
    """
    plt.figure(figsize=(10, 6))
    
    for summary in summaries:
        seed = summary['metadata']['seed']
        epochs = summary['epochs']
        
        # Extract epoch numbers and AUC values
        epoch_nums = []
        auc_values = []
        
        for epoch in epochs:
            if epoch['AUC'] is not None and np.isfinite(epoch['AUC']):
                epoch_nums.append(epoch['epoch'])
                auc_values.append(epoch['AUC'])
        
        if epoch_nums:
            plt.plot(epoch_nums, auc_values, marker='o', label=f'Seed {seed}', alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC per Epoch (All Seeds)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    output_file = output_path / 'auc_per_epoch_all_seeds.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=200)
    plt.close()
    
    print(f"  → Saved: {output_file.name}")


def plot_auc_last_epoch(summaries: List[Dict[str, Any]], 
                         final_summary: Optional[Dict[str, Any]],
                         output_path: Path) -> None:
    """
    Plot last-epoch AUC across seeds (bar chart).
    """
    # Collect last epoch AUC for each seed
    seed_auc_pairs = []
    
    for summary in summaries:
        seed = summary['metadata']['seed']
        epochs = summary['epochs']
        
        if not epochs:
            continue
        
        # Get last epoch
        last_epoch = max(epochs, key=lambda e: e['epoch'])
        if last_epoch['AUC'] is not None and np.isfinite(last_epoch['AUC']):
            seed_auc_pairs.append((seed, last_epoch['AUC']))
    
    if not seed_auc_pairs:
        print("  Warning: No valid last-epoch AUC data to plot")
        return
    
    # Sort by AUC descending
    seed_auc_pairs.sort(key=lambda x: x[1], reverse=True)
    
    seeds = [str(pair[0]) for pair in seed_auc_pairs]
    aucs = [pair[1] for pair in seed_auc_pairs]
    
    plt.figure(figsize=(max(8, len(seeds) * 0.6), 6))
    bars = plt.bar(seeds, aucs, alpha=0.7, edgecolor='black')
    
    # Color bars by value
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(aucs)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Seed')
    plt.ylabel('AUC')
    
    # Add subtitle with mean/std if available
    title = 'Last Epoch AUC (All Seeds)'
    if final_summary and final_summary.get('AUC_mean') is not None:
        auc_mean = final_summary['AUC_mean']
        auc_std = final_summary['AUC_std']
        subtitle = f'Mean: {auc_mean:.3f} ± {auc_std:.3f}'
        plt.title(f'{title}\n{subtitle}', fontsize=12)
    else:
        plt.title(title)
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_file = output_path / 'auc_last_epoch_all_seeds.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=200)
    plt.close()
    
    print(f"  → Saved: {output_file.name}")


def plot_metric_per_epoch(summaries: List[Dict[str, Any]], 
                          metric_name: str,
                          metric_key: str,
                          ylabel: str,
                          output_filename: str,
                          output_path: Path,
                          is_diagnostics: bool = False) -> None:
    """
    Plot a single metric per epoch for all seeds (line chart).
    
    Args:
        metric_name: Display name for the metric
        metric_key: Key to extract from node_metrics or diagnostics dict
        ylabel: Y-axis label
        output_filename: Output file name
        is_diagnostics: If True, look in diagnostics dict; else node_metrics
    """
    plt.figure(figsize=(10, 6))
    
    has_data = False
    
    for summary in summaries:
        seed = summary['metadata']['seed']
        epochs = summary['epochs']
        
        # Extract epoch numbers and metric values
        epoch_nums = []
        metric_values = []
        
        for epoch in epochs:
            if is_diagnostics:
                value = epoch['diagnostics'].get(metric_key)
            else:
                value = epoch['node_metrics'].get(metric_key)
            
            if value is not None and np.isfinite(value):
                epoch_nums.append(epoch['epoch'])
                metric_values.append(value)
        
        if epoch_nums:
            plt.plot(epoch_nums, metric_values, marker='o', label=f'Seed {seed}', alpha=0.7)
            has_data = True
    
    if not has_data:
        print(f"  Warning: No valid data for {metric_name}")
        plt.close()
        return
    
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(f'{metric_name} per Epoch (All Seeds)')
    
    # Handle legend - if too many seeds, omit legend
    if len(summaries) <= 12:
        plt.legend(loc='best', fontsize=8)
    
    plt.grid(True, alpha=0.3)
    
    output_file = output_path / output_filename
    plt.savefig(output_file, bbox_inches='tight', dpi=200)
    plt.close()
    
    print(f"  → Saved: {output_file.name}")


def generate_all_plots(summaries: List[Dict[str, Any]], 
                       final_summary: Optional[Dict[str, Any]],
                       output_path: Path) -> None:
    """
    Generate all required plots.
    """
    print("\nGenerating plots:")
    
    # 1. AUC per epoch (all seeds)
    plot_auc_per_epoch(summaries, output_path)
    
    # 2. Last epoch AUC (bar chart)
    plot_auc_last_epoch(summaries, final_summary, output_path)
    
    # 3. Node-level metrics
    node_metrics = [
        ('p_loss_mean', 'P Loss', 'P Loss (mean)', 'loss_p_loss_per_epoch_all_seeds.png'),
        ('size_loss_mean', 'Size Loss', 'Size Loss (mean)', 'loss_size_loss_per_epoch_all_seeds.png'),
        ('sim_loss_mean', 'Sim Loss', 'Sim Loss (mean)', 'loss_sim_loss_per_epoch_all_seeds.png'),
        ('r_p_mean', 'R_P', 'R_P (mean)', 'loss_r_p_per_epoch_all_seeds.png'),
    ]
    
    for metric_key, metric_name, ylabel, filename in node_metrics:
        plot_metric_per_epoch(
            summaries, metric_name, metric_key, ylabel, filename, output_path, is_diagnostics=False
        )
    
    # 4. Diagnostics
    diagnostics = [
        ('KL_mean', 'KL Divergence', 'KL (mean)', 'diag_KL_per_epoch_all_seeds.png'),
        ('Entropy_mean', 'Entropy', 'Entropy (mean)', 'diag_Entropy_per_epoch_all_seeds.png'),
        ('ExplainedVar_mean', 'Explained Variance', 'Explained Var (mean)', 'diag_ExplainedVar_per_epoch_all_seeds.png'),
        ('PolicyLoss_mean', 'Policy Loss', 'Policy Loss (mean)', 'diag_PolicyLoss_per_epoch_all_seeds.png'),
        ('ValueLoss_mean', 'Value Loss', 'Value Loss (mean)', 'diag_ValueLoss_per_epoch_all_seeds.png'),
    ]
    
    for metric_key, metric_name, ylabel, filename in diagnostics:
        plot_metric_per_epoch(
            summaries, metric_name, metric_key, ylabel, filename, output_path, is_diagnostics=True
        )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate visualization plots from JSON summaries'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='syn3',
        help='Dataset name (default: syn3)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Results directory (default: results) [unused but kept for consistency]'
    )
    parser.add_argument(
        '--viz-dir',
        type=str,
        default='visualization',
        help='Visualization directory (default: visualization)'
    )
    
    args = parser.parse_args()
    
    # Construct paths
    viz_path = Path(args.viz_dir) / args.dataset
    plots_path = viz_path / 'plots'
    
    print(f"Processing dataset: {args.dataset}")
    print(f"Visualization directory: {viz_path}")
    print(f"Output plots directory: {plots_path}")
    
    # Check if visualization directory exists
    if not viz_path.exists():
        print(f"\nNo visualization directory found at {viz_path}")
        print("Please run build_json_summaries.py first.")
        return
    
    # Load seed summaries
    print("\nLoading JSON summaries:")
    summaries = load_seed_summaries(viz_path)
    
    if not summaries:
        print(f"No seed summary JSONs found in {viz_path}")
        print("Please run build_json_summaries.py first.")
        return
    
    print(f"  Loaded {len(summaries)} seed summaries")
    
    # Load final summary (optional)
    final_summary = load_final_summary(viz_path)
    if final_summary:
        print(f"  Loaded final epoch summary")
    
    # Ensure plots directory exists
    ensure_dir(str(plots_path))
    
    # Generate all plots
    generate_all_plots(summaries, final_summary, plots_path)
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()

