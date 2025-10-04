"""
Plotting functions for PPO + Causal GNN Explainer analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any


def create_plots(epoch_metrics: pd.DataFrame, output_dir: str, config: Dict[str, Any]) -> None:
    """Create all plots and save to output directory."""
    figures_dir = Path(output_dir) / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Set matplotlib style
    plt.style.use(config.get('plots', {}).get('style', 'default'))
    dpi = config.get('plots', {}).get('dpi', 160)
    
    # Create all plots
    plot_auc(epoch_metrics, figures_dir, dpi)
    plot_kl_clipfrac(epoch_metrics, figures_dir, dpi, config)
    plot_entropy_expvar(epoch_metrics, figures_dir, dpi)
    plot_loss_decomp(epoch_metrics, figures_dir, dpi)
    plot_length_trends(epoch_metrics, figures_dir, dpi)
    plot_earlystop_heatmap(epoch_metrics, figures_dir, dpi)


def plot_auc(epoch_metrics: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot AUC vs epoch with horizontal line at 0.5."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'auc' in epoch_metrics.columns:
        valid_auc = epoch_metrics.dropna(subset=['auc'])
        if not valid_auc.empty:
            ax.plot(valid_auc['epoch'], valid_auc['auc'], 'b-', linewidth=2, label='AUC')
            
            # Add EMA if available
            if 'auc_ema' in epoch_metrics.columns:
                valid_ema = epoch_metrics.dropna(subset=['auc_ema'])
                if not valid_ema.empty:
                    ax.plot(valid_ema['epoch'], valid_ema['auc_ema'], 'r--', 
                           linewidth=1.5, alpha=0.7, label='AUC (EMA)')
            
            # Horizontal line at 0.5
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Random (0.5)')
            
            # Annotate first epoch where AUC >= 0.6
            auc_above_06 = valid_auc[valid_auc['auc'] >= 0.6]
            if not auc_above_06.empty:
                first_epoch = auc_above_06['epoch'].iloc[0]
                first_auc = auc_above_06['auc'].iloc[0]
                ax.annotate(f'AUC ≥ 0.6\n(Epoch {first_epoch})', 
                           xy=(first_epoch, first_auc), 
                           xytext=(first_epoch + 2, first_auc + 0.05),
                           arrowprops=dict(arrowstyle='->', color='red'),
                           fontsize=10, ha='center')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('AUC vs Epoch')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'auc.png', dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_kl_clipfrac(epoch_metrics: pd.DataFrame, output_dir: Path, dpi: int, config: Dict) -> None:
    """Plot KL divergence and clip fraction vs epoch."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # KL divergence
    if 'mean_kl' in epoch_metrics.columns:
        valid_kl = epoch_metrics.dropna(subset=['mean_kl'])
        if not valid_kl.empty:
            ax1.plot(valid_kl['epoch'], valid_kl['mean_kl'], 'b-', linewidth=2, label='Mean KL')
            
            # KL target line
            kl_target = config.get('kl_target', 0.01)
            ax1.axhline(y=kl_target, color='red', linestyle='--', alpha=0.7, label=f'KL Target ({kl_target})')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('KL Divergence', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # Clip fraction on secondary axis
    ax2 = ax1.twinx()
    if 'mean_clipfrac' in epoch_metrics.columns:
        valid_clip = epoch_metrics.dropna(subset=['mean_clipfrac'])
        if not valid_clip.empty:
            ax2.plot(valid_clip['epoch'], valid_clip['mean_clipfrac'], 'r-', linewidth=2, label='Mean ClipFrac')
            
            # Clip fraction warning line
            clip_warn = config.get('warn', {}).get('clipfrac_hi', 0.4)
            ax2.axhline(y=clip_warn, color='orange', linestyle=':', alpha=0.7, label=f'Warning ({clip_warn})')
    
    ax2.set_ylabel('Clip Fraction', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0, 1)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('KL Divergence and Clip Fraction vs Epoch')
    plt.tight_layout()
    plt.savefig(output_dir / 'kl_clipfrac.png', dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_entropy_expvar(epoch_metrics: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot entropy and explained variance vs epoch."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Entropy
    if 'entropy_median' in epoch_metrics.columns:
        valid_entropy = epoch_metrics.dropna(subset=['entropy_median'])
        if not valid_entropy.empty:
            ax1.plot(valid_entropy['epoch'], valid_entropy['entropy_median'], 'b-', linewidth=2, label='Entropy (median)')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Entropy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # Explained variance on secondary axis
    ax2 = ax1.twinx()
    if 'expv_median' in epoch_metrics.columns:
        valid_expv = epoch_metrics.dropna(subset=['expv_median'])
        if not valid_expv.empty:
            ax2.plot(valid_expv['epoch'], valid_expv['expv_median'], 'r-', linewidth=2, label='Explained Var (median)')
    
    ax2.set_ylabel('Explained Variance', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0, 1)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Entropy and Explained Variance vs Epoch')
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_expvar.png', dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_loss_decomp(epoch_metrics: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot stacked bars of loss components per epoch."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Loss components
    loss_cols = ['eval_avg_p', 'eval_avg_size', 'eval_avg_sim', 'eval_avg_rp']
    loss_labels = ['Prediction Loss', 'Size Loss', 'Similarity Loss', 'Radius Penalty']
    
    # Filter valid data
    valid_data = epoch_metrics.dropna(subset=loss_cols)
    if valid_data.empty:
        ax.text(0.5, 0.5, 'No loss data available', ha='center', va='center', transform=ax.transAxes)
        plt.title('Loss Decomposition')
        plt.tight_layout()
        plt.savefig(output_dir / 'loss_decomp.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        return
    
    # Create stacked bar plot
    bottom = np.zeros(len(valid_data))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (col, label, color) in enumerate(zip(loss_cols, loss_labels, colors)):
        if col in valid_data.columns:
            values = valid_data[col].fillna(0)
            ax.bar(valid_data['epoch'], values, bottom=bottom, label=label, color=color, alpha=0.8)
            bottom += values
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss Decomposition by Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_decomp.png', dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_length_trends(epoch_metrics: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot average length and length efficiency vs epoch."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Average length
    if 'avg_len' in epoch_metrics.columns:
        valid_len = epoch_metrics.dropna(subset=['avg_len'])
        if not valid_len.empty:
            ax1.plot(valid_len['epoch'], valid_len['avg_len'], 'b-', linewidth=2, label='Avg Length')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Length', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # Length efficiency on secondary axis
    ax2 = ax1.twinx()
    if 'length_efficiency' in epoch_metrics.columns:
        valid_eff = epoch_metrics.dropna(subset=['length_efficiency'])
        if not valid_eff.empty:
            ax2.plot(valid_eff['epoch'], valid_eff['length_efficiency'], 'r-', linewidth=2, label='Length Efficiency')
    
    ax2.set_ylabel('Length Efficiency (AUC/Length)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Length Trends vs Epoch')
    plt.tight_layout()
    plt.savefig(output_dir / 'length_trends.png', dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_earlystop_heatmap(epoch_metrics: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot early stopping heatmap or scatter."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'early_stop_rate' in epoch_metrics.columns:
        valid_data = epoch_metrics.dropna(subset=['early_stop_rate'])
        if not valid_data.empty:
            # Use scatter plot for early stop rates
            scatter = ax.scatter(valid_data['epoch'], [1] * len(valid_data), 
                               c=valid_data['early_stop_rate'], s=100, 
                               cmap='Reds', alpha=0.7, vmin=0, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Early Stop Rate')
            
            # Add text annotations for high early stop rates
            high_rates = valid_data[valid_data['early_stop_rate'] > 0.5]
            for _, row in high_rates.iterrows():
                ax.annotate(f'{row["early_stop_rate"]:.2f}', 
                           xy=(row['epoch'], 1), 
                           xytext=(row['epoch'], 1.1),
                           ha='center', fontsize=8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Early Stop Indicator')
    ax.set_title('Early Stopping Rate by Epoch')
    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([1])
    ax.set_yticklabels(['Early Stop'])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'earlystop_heatmap.png', dpi=dpi, bbox_inches='tight')
    plt.close()
