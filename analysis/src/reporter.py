"""
Reporter for generating levers analysis and hyperparameter suggestions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


def generate_levers_report(epoch_metrics: pd.DataFrame, scalar_metrics: Dict, 
                          config: Dict, output_path: str) -> None:
    """Generate levers.md with hyperparameter suggestions."""
    
    suggestions = []
    warnings = []
    
    # Analyze metrics and generate suggestions
    suggestions.extend(_analyze_kl_clipfrac(epoch_metrics, scalar_metrics, config))
    suggestions.extend(_analyze_reward_auc_alignment(epoch_metrics, scalar_metrics))
    suggestions.extend(_analyze_value_function(epoch_metrics, scalar_metrics, config))
    suggestions.extend(_analyze_length_efficiency(epoch_metrics, scalar_metrics))
    suggestions.extend(_analyze_early_stopping(epoch_metrics, scalar_metrics))
    
    # Generate warnings
    warnings.extend(_generate_warnings(epoch_metrics, scalar_metrics, config))
    
    # Create summary table
    summary_table = _create_summary_table(epoch_metrics)
    
    # Write report
    _write_report(suggestions, warnings, summary_table, scalar_metrics, output_path)


def _analyze_kl_clipfrac(epoch_metrics: pd.DataFrame, scalar_metrics: Dict, config: Dict) -> List[str]:
    """Analyze KL divergence and clip fraction patterns."""
    suggestions = []
    
    mean_kl = scalar_metrics.get('mean_kl', 0)
    mean_clipfrac = scalar_metrics.get('mean_clipfrac', 0)
    pct_kl_above_target = scalar_metrics.get('pct_kl_above_target', 0)
    kl_target = scalar_metrics.get('kl_target', 0.01)
    
    if mean_kl > kl_target * 2 and mean_clipfrac > 0.4:
        suggestions.append(
            f"**High KL divergence ({mean_kl:.4f}) and clip fraction ({mean_clipfrac:.3f})**: "
            f"Consider reducing learning rate or clip_range. Current KL target ({kl_target}) may be too strict."
        )
    elif mean_kl > kl_target * 1.5:
        suggestions.append(
            f"**Elevated KL divergence ({mean_kl:.4f})**: "
            f"Policy updates are large. Consider smaller learning rate or looser KL target."
        )
    elif mean_clipfrac > 0.6:
        suggestions.append(
            f"**High clip fraction ({mean_clipfrac:.3f})**: "
            f"Many policy updates are being clipped. Consider increasing clip_range or reducing learning rate."
        )
    
    if pct_kl_above_target > 50:
        suggestions.append(
            f"**Frequent KL violations ({pct_kl_above_target:.1f}%)**: "
            f"KL target ({kl_target}) is frequently exceeded. Consider increasing target to {kl_target * 1.5:.4f}."
        )
    
    return suggestions


def _analyze_reward_auc_alignment(epoch_metrics: pd.DataFrame, scalar_metrics: Dict) -> List[str]:
    """Analyze reward-AUC alignment."""
    suggestions = []
    
    reward_auc_corr = scalar_metrics.get('reward_auc_spearman', 0)
    final_auc = scalar_metrics.get('final_auc', 0)
    max_auc = scalar_metrics.get('max_auc', 0)
    
    if reward_auc_corr < 0.3 and final_auc < 0.7:
        suggestions.append(
            f"**Poor reward-AUC alignment (corr={reward_auc_corr:.3f})**: "
            f"Reward signal may be misaligned with AUC. Consider increasing fidelity weight or reducing size/similarity penalties."
        )
    
    if max_auc > final_auc + 0.1:
        suggestions.append(
            f"**AUC degradation**: Max AUC ({max_auc:.3f}) > Final AUC ({final_auc:.3f}). "
            f"Consider early stopping or reducing learning rate in later epochs."
        )
    
    return suggestions


def _analyze_value_function(epoch_metrics: pd.DataFrame, scalar_metrics: Dict, config: Dict) -> List[str]:
    """Analyze value function performance."""
    suggestions = []
    
    expv_median = scalar_metrics.get('expv_median', 0)
    expv_warn_threshold = config.get('warn', {}).get('expv_lo', 0.1)
    
    if expv_median < expv_warn_threshold:
        suggestions.append(
            f"**Poor value function (explained var={expv_median:.3f})**: "
            f"Value function explains little variance. Consider increasing value loss coefficient or simplifying value network."
        )
    
    # Check if value function improves over time
    if 'expv_median' in epoch_metrics.columns:
        late_epochs = epoch_metrics[epoch_metrics['epoch'] > epoch_metrics['epoch'].max() * 0.7]
        if not late_epochs.empty:
            late_expv = late_epochs['expv_median'].mean()
            if late_expv < expv_warn_threshold:
                suggestions.append(
                    f"**Value function not learning**: Late-epoch explained variance ({late_expv:.3f}) remains low. "
                    f"Consider value function architecture changes or higher value loss weight."
                )
    
    return suggestions


def _analyze_length_efficiency(epoch_metrics: pd.DataFrame, scalar_metrics: Dict) -> List[str]:
    """Analyze length efficiency patterns."""
    suggestions = []
    
    length_efficiency = scalar_metrics.get('length_efficiency_mean', 0)
    final_auc = scalar_metrics.get('final_auc', 0)
    
    if length_efficiency < 0.02 and final_auc > 0.6:
        suggestions.append(
            f"**Low length efficiency ({length_efficiency:.4f})**: "
            f"Generated subgraphs are long relative to AUC. Consider increasing size penalty or improving STOP policy weighting."
        )
    
    # Check length trend
    if 'avg_len' in epoch_metrics.columns and 'auc' in epoch_metrics.columns:
        valid_data = epoch_metrics.dropna(subset=['avg_len', 'auc'])
        if len(valid_data) > 5:
            length_trend = np.polyfit(valid_data['epoch'], valid_data['avg_len'], 1)[0]
            auc_trend = np.polyfit(valid_data['epoch'], valid_data['auc'], 1)[0]
            
            if length_trend > 0.5 and auc_trend < 0.01:
                suggestions.append(
                    f"**Length increasing without AUC gain**: Length trend (+{length_trend:.2f}/epoch) > AUC trend (+{auc_trend:.4f}/epoch). "
                    f"Consider stronger size regularization."
                )
    
    return suggestions


def _analyze_early_stopping(epoch_metrics: pd.DataFrame, scalar_metrics: Dict) -> List[str]:
    """Analyze early stopping patterns."""
    suggestions = []
    
    avg_early_stop_rate = scalar_metrics.get('avg_early_stop_rate', 0)
    
    if avg_early_stop_rate > 0.8:
        suggestions.append(
            f"**Frequent early stopping ({avg_early_stop_rate:.1%})**: "
            f"KL target may be too strict. Consider increasing target or reducing learning rate."
        )
    elif avg_early_stop_rate < 0.1:
        suggestions.append(
            f"**Rare early stopping ({avg_early_stop_rate:.1%})**: "
            f"KL target may be too loose. Consider decreasing target for more stable updates."
        )
    
    return suggestions


def _generate_warnings(epoch_metrics: pd.DataFrame, scalar_metrics: Dict, config: Dict) -> List[str]:
    """Generate warning messages."""
    warnings = []
    
    # Entropy collapse warning
    entropy_median = scalar_metrics.get('entropy_median', 0)
    entropy_collapse_threshold = config.get('warn', {}).get('entropy_collapse', 0.2)
    
    if entropy_median < entropy_collapse_threshold:
        warnings.append(
            f"⚠️ **Entropy collapse**: Median entropy ({entropy_median:.3f}) is very low. "
            f"Policy may be over-exploiting and losing exploration."
        )
    
    # AUC performance warning
    final_auc = scalar_metrics.get('final_auc', 0)
    if final_auc < 0.5:
        warnings.append(
            f"⚠️ **Poor AUC performance**: Final AUC ({final_auc:.3f}) is below random baseline (0.5). "
            f"Check reward function and hyperparameters."
        )
    
    # Training instability warning
    if 'mean_kl' in epoch_metrics.columns:
        kl_std = epoch_metrics['mean_kl'].std()
        if kl_std > 0.1:
            warnings.append(
                f"⚠️ **Training instability**: High KL divergence variance ({kl_std:.3f}). "
                f"Consider reducing learning rate or increasing batch size."
            )
    
    return warnings


def _create_summary_table(epoch_metrics: pd.DataFrame) -> str:
    """Create summary table for key metrics."""
    if epoch_metrics.empty:
        return "No data available for summary table."
    
    # Select key epochs (every 5th epoch + final)
    key_epochs = []
    if len(epoch_metrics) > 0:
        key_epochs.extend(epoch_metrics['epoch'].iloc[::5].tolist())
        if epoch_metrics['epoch'].iloc[-1] not in key_epochs:
            key_epochs.append(epoch_metrics['epoch'].iloc[-1])
    
    # Filter to key epochs
    summary_data = epoch_metrics[epoch_metrics['epoch'].isin(key_epochs)].copy()
    
    if summary_data.empty:
        return "No summary data available."
    
    # Create table
    table_lines = ["| Epoch | AUC | Reward | AvgLen | MeanKL | ClipFrac | EarlyStopRate |"]
    table_lines.append("|------:|----:|-------:|-------:|------:|---------:|--------------:|")
    
    for _, row in summary_data.iterrows():
        epoch = int(row['epoch'])
        auc = f"{row.get('auc', 0):.3f}" if not pd.isna(row.get('auc')) else "N/A"
        reward = f"{row.get('reward', 0):.2f}" if not pd.isna(row.get('reward')) else "N/A"
        avg_len = f"{row.get('avg_len', 0):.1f}" if not pd.isna(row.get('avg_len')) else "N/A"
        mean_kl = f"{row.get('mean_kl', 0):.4f}" if not pd.isna(row.get('mean_kl')) else "N/A"
        clipfrac = f"{row.get('mean_clipfrac', 0):.3f}" if not pd.isna(row.get('mean_clipfrac')) else "N/A"
        early_stop = f"{row.get('early_stop_rate', 0):.2f}" if not pd.isna(row.get('early_stop_rate')) else "N/A"
        
        table_lines.append(f"| {epoch} | {auc} | {reward} | {avg_len} | {mean_kl} | {clipfrac} | {early_stop} |")
    
    return "\n".join(table_lines)


def _write_report(suggestions: List[str], warnings: List[str], summary_table: str, 
                 scalar_metrics: Dict, output_path: str) -> None:
    """Write the complete levers report."""
    
    with open(output_path, 'w') as f:
        f.write("# PPO + Causal GNN Explainer - Levers Analysis\n\n")
        
        # Executive summary
        f.write("## Executive Summary\n\n")
        final_auc = scalar_metrics.get('final_auc', 0)
        total_epochs = scalar_metrics.get('total_epochs', 0)
        f.write(f"- **Final AUC**: {final_auc:.3f}\n")
        f.write(f"- **Total Epochs**: {total_epochs}\n")
        f.write(f"- **Training Status**: {'✅ Complete' if total_epochs > 0 else '❌ Incomplete'}\n\n")
        
        # Warnings
        if warnings:
            f.write("## ⚠️ Warnings\n\n")
            for warning in warnings:
                f.write(f"{warning}\n\n")
        
        # Suggestions
        if suggestions:
            f.write("## 🔧 Hyperparameter Suggestions\n\n")
            for i, suggestion in enumerate(suggestions, 1):
                f.write(f"{i}. {suggestion}\n\n")
        else:
            f.write("## 🔧 Hyperparameter Suggestions\n\n")
            f.write("No specific suggestions based on current metrics. Training appears to be proceeding normally.\n\n")
        
        # Summary table
        f.write("## 📊 Key Metrics Summary\n\n")
        f.write(summary_table)
        f.write("\n\n")
        
        # Additional metrics
        f.write("## 📈 Additional Metrics\n\n")
        f.write("| Metric | Value | Description |\n")
        f.write("|--------|-------|-------------|\n")
        
        metric_descriptions = {
            'mean_kl': 'Average KL divergence',
            'mean_clipfrac': 'Average clip fraction',
            'entropy_median': 'Median policy entropy',
            'expv_median': 'Median explained variance',
            'reward_auc_spearman': 'Reward-AUC correlation',
            'length_efficiency_mean': 'Average length efficiency',
            'dAUC_5epoch_slope': '5-epoch AUC trend slope'
        }
        
        for metric, description in metric_descriptions.items():
            value = scalar_metrics.get(metric, 'N/A')
            if isinstance(value, float):
                value = f"{value:.4f}"
            f.write(f"| {metric} | {value} | {description} |\n")
        
        f.write("\n")
        
        # Footer
        f.write("---\n")
        f.write("*Generated by PPO + Causal GNN Explainer Log Analysis Toolkit*\n")
