"""
Computed metrics and aggregations for PPO + Causal GNN Explainer analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple
from utils import ema


def compute_metrics(ppo_diags: pd.DataFrame, epoch_summaries: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """
    Compute additional metrics and merge with epoch summaries.
    
    Returns:
        Tuple of (scalar_metrics_dict, merged_epoch_metrics_df)
    """
    if ppo_diags.empty or epoch_summaries.empty:
        return {}, epoch_summaries.copy()
    
    # Compute scalar metrics
    scalar_metrics = {}
    
    # Overall statistics
    scalar_metrics['total_epochs'] = len(epoch_summaries)
    scalar_metrics['total_diagnostics'] = len(ppo_diags)
    scalar_metrics['avg_early_stop_rate'] = ppo_diags['early_stop'].mean() if 'early_stop' in ppo_diags.columns else 0.0
    
    # KL statistics
    if 'kl' in ppo_diags.columns:
        scalar_metrics['mean_kl'] = ppo_diags['kl'].mean()
        scalar_metrics['max_kl'] = ppo_diags['kl'].max()
        scalar_metrics['median_kl'] = ppo_diags['kl'].median()
        
        # Percentage of KL values above target
        kl_target = ppo_diags['kl_target'].iloc[0] if not ppo_diags['kl_target'].isna().all() else 0.01
        scalar_metrics['pct_kl_above_target'] = (ppo_diags['kl'] > kl_target).mean() * 100
        scalar_metrics['kl_target'] = kl_target
    
    # Clip fraction statistics
    if 'clip' in ppo_diags.columns:
        scalar_metrics['mean_clipfrac'] = ppo_diags['clip'].mean()
        scalar_metrics['max_clipfrac'] = ppo_diags['clip'].max()
    
    # Entropy and explained variance
    if 'entropy' in ppo_diags.columns:
        scalar_metrics['entropy_median'] = ppo_diags['entropy'].median()
        scalar_metrics['entropy_mean'] = ppo_diags['entropy'].mean()
        scalar_metrics['entropy_std'] = ppo_diags['entropy'].std()
    
    if 'expv' in ppo_diags.columns:
        scalar_metrics['expv_median'] = ppo_diags['expv'].median()
        scalar_metrics['expv_mean'] = ppo_diags['expv'].mean()
    
    # Reward-AUC correlation
    if 'reward' in epoch_summaries.columns and 'auc' in epoch_summaries.columns:
        valid_mask = ~(epoch_summaries['reward'].isna() | epoch_summaries['auc'].isna())
        if valid_mask.sum() > 1:
            corr, p_value = stats.spearmanr(epoch_summaries.loc[valid_mask, 'reward'], 
                                          epoch_summaries.loc[valid_mask, 'auc'])
            scalar_metrics['reward_auc_spearman'] = corr
            scalar_metrics['reward_auc_pvalue'] = p_value
    
    # AUC trend analysis
    if 'auc' in epoch_summaries.columns:
        valid_auc = epoch_summaries['auc'].dropna()
        if len(valid_auc) > 1:
            scalar_metrics['final_auc'] = valid_auc.iloc[-1]
            scalar_metrics['max_auc'] = valid_auc.max()
            scalar_metrics['auc_improvement'] = valid_auc.iloc[-1] - valid_auc.iloc[0]
            
            # Rolling slope (5-epoch window)
            if len(valid_auc) >= 5:
                window = 5
                slopes = []
                for i in range(window, len(valid_auc)):
                    y = valid_auc.iloc[i-window:i].values
                    x = np.arange(len(y))
                    slope, _, _, _, _ = stats.linregress(x, y)
                    slopes.append(slope)
                scalar_metrics['dAUC_5epoch_slope'] = np.mean(slopes) if slopes else 0.0
            else:
                scalar_metrics['dAUC_5epoch_slope'] = 0.0
    
    # Length efficiency
    if 'auc' in epoch_summaries.columns and 'avg_len' in epoch_summaries.columns:
        valid_mask = ~(epoch_summaries['auc'].isna() | epoch_summaries['avg_len'].isna())
        if valid_mask.sum() > 0:
            length_efficiency = epoch_summaries.loc[valid_mask, 'auc'] / epoch_summaries.loc[valid_mask, 'avg_len']
            scalar_metrics['length_efficiency_mean'] = length_efficiency.mean()
            scalar_metrics['length_efficiency_median'] = length_efficiency.median()
    
    # Per-epoch metrics
    epoch_metrics = epoch_summaries.copy()
    
    # Early stop rate per epoch
    if not ppo_diags.empty:
        early_stop_by_epoch = ppo_diags.groupby('epoch')['early_stop'].agg(['count', 'sum']).reset_index()
        early_stop_by_epoch.columns = ['epoch', 'diag_count', 'early_stop_count']
        early_stop_by_epoch['early_stop_rate'] = early_stop_by_epoch['early_stop_count'] / early_stop_by_epoch['diag_count']
        
        epoch_metrics = epoch_metrics.merge(early_stop_by_epoch[['epoch', 'early_stop_rate']], 
                                          on='epoch', how='left')
        epoch_metrics['early_stop_rate'] = epoch_metrics['early_stop_rate'].fillna(0.0)
    else:
        epoch_metrics['early_stop_rate'] = 0.0
    
    # KL statistics per epoch
    if not ppo_diags.empty and 'kl' in ppo_diags.columns:
        kl_by_epoch = ppo_diags.groupby('epoch')['kl'].agg(['mean', 'max', 'median', 'std']).reset_index()
        kl_by_epoch.columns = ['epoch', 'mean_kl', 'max_kl', 'median_kl', 'std_kl']
        epoch_metrics = epoch_metrics.merge(kl_by_epoch, on='epoch', how='left')
        
        # Percentage above target per epoch
        kl_target = scalar_metrics.get('kl_target', 0.01)
        pct_above_target = ppo_diags.groupby('epoch').apply(
            lambda x: (x['kl'] > kl_target).mean() * 100
        ).reset_index()
        pct_above_target.columns = ['epoch', 'pct_kl_above_target']
        epoch_metrics = epoch_metrics.merge(pct_above_target, on='epoch', how='left')
        epoch_metrics['pct_kl_above_target'] = epoch_metrics['pct_kl_above_target'].fillna(0.0)
    else:
        epoch_metrics['mean_kl'] = np.nan
        epoch_metrics['max_kl'] = np.nan
        epoch_metrics['median_kl'] = np.nan
        epoch_metrics['std_kl'] = np.nan
        epoch_metrics['pct_kl_above_target'] = 0.0
    
    # Clip fraction per epoch
    if not ppo_diags.empty and 'clip' in ppo_diags.columns:
        clip_by_epoch = ppo_diags.groupby('epoch')['clip'].agg(['mean', 'max']).reset_index()
        clip_by_epoch.columns = ['epoch', 'mean_clipfrac', 'max_clipfrac']
        epoch_metrics = epoch_metrics.merge(clip_by_epoch, on='epoch', how='left')
    else:
        epoch_metrics['mean_clipfrac'] = np.nan
        epoch_metrics['max_clipfrac'] = np.nan
    
    # Entropy and explained variance per epoch
    if not ppo_diags.empty:
        if 'entropy' in ppo_diags.columns:
            entropy_by_epoch = ppo_diags.groupby('epoch')['entropy'].agg(['median', 'mean']).reset_index()
            entropy_by_epoch.columns = ['epoch', 'entropy_median', 'entropy_mean']
            epoch_metrics = epoch_metrics.merge(entropy_by_epoch, on='epoch', how='left')
        else:
            epoch_metrics['entropy_median'] = np.nan
            epoch_metrics['entropy_mean'] = np.nan
        
        if 'expv' in ppo_diags.columns:
            expv_by_epoch = ppo_diags.groupby('epoch')['expv'].agg(['median', 'mean']).reset_index()
            expv_by_epoch.columns = ['epoch', 'expv_median', 'expv_mean']
            epoch_metrics = epoch_metrics.merge(expv_by_epoch, on='epoch', how='left')
        else:
            epoch_metrics['expv_median'] = np.nan
            epoch_metrics['expv_mean'] = np.nan
    
    # Length efficiency per epoch
    if 'auc' in epoch_metrics.columns and 'avg_len' in epoch_metrics.columns:
        valid_mask = ~(epoch_metrics['auc'].isna() | epoch_metrics['avg_len'].isna())
        epoch_metrics.loc[valid_mask, 'length_efficiency'] = (
            epoch_metrics.loc[valid_mask, 'auc'] / epoch_metrics.loc[valid_mask, 'avg_len']
        )
    else:
        epoch_metrics['length_efficiency'] = np.nan
    
    # EMA smoothing for key metrics
    if 'auc' in epoch_metrics.columns:
        epoch_metrics['auc_ema'] = ema(epoch_metrics['auc'], span=5)
    if 'reward' in epoch_metrics.columns:
        epoch_metrics['reward_ema'] = ema(epoch_metrics['reward'], span=5)
    
    return scalar_metrics, epoch_metrics
