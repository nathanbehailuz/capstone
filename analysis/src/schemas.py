"""
Data schemas and type definitions for the log analysis toolkit.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import pandas as pd


@dataclass
class RunMetadata:
    """Schema for run metadata."""
    dataset: str
    train_acc: float
    val_acc: float
    test_acc: float
    tot_nodes: int
    tot_edges: int
    pred_err: float
    n_classes: int
    causal_w: float
    seed: Optional[int] = None
    job_id: Optional[int] = None
    device: Optional[str] = None
    code_hash: Optional[str] = None


@dataclass
class PretrainedRow:
    """Schema for pretrained evaluation rows."""
    g_id: int
    n_id: int
    length: int
    p_loss: float
    size_loss: float
    sim_loss: float
    radius_pen: float


@dataclass
class PPODiagnostic:
    """Schema for PPO diagnostics."""
    epoch: int
    step_idx: int
    kl: float
    clip: float
    entropy: float
    expv: float
    policy_loss: float
    value_loss: float
    early_stop: bool
    kl_target: float


@dataclass
class EpochSummary:
    """Schema for epoch summaries."""
    epoch: int
    entropy_coeff: float
    reward: float
    policy_loss: float
    avg_len: float
    time_sec: float
    early_stop_cnt: int
    eval_avg_p: float
    eval_avg_size: float
    eval_avg_sim: float
    eval_avg_rp: float
    auc: float


# DataFrame column schemas
RUN_META_COLUMNS = [
    'dataset', 'train_acc', 'val_acc', 'test_acc', 'tot_nodes', 'tot_edges',
    'pred_err', 'n_classes', 'causal_w', 'seed', 'job_id', 'device', 'code_hash'
]

PRETRAINED_ROWS_COLUMNS = [
    'g_id', 'n_id', 'length', 'p_loss', 'size_loss', 'sim_loss', 'radius_pen'
]

PPO_DIAGS_COLUMNS = [
    'epoch', 'step_idx', 'kl', 'clip', 'entropy', 'expv', 
    'policy_loss', 'value_loss', 'early_stop', 'kl_target'
]

EPOCH_SUMMARIES_COLUMNS = [
    'epoch', 'entropy_coeff', 'reward', 'policy_loss', 'avg_len', 'time_sec',
    'early_stop_cnt', 'eval_avg_p', 'eval_avg_size', 'eval_avg_sim', 
    'eval_avg_rp', 'auc'
]

# Expected data types
RUN_META_DTYPES = {
    'dataset': 'object',
    'train_acc': 'float64',
    'val_acc': 'float64', 
    'test_acc': 'float64',
    'tot_nodes': 'int64',
    'tot_edges': 'int64',
    'pred_err': 'float64',
    'n_classes': 'int64',
    'causal_w': 'float64',
    'seed': 'float64',
    'job_id': 'float64',
    'device': 'object',
    'code_hash': 'object'
}

PRETRAINED_ROWS_DTYPES = {
    'g_id': 'int64',
    'n_id': 'int64',
    'length': 'int64',
    'p_loss': 'float64',
    'size_loss': 'float64',
    'sim_loss': 'float64',
    'radius_pen': 'float64'
}

PPO_DIAGS_DTYPES = {
    'epoch': 'int64',
    'step_idx': 'int64',
    'kl': 'float64',
    'clip': 'float64',
    'entropy': 'float64',
    'expv': 'float64',
    'policy_loss': 'float64',
    'value_loss': 'float64',
    'early_stop': 'bool',
    'kl_target': 'float64'
}

EPOCH_SUMMARIES_DTYPES = {
    'epoch': 'int64',
    'entropy_coeff': 'float64',
    'reward': 'float64',
    'policy_loss': 'float64',
    'avg_len': 'float64',
    'time_sec': 'float64',
    'early_stop_cnt': 'int64',
    'eval_avg_p': 'float64',
    'eval_avg_size': 'float64',
    'eval_avg_sim': 'float64',
    'eval_avg_rp': 'float64',
    'auc': 'float64'
}
