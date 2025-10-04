"""
Log parser for PPO + Causal GNN Explainer training logs.
Extracts structured metrics from training logs and returns pandas DataFrames.
"""

import regex
import pandas as pd
from typing import Dict, List, Optional, Tuple
import numpy as np


class LogParser:
    """Parser for PPO + Causal GNN Explainer training logs."""
    
    def __init__(self):
        # Compile regex patterns once for efficiency
        self.re_init = regex.compile(
            r'Loading (?P<dataset>\w+) dataset.*?'
            r'Train Acc:\s*(?P<train_acc>\d\.\d+), Val Acc:\s*(?P<val_acc>\d\.\d+), Test Acc:\s*(?P<test_acc>\d\.\d+).*?'
            r'total nodes\s*(?P<tot_nodes>\d+).*?total edges\s*(?P<tot_edges>\d+).*?'
            r'Prediction Error for Graphs:\s*(?P<pred_err>[-\d\.eE+]+).*?'
            r'\[CAUSAL\] Initialized .*? with (?P<n_classes>\d+) classes, weight (?P<causal_w>[-\d\.eE+]+)',
            regex.S
        )
        
        self.re_pretrained_row = regex.compile(
            r'g_id\s+(?P<g_id>\d+)\s+n_id\s+(?P<n_id>\d+)\s+len\s+(?P<length>\d+)\s+'
            r'p_loss=(?P<p_loss>[-\d\.eE+]+)\s+size_loss=(?P<size_loss>[-\d\.eE+]+)\s+'
            r'sim_loss=(?P<sim_loss>[-\d\.eE+]+)\s+r_p=(?P<radius_pen>[-\d\.eE+]+)'
        )
        
        self.re_pretrained_sum = regex.compile(
            r'avg_p_loss=(?P<avg_p>[-\d\.eE+]+)\s+avg_size_loss=(?P<avg_size>[-\d\.eE+]+)\s+'
            r'avg_sim_loss=(?P<avg_sim>[-\d\.eE+]+)\s+r_p=(?P<avg_rp>[-\d\.eE+]+)\s*\n\[EVAL-Pretrained\]\s*AUC=(?P<auc>[-\d\.eE+]+)'
        )
        
        self.re_epoch_header = regex.compile(r'^\[Epoch\s+(?P<epoch>\d+)\]\s*$', regex.M)
        self.re_entropy_coef = regex.compile(r'Entropy coefficient:\s*(?P<ent_coef>[-\d\.eE+]+)')
        self.re_diag = regex.compile(
            r'\[Diagnostics\]KL=(?P<kl>[-\d\.eE+]+),\s*ClipFrac=(?P<clip>[-\d\.eE+]+),\s*'
            r'Entropy=(?P<entropy>[-\d\.eE+]+),\s*ExplainedVar=(?P<expv>[-\d\.eE+]+),\s*'
            r'PolicyLoss=(?P<ploss>[-\d\.eE+]+),\s*ValueLoss=(?P<vloss>[-\d\.eE+]+)'
        )
        self.re_early_stop = regex.compile(r'Early stopping:\s*KL divergence\s*(?P<kl>[-\d\.eE+]+)\s*>\s*target\s*(?P<kl_tgt>[-\d\.eE+]+)')
        self.re_train_sum = regex.compile(
            r'Reward=(?P<reward>[-\d\.eE+]+)\s+PLoss=(?P<ploss>[-\d\.eE+]+)\s+Length=(?P<length>[-\d\.eE+]+)\s*\n'
            r'Elapsed Time:\s*(?P<secs>[-\d\.eE+]+)s'
        )
        self.re_eval_sum = regex.compile(
            r'avg_p_loss=(?P<avg_p>[-\d\.eE+]+)\s+avg_size_loss=(?P<avg_size>[-\d\.eE+]+)\s+'
            r'avg_sim_loss=(?P<avg_sim>[-\d\.eE+]+)\s+r_p=(?P<avg_rp>[-\d\.eE+]+)'
        )
        self.re_eval_auc = regex.compile(
            r'\[EVAL-(?:Pretrained|Epoch\s+\d+)\]\s*AUC=(?P<auc>[-\d\.eE+]+)'
        )
        
        # Additional patterns for metadata
        self.re_seed = regex.compile(r'Seed:\s*(?P<seed>\d+)')
        self.re_job_id = regex.compile(r'Job ID:\s*(?P<job_id>\d+)')
        self.re_device = regex.compile(r'CUDA_VISIBLE_DEVICES:\s*(?P<device>[-\d]+)')
        self.re_code_hash = regex.compile(r'CODE:\s*(?P<code_hash>[A-F0-9]+)')
    
    def parse_log(self, text: str) -> Dict[str, pd.DataFrame]:
        """Parse log text and return structured DataFrames."""
        lines = text.split('\n')
        
        # Initialize results
        results = {
            'run_meta': self._parse_init_metadata(text),
            'pretrained_rows': self._parse_pretrained_rows(lines),
            'ppo_diags': self._parse_ppo_diagnostics(lines),
            'epoch_summaries': self._parse_epoch_summaries(lines)
        }
        
        return results
    
    def _parse_init_metadata(self, text: str) -> pd.DataFrame:
        """Parse initialization metadata."""
        match = self.re_init.search(text)
        if not match:
            # Return empty DataFrame with correct schema
            return pd.DataFrame({
                'dataset': [np.nan], 'train_acc': [np.nan], 'val_acc': [np.nan], 
                'test_acc': [np.nan], 'tot_nodes': [np.nan], 'tot_edges': [np.nan],
                'pred_err': [np.nan], 'n_classes': [np.nan], 'causal_w': [np.nan],
                'seed': [np.nan], 'job_id': [np.nan], 'device': [np.nan], 'code_hash': [np.nan]
            })
        
        data = match.groupdict()
        
        # Extract additional metadata
        seed_match = self.re_seed.search(text)
        job_id_match = self.re_job_id.search(text)
        device_match = self.re_device.search(text)
        code_hash_match = self.re_code_hash.search(text)
        
        data.update({
            'seed': float(seed_match.group('seed')) if seed_match else np.nan,
            'job_id': float(job_id_match.group('job_id')) if job_id_match else np.nan,
            'device': device_match.group('device') if device_match else np.nan,
            'code_hash': code_hash_match.group('code_hash') if code_hash_match else np.nan
        })
        
        # Convert numeric fields
        for key in ['train_acc', 'val_acc', 'test_acc', 'tot_nodes', 'tot_edges', 
                   'pred_err', 'n_classes', 'causal_w']:
            if key in data and data[key] is not None:
                data[key] = float(data[key])
            else:
                data[key] = np.nan
        
        return pd.DataFrame([data])
    
    def _parse_pretrained_rows(self, lines: List[str]) -> pd.DataFrame:
        """Parse pretrained evaluation rows."""
        rows = []
        for line in lines:
            match = self.re_pretrained_row.search(line)
            if match:
                data = match.groupdict()
                # Convert to appropriate types
                for key in ['g_id', 'n_id', 'length']:
                    data[key] = int(data[key])
                for key in ['p_loss', 'size_loss', 'sim_loss', 'radius_pen']:
                    data[key] = float(data[key])
                rows.append(data)
        
        if not rows:
            return pd.DataFrame(columns=['g_id', 'n_id', 'length', 'p_loss', 'size_loss', 'sim_loss', 'radius_pen'])
        
        return pd.DataFrame(rows)
    
    def _parse_ppo_diagnostics(self, lines: List[str]) -> pd.DataFrame:
        """Parse PPO diagnostics from training."""
        rows = []
        current_epoch = None
        step_idx = 0
        kl_target = None
        
        for line in lines:
            # Check for epoch header
            epoch_match = self.re_epoch_header.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group('epoch'))
                step_idx = 0
                continue
            
            # Check for early stopping (to capture KL target)
            early_stop_match = self.re_early_stop.search(line)
            if early_stop_match:
                kl_target = float(early_stop_match.group('kl_tgt'))
                continue
            
            # Check for diagnostics
            diag_match = self.re_diag.search(line)
            if diag_match and current_epoch is not None:
                data = diag_match.groupdict()
                data['epoch'] = current_epoch
                data['step_idx'] = step_idx
                data['early_stop'] = False
                data['kl_target'] = kl_target if kl_target is not None else np.nan
                
                # Convert to float
                for key in ['kl', 'clip', 'entropy', 'expv', 'ploss', 'vloss']:
                    data[key] = float(data[key])
                
                rows.append(data)
                step_idx += 1
        
        if not rows:
            return pd.DataFrame(columns=['epoch', 'step_idx', 'kl', 'clip', 'entropy', 'expv', 
                                       'policy_loss', 'value_loss', 'early_stop', 'kl_target'])
        
        return pd.DataFrame(rows)
    
    def _parse_epoch_summaries(self, lines: List[str]) -> pd.DataFrame:
        """Parse epoch summaries."""
        rows = []
        current_epoch = None
        entropy_coeff = None
        train_data = {}
        eval_data = {}
        early_stop_cnt = 0
        
        for i, line in enumerate(lines):
            # Check for epoch header
            epoch_match = self.re_epoch_header.search(line)
            if epoch_match:
                # Save previous epoch if exists
                if current_epoch is not None:
                    row = {
                        'epoch': current_epoch,
                        'entropy_coeff': entropy_coeff,
                        'reward': train_data.get('reward', np.nan),
                        'policy_loss': train_data.get('ploss', np.nan),
                        'avg_len': train_data.get('length', np.nan),
                        'time_sec': train_data.get('secs', np.nan),
                        'early_stop_cnt': early_stop_cnt,
                        'eval_avg_p': eval_data.get('avg_p', np.nan),
                        'eval_avg_size': eval_data.get('avg_size', np.nan),
                        'eval_avg_sim': eval_data.get('avg_sim', np.nan),
                        'eval_avg_rp': eval_data.get('avg_rp', np.nan),
                        'auc': eval_data.get('auc', np.nan)
                    }
                    rows.append(row)
                
                # Reset for new epoch
                current_epoch = int(epoch_match.group('epoch'))
                entropy_coeff = None
                train_data = {}
                eval_data = {}
                early_stop_cnt = 0
                continue
            
            # Check for entropy coefficient
            entropy_match = self.re_entropy_coef.search(line)
            if entropy_match:
                entropy_coeff = float(entropy_match.group('ent_coef'))
                continue
            
            # Check for early stopping
            if 'Early stopping:' in line:
                early_stop_cnt += 1
                continue
            
            # Check for training summary
            train_match = self.re_train_sum.search(line + '\n' + (lines[i+1] if i+1 < len(lines) else ''))
            if train_match:
                train_data = {k: float(v) for k, v in train_match.groupdict().items()}
                continue
            
            # Check for evaluation summary (avg_p_loss line)
            eval_match = self.re_eval_sum.search(line)
            if eval_match:
                eval_data = {k: float(v) for k, v in eval_match.groupdict().items()}
                continue
            
            # Check for AUC line
            auc_match = self.re_eval_auc.search(line)
            if auc_match:
                eval_data['auc'] = float(auc_match.group('auc'))
                continue
        
        # Don't forget the last epoch
        if current_epoch is not None:
            row = {
                'epoch': current_epoch,
                'entropy_coeff': entropy_coeff,
                'reward': train_data.get('reward', np.nan),
                'policy_loss': train_data.get('ploss', np.nan),
                'avg_len': train_data.get('length', np.nan),
                'time_sec': train_data.get('secs', np.nan),
                'early_stop_cnt': early_stop_cnt,
                'eval_avg_p': eval_data.get('avg_p', np.nan),
                'eval_avg_size': eval_data.get('avg_size', np.nan),
                'eval_avg_sim': eval_data.get('avg_sim', np.nan),
                'eval_avg_rp': eval_data.get('avg_rp', np.nan),
                'auc': eval_data.get('auc', np.nan)
            }
            rows.append(row)
        
        if not rows:
            return pd.DataFrame(columns=['epoch', 'entropy_coeff', 'reward', 'policy_loss', 'avg_len', 
                                       'time_sec', 'early_stop_cnt', 'eval_avg_p', 'eval_avg_size', 
                                       'eval_avg_sim', 'eval_avg_rp', 'auc'])
        
        return pd.DataFrame(rows)


def parse_log(text: str) -> Dict[str, pd.DataFrame]:
    """Parse log text and return structured DataFrames."""
    parser = LogParser()
    return parser.parse_log(text)
