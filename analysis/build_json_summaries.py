#!/usr/bin/env python3
"""
Parse results/*.txt logs and generate JSON summaries under visualization/<dataset>/

For each seed, creates:
- seed_<seed>_epoch_summary.json with per-epoch statistics
- final_epoch_summary.json aggregating last epoch across all seeds
"""

import argparse
import json
import os
import re
import hashlib
import statistics
from pathlib import Path
from typing import Optional, Dict, List, Any
from collections import defaultdict


def safe_float(x: str) -> Optional[float]:
    """Convert string to float, return None on failure."""
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def extract_seed(filepath: str) -> str:
    """
    Extract seed from filename using regex.
    If not found, return stable hash of filename.
    """
    filename = os.path.basename(filepath)
    
    # Try to find seed in filename
    patterns = [
        r'seed[_\-]?(\d+)',
        r'run[_\-]?seed[_\-]?(\d+)',
        r's(\d+)(?:_|\.)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Fallback: hash the filename
    hash_obj = hashlib.md5(filename.encode())
    return f"hash_{hash_obj.hexdigest()[:4]}"


def compute_stats(values: List[float]) -> tuple[float, float]:
    """
    Compute mean and standard deviation.
    Returns (mean, std). If single value, std=0.0.
    """
    if not values:
        return 0.0, 0.0
    
    mean_val = statistics.mean(values)
    
    if len(values) == 1:
        std_val = 0.0
    else:
        std_val = statistics.stdev(values)
    
    return mean_val, std_val


def parse_log_file(filepath: str, dataset: str) -> Dict[str, Any]:
    """
    Parse a single log file and extract per-epoch statistics.
    
    Returns a dictionary with metadata and epochs list.
    """
    seed = extract_seed(filepath)
    
    # Regex patterns (case-sensitive)
    epoch_header_re = re.compile(r'^\[Epoch\s+(\d+)\]')
    eval_auc_re = re.compile(r'^\[EVAL-Epoch\s+(\d+)\]\s+AUC=([0-9eE.\-+]+)')
    node_row_re = re.compile(
        r'^\s*n_id\s+\d+\s+len\s+\d+\s+p_loss=([0-9eE.\-+]+)\s+'
        r'size_loss=([0-9eE.\-+]+)\s+sim_loss=([0-9eE.\-+]+)\s+'
        r'r_p=([0-9eE.\-+]+)'
    )
    diagnostics_re = re.compile(
        r'^\[Diagnostics\]KL=([0-9eE.\-+]+),\s*ClipFrac=([0-9eE.\-+]+),\s*'
        r'Entropy=([0-9eE.\-+]+),\s*ExplainedVar=([0-9eE.\-+]+),\s*'
        r'PolicyLoss=([0-9eE.\-+]+),\s*ValueLoss=([0-9eE.\-+]+)'
    )
    
    # State tracking
    current_epoch = None
    epochs_data = defaultdict(lambda: {
        'epoch': None,
        'AUC': None,
        'node_metrics': {
            'p_loss': [],
            'size_loss': [],
            'sim_loss': [],
            'r_p': []
        },
        'diagnostics': {
            'KL': [],
            'ClipFrac': [],
            'Entropy': [],
            'ExplainedVar': [],
            'PolicyLoss': [],
            'ValueLoss': []
        }
    })
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.rstrip('\n')
            
            # Check for epoch header
            match = epoch_header_re.match(line)
            if match:
                current_epoch = int(match.group(1))
                epochs_data[current_epoch]['epoch'] = current_epoch
                continue
            
            # Check for AUC line
            match = eval_auc_re.match(line)
            if match:
                epoch_num = int(match.group(1))
                auc_val = safe_float(match.group(2))
                if auc_val is not None:
                    epochs_data[epoch_num]['AUC'] = auc_val
                    epochs_data[epoch_num]['epoch'] = epoch_num
                continue
            
            # Check for node metrics (only if we have a current epoch)
            if current_epoch is not None:
                match = node_row_re.match(line)
                if match:
                    p_loss = safe_float(match.group(1))
                    size_loss = safe_float(match.group(2))
                    sim_loss = safe_float(match.group(3))
                    r_p = safe_float(match.group(4))
                    
                    if p_loss is not None:
                        epochs_data[current_epoch]['node_metrics']['p_loss'].append(p_loss)
                    if size_loss is not None:
                        epochs_data[current_epoch]['node_metrics']['size_loss'].append(size_loss)
                    if sim_loss is not None:
                        epochs_data[current_epoch]['node_metrics']['sim_loss'].append(sim_loss)
                    if r_p is not None:
                        epochs_data[current_epoch]['node_metrics']['r_p'].append(r_p)
                    continue
                
                # Check for diagnostics
                match = diagnostics_re.match(line)
                if match:
                    kl = safe_float(match.group(1))
                    clip_frac = safe_float(match.group(2))
                    entropy = safe_float(match.group(3))
                    explained_var = safe_float(match.group(4))
                    policy_loss = safe_float(match.group(5))
                    value_loss = safe_float(match.group(6))
                    
                    if kl is not None:
                        epochs_data[current_epoch]['diagnostics']['KL'].append(kl)
                    if clip_frac is not None:
                        epochs_data[current_epoch]['diagnostics']['ClipFrac'].append(clip_frac)
                    if entropy is not None:
                        epochs_data[current_epoch]['diagnostics']['Entropy'].append(entropy)
                    if explained_var is not None:
                        epochs_data[current_epoch]['diagnostics']['ExplainedVar'].append(explained_var)
                    if policy_loss is not None:
                        epochs_data[current_epoch]['diagnostics']['PolicyLoss'].append(policy_loss)
                    if value_loss is not None:
                        epochs_data[current_epoch]['diagnostics']['ValueLoss'].append(value_loss)
                    continue
    
    # Convert to final format with statistics
    epochs_list = []
    for epoch_num in sorted(epochs_data.keys()):
        epoch_info = epochs_data[epoch_num]
        
        # Compute node metrics statistics
        node_stats = {}
        for metric in ['p_loss', 'size_loss', 'sim_loss', 'r_p']:
            values = epoch_info['node_metrics'][metric]
            if values:
                mean_val, std_val = compute_stats(values)
                node_stats[f'{metric}_mean'] = mean_val
                node_stats[f'{metric}_std'] = std_val
            else:
                node_stats[f'{metric}_mean'] = None
                node_stats[f'{metric}_std'] = None
        
        # Compute diagnostics statistics
        diag_stats = {}
        for metric in ['KL', 'ClipFrac', 'Entropy', 'ExplainedVar', 'PolicyLoss', 'ValueLoss']:
            values = epoch_info['diagnostics'][metric]
            if values:
                mean_val, std_val = compute_stats(values)
                diag_stats[f'{metric}_mean'] = mean_val
                diag_stats[f'{metric}_std'] = std_val
            else:
                diag_stats[f'{metric}_mean'] = None
                diag_stats[f'{metric}_std'] = None
        
        epochs_list.append({
            'epoch': epoch_num,
            'AUC': epoch_info['AUC'],
            'node_metrics': node_stats,
            'diagnostics': diag_stats
        })
    
    return {
        'metadata': {
            'dataset': dataset,
            'seed': seed,
            'source_file': filepath
        },
        'epochs': epochs_list
    }


def aggregate_final_epochs(seed_summaries: List[Dict[str, Any]], dataset: str) -> Dict[str, Any]:
    """
    Aggregate statistics from the last epoch of each seed.
    
    Returns a summary dictionary.
    """
    if not seed_summaries:
        return {
            'dataset': dataset,
            'summary': 'last_epoch_across_seeds',
            'seed_count': 0
        }
    
    last_epoch_data = []
    
    for summary in seed_summaries:
        epochs = summary['epochs']
        if not epochs:
            continue
        
        # Get the last epoch (highest epoch number)
        last_epoch = max(epochs, key=lambda e: e['epoch'])
        last_epoch_data.append(last_epoch)
    
    if not last_epoch_data:
        return {
            'dataset': dataset,
            'summary': 'last_epoch_across_seeds',
            'seed_count': 0
        }
    
    # Aggregate AUC
    auc_values = [e['AUC'] for e in last_epoch_data if e['AUC'] is not None]
    if auc_values:
        auc_mean, auc_std = compute_stats(auc_values)
    else:
        auc_mean, auc_std = None, None
    
    # Aggregate node metrics
    node_metrics = {}
    for metric in ['p_loss', 'size_loss', 'sim_loss', 'r_p']:
        mean_key = f'{metric}_mean'
        values = [e['node_metrics'][mean_key] for e in last_epoch_data 
                  if e['node_metrics'][mean_key] is not None]
        if values:
            mean_val, std_val = compute_stats(values)
            node_metrics[mean_key] = mean_val
            node_metrics[f'{metric}_std'] = std_val
        else:
            node_metrics[mean_key] = None
            node_metrics[f'{metric}_std'] = None
    
    # Aggregate diagnostics
    diagnostics = {}
    for metric in ['KL', 'ClipFrac', 'Entropy', 'ExplainedVar', 'PolicyLoss', 'ValueLoss']:
        mean_key = f'{metric}_mean'
        values = [e['diagnostics'][mean_key] for e in last_epoch_data 
                  if e['diagnostics'][mean_key] is not None]
        if values:
            mean_val, std_val = compute_stats(values)
            diagnostics[mean_key] = mean_val
            diagnostics[f'{metric}_std'] = std_val
        else:
            diagnostics[mean_key] = None
            diagnostics[f'{metric}_std'] = None
    
    return {
        'dataset': dataset,
        'summary': 'last_epoch_across_seeds',
        'seed_count': len(last_epoch_data),
        'AUC_mean': auc_mean,
        'AUC_std': auc_std,
        'node_metrics': node_metrics,
        'diagnostics': diagnostics
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Parse result logs and generate JSON summaries'
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
        help='Results directory (default: results)'
    )
    parser.add_argument(
        '--viz-dir',
        type=str,
        default='visualization',
        help='Visualization output directory (default: visualization)'
    )
    
    args = parser.parse_args()
    
    # Construct paths
    results_path = Path(args.results_dir)
    viz_path = Path(args.viz_dir) / args.dataset
    
    print(f"Processing dataset: {args.dataset}")
    print(f"Results directory: {results_path}")
    print(f"Output directory: {viz_path}")
    
    # Ensure output directory exists
    ensure_dir(str(viz_path))
    
    # Find all .txt files in results directory that start with dataset name
    if not results_path.exists():
        print(f"Warning: Results directory {results_path} does not exist.")
        return
    
    # Filter files by dataset prefix (e.g., syn3_*.txt for dataset syn3)
    txt_files = list(results_path.glob(f'{args.dataset}_*.txt'))
    
    if not txt_files:
        print(f"No .txt files found in {results_path}")
        return
    
    print(f"Found {len(txt_files)} log file(s)")
    
    # Parse each file and save per-seed JSON
    seed_summaries = []
    for txt_file in txt_files:
        print(f"  Parsing: {txt_file.name}")
        
        try:
            summary = parse_log_file(str(txt_file), args.dataset)
            seed_summaries.append(summary)
            
            # Save per-seed JSON
            seed = summary['metadata']['seed']
            output_file = viz_path / f"seed_{seed}_epoch_summary.json"
            
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"    → Saved: {output_file.name} ({len(summary['epochs'])} epochs)")
        
        except Exception as e:
            print(f"    ✗ Error parsing {txt_file.name}: {e}")
            continue
    
    # Generate final epoch aggregate
    if seed_summaries:
        print(f"\nAggregating final epochs across {len(seed_summaries)} seed(s)")
        final_summary = aggregate_final_epochs(seed_summaries, args.dataset)
        
        output_file = viz_path / "final_epoch_summary.json"
        with open(output_file, 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        print(f"  → Saved: {output_file.name}")
        if final_summary.get('AUC_mean') is not None:
            print(f"    AUC: {final_summary['AUC_mean']:.4f} ± {final_summary['AUC_std']:.4f}")
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()

