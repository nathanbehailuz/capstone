#!/usr/bin/env python3
"""
Extract PPO performance metrics from log files and save to JSON.

This script extracts detailed PPO performance information including:
- PPO Stats (Policy loss, Value loss, Entropy, KL divergence, etc.)
- PPO Debug metrics (MeanDeltaReward, StdDeltaReward, AvgSubgraphSize, StopRate)
- Reward summaries
- AUC scores
- Lambda schedules

Usage:
    python extract_ppo_performance.py <dataset_name> [seed] [--casual 0|1]

Examples:
    python extract_ppo_performance.py syn1                    # Extract all seeds for syn1
    python extract_ppo_performance.py syn1 0                  # Extract seed 0 for syn1
    python extract_ppo_performance.py syn1 --casual 1         # Extract with casual filename pattern
"""

import os
import re
import json
import argparse
from typing import Dict, Any, Optional, List


def extract_ppo_performance(file_path: str) -> Dict[str, Any]:
    """
    Extract PPO performance metrics from a log file.
    
    Args:
        file_path: Path to the log file
        
    Returns:
        Dictionary with PPO performance data organized by epoch
    """
    performance_data = {
        'epochs': {},
        'summary': {}
    }
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return performance_data
    
    current_epoch = None
    current_epoch_data = {}
    
    # Track overall metrics
    auc_scores = []
    
    for line in lines:
        line = line.strip()
        
        # Check for epoch start
        epoch_match = re.match(r'\[Epoch\s+(\d+)\]', line)
        if epoch_match:
            new_epoch = int(epoch_match.group(1))
            
            # If this is a new epoch (not the same epoch appearing again)
            if current_epoch != new_epoch:
                # Save previous epoch data if exists
                if current_epoch is not None and current_epoch_data:
                    performance_data['epochs'][f"epoch_{current_epoch}"] = current_epoch_data
                
                # Start new epoch
                current_epoch = new_epoch
                current_epoch_data = {
                    'epoch_num': current_epoch
                }
            # If same epoch number, continue adding to current_epoch_data
            continue
        
        # Process PPO performance lines
        if current_epoch is not None:
            # Parse λ Schedule line
            if 'λ Schedule:' in line:
                schedule_match = re.search(
                    r'size=([\d.-]+),\s*radius=([\d.-]+),\s*sim=([\d.-]+),\s*entropy=([\d.-]+)', 
                    line
                )
                if schedule_match:
                    current_epoch_data['lambda_schedule'] = {
                        'size': float(schedule_match.group(1)),
                        'radius': float(schedule_match.group(2)),
                        'sim': float(schedule_match.group(3)),
                        'entropy': float(schedule_match.group(4))
                    }
            
            # Parse PPO Stats line (most detailed technical metrics)
            elif '[PPO Stats]' in line:
                stats_match = re.search(
                    r'Policy=([\d.-]+),\s*Value=([\d.-]+),\s*Entropy=([\d.-]+),\s*'
                    r'KL=([\d.-]+),\s*ClipFrac=([\d.-]+),\s*EV=([\d.-]+),\s*'
                    r'GradNorm=([\d.-]+),\s*Updates=(\d+),\s*EpsPolicy=([\d.-]+)',
                    line
                )
                if stats_match:
                    current_epoch_data['ppo_stats'] = {
                        'policy_loss': float(stats_match.group(1)),
                        'value_loss': float(stats_match.group(2)),
                        'entropy': float(stats_match.group(3)),
                        'kl_divergence': float(stats_match.group(4)),
                        'clip_fraction': float(stats_match.group(5)),
                        'explained_variance': float(stats_match.group(6)),
                        'grad_norm': float(stats_match.group(7)),
                        'updates': int(stats_match.group(8)),
                        'eps_policy': float(stats_match.group(9))
                    }
            
            # Parse PPO Debug line (behavioral metrics)
            elif '[PPO Debug]' in line:
                debug_match = re.search(
                    r'MeanDeltaReward=([\d.-]+),\s*StdDeltaReward=([\d.-]+),\s*'
                    r'AvgSubgraphSize=([\d.-]+),\s*StopRate=([\d.-]+)',
                    line
                )
                if debug_match:
                    current_epoch_data['ppo_debug'] = {
                        'mean_delta_reward': float(debug_match.group(1)),
                        'std_delta_reward': float(debug_match.group(2)),
                        'avg_subgraph_size': float(debug_match.group(3)),
                        'stop_rate': float(debug_match.group(4))
                    }
            
            # Parse Reward summary line
            elif line.startswith('Reward='):
                reward_match = re.search(
                    r'Reward=([\d.-]+)\s+PLoss=\s*([\d.-]+)\s+Length=([\d.-]+)\s+'
                    r'EarlyStopRate=([\d.-]+)\s+AvgSize=([\d.-]+)',
                    line
                )
                if reward_match:
                    current_epoch_data['reward_summary'] = {
                        'reward': float(reward_match.group(1)),
                        'policy_loss': float(reward_match.group(2)),
                        'length': float(reward_match.group(3)),
                        'early_stop_rate': float(reward_match.group(4)),
                        'avg_size': float(reward_match.group(5))
                    }
            
            # Parse average losses line
            elif 'avg_p_loss=' in line:
                losses_match = re.search(
                    r'avg_p_loss=([\d.-]+)\s+avg_size_loss=([\d.-]+)\s+'
                    r'avg_sim_loss=([\d.-]+)\s+r_p=([\d.-]+)',
                    line
                )
                if losses_match:
                    current_epoch_data['average_losses'] = {
                        'avg_p_loss': float(losses_match.group(1)),
                        'avg_size_loss': float(losses_match.group(2)),
                        'avg_sim_loss': float(losses_match.group(3)),
                        'r_p': float(losses_match.group(4))
                    }
            
            # Parse AUC line (model performance evaluation)
            elif '[EVAL-Epoch' in line and 'AUC=' in line:
                auc_match = re.search(r'AUC=([\d.-]+)', line)
                if auc_match:
                    auc_score = float(auc_match.group(1))
                    current_epoch_data['auc'] = auc_score
                    auc_scores.append({
                        'epoch': current_epoch,
                        'auc': auc_score
                    })
    
    # Save the last epoch if it exists
    if current_epoch is not None and current_epoch_data:
        performance_data['epochs'][f"epoch_{current_epoch}"] = current_epoch_data
    
    # Add summary statistics
    if auc_scores:
        performance_data['summary']['auc_progression'] = auc_scores
        performance_data['summary']['final_auc'] = auc_scores[-1]['auc'] if auc_scores else None
        performance_data['summary']['initial_auc'] = auc_scores[0]['auc'] if auc_scores else None
        performance_data['summary']['auc_improvement'] = (
            auc_scores[-1]['auc'] - auc_scores[0]['auc'] if len(auc_scores) > 1 else 0
        )
    
    performance_data['summary']['total_epochs'] = len(performance_data['epochs'])
    
    return performance_data


def extract_multiple_seeds(results_dir: str, dataset_name: str, seed: Optional[int] = None, casual: int = 0) -> Dict[str, Any]:
    """
    Extract PPO performance data for one or all seeds.
    
    Args:
        results_dir: Path to results directory
        dataset_name: Name of the dataset
        seed: Specific seed to extract, or None for all seeds
        casual: If 1, use 'casual' in filename pattern (default: 0)
        
    Returns:
        Dictionary with performance data for each seed
    """
    all_data = {
        'dataset': dataset_name,
        'seeds': {}
    }
    
    # Determine the pattern based on casual flag
    if casual == 1:
        base_pattern = f"{dataset_name}_ppo_casual_seed_"
    else:
        base_pattern = f"{dataset_name}_ppo_seed_"
    
    # Find all files matching the dataset pattern
    files = []
    for filename in os.listdir(results_dir):
        if filename.startswith(base_pattern) and filename.endswith('.txt'):
            if seed is not None:
                if filename == f"{base_pattern}{seed}.txt":
                    files.append(filename)
            else:
                files.append(filename)
    
    if not files:
        print(f"No files found matching pattern: {base_pattern}*.txt")
        return all_data
    
    # Extract data from each file
    for filename in sorted(files):
        # Extract seed number from filename
        seed_match = re.search(r'_seed_(\d+)\.txt', filename)
        if seed_match:
            seed_num = int(seed_match.group(1))
            file_path = os.path.join(results_dir, filename)
            print(f"Processing {filename}...")
            
            performance_data = extract_ppo_performance(file_path)
            all_data['seeds'][f"seed_{seed_num}"] = performance_data
    
    # Add cross-seed summary if multiple seeds
    if len(all_data['seeds']) > 1:
        final_aucs = []
        for seed_key, seed_data in all_data['seeds'].items():
            if 'summary' in seed_data and 'final_auc' in seed_data['summary'] and seed_data['summary']['final_auc'] is not None:
                final_aucs.append(seed_data['summary']['final_auc'])
        
        if final_aucs:
            all_data['cross_seed_summary'] = {
                'num_seeds': len(final_aucs),
                'mean_final_auc': sum(final_aucs) / len(final_aucs),
                'min_final_auc': min(final_aucs),
                'max_final_auc': max(final_aucs),
                'std_final_auc': (sum((x - sum(final_aucs)/len(final_aucs))**2 for x in final_aucs) / len(final_aucs))**0.5
            }
    
    return all_data


def save_json(data: Dict[str, Any], output_path: str):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        output_path: Path to output JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Data saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract PPO performance metrics from log files and save as JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_ppo_performance.py syn1                 # Extract all seeds for syn1
  python extract_ppo_performance.py syn1 0               # Extract seed 0 for syn1
  python extract_ppo_performance.py syn1 --casual 1      # Extract with casual pattern
  python extract_ppo_performance.py mutag 2              # Extract seed 2 for mutag
        """
    )
    parser.add_argument('dataset_name', help='Name of the dataset (e.g., syn1, syn4, mutag)')
    parser.add_argument('seed', nargs='?', type=int, help='Specific seed number (optional, extracts all if not specified)')
    parser.add_argument('--casual', type=int, default=0, choices=[0, 1], 
                       help='If 1, use "casual" in filename pattern (default: 0)')
    
    args = parser.parse_args()
    
    # Set up paths
    results_dir = "/scratch/nz2212/capstone/results"
    output_dir = "/scratch/nz2212/capstone/analysis/jsons"
    
    # Determine output filename
    seed_str = str(args.seed) if args.seed is not None else "all"
    pattern_str = "ppo_casual" if args.casual == 1 else "ppo"
    output_filename = f"{args.dataset_name}_{pattern_str}_performance_seed_{seed_str}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    print("=" * 60)
    print(f"Extracting PPO Performance Metrics")
    print("=" * 60)
    print(f"Dataset:      {args.dataset_name}")
    print(f"Seed:         {seed_str}")
    print(f"Casual mode:  {args.casual}")
    print(f"Output file:  {output_filename}")
    print("=" * 60)
    
    # Extract data
    data = extract_multiple_seeds(results_dir, args.dataset_name, args.seed, args.casual)
    
    # Save to JSON
    save_json(data, output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    print(f"Seeds processed: {len(data['seeds'])}")
    for seed_key, seed_data in data['seeds'].items():
        epochs = seed_data['summary'].get('total_epochs', 0)
        final_auc = seed_data['summary'].get('final_auc')
        if final_auc is not None:
            print(f"  {seed_key}: {epochs} epochs, final AUC = {final_auc:.4f}")
        else:
            print(f"  {seed_key}: {epochs} epochs")
    
    if 'cross_seed_summary' in data:
        print(f"\nCross-seed statistics:")
        print(f"  Mean final AUC: {data['cross_seed_summary']['mean_final_auc']:.4f}")
        print(f"  Std final AUC:  {data['cross_seed_summary']['std_final_auc']:.4f}")
        print(f"  Min final AUC:  {data['cross_seed_summary']['min_final_auc']:.4f}")
        print(f"  Max final AUC:  {data['cross_seed_summary']['max_final_auc']:.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

