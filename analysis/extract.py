#!/usr/bin/env python3
"""
This script is used to extract the results from the log files and save them in a json file.

Usage:
    python extract.py <dataset_name> [seed] [epoch] [--casual 0|1]

Examples:
    python extract.py syn1                    # Extract all seeds and epochs for syn1
    python extract.py syn1 0                  # Extract all epochs for syn1 seed 0
    python extract.py syn1 0 5                # Extract epoch 5 for syn1 seed 0
    python extract.py syn1 --casual 1         # Extract with casual filename pattern
    python extract.py syn1 0 --casual 1       # Extract seed 0 with casual pattern
"""

import os
import re
import json
import argparse
import ast
from typing import Dict, Any, Optional, List


def extract_hyperparameters(file_path: str) -> Dict[str, Any]:
    """
    Extract hyperparameters from the Namespace line in a log file.
    
    Args:
        file_path: Path to the log file
        
    Returns:
        Dictionary with hyperparameters
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    
    for line in lines:
        line = line.strip()
        # Look for Namespace line
        if line.startswith('Namespace(') and line.endswith(')'):
            try:
                # Parse the Namespace arguments
                # Remove 'Namespace(' and ')' and parse as a dictionary
                namespace_content = line[10:-1]  # Remove 'Namespace(' and ')'
                
                # Parse the arguments into a dictionary
                hparams = {}
                # Split by comma but be careful with nested structures
                # This is a simple approach - might need refinement for complex cases
                parts = []
                current_part = ""
                paren_count = 0
                
                for char in namespace_content:
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                    elif char == ',' and paren_count == 0:
                        parts.append(current_part.strip())
                        current_part = ""
                        continue
                    current_part += char
                
                if current_part.strip():
                    parts.append(current_part.strip())
                
                # Parse each key=value pair
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Try to parse the value
                        try:
                            # Try to evaluate as a Python literal
                            parsed_value = ast.literal_eval(value)
                            hparams[key] = parsed_value
                        except (ValueError, SyntaxError):
                            # If parsing fails, keep as string
                            hparams[key] = value
                
                return hparams
            except Exception as e:
                print(f"Error parsing hyperparameters from {file_path}: {e}")
                return {}
    
    return {}


def extract_per_epoch(file_path: str, epoch_num: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """
    Extract data for specific epoch(s) from a log file.
    
    Args:
        file_path: Path to the log file
        epoch_num: Specific epoch number to extract, or None for all epochs
        
    Returns:
        Dictionary with epoch data
    """
    epoch_data = {}
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return epoch_data
    
    current_epoch = None
    current_epoch_data = {}
    
    for line in lines:
        line = line.strip()
        
        # Check for epoch start
        epoch_match = re.match(r'\[Epoch\s+(\d+)\]', line)
        if epoch_match:
            # Save previous epoch data if exists
            if current_epoch is not None:
                epoch_data[f"epoch_{current_epoch}"] = {"data": current_epoch_data}
            
            # Start new epoch
            current_epoch = int(epoch_match.group(1))
            current_epoch_data = {}
            
            # If specific epoch requested and we found it, continue processing
            if epoch_num is not None and current_epoch != epoch_num:
                current_epoch = None
                continue
                
        # Process data lines only if we're in an epoch we want
        if current_epoch is not None:
            # Parse λ Schedule line
            if 'λ Schedule:' in line:
                schedule_match = re.search(r'size=([\d.-]+),\s*radius=([\d.-]+),\s*sim=([\d.-]+),\s*entropy=([\d.-]+)', line)
                if schedule_match:
                    current_epoch_data['size'] = float(schedule_match.group(1))
                    current_epoch_data['radius'] = float(schedule_match.group(2))
                    current_epoch_data['sim'] = float(schedule_match.group(3))
                    current_epoch_data['entropy'] = float(schedule_match.group(4))
            
            # Parse PPO Stats line
            elif '[PPO Stats]' in line:
                stats_match = re.search(r'Policy=([\d.-]+),\s*Value=([\d.-]+),\s*Entropy=([\d.-]+),\s*KL=([\d.-]+),\s*ClipFrac=([\d.-]+),\s*EV=([\d.-]+),\s*GradNorm=([\d.-]+),\s*Updates=(\d+),\s*EpsPolicy=([\d.-]+)', line)
                if stats_match:
                    current_epoch_data['Policy'] = float(stats_match.group(1))
                    current_epoch_data['Value'] = float(stats_match.group(2))
                    current_epoch_data['Entropy'] = float(stats_match.group(3))
                    current_epoch_data['KL'] = float(stats_match.group(4))
                    current_epoch_data['ClipFrac'] = float(stats_match.group(5))
                    current_epoch_data['EV'] = float(stats_match.group(6))
                    current_epoch_data['GradNorm'] = float(stats_match.group(7))
                    current_epoch_data['Updates'] = int(stats_match.group(8))
                    current_epoch_data['EpsPolicy'] = float(stats_match.group(9))
            
            # Parse PPO Debug line
            elif '[PPO Debug]' in line:
                debug_match = re.search(r'MeanDeltaReward=([\d.-]+),\s*StdDeltaReward=([\d.-]+),\s*AvgSubgraph\s+Size=([\d.-]+),\s*StopRate=([\d.-]+)', line)
                if debug_match:
                    current_epoch_data['MeanDeltaReward'] = float(debug_match.group(1))
                    current_epoch_data['StdDeltaReward'] = float(debug_match.group(2))
                    current_epoch_data['AvgSubgraphSize'] = float(debug_match.group(3))
                    current_epoch_data['StopRate'] = float(debug_match.group(4))
            
            # Parse Reward/PLoss/Length line
            elif line.startswith('Reward='):
                reward_match = re.search(r'Reward=([\d.-]+)\s+PLoss=\s*([\d.-]+)\s+Length=([\d.-]+)\s+EarlyStopRate=([\d.-]+)\s+AvgSize=([\d.-]+)', line)
                if reward_match:
                    current_epoch_data['Reward'] = float(reward_match.group(1))
                    current_epoch_data['PLoss'] = float(reward_match.group(2))
                    current_epoch_data['Length'] = float(reward_match.group(3))
                    current_epoch_data['EarlyStopRate'] = float(reward_match.group(4))
                    current_epoch_data['AvgSize'] = float(reward_match.group(5))
            
            # Parse avg losses line
            elif 'avg_p_loss=' in line:
                losses_match = re.search(r'avg_p_loss=([\d.-]+)\s+avg_size_loss=([\d.-]+)\s+avg_sim_loss=([\d.-]+)\s+r_p=([\d.-]+)', line)
                if losses_match:
                    current_epoch_data['avg_p_loss'] = float(losses_match.group(1))
                    current_epoch_data['avg_size_loss'] = float(losses_match.group(2))
                    current_epoch_data['avg_sim_loss'] = float(losses_match.group(3))
                    current_epoch_data['r_p'] = float(losses_match.group(4))
            
            # Parse AUC line
            elif '[EVAL-Epoch' in line and 'AUC=' in line:
                auc_match = re.search(r'AUC=([\d.-]+)', line)
                if auc_match:
                    current_epoch_data['AUC'] = float(auc_match.group(1))
    
    # Save the last epoch if it exists
    if current_epoch is not None:
        epoch_data[f"epoch_{current_epoch}"] = {"data": current_epoch_data}
    
    return epoch_data


def extract_per_seed(results_dir: str, dataset_name: str, seed: Optional[int] = None, casual: int = 0) -> Dict[str, Any]:
    """
    Extract data for specific seed(s) from results directory.
    
    Args:
        results_dir: Path to results directory
        dataset_name: Name of the dataset
        seed: Specific seed to extract, or None for all seeds
        casual: If 1, use 'casual' in filename pattern (default: 0)
        
    Returns:
        Dictionary with seed data and hyperparameters
    """
    seed_data = {}
    hyperparameters = {}
    
    # Determine the pattern based on casual flag
    if casual == 1:
        base_pattern = f"{dataset_name}_ppo_casual_seed_"
    else:
        base_pattern = f"{dataset_name}_ppo_seed_"
    
    # Find all files matching the dataset pattern
    pattern = base_pattern
    if seed is not None:
        pattern += f"{seed}.txt"
    else:
        pattern += "*.txt"
    
    files = []
    for filename in os.listdir(results_dir):
        if filename.startswith(base_pattern) and filename.endswith('.txt'):
            if seed is not None:
                if filename == f"{base_pattern}{seed}.txt":
                    files.append(filename)
            else:
                files.append(filename)
    
    for filename in files:
        # Extract seed number from filename
        seed_match = re.search(r'_seed_(\d+)\.txt', filename)
        if seed_match:
            seed_num = int(seed_match.group(1))
            file_path = os.path.join(results_dir, filename)
            epoch_data = extract_per_epoch(file_path)
            
            # Extract hyperparameters from the first file (they should be the same for all seeds)
            if not hyperparameters:
                hyperparameters = extract_hyperparameters(file_path)
            
            if epoch_data:
                seed_data[f"seed_{seed_num}"] = epoch_data
    
    return {
        "hparams": hyperparameters,
        "seeds": seed_data
    }


def extract_all(results_dir: str, dataset_name: str, casual: int = 0) -> Dict[str, Any]:
    """
    Extract all data for a dataset.
    
    Args:
        results_dir: Path to results directory
        dataset_name: Name of the dataset
        casual: If 1, use 'casual' in filename pattern (default: 0)
        
    Returns:
        Dictionary with all data including hyperparameters
    """
    return {dataset_name: extract_per_seed(results_dir, dataset_name, casual=casual)}


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


def main():
    parser = argparse.ArgumentParser(description='Extract results from log files and save as JSON')
    parser.add_argument('dataset_name', help='Name of the dataset')
    parser.add_argument('seed', nargs='?', type=int, help='Specific seed number (optional)')
    parser.add_argument('epoch', nargs='?', type=int, help='Specific epoch number (optional)')
    parser.add_argument('--casual', type=int, default=0, choices=[0, 1], help='If 1, use "casual" in filename pattern (default: 0)')
    
    args = parser.parse_args()
    
    # Set up paths
    results_dir = "/scratch/nz2212/capstone/results"
    output_dir = "/scratch/nz2212/capstone/analysis/jsons"
    
    # Determine the file pattern based on casual flag
    if args.casual == 1:
        pattern_str = "ppo_casual"
    else:
        pattern_str = "ppo"
    
    # Determine output filename
    seed_str = str(args.seed) if args.seed is not None else "all"
    epoch_str = str(args.epoch) if args.epoch is not None else "all"
    if args.casual == 1:
        output_filename = f"{args.dataset_name}_{pattern_str}_seed_{seed_str}_epoch_{epoch_str}.json"
    else:
        output_filename = f"{args.dataset_name}_{pattern_str}_seed_{seed_str}_epoch_{epoch_str}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Extracting data for dataset: {args.dataset_name}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    if args.epoch is not None:
        print(f"Epoch: {args.epoch}")
    print(f"Casual mode: {args.casual}")
    print(f"Output file: {output_path}")
    
    # Extract data based on arguments
    if args.seed is not None and args.epoch is not None:
        # Extract specific seed and epoch
        if args.casual == 1:
            file_path = os.path.join(results_dir, f"{args.dataset_name}_ppo_casual_seed_{args.seed}.txt")
        else:
            file_path = os.path.join(results_dir, f"{args.dataset_name}_ppo_seed_{args.seed}.txt")
        epoch_data = extract_per_epoch(file_path, args.epoch)
        hyperparameters = extract_hyperparameters(file_path)
        data = {
            args.dataset_name: {
                "hparams": hyperparameters,
                "seeds": {f"seed_{args.seed}": epoch_data}
            }
        }
    elif args.seed is not None:
        # Extract all epochs for specific seed
        seed_data = extract_per_seed(results_dir, args.dataset_name, args.seed, args.casual)
        data = {args.dataset_name: seed_data}
    else:
        # Extract all seeds and epochs
        data = extract_all(results_dir, args.dataset_name, args.casual)
    
    # Save to JSON
    save_json(data, output_path)
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    main()