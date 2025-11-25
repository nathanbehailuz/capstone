#!/usr/bin/env python3
'''
This script analyzes {dataset}_ppo txt files and extracts average AUC for the last epoch across all seeds.
'''

import os
import re
import sys
import numpy as np
from pathlib import Path


def extract_auc_from_file(file_path):
    """
    Extract all epoch AUC values from a single txt file.
    Returns a dict mapping epoch_num -> AUC value
    """
    epoch_aucs = {}
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return epoch_aucs
    
    for line in lines:
        line = line.strip()
        # Look for lines like: [EVAL-Epoch    1] AUC=0.7321939623544282
        auc_match = re.search(r'\[EVAL-Epoch\s+(\d+)\]\s+AUC=([\d.-]+)', line)
        if auc_match:
            epoch_num = int(auc_match.group(1))
            auc_value = float(auc_match.group(2))
            epoch_aucs[epoch_num] = auc_value
    
    return epoch_aucs


def analyze_ppo_dataset(dataset_name):
    """
    Analyze all {dataset}_ppo_seed_*.txt files and compute average AUC at last epoch.
    """
    # hard coded values
    results_dir = '/scratch/nz2212/capstone/results'
    output_file = f'/scratch/nz2212/capstone/analysis/txts/{dataset_name}_ppo_auc.txt'
    
    # create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Find all {dataset}_ppo_seed_*.txt files
    txt_files = []
    for filename in os.listdir(results_dir):
        if filename.startswith(f'{dataset_name}_ppo_seed_') and filename.endswith('.txt'):
            # Skip the causal file if it exists
            if 'causal' not in filename:
                txt_files.append(filename)
    
    if not txt_files:
        print(f"No {dataset_name}_ppo_seed_*.txt files found in {results_dir}")
        return None
    
    # Sort files by seed number
    txt_files.sort()
    
    print(f"Found {len(txt_files)} files: {txt_files}")
    
    # Extract AUC values for each seed
    seed_data = {}
    all_last_epoch_aucs = []
    
    for filename in txt_files:
        # Extract seed number
        seed_match = re.search(r'_seed_(\d+)\.txt', filename)
        if not seed_match:
            continue
        
        seed_num = int(seed_match.group(1))
        file_path = os.path.join(results_dir, filename)
        
        print(f"Processing seed {seed_num}: {filename}")
        
        # Extract all epoch AUCs
        epoch_aucs = extract_auc_from_file(file_path)
        
        if epoch_aucs:
            # Get the last (maximum) epoch
            last_epoch = max(epoch_aucs.keys())
            last_auc = epoch_aucs[last_epoch]
            
            seed_data[seed_num] = {
                'last_epoch': last_epoch,
                'last_auc': last_auc,
                'all_epochs': epoch_aucs
            }
            all_last_epoch_aucs.append(last_auc)
            
            print(f"  Seed {seed_num}: Last epoch = {last_epoch}, AUC = {last_auc:.6f}")
        else:
            print(f"  Warning: No AUC values found in {filename}")
    
    # Calculate statistics
    if all_last_epoch_aucs:
        avg_auc = np.mean(all_last_epoch_aucs)
        std_auc = np.std(all_last_epoch_aucs)
        min_auc = np.min(all_last_epoch_aucs)
        max_auc = np.max(all_last_epoch_aucs)
        
        # Write results to file
        with open(output_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"{dataset_name.upper()} PPO Results - Average AUC at Last Epoch\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Dataset: {dataset_name}_ppo\n")
            f.write(f"Number of seeds: {len(all_last_epoch_aucs)}\n\n")
            
            f.write(f"Average AUC: {avg_auc:.6f}\n")
            f.write(f"Std AUC: {std_auc:.6f}\n")
            f.write(f"Min AUC: {min_auc:.6f}\n")
            f.write(f"Max AUC: {max_auc:.6f}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("Per-Seed Results:\n")
            f.write("-" * 70 + "\n")
            
            for seed_num in sorted(seed_data.keys()):
                data = seed_data[seed_num]
                f.write(f"\nSeed {seed_num}:\n")
                f.write(f"  Last Epoch: {data['last_epoch']}\n")
                f.write(f"  Last AUC: {data['last_auc']:.6f}\n")
                
                # Write all epochs
                f.write(f"  All Epochs:\n")
                for epoch in sorted(data['all_epochs'].keys()):
                    f.write(f"    Epoch {epoch:2d}: AUC = {data['all_epochs'][epoch]:.6f}\n")
        
        print(f"\n{'=' * 70}")
        print("SUMMARY:")
        print(f"{'=' * 70}")
        print(f"Dataset: {dataset_name}_ppo")
        print(f"Number of seeds: {len(all_last_epoch_aucs)}")
        print(f"Average AUC: {avg_auc:.6f}")
        print(f"Std AUC: {std_auc:.6f}")
        print(f"Min AUC: {min_auc:.6f}")
        print(f"Max AUC: {max_auc:.6f}")
        print(f"\nResults written to {output_file}")
        
        return {
            'avg_auc': avg_auc,
            'std_auc': std_auc,
            'min_auc': min_auc,
            'max_auc': max_auc,
            'num_seeds': len(all_last_epoch_aucs),
            'seed_data': seed_data
        }
    else:
        print("No valid AUC values found across all files!")
        return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_ppo_dataset.py <dataset_name>")
        print("Example: python analyze_ppo_dataset.py syn1")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    analyze_ppo_dataset(dataset_name)

