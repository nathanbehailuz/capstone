import re
from pathlib import Path
import numpy as np
import argparse
import os

avg_p_loss_re = re.compile(r'avg_p_loss=([-\d.]+)')
pair_re = re.compile(r'(\w+)=([-\d.]+)')
auc_re = re.compile(r'AUC=([-\d.]+)')
diag_re = re.compile(
    r'\[Diagnostics\]'
    r'KL=(-?\d+(?:\.\d+)?), '
    r'ClipFrac=(-?\d+(?:\.\d+)?), '
    r'Entropy=(-?\d+(?:\.\d+)?), '
    r'ExplainedVar=(-?\d+(?:\.\d+)?), '
    r'PolicyLoss=(-?\d+(?:\.\d+)?), '
    r'ValueLoss=(-?\d+(?:\.\d+)?)'
)


def extract_metrics(file_path):
    results = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                if line.lstrip().startswith("avg_p_loss"):
                    # capture all key=value pairs in this line
                    next_line = file.readline()
                    match = auc_re.search(next_line)
                    if match:
                        for key, value in pair_re.findall(line):
                            results[key] = results.get(key, []) + [float(value)]                
                        results['auc'] = results.get('auc', []) + [float(match.group(1))]
                elif line.lstrip().startswith("[Diagnostics]"):
                    match = diag_re.search(line)
                    if match:
                        results['kl'] = results.get('kl', []) + [float(match.group(1))]
                        results['clip_frac'] = results.get('clip_frac', []) + [float(match.group(2))]
                        results['entropy'] = results.get('entropy', []) + [float(match.group(3))]
                        results['explained_var'] = results.get('explained_var', []) + [float(match.group(4))]
                        results['policy_loss'] = results.get('policy_loss', []) + [float(match.group(5))]
                        results['value_loss'] = results.get('value_loss', []) + [float(match.group(6))]
    
    return results

def auc_results(datasets):
    for dataset in datasets:
        print("Dataset: ", dataset)
        for policy in ['reinforce', 'ppo']:
            print(f"Policy = {policy}")
            results = []
            for i in range(10):
                file_path = f'./results/{dataset}_{policy}_seed_{i}.txt'
                metrics = extract_metrics(Path(file_path))
                if metrics:
                    results.append(metrics)
            
            # calculate the average AUC with std 
            auc_values = [result['auc'][-1] for result in results]
            print(f"Average AUC: {np.mean(auc_values)} with std {np.std(auc_values)}")
        print("=" * 100)


auc_results(['ba2','mutag','syn1','syn2','syn3' ])