'''
given 2 json files, one for the RL and one for the PPO, plot the arg  for the RL and PPO for each epoch
if arg = policy_loss 
plot the policy loss for the RL and PPO for each epoch
if arg = reward
plot the reward for the RL and PPO for each epoch
'''

# Set OpenBLAS threading to avoid resource issues on HPC systems
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import json
import matplotlib.pyplot as plt
import argparse

def load_reinforce_data(file_path, metric):
    """Load REINFORCE data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    epochs = []
    values = []
    
    # Navigate through the structure: syn1.seeds.seed_0.epoch_X.generator
    seed_data = data['syn1']['seeds']['seed_0']
    
    for epoch_key in sorted(seed_data.keys(), key=lambda x: int(x.split('_')[1])):
        if epoch_key.startswith('epoch_'):
            epoch_num = int(epoch_key.split('_')[1])
            generator_data = seed_data[epoch_key].get('generator', {})
            
            # Skip empty epochs
            if not generator_data:
                continue
                
            if metric in generator_data:
                epochs.append(epoch_num)
                values.append(generator_data[metric])
    
    return epochs, values

def load_ppo_data(file_path, metric):
    """Load PPO data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    epochs = []
    values = []
    
    # Navigate through the structure: seeds.seed_0.epochs.epoch_X
    epoch_data = data['seeds']['seed_0']['epochs']
    
    for epoch_key in sorted(epoch_data.keys(), key=lambda x: int(x.split('_')[1])):
        if epoch_key.startswith('epoch_'):
            epoch_num = int(epoch_key.split('_')[1])
            
            # For PPO, policy_loss is in ppo_stats, reward is in reward_summary
            if metric == 'policy_loss':
                value = epoch_data[epoch_key]['ppo_stats'].get('policy_loss')
            elif metric == 'reward':
                value = epoch_data[epoch_key]['reward_summary'].get('reward')
            else:
                continue
            
            if value is not None:
                epochs.append(epoch_num)
                values.append(value)
    
    return epochs, values

def plot_comparison(reinforce_file, ppo_file, metric, output_file=None):
    """Plot comparison of REINFORCE and PPO for a given metric"""
    
    # Load data
    reinforce_epochs, reinforce_values = load_reinforce_data(reinforce_file, metric)
    ppo_epochs, ppo_values = load_ppo_data(ppo_file, metric)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(reinforce_epochs, reinforce_values, marker='o', label='REINFORCE', 
             color='blue', linewidth=2, markersize=6)
    plt.plot(ppo_epochs, ppo_values, marker='s', label='PPO', 
             color='red', linewidth=2, markersize=6)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'{metric.replace("_", " ").title()} Comparison: REINFORCE vs PPO', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot REINFORCE vs PPO comparison')
    parser.add_argument('--metric', type=str, default='policy_loss', 
                        choices=['policy_loss', 'reward'],
                        help='Metric to plot (policy_loss or reward)')
    parser.add_argument('--reinforce_file', type=str, 
                        default='analysis/jsons/syn1_performance_seed_0_epoch_all.json',
                        help='Path to REINFORCE JSON file')
    parser.add_argument('--ppo_file', type=str, 
                        default='analysis/jsons/syn1_ppo_performance_seed_0.json',
                        help='Path to PPO JSON file')
    parser.add_argument('--dataset', type=str, default='syn1',
                        help='Dataset name for output file')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed number for output file')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = 'analysis/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    file_name = f'{args.dataset}_{args.metric}_seed_{args.seed}_comparison.png'
    output_file = os.path.join(output_dir, file_name)
    
    plot_comparison(args.reinforce_file, args.ppo_file, args.metric, output_file)
