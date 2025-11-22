'''
This script is used to analyze the results of the experiments using the json files.

1. write the average auc for each dataset to .txt file.
How to do this:
0. for all jsons in file paths and 
1. read the json file
2. for each seed, get the auc of the last (largest) epoch
3. get the average auc and std for all seeds
4. write the average auc and std to the .txt file


hard coded values:
1. file paths = './analysis/jsons'
2. output file path = './analysis/txts/aug_auc.txt'

user input:
1. function name = 'aug_auc'

'''

import json
import os
import numpy as np
from pathlib import Path


def aug_auc():
    # hard coded values
    json_dir = './analysis/jsons'
    output_file = './analysis/txts/aug_auc.txt'
    
    # create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # get all json files in the directory
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    results = []
    
    # process each json file
    for json_file in sorted(json_files):
        json_path = os.path.join(json_dir, json_file)
        
        # read the json file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # get the dataset name (first key in the json)
        dataset_name = list(data.keys())[0]
        
        # get seeds data
        seeds_data = data[dataset_name].get('seeds', {})
        
        # skip if no seeds data
        if not seeds_data:
            print(f"Warning: No seeds data found in {json_file}, skipping...")
            continue
        
        aucs = []
        
        # for each seed, get the auc of the last (largest) epoch
        for seed_key in seeds_data.keys():
            seed_data = seeds_data[seed_key]
            
            # get all epoch keys and find the last one
            epoch_keys = [k for k in seed_data.keys() if k.startswith('epoch_')]
            
            if epoch_keys:
                # extract epoch numbers and find the maximum
                epoch_numbers = [int(k.split('_')[1]) for k in epoch_keys]
                last_epoch_num = max(epoch_numbers)
                last_epoch_key = f'epoch_{last_epoch_num}'
                
                # get the AUC value with error handling
                try:
                    epoch_data = seed_data[last_epoch_key]
                    if 'data' in epoch_data and 'AUC' in epoch_data['data']:
                        auc = epoch_data['data']['AUC']
                        aucs.append(auc)
                    else:
                        print(f"Warning: Missing 'data' or 'AUC' in {json_file}, {seed_key}, {last_epoch_key}")
                except (KeyError, TypeError) as e:
                    print(f"Warning: Error reading AUC from {json_file}, {seed_key}, {last_epoch_key}: {e}")
        
        # get the average auc and std for all seeds
        if aucs:
            avg_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            results.append({
                'dataset': dataset_name,
                'avg_auc': avg_auc,
                'std_auc': std_auc,
                'num_seeds': len(aucs)
            })
        else:
            print(f"Warning: No valid AUC values found in {json_file}, skipping...")
    
    # write the average auc and std to the .txt file
    with open(output_file, 'w') as f:
        f.write("Dataset Results - Average AUC at Last Epoch\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results:
            f.write(f"Dataset: {result['dataset']}\n")
            f.write(f"  Average AUC: {result['avg_auc']:.6f}\n")
            f.write(f"  Std AUC: {result['std_auc']:.6f}\n")
            f.write(f"  Number of seeds: {result['num_seeds']}\n")
            f.write("\n")
    
    print(f"Results written to {output_file}")
    return results


if __name__ == '__main__':
    aug_auc()
