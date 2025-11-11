#!/usr/bin/env python3
"""
Verification script for causal reward integration fix.

This script checks if the causal reward is actually being computed
by inspecting the logs and reward statistics.

Usage:
    python verify_causal_fix.py results/syn4_ppo_causal_seed_0.txt
"""

import sys
import re
import numpy as np


def parse_log_file(filepath):
    """Parse a log file and extract relevant statistics."""
    
    stats = {
        'causal_init': False,
        'causal_reward_calls': 0,
        'causal_deltas': [],
        'auc_scores': [],
        'avg_p_losses': [],
        'avg_size_losses': [],
        'has_causal_logs': False,
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            # Check for causal initialization
            if 'Initialized causal reward calculator' in line:
                stats['causal_init'] = True
            
            # Check for causal reward computation
            if '[CAUSAL REWARD]' in line:
                stats['has_causal_logs'] = True
                stats['causal_reward_calls'] += 1
                
                # Extract delta value
                delta_match = re.search(r'Delta:\s*([-\d.]+)', line)
                if delta_match:
                    stats['causal_deltas'].append(float(delta_match.group(1)))
            
            # Extract AUC scores
            auc_match = re.search(r'\[EVAL-Epoch\s+\d+\]\s+AUC=([\d.]+)', line)
            if auc_match:
                stats['auc_scores'].append(float(auc_match.group(1)))
            
            # Extract average losses
            p_loss_match = re.search(r'avg_p_loss=([\d.]+)', line)
            if p_loss_match:
                stats['avg_p_losses'].append(float(p_loss_match.group(1)))
            
            size_loss_match = re.search(r'avg_size_loss=([\d.]+)', line)
            if size_loss_match:
                stats['avg_size_losses'].append(float(size_loss_match.group(1)))
    
    return stats


def verify_causal_integration(stats):
    """Verify that causal reward integration is working."""
    
    print("=" * 70)
    print("CAUSAL REWARD INTEGRATION VERIFICATION")
    print("=" * 70)
    print()
    
    # Check 1: Causal reward calculator initialized
    print("✓ CHECK 1: Causal Reward Calculator Initialization")
    if stats['causal_init']:
        print("  ✅ PASS - Causal reward calculator initialized")
    else:
        print("  ❌ FAIL - Causal reward calculator NOT initialized")
        print("  → Check if use_causal=True in config")
    print()
    
    # Check 2: Causal reward calls
    print("✓ CHECK 2: Causal Reward Computation")
    if stats['causal_reward_calls'] > 0:
        print(f"  ✅ PASS - Causal reward computed {stats['causal_reward_calls']} times")
        print(f"  → This confirms the fix is working!")
    else:
        print("  ❌ FAIL - Causal reward NEVER computed")
        print("  → This indicates the bug is still present")
        print("  → The causal reward calculator is not being called in the reward loop")
    print()
    
    # Check 3: Causal delta statistics
    if stats['causal_deltas']:
        print("✓ CHECK 3: Causal Delta Statistics")
        print(f"  Mean causal delta: {np.mean(stats['causal_deltas']):.4f}")
        print(f"  Std causal delta:  {np.std(stats['causal_deltas']):.4f}")
        print(f"  Min causal delta:  {np.min(stats['causal_deltas']):.4f}")
        print(f"  Max causal delta:  {np.max(stats['causal_deltas']):.4f}")
        
        if np.mean(np.abs(stats['causal_deltas'])) > 0.001:
            print("  ✅ Causal deltas have meaningful variation")
        else:
            print("  ⚠️  Causal deltas are very small - check if this is expected")
        print()
    
    # Check 4: AUC scores
    if stats['auc_scores']:
        print("✓ CHECK 4: Final AUC Score")
        final_auc = stats['auc_scores'][-1]
        print(f"  Final AUC: {final_auc:.4f}")
        
        if final_auc > 0.85:
            print("  ✅ Good AUC score")
        elif final_auc > 0.70:
            print("  ⚠️  Moderate AUC score - room for improvement")
        else:
            print("  ❌ Low AUC score - check hyperparameters")
        print()
    
    # Check 5: Training stability
    if stats['avg_p_losses']:
        print("✓ CHECK 5: Training Stability (Prediction Loss)")
        print(f"  Mean prediction loss: {np.mean(stats['avg_p_losses']):.4f}")
        print(f"  Std prediction loss:  {np.std(stats['avg_p_losses']):.4f}")
        
        if np.std(stats['avg_p_losses']) < 2.0:
            print("  ✅ Training appears stable")
        else:
            print("  ⚠️  High variance in prediction loss - check stability")
        print()
    
    # Overall verdict
    print("=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)
    
    if stats['has_causal_logs'] and stats['causal_reward_calls'] > 0:
        print("✅ SUCCESS - Causal reward integration is working!")
        print()
        print("The fix has been applied successfully. The causal reward calculator")
        print("is being called during training and contributing to the reward signal.")
    else:
        print("❌ FAILURE - Causal reward integration is NOT working!")
        print()
        print("The causal reward calculator is not being called in the reward loop.")
        print("This suggests either:")
        print("  1. The bug fix was not applied correctly")
        print("  2. use_causal=False in the config")
        print("  3. The code is using an old version without the fix")
    
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_causal_fix.py <log_file>")
        print()
        print("Example:")
        print("  python verify_causal_fix.py results/syn4_ppo_causal_seed_0.txt")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    try:
        stats = parse_log_file(log_file)
        verify_causal_integration(stats)
        
    except FileNotFoundError:
        print(f"Error: File not found: {log_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

