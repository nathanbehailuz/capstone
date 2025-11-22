#!/usr/bin/env python3
"""
Quick test to verify the edge dropping fix works for pretraining.
This tests that pretraining can complete at least one iteration with edge dropping enabled.
"""

import sys
import argparse
import torch
import numpy as np
from run_RG_explainer import Runner, drop_edges
from utils import seed_all

def test_edge_drop_pretraining():
    """Test that pretraining works with edge dropping without dimension mismatch."""
    
    # Setup basic args
    parser = argparse.ArgumentParser()
    
    # Minimal args needed for testing
    parser.add_argument('--dataset', type=str, default='syn1')
    parser.add_argument('--model', type=str, default='GNN')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--with_attr', action='store_true', default=True)
    parser.add_argument('--n_hop', type=int, default=3)
    parser.add_argument('--max_size', type=int, default=20)
    
    # Pretraining args - use very small values for fast test
    parser.add_argument('--pretrain_list', type=int, default=2)  # Just 2 iterations
    parser.add_argument('--pretrain_set', type=int, default=2)   # Just 2 iterations
    parser.add_argument('--pretrain_l_iter', type=int, default=1)
    parser.add_argument('--pretrain_l_sample_rate', type=float, default=1.0)
    parser.add_argument('--pretrain_g_batch_size', type=int, default=30)
    
    # Edge dropping - this is what we're testing!
    parser.add_argument('--random_edge_drop', type=float, default=0.10)  # 10% drop
    
    # Other required args
    parser.add_argument('--hidden_size', type=int, default=20)
    parser.add_argument('--g_lr', type=float, default=1e-3)
    parser.add_argument('--l_lr', type=float, default=1e-3)
    parser.add_argument('--policy', type=str, default='PPO')
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--minibatch_size', type=int, default=256)
    parser.add_argument('--n_rollouts', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--use_causal', type=str, default='True')
    parser.add_argument('--causal_weight', type=float, default=0.3)
    parser.add_argument('--clip_range_policy', type=float, default=0.2)
    parser.add_argument('--clip_range_value', type=float, default=0.5)
    parser.add_argument('--entropy_coeff_start', type=float, default=0.01)
    parser.add_argument('--entropy_coeff_end', type=float, default=0.001)
    parser.add_argument('--reward_scale', type=float, default=10.0)
    parser.add_argument('--lambda_size_final', type=float, default=0.02)
    parser.add_argument('--lambda_radius_final', type=float, default=0.10)
    parser.add_argument('--lambda_sim_start', type=float, default=0.0)
    parser.add_argument('--lambda_sim_final', type=float, default=0.0)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--schedule_epochs', type=int, default=12)
    parser.add_argument('--target_size_k', type=int, default=8)
    parser.add_argument('--causal_warmup_epochs', type=int, default=5)
    parser.add_argument('--debug_mode', type=int, default=0)
    parser.add_argument('--ds_name', type=str, default='syn1')
    
    # Required but not used in pretraining
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--adaptive_kl_penalty', action='store_true', default=False)
    parser.add_argument('--kl_target', type=float, default=0.015)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--use_value_clip', action='store_true', default=True)
    parser.add_argument('--vf_clip', type=float, default=0.5)
    parser.add_argument('--use_gae', action='store_true', default=True)
    parser.add_argument('--reward_norm', action='store_true', default=True)
    parser.add_argument('--entropy_decay_start', type=float, default=1e-3)
    parser.add_argument('--entropy_decay_end', type=float, default=1e-4)
    parser.add_argument('--entropy_decay_final', type=float, default=0.0)
    parser.add_argument('--delta_reward', type=int, default=1)
    parser.add_argument('--stop_bonus_ok', type=float, default=0.10)
    parser.add_argument('--stop_penalty_bad', type=float, default=0.40)
    parser.add_argument('--fidelity_thresh', type=float, default=0.05)
    parser.add_argument('--frontier_shaping_beta', type=float, default=0.0)
    parser.add_argument('--ppo_clip', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--value_loss_coef', type=float, default=0.3)
    parser.add_argument('--value_loss_scale', type=float, default=0.5)
    parser.add_argument('--confidence_threshold', type=float, default=0.5)
    parser.add_argument('--confidence_tolerance', type=float, default=0.05)
    parser.add_argument('--causal_action_weight', type=float, default=0.2)
    parser.add_argument('--causal_gamma', type=float, default=0.5)
    parser.add_argument('--ice_score_default', type=float, default=-5.0)
    parser.add_argument('--stop_mask_value', type=float, default=-1e9)
    parser.add_argument('--min_steps_before_stop', type=int, default=8)
    parser.add_argument('--kl_adapt_high_multiplier', type=float, default=2.0)
    parser.add_argument('--kl_adapt_low_multiplier', type=float, default=0.5)
    parser.add_argument('--kl_adapt_high_threshold', type=float, default=2.0)
    parser.add_argument('--kl_adapt_low_threshold', type=float, default=0.5)
    parser.add_argument('--entropy_decay_fraction', type=float, default=0.7)
    parser.add_argument('--size_reg', type=float, default=0.0)
    parser.add_argument('--sim_reg', type=float, default=0.0)
    parser.add_argument('--radius_penalty', type=float, default=0.0)
    parser.add_argument('--step_cost', type=float, default=0.001)
    parser.add_argument('--update_l_sample_rate', type=float, default=0.2)
    parser.add_argument('--n_g_updates', type=int, default=1)
    parser.add_argument('--n_l_updates', type=int, default=10)
    parser.add_argument('--edge_score_agg', type=str, default='max')
    parser.add_argument('--debug_eval_graph', type=int, default=-1)
    parser.add_argument('--dry_run_debug', action='store_true', default=False)
    
    args = parser.parse_args([])  # Empty list = use defaults
    
    # Set seeds for reproducibility
    seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("=" * 60)
    print("EDGE DROPPING FIX TEST")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Edge Drop: {args.random_edge_drop * 100:.0f}%")
    print(f"Pretrain Iterations: List={args.pretrain_list}, Set={args.pretrain_set}")
    print("=" * 60)
    print()
    
    try:
        # Initialize Runner
        print("Initializing Runner...", flush=True)
        runner = Runner(args)
        
        # Run pretraining - this is where the dimension mismatch occurred
        print("\nRunning pretraining with edge dropping...", flush=True)
        pretrain_metrics = runner._pretrain()
        
        # If we get here without exception, the fix worked!
        print("\n" + "=" * 60)
        print("✓ TEST PASSED!")
        print("=" * 60)
        print(f"List-based pretraining completed: {len(pretrain_metrics['list_based']['iterations'])} iterations")
        print(f"Set-based pretraining completed: {len(pretrain_metrics['set_based']['iterations'])} iterations")
        print(f"Final List Loss: {pretrain_metrics['list_based']['avg_losses'][-1]:.4f}")
        print(f"Final Set Loss: {pretrain_metrics['set_based']['avg_losses'][-1]:.4f}")
        print("\nThe dimension mismatch bug has been fixed!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(test_edge_drop_pretraining())






