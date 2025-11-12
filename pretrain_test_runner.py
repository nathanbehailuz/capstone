#!/usr/bin/env python3
"""
Pretraining Test Runner
-----------------------
Standalone script for testing generator pretraining with edge dropping.
Evaluates robustness and optionally runs short PPO warmup phase.

Usage:
    python pretrain_test_runner.py --dataset syn2 --seed 0 --random_edge_drop 0.2
"""

import argparse
import datetime
import json
import os
import pathlib
import sys
import time
import numpy as np
import torch

# Import from existing codebase
from run_RG_explainer import Runner, drop_edges, linear_schedule, get_lambda_schedule
from utils import seed_all
from datasets import dataset_loaders, ground_truth_loaders
from models import model_selector
from components import Graph, Generator


class TeeLogger:
    """Helper to write to both stdout and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def initialize_test_env(args):
    """
    Initialize test environment: set seeds, create directories, setup logging.
    
    Returns:
        log_path: Path to log file
        start_time: Start timestamp
    """
    # Set random seeds
    seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    drop_str = f"drop{int(args.random_edge_drop*100):02d}" if args.random_edge_drop > 0 else "nodrop"
    log_filename = f"{args.dataset}_{drop_str}_seed{args.seed}_{timestamp}.log"
    log_path = output_dir / log_filename
    
    # Setup dual logging (stdout + file)
    logger = TeeLogger(log_path)
    sys.stdout = logger
    
    # Print configuration
    print("=" * 60)
    print("PRETRAINING TEST RUNNER")
    print("=" * 60)
    print(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Edge Drop Probability: {args.random_edge_drop}")
    print(f"Pretrain Iterations: List={args.pretrain_list}, Set={args.pretrain_set}")
    print(f"PPO Test Epochs: {args.ppo_test_epochs if not args.skip_ppo_test else 'Skipped'}")
    print(f"Robustness Eval: {'Enabled' if not args.skip_eval_robustness else 'Skipped'}")
    print(f"Log File: {log_path}")
    print(f"SLURM Job ID: {os.environ.get('SLURM_JOB_ID', 'Not running on SLURM')}")
    print("=" * 60)
    print(flush=True)
    
    start_time = time.time()
    return log_path, start_time


def run_pretraining_test(args):
    """
    Run pretraining using existing Runner class.
    
    Returns:
        runner: Initialized Runner instance with pretrained model
        pretrain_metrics: Dictionary with pretraining metrics
    """
    print("\n" + "=" * 60)
    print("PHASE 1: PRETRAINING")
    print("=" * 60)
    
    # Initialize Runner
    print("Initializing Runner...", flush=True)
    runner = Runner(args)
    
    # Run pretraining
    print("\nStarting pretraining phase...", flush=True)
    pretrain_start = time.time()
    runner.pretrain()
    pretrain_duration = time.time() - pretrain_start
    
    print(f"\nPretraining completed in {pretrain_duration:.1f} seconds ({pretrain_duration/60:.1f} minutes)", flush=True)
    
    # Get final evaluation
    print("\nEvaluating pretrained model...", flush=True)
    if runner.task == "graph_task":
        runner.eval_seeds = runner.l.get_eval_seed_deterministic()
    auc_score = runner.evaluate_and_print('Pretrained')
    
    pretrain_metrics = {
        'duration_seconds': pretrain_duration,
        'auc_score': auc_score,
        'dataset': args.dataset,
        'seed': args.seed,
        'edge_drop_prob': args.random_edge_drop
    }
    
    return runner, pretrain_metrics


def evaluate_robustness(runner, args):
    """
    Evaluate model robustness on edge-dropped graphs.
    
    Returns:
        robust_metrics: Dictionary with robustness test results
    """
    print("\n" + "=" * 60)
    print("PHASE 2: ROBUSTNESS EVALUATION")
    print("=" * 60)
    
    drop_probs = [0.0, 0.1, 0.2, 0.3]
    results = {}
    
    # Save original graph
    original_graph = runner.g.graph
    original_edge_index = original_graph.edge_index
    
    for drop_prob in drop_probs:
        print(f"\nTesting with drop probability: {drop_prob:.2f}", flush=True)
        
        if drop_prob > 0:
            # Apply edge dropping
            dropped_edge_index = drop_edges(original_edge_index, drop_prob, device=original_edge_index.device)
            
            # Create temporary graph
            dropped_edges_np = dropped_edge_index.cpu().numpy()
            temp_graph = Graph(dropped_edges_np)
            temp_graph.edge_index = dropped_edge_index
            
            # Replace graph temporarily
            runner.g.graph = temp_graph
            
            original_edges = original_edge_index.size(1)
            remaining_edges = dropped_edge_index.size(1)
            print(f"  Edges: {original_edges} -> {remaining_edges} (retained: {100*remaining_edges/original_edges:.1f}%)", flush=True)
        
        # Evaluate
        pred_sgs = runner.g.generate(runner.eval_seeds)
        p_loss, s_loss, sim_loss, r_p = runner.eval_loss(pred_sgs)
        
        avg_p_loss = np.mean(p_loss)
        avg_s_loss = np.mean(s_loss)
        avg_sim_loss = np.mean(sim_loss)
        avg_r_p = np.mean(r_p)
        
        results[f'drop_{int(drop_prob*100):02d}'] = {
            'pred_loss': float(avg_p_loss),
            'size_loss': float(avg_s_loss),
            'sim_loss': float(avg_sim_loss),
            'radius_penalty': float(avg_r_p),
            'total_loss': float(avg_p_loss + avg_s_loss + avg_sim_loss + avg_r_p)
        }
        
        print(f"  [EdgeDrop-Test] Pred Loss: {avg_p_loss:.4f} | Size Loss: {avg_s_loss:.4f} | Total: {results[f'drop_{int(drop_prob*100):02d}']['total_loss']:.4f}", flush=True)
    
    # Restore original graph
    runner.g.graph = original_graph
    
    # Compute deltas
    baseline_loss = results['drop_00']['total_loss']
    deltas = {}
    for drop_prob in [0.1, 0.2, 0.3]:
        key = f'drop_{int(drop_prob*100):02d}'
        delta = results[key]['total_loss'] - baseline_loss
        deltas[key] = delta
        print(f"\nΔLoss (drop {drop_prob:.1f} - baseline): {delta:+.4f}", flush=True)
    
    robust_metrics = {
        'results': results,
        'deltas': deltas,
        'baseline_loss': baseline_loss
    }
    
    return robust_metrics


def run_ppo_warmup(runner, args):
    """
    Run short PPO warmup phase to check entropy/KL stats.
    
    Returns:
        ppo_metrics: Dictionary with PPO warmup metrics
    """
    print("\n" + "=" * 60)
    print("PHASE 3: PPO WARMUP TEST")
    print("=" * 60)
    
    ppo_metrics = {
        'epochs': [],
        'rewards': [],
        'policy_losses': [],
        'lengths': [],
        'early_stop_rates': [],
        'avg_sizes': []
    }
    
    print(f"Running {args.ppo_test_epochs} PPO warmup epochs...\n", flush=True)
    
    for epoch in range(args.ppo_test_epochs):
        print(f"[PPO-Test Epoch {epoch+1}/{args.ppo_test_epochs}]", flush=True)
        
        # Get lambda schedule for this epoch
        lam_size, lam_radius, lam_sim, entropy_c = get_lambda_schedule(epoch, args)
        
        # Update epoch counter
        runner.g.update_epoch(epoch)
        
        # Run one training step
        import random
        shuffle_index = random.choices(list(range(len(runner.eval_seeds))), k=args.g_batch_size)
        selected_nodes, r, policy_loss, length = runner.g.update_ppo(
            np.array(runner.eval_seeds)[shuffle_index],
            runner.score_fn,
            lam_size=lam_size,
            lam_radius=lam_radius,
            lam_sim=lam_sim,
            entropy_coef=entropy_c
        )
        
        # Calculate metrics
        early_stop_rate = runner.g.get_early_stop_rate(selected_nodes)
        avg_size = np.mean([len([x for x in traj if x != 'EOS']) for traj in selected_nodes])
        
        # Store metrics
        ppo_metrics['epochs'].append(epoch + 1)
        ppo_metrics['rewards'].append(float(r))
        ppo_metrics['policy_losses'].append(float(policy_loss))
        ppo_metrics['lengths'].append(float(length))
        ppo_metrics['early_stop_rates'].append(float(early_stop_rate))
        ppo_metrics['avg_sizes'].append(float(avg_size))
        
        # Log stats
        print(f"  Reward: {r:.3f} | Policy Loss: {policy_loss:.4f} | Length: {length:.1f}", flush=True)
        print(f"  Early Stop Rate: {early_stop_rate:.3f} | Avg Size: {avg_size:.1f}", flush=True)
        print(f"  λ Schedule: size={lam_size:.4f}, radius={lam_radius:.4f}, entropy={entropy_c:.4f}", flush=True)
        
        # Get entropy and KL from generator if available
        if hasattr(runner.g, 'last_entropy'):
            print(f"  Entropy: {runner.g.last_entropy:.4f}", flush=True)
        if hasattr(runner.g, 'last_kl'):
            print(f"  KL Divergence: {runner.g.last_kl:.4f}", flush=True)
        
        print(flush=True)
    
    return ppo_metrics


def print_summary(pretrain_metrics, robust_metrics, ppo_metrics, args, total_duration):
    """
    Print and save comprehensive summary of test results.
    
    Returns:
        summary_path: Path to saved summary JSON file
    """
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    # Console summary
    print(f"Dataset: {args.dataset} | Seed: {args.seed}")
    print(f"Edge Drop Probability: {args.random_edge_drop}")
    
    if pretrain_metrics:
        print(f"\nPretraining:")
        print(f"  Duration: {pretrain_metrics['duration_seconds']/60:.1f} minutes")
        print(f"  AUC Score: {pretrain_metrics['auc_score']:.4f}")
    
    if robust_metrics:
        print(f"\nRobustness Evaluation:")
        print(f"  Baseline Loss (no drop): {robust_metrics['baseline_loss']:.4f}")
        for drop_prob in [0.1, 0.2, 0.3]:
            key = f'drop_{int(drop_prob*100):02d}'
            delta = robust_metrics['deltas'][key]
            loss = robust_metrics['results'][key]['total_loss']
            print(f"  Drop {drop_prob:.1f}: Loss={loss:.4f}, Δ={delta:+.4f}")
    
    if ppo_metrics and not args.skip_ppo_test:
        print(f"\nPPO Warmup ({args.ppo_test_epochs} epochs):")
        final_epoch = ppo_metrics['epochs'][-1]
        print(f"  Final Reward (Epoch {final_epoch}): {ppo_metrics['rewards'][-1]:.3f}")
        print(f"  Final Policy Loss: {ppo_metrics['policy_losses'][-1]:.4f}")
        print(f"  Final Early Stop Rate: {ppo_metrics['early_stop_rates'][-1]:.3f}")
        print(f"  Final Avg Size: {ppo_metrics['avg_sizes'][-1]:.1f}")
    
    print(f"\nTotal Test Duration: {total_duration/60:.1f} minutes")
    print("=" * 60)
    print(flush=True)
    
    # Save to JSON
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    drop_str = f"drop{int(args.random_edge_drop*100):02d}" if args.random_edge_drop > 0 else "nodrop"
    json_filename = f"{args.dataset}_{drop_str}_seed{args.seed}_{timestamp}.json"
    json_path = pathlib.Path(args.output_dir) / json_filename
    
    summary_data = {
        'config': vars(args),
        'pretrain_metrics': pretrain_metrics,
        'robustness_metrics': robust_metrics,
        'ppo_metrics': ppo_metrics,
        'total_duration_seconds': total_duration,
        'timestamp': timestamp
    }
    
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Summary saved to: {json_path}", flush=True)
    
    return json_path


def main():
    """Main entry point for pretraining test runner."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Pretraining Test Runner')
    
    # Dataset and model
    parser.add_argument('--dataset', type=str, default='syn2',
                       help='Dataset name (syn1/syn2/syn3/syn4/ba2/mutag)')
    parser.add_argument('--model', type=str, default='GNN',
                       help='Model type')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    
    # Architecture
    parser.add_argument('--with_attr', action='store_true', default=True,
                       help='Use node attributes')
    parser.add_argument('--n_hop', type=int, default=3,
                       help='Number of hops for subgraph')
    parser.add_argument('--max_size', type=int, default=15,
                       help='Maximum subgraph size')
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='Hidden layer size')
    
    # Pretraining parameters (reduced defaults for quick testing)
    parser.add_argument('--pretrain_list', type=int, default=5,
                       help='List-based pretraining iterations')
    parser.add_argument('--pretrain_set', type=int, default=10,
                       help='Set-based pretraining iterations')
    parser.add_argument('--pretrain_g_batch_size', type=int, default=32,
                       help='Pretraining batch size')
    parser.add_argument('--pretrain_l_iter', type=int, default=50,
                       help='Locator pretraining iterations')
    parser.add_argument('--pretrain_l_sample_rate', type=float, default=1.0,
                       help='Locator sample rate')
    parser.add_argument('--l_lr', type=float, default=1e-2,
                       help='Locator learning rate')
    parser.add_argument('--g_lr', type=float, default=1e-2,
                       help='Generator learning rate')
    
    # Edge dropping
    parser.add_argument('--random_edge_drop', type=float, default=0.0,
                       help='Edge drop probability during pretraining (0.0-1.0)')
    
    # Testing parameters
    parser.add_argument('--ppo_test_epochs', type=int, default=2,
                       help='Number of PPO warmup epochs to run')
    parser.add_argument('--skip_ppo_test', action='store_true',
                       help='Skip PPO warmup phase')
    parser.add_argument('--skip_eval_robustness', action='store_true',
                       help='Skip robustness evaluation')
    
    # PPO parameters (needed for warmup phase)
    parser.add_argument('--n_epochs', type=int, default=5,
                       help='Total epochs (for schedule calculation)')
    parser.add_argument('--g_batch_size', type=int, default=128,
                       help='Batch size for PPO')
    parser.add_argument('--n_rollouts', type=int, default=5,
                       help='Number of rollouts')
    parser.add_argument('--ppo_epochs', type=int, default=4,
                       help='PPO optimization epochs')
    parser.add_argument('--minibatch_size', type=int, default=256,
                       help='PPO minibatch size')
    
    # Regularization and scheduling
    parser.add_argument('--warmup_epochs', type=int, default=3,
                       help='Warmup epochs for lambda schedule')
    parser.add_argument('--schedule_epochs', type=int, default=12,
                       help='Schedule ramp epochs')
    parser.add_argument('--entropy_coeff_start', type=float, default=3e-3,
                       help='Starting entropy coefficient')
    parser.add_argument('--entropy_coeff_end', type=float, default=5e-4,
                       help='Ending entropy coefficient')
    parser.add_argument('--lambda_size_final', type=float, default=0.02,
                       help='Final size penalty')
    parser.add_argument('--lambda_radius_final', type=float, default=0.10,
                       help='Final radius penalty')
    parser.add_argument('--lambda_sim_start', type=float, default=0.0,
                       help='Starting similarity penalty')
    parser.add_argument('--lambda_sim_final', type=float, default=0.005,
                       help='Final similarity penalty')
    
    # Other necessary parameters
    parser.add_argument('--size_reg', type=float, default=0.01,
                       help='Size regularization')
    parser.add_argument('--sim_reg', type=float, default=0.01,
                       help='Similarity regularization')
    parser.add_argument('--radius_penalty', type=float, default=0.1,
                       help='Radius penalty')
    parser.add_argument('--entropy_coef', type=float, default=0.,
                       help='Entropy coefficient')
    
    # PPO stability parameters
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--clip_range_policy', type=float, default=0.2)
    parser.add_argument('--clip_range_value', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--kl_target', type=float, default=0.0075)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--use_value_clip', action='store_true', default=True)
    parser.add_argument('--adaptive_kl_penalty', action='store_true', default=False)
    parser.add_argument('--ppo_clip', type=float, default=0.1)
    parser.add_argument('--vf_clip', type=float, default=0.5)
    parser.add_argument('--use_gae', action='store_true', default=True)
    parser.add_argument('--reward_norm', action='store_true', default=True)
    
    # STOP action parameters
    parser.add_argument('--target_size_k', type=int, default=8)
    parser.add_argument('--lambda_size', type=float, default=0.5)
    parser.add_argument('--lambda_radius', type=float, default=1.0)
    parser.add_argument('--lambda_radius_min', type=float, default=0.2)
    parser.add_argument('--step_cost', type=float, default=0.001)
    parser.add_argument('--stop_bias', type=float, default=0.5)
    parser.add_argument('--fidelity_thresh', type=float, default=0.05)
    parser.add_argument('--reward_scale', type=float, default=10.0)
    parser.add_argument('--stop_bonus_ok', type=float, default=0.10)
    parser.add_argument('--stop_penalty_bad', type=float, default=0.40)
    
    # Causal parameters
    parser.add_argument('--use_causal', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--causal_weight', type=float, default=0.5)
    parser.add_argument('--causal_warmup_epochs', type=int, default=5)
    
    # Additional hyperparameters
    parser.add_argument('--value_loss_coef', type=float, default=0.3)
    parser.add_argument('--value_loss_scale', type=float, default=0.5)
    parser.add_argument('--confidence_threshold', type=float, default=0.5)
    parser.add_argument('--confidence_tolerance', type=float, default=0.05)
    parser.add_argument('--causal_action_weight', type=float, default=0.2)
    parser.add_argument('--causal_gamma', type=float, default=0.5)
    parser.add_argument('--lambda_conf', type=float, default=0.1)
    parser.add_argument('--ice_score_default', type=float, default=-5.0)
    parser.add_argument('--stop_mask_value', type=float, default=-1e9)
    parser.add_argument('--min_steps_before_stop', type=int, default=8)
    parser.add_argument('--kl_adapt_high_multiplier', type=float, default=2.0)
    parser.add_argument('--kl_adapt_low_multiplier', type=float, default=0.5)
    parser.add_argument('--kl_adapt_high_threshold', type=float, default=2.0)
    parser.add_argument('--kl_adapt_low_threshold', type=float, default=0.5)
    parser.add_argument('--entropy_decay_fraction', type=float, default=0.7)
    parser.add_argument('--entropy_decay_start', type=float, default=1e-3)
    parser.add_argument('--entropy_decay_end', type=float, default=1e-4)
    parser.add_argument('--entropy_decay_final', type=float, default=0.0)
    parser.add_argument('--delta_reward', type=int, default=1)
    parser.add_argument('--traj_reward_norm', type=int, default=1)
    parser.add_argument('--std_advantage', type=int, default=1)
    
    # Locator parameters
    parser.add_argument('--update_l_sample_rate', type=float, default=0.2)
    parser.add_argument('--n_g_updates', type=int, default=1)
    parser.add_argument('--n_l_updates', type=int, default=10)
    
    # Evaluation
    parser.add_argument('--eval_every', type=int, default=1)
    
    # Debug
    parser.add_argument('--debug_mode', type=int, default=0)
    parser.add_argument('--ppo_debug', action='store_true', default=False)
    parser.add_argument('--dry_run_debug', action='store_true', default=False)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/pretraining_tests',
                       help='Directory for test outputs')
    parser.add_argument('--savedir', type=str, default='',
                       help='Save directory (auto-generated if empty)')
    
    # Policy (needed for Runner)
    parser.add_argument('--policy', type=str, default='PPO')
    parser.add_argument('--frontier_shaping_beta', type=float, default=0.0)
    
    # Edge score aggregation
    parser.add_argument('--edge_score_agg', type=str, default='max')
    parser.add_argument('--debug_eval_graph', type=int, default=-1)
    
    args = parser.parse_args()
    
    # Auto-generate savedir if not provided
    if not args.savedir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        args.savedir = f'ckpts/{args.dataset}/test_{timestamp}/'
    
    # Set dataset-specific defaults for target_size_k
    if args.target_size_k == 8:
        if args.dataset in ['syn1', 'syn2', 'syn3', 'syn4']:
            args.target_size_k = 8
        elif args.dataset == 'ba2':
            args.target_size_k = 10
        elif args.dataset == 'mutag':
            args.target_size_k = 6
    
    # Initialize environment
    log_path, start_time = initialize_test_env(args)
    
    try:
        # Phase 1: Pretraining
        runner, pretrain_metrics = run_pretraining_test(args)
        
        # Phase 2: Robustness Evaluation
        robust_metrics = None
        if not args.skip_eval_robustness:
            robust_metrics = evaluate_robustness(runner, args)
        else:
            print("\nSkipping robustness evaluation (--skip_eval_robustness)", flush=True)
        
        # Phase 3: PPO Warmup
        ppo_metrics = None
        if not args.skip_ppo_test:
            ppo_metrics = run_ppo_warmup(runner, args)
        else:
            print("\nSkipping PPO warmup test (--skip_ppo_test)", flush=True)
        
        # Print summary
        total_duration = time.time() - start_time
        summary_path = print_summary(pretrain_metrics, robust_metrics, ppo_metrics, args, total_duration)
        
        print(f"\n✓ Test completed successfully!")
        print(f"End Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60, flush=True)
        
        # Close runner
        runner.close()
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Restore stdout and close log file
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
        sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()

