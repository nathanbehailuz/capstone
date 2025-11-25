#!/usr/bin/env python3
"""
Per-Graph AUC Analysis Tool
Computes AUC for each graph from the last epoch and creates a scatter plot.

EXACT REPLICATION OF TRAINING EVALUATION:
This script uses the EXACT methods from the training code to ensure computed AUC
matches the logged AUC values:

1. Edge Scoring: prepare_mask() replicates run_RG_explainer.py lines 658-676
   - Creates edge scores from node selection sequence
   - Earlier selections get higher scores: rank[edge] = max_size - i
   - Normalizes to [0,1]: mask = rank / (max_size - 1)

2. AUC Calculation (TWO DIFFERENT METHODS):
   
   a) NODE TASKS (syn1-syn4): compute_auc_from_mask_node() 
      Replicates evaluation/utils.py lines 61-96 (evaluation_auc_node)
      - Loops over EXPLANATION edges (subgraph)
      - For each edge, checks if it exists in ground truth (either direction)
      - Labels as 1 if in ground truth, 0 otherwise
      - Uses k-hop subgraph, sgs are relative to seed node
   
   b) GRAPH TASKS (ba2, mutag): compute_auc_from_mask_graph()
      Replicates evaluation/utils.py lines 24-58 (evaluation_auc_graph)
      - Loops over GROUND TRUTH edges
      - Finds each GT edge in explanation
      - Skips self-loops
      - Retrieves mask score for that edge
      - Uses full graph, sgs are graph-relative

   Both use manual ROC-AUC calculation to avoid sklearn GLIBCXX library issues.

Usage:
    python analysis/plot_auc_per_graph.py --dataset ba2 --seed 4 [--causal] [--max_size 20]
    python analysis/plot_auc_per_graph.py --dataset syn1 --seed 0 [--causal] [--n_hop 3]
"""

import os
# Limit OpenBLAS threads before importing numpy
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import json
import re
import numpy as np
import sys

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import functions from visualize_explanations.py
from analysis.visualize_explanations import (
    parse_results_file,
    load_ground_truth,
    load_graph_structure,
    compute_auc_roc  # Fallback manual AUC calculation if sklearn unavailable
)

# Import evaluation utilities - these use sklearn but we'll handle import errors
try:
    from evaluation.utils import evaluation_auc_graph, evaluation_auc_node
    USE_EVALUATION_FUNCTIONS = True
except (ImportError, ModuleNotFoundError):
    USE_EVALUATION_FUNCTIONS = False
    print("Warning: Could not import evaluation functions, using manual AUC calculation")


def prepare_mask(edge_index, selected_nodes, max_size=20):
    """
    Prepare edge mask exactly as done in run_RG_explainer.py (lines 658-676).
    
    This replicates the exact logic used during training to create edge importance scores
    from the node selection sequence. The scores are based on when nodes were selected:
    earlier selections get higher scores.
    
    Args:
        edge_index: Full graph or subgraph edge index [2, num_edges]
        selected_nodes: List of selected node IDs (absolute node IDs)
        max_size: Maximum size for normalization (default: 20, matches args.max_size)
        
    Returns:
        tuple: (edge_index, mask) where mask is [num_edges] with scores in [0, 1]
    """
    n_edges = edge_index.shape[1]
    
    # Create edge index lookup (lines 659-662 of run_RG_explainer.py)
    n_edges_index = {}
    for i in range(n_edges):
        n_edges_index[(int(edge_index[0, i]), int(edge_index[1, i]))] = i
    
    # Initialize ranks to 0 (line 664 of run_RG_explainer.py)
    rank = np.zeros(n_edges)
    
    # Assign ranks based on node selection order (lines 666-673 of run_RG_explainer.py)
    for i in range(len(selected_nodes)):
        for j in range(i):
            node_i = int(selected_nodes[i])
            node_j = int(selected_nodes[j])
            
            # Check if BOTH directions exist (lines 668-670 of run_RG_explainer.py)
            if (node_i, node_j) in n_edges_index and (node_j, node_i) in n_edges_index:
                rank[n_edges_index[(node_i, node_j)]] = max_size - i
                rank[n_edges_index[(node_j, node_i)]] = max_size - i
            elif (node_i, node_j) in n_edges_index or (node_j, node_i) in n_edges_index:
                # One direction exists but not both (line 672-673 edge pair error case)
                # Handle gracefully instead of printing error
                if (node_i, node_j) in n_edges_index:
                    rank[n_edges_index[(node_i, node_j)]] = max_size - i
                if (node_j, node_i) in n_edges_index:
                    rank[n_edges_index[(node_j, node_i)]] = max_size - i
    
    # Normalize to [0, 1] (line 675 of run_RG_explainer.py)
    mask = rank / (max_size - 1) if max_size > 1 else rank
    
    return edge_index, mask


def compute_auc_from_mask_node(edge_index, mask, gt_edge_index, gt_edge_labels):
    """
    Compute AUC for NODE tasks using evaluation/utils.py evaluation_auc_node logic.
    
    This function replicates the exact logic from evaluation/utils.py (lines 61-96)
    to match the evaluation code structure.
    
    CRITICAL DIFFERENCE FROM GRAPH TASKS:
    - NODE tasks loop over EXPLANATION edges (lines 77-90)
    - GRAPH tasks loop over GROUND TRUTH edges (lines 47-55)
    
    Why? For node tasks, the explanation is a k-hop SUBGRAPH which only contains a subset
    of edges. Many ground truth edges won't be in the subgraph, so we can't loop over GT edges.
    Instead, we loop over explanation edges and check if each one is in the ground truth.
    
    Args:
        edge_index: Subgraph edge index [2, num_subgraph_edges]
        mask: Edge scores [num_subgraph_edges]
        gt_edge_index: Full graph ground truth edge index [2, num_gt_edges]
        gt_edge_labels: Ground truth labels [num_gt_edges] (1=motif, 0=non-motif)
        
    Returns:
        float: AUC score
    """
    import torch
    
    # Convert to torch tensors to match evaluation code format
    edge_index_torch = torch.from_numpy(edge_index)
    mask_torch = torch.from_numpy(mask)
    
    # Format explanation as expected by evaluation_auc_node
    # Each explanation is (edge_index, mask) where mask is tensor
    explanation = (edge_index_torch, mask_torch)
    explanations = [explanation]
    
    # Format ground truth as expected by evaluation_auc_node
    explanation_labels = (gt_edge_index, gt_edge_labels)
    
    # Use evaluation function directly (matches evaluation/utils.py lines 61-96)
    if USE_EVALUATION_FUNCTIONS:
        try:
            auc = evaluation_auc_node(explanations, explanation_labels)
            return auc
        except (IndexError, RuntimeError, AttributeError) as e:
            # Fallback if there's an issue with the evaluation function
            # (e.g., edge not found, tensor issues, etc.)
            pass
    
    # Fallback: manual calculation matching evaluation logic exactly
    ground_truth = []
    predictions = []
    
    # Loop over all edges in the explanation subgraph (line 77 of evaluation/utils.py)
    for i in range(edge_index.shape[1]):
        # Get prediction score for this edge (line 78 of evaluation/utils.py)
        predictions.append(float(mask[i]))
        
        # Get the edge pair (line 81 of evaluation/utils.py)
        pair = edge_index.T[i]
        
        # Check both directions (lines 82-83 of evaluation/utils.py)
        idx_edge = np.where((gt_edge_index.T == pair).all(axis=1))[0]
        idx_edge_rev = np.where((gt_edge_index.T == [pair[1], pair[0]]).all(axis=1))[0]
        
        # If edge exists in ground truth (either direction), label as 1 (lines 86-90)
        gt = gt_edge_labels[idx_edge].sum() if len(idx_edge) > 0 else 0
        gt += gt_edge_labels[idx_edge_rev].sum() if len(idx_edge_rev) > 0 else 0
        
        if gt == 0:
            ground_truth.append(0)
        else:
            ground_truth.append(1)
    
    # Compute AUC (line 95 of evaluation/utils.py)
    if len(np.unique(ground_truth)) > 1:
        auc = compute_auc_roc(ground_truth, predictions)
    else:
        auc = 0.5  # Only one class present
    
    return auc


def compute_auc_from_mask_graph(edge_index, mask, gt_edge_index, gt_edge_labels):
    """
    Compute AUC for GRAPH tasks using evaluation/utils.py evaluation_auc_graph logic.
    
    This function replicates the exact logic from evaluation/utils.py (lines 24-58)
    to match the evaluation code structure.
    
    For graph tasks, we loop over GROUND TRUTH edges and find them in the explanation.
    This is used for ba2, mutag, etc.
    
    Args:
        edge_index: Full graph edge index [2, num_edges] (called 'graph' in evaluation)
        mask: Edge scores [num_edges] (called 'mask' in evaluation)
        gt_edge_index: Ground truth edge index [2, num_gt_edges] (called 'edge_list' in evaluation)
        gt_edge_labels: Ground truth labels [num_gt_edges] (1=motif, 0=non-motif)
        
    Returns:
        float: AUC score
    """
    import torch
    
    # Convert to torch tensors to match evaluation code format
    edge_index_torch = torch.from_numpy(edge_index)
    mask_torch = torch.from_numpy(mask)
    
    # Format explanation as expected by evaluation_auc_graph
    # Each explanation is (edge_index, mask) where mask is tensor
    explanation = (edge_index_torch, mask_torch)
    explanations = [explanation]
    
    # Format ground truth as expected by evaluation_auc_graph
    # explanation_labels[0] is list of edge indices per graph
    # explanation_labels[1] is list of edge labels per graph
    explanation_labels = ([gt_edge_index], [gt_edge_labels])
    
    # Use evaluation function directly (matches evaluation/utils.py lines 24-58)
    # indices=[0] means we evaluate graph index 0
    if USE_EVALUATION_FUNCTIONS:
        try:
            auc = evaluation_auc_graph(explanations, explanation_labels, indices=[0])
            return auc
        except (IndexError, RuntimeError, AttributeError) as e:
            # Fallback if there's an issue with the evaluation function
            # (e.g., edge not found, tensor issues, etc.)
            pass
    
    # Fallback: manual calculation matching evaluation logic exactly
    ground_truth = []
    predictions = []
    
    # For each ground truth edge (exactly as in evaluation/utils.py lines 47-55)
    for edge_idx in range(gt_edge_labels.shape[0]):
        edge_ = gt_edge_index.T[edge_idx]
        
        # Skip self-loops (line 49-50 of evaluation/utils.py)
        if edge_[0] == edge_[1]:
            continue
        
        # Find this edge in the explanation edge list (line 51 of evaluation/utils.py)
        t = np.where((edge_index.T == edge_.T).all(axis=1))
        
        # Check if edge found (evaluation code assumes it exists, but handle gracefully)
        if len(t[0]) == 0:
            continue  # Skip if edge not found in explanation
        
        # Retrieve predictions and ground truth (lines 54-55 of evaluation/utils.py)
        # Note: evaluation code does mask[t][0] where t is tuple from np.where
        # mask[t] works when mask is numpy array and t is tuple of indices
        predictions.append(float(mask[t[0][0]]))
        ground_truth.append(int(gt_edge_labels[edge_idx]))
    
    # Compute AUC using manual calculation (avoids sklearn/scipy library issues on HPC)
    if len(ground_truth) == 0:
        auc = 0.5  # No edges to evaluate
    elif len(np.unique(ground_truth)) > 1:
        auc = compute_auc_roc(ground_truth, predictions)
    else:
        auc = 0.5  # Only one class present
    
    return auc


def extract_logged_auc(filepath):
    """
    Extract the AUC value logged in the results file for the last epoch.
    
    Args:
        filepath: Path to results file
        
    Returns:
        float: Logged AUC value or None if not found
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find all EVAL lines with AUC
    auc_lines = []
    for line in lines:
        if '[EVAL-Epoch' in line and 'AUC=' in line:
            match = re.search(r'AUC=([\d.]+)', line)
            if match:
                auc_lines.append(float(match.group(1)))
    
    # Return the last one (most recent epoch)
    return auc_lines[-1] if auc_lines else None


def get_khop_subgraph(seed_node, edge_index, n_hop=3):
    """
    Extract k-hop subgraph around a seed node.
    
    Args:
        seed_node: Central node ID
        edge_index: Full graph edge index [2, num_edges]
        n_hop: Number of hops (default: 3)
        
    Returns:
        numpy array: Subgraph edge index [2, num_subgraph_edges]
    """
    import torch
    import torch_geometric as ptgeom
    
    # Convert to torch tensors
    seed_tensor = torch.LongTensor([seed_node])
    edge_index_tensor = torch.LongTensor(edge_index)
    
    # Extract k-hop subgraph
    _, subgraph_edge_index, *_ = ptgeom.utils.k_hop_subgraph(
        seed_tensor, n_hop, edge_index_tensor
    )
    
    return subgraph_edge_index.numpy()


def compute_all_graph_aucs(graphs_data, gt_edge_indices, gt_edge_labels, dataset_name, max_size=20, n_hop=3):
    """
    Compute AUC for all graphs in the dataset using the exact evaluation method.
    
    Args:
        graphs_data: Dict of parsed graph data from last epoch
        gt_edge_indices: Ground truth edge indices
        gt_edge_labels: Ground truth edge labels
        dataset_name: Name of dataset
        max_size: Maximum size for edge score normalization
        n_hop: Number of hops for subgraph extraction (node tasks only)
        
    Returns:
        dict: {graph_id: auc_score}
    """
    auc_scores = {}
    is_node_task = dataset_name in ['syn1', 'syn2', 'syn3', 'syn4']
    
    print(f"\nComputing AUC for {len(graphs_data)} {'nodes' if is_node_task else 'graphs'}...")
    
    # Load full graph structure once for node tasks
    if is_node_task:
        full_edge_index = load_graph_structure(dataset_name, None)
    
    for idx, (node_or_graph_id, data) in enumerate(sorted(graphs_data.items())):
        sgs = data['sgs']
        n_id = data['n_id']  # Seed node ID
        
        if is_node_task:
            # NODE TASK EVALUATION (like syn1-syn4)
            # In the log file, sgs are printed as [x - seed_node for x in pred_sgs]
            # So sgs contains RELATIVE node IDs, and n_id is the seed node
            seed_node = n_id
            
            # Convert relative node IDs to absolute
            selected_nodes = [seed_node + int(rel_id) for rel_id in sgs]
            
            # Extract k-hop subgraph around seed node (as done in run_RG_explainer.py line 763)
            subgraph_edge_index = get_khop_subgraph(seed_node, full_edge_index, n_hop)
            
            # Prepare mask using the subgraph edge index and absolute selected nodes
            _, mask = prepare_mask(subgraph_edge_index, selected_nodes, max_size)
            
            # For node tasks, use the same ground truth for all nodes
            gt_edge_idx = gt_edge_indices
            gt_labels = gt_edge_labels
            
            # Compute AUC using NODE task method (loops over explanation edges)
            auc = compute_auc_from_mask_node(subgraph_edge_index, mask, gt_edge_idx, gt_labels)
            
        else:
            # GRAPH TASK EVALUATION (like ba2, mutag)
            # sgs contains node IDs relative to the graph's node offset
            # We need to load this specific graph's structure
            graph_id = node_or_graph_id
            edge_index = load_graph_structure(dataset_name, graph_id)
            
            # selected_nodes are already in the correct coordinate system for this graph
            selected_nodes = [int(n) for n in sgs]
            
            # Prepare mask using exact method from run_RG_explainer.py
            _, mask = prepare_mask(edge_index, selected_nodes, max_size)
            
            # Get ground truth for this specific graph
            gt_edge_idx = gt_edge_indices[graph_id]
            gt_labels = gt_edge_labels[graph_id]
            
            # Compute AUC using GRAPH task method (loops over ground truth edges)
            auc = compute_auc_from_mask_graph(edge_index, mask, gt_edge_idx, gt_labels)
        
        auc_scores[node_or_graph_id] = auc
        
        # Print progress every 50 items
        if ((idx + 1) % 50 == 0) or ((idx + 1) == len(graphs_data)):
            print(f"  Processed {idx + 1}/{len(graphs_data)} {'nodes' if is_node_task else 'graphs'}...")
    
    return auc_scores


def save_results_to_json(auc_scores, dataset_name, seed, causal, logged_auc, output_path):
    """
    Save AUC results to JSON file with statistics.
    
    Args:
        auc_scores: Dict of {node_id/graph_id: auc_score}
        dataset_name: Name of dataset
        seed: Random seed
        causal: Whether causal mode was used
        logged_auc: AUC value logged in the results file
        output_path: Path to save JSON file
    """
    # Compute statistics
    auc_values = list(auc_scores.values())
    mean_auc = np.mean(auc_values)
    std_auc = np.std(auc_values)
    min_auc = np.min(auc_values)
    max_auc = np.max(auc_values)
    median_auc = np.median(auc_values)
    
    # Count performance levels
    low_perf = sum(1 for auc in auc_values if auc < 0.5)
    medium_perf = sum(1 for auc in auc_values if 0.5 <= auc < 0.8)
    high_perf = sum(1 for auc in auc_values if auc >= 0.8)
    
    is_node_task = dataset_name in ['syn1', 'syn2', 'syn3', 'syn4']
    
    # Create results dictionary
    results = {
        'dataset': dataset_name,
        'task_type': 'node_task' if is_node_task else 'graph_task',
        'seed': seed,
        'causal': causal,
        'num_items': len(auc_scores),
        'num_graphs': len(auc_scores),  # Keep for backward compatibility
        'statistics': {
            'mean_auc': float(mean_auc),
            'std_auc': float(std_auc),
            'min_auc': float(min_auc),
            'max_auc': float(max_auc),
            'median_auc': float(median_auc),
            'logged_auc': float(logged_auc) if logged_auc is not None else None,
            'auc_difference': float(abs(mean_auc - logged_auc)) if logged_auc is not None else None
        },
        'performance_distribution': {
            'low_performance_count': low_perf,  # AUC < 0.5
            'medium_performance_count': medium_perf,  # 0.5 <= AUC < 0.8
            'high_performance_count': high_perf  # AUC >= 0.8
        },
        'per_item_auc': {str(k): float(v) for k, v in sorted(auc_scores.items())},
        'per_graph_auc': {str(k): float(v) for k, v in sorted(auc_scores.items())}  # Keep for backward compatibility
    }
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    return results


def create_scatter_plot(auc_scores, dataset_name, seed, causal, results, output_path):
    """
    Create a histogram showing AUC distribution across graphs.
    
    Args:
        auc_scores: Dict of {graph_id: auc_score}
        dataset_name: Name of dataset
        seed: Random seed
        causal: Whether causal mode was used
        results: Full results dictionary with statistics and performance_distribution
        output_path: Path to save plot
    """
    statistics = results['statistics']
    performance_dist = results['performance_distribution']
    
    # Prepare data
    auc_values = list(auc_scores.values())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define bins for histogram
    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
    
    # Color code by performance level - create separate histograms for each category
    low_auc = [a for a in auc_values if a < 0.5]
    medium_auc = [a for a in auc_values if 0.5 <= a < 0.8]
    high_auc = [a for a in auc_values if a >= 0.8]
    
    # Create histogram with stacked colors
    ax.hist([low_auc, medium_auc, high_auc], bins=bins, 
            color=['#FF4444', '#FFB84D', '#44AA44'], 
            alpha=0.7, edgecolor='black', linewidth=1.2,
            label=[
                f'Low (<0.5): {performance_dist["low_performance_count"]}',
                f'Medium (0.5-0.8): {performance_dist["medium_performance_count"]}',
                f'High (≥0.8): {performance_dist["high_performance_count"]}'
            ],
            stacked=False)
    
    # Add mean line
    mean_auc = statistics['mean_auc']
    ax.axvline(x=mean_auc, color='blue', linestyle='--', linewidth=2, 
               label=f'Mean AUC: {mean_auc:.4f}', alpha=0.8)
    
    # Add median line
    median_auc = statistics['median_auc']
    ax.axvline(x=median_auc, color='cyan', linestyle='--', linewidth=2, 
               label=f'Median AUC: {median_auc:.4f}', alpha=0.8)
    
    # Add logged AUC line if available
    logged_auc = statistics.get('logged_auc')
    if logged_auc is not None:
        ax.axvline(x=logged_auc, color='purple', linestyle=':', linewidth=2, 
                   label=f'Logged AUC: {logged_auc:.4f}', alpha=0.8)
    
    # Add performance threshold lines
    ax.axvline(x=0.5, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(x=0.8, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Labels and title
    is_node_task = dataset_name in ['syn1', 'syn2', 'syn3', 'syn4']
    
    ax.set_xlabel('AUC Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    
    mode_str = 'Causal' if causal else 'Standard'
    task_str = 'Per-Node' if is_node_task else 'Per-Graph'
    title = f'{dataset_name.upper()} - {task_str} AUC Distribution ({mode_str})\nSeed: {seed}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add statistics text box
    item_type = 'nodes' if is_node_task else 'graphs'
    stats_text = (
        f"Statistics:\n"
        f"  Mean: {statistics['mean_auc']:.4f}\n"
        f"  Std: {statistics['std_auc']:.4f}\n"
        f"  Min: {statistics['min_auc']:.4f}\n"
        f"  Max: {statistics['max_auc']:.4f}\n"
        f"  Median: {statistics['median_auc']:.4f}\n"
        f"  Total {item_type}: {len(auc_values)}\n"
    )
    
    if logged_auc is not None:
        auc_diff = statistics.get('auc_difference', 0)
        stats_text += f"  Logged vs Mean Diff: {auc_diff:.6f}\n"
    
    # Add text box in upper right
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props,
            family='monospace')
    
    # Add legend
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Set x-axis limits
    ax.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    plt.close()


def main():
    """
    Main function for per-graph/per-node AUC analysis.
    
    Key differences between node and graph tasks:
    
    NODE TASKS (syn1-syn4):
    - sgs in log file are RELATIVE to seed node: [x - seed_node for x in pred_sgs]
    - Uses k-hop subgraph around seed node (not full graph)
    - prepare_mask is called with subgraph_edge_index
    - n_id in log is the seed node (first node in sequence)
    
    GRAPH TASKS (ba2, mutag):
    - sgs in log file are graph-relative node IDs
    - Uses full graph structure for that specific graph
    - prepare_mask is called with full graph edge_index
    - g_id in log identifies the graph, n_id is relative seed in that graph
    """
    parser = argparse.ArgumentParser(
        description='Compute per-graph/per-node AUC from last epoch and create scatter plot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Graph tasks
  python analysis/plot_auc_per_graph.py --dataset ba2 --seed 4
  python analysis/plot_auc_per_graph.py --dataset mutag --seed 2
  
  # Node tasks
  python analysis/plot_auc_per_graph.py --dataset syn1 --seed 0 --causal --n_hop 3
  python analysis/plot_auc_per_graph.py --dataset syn4 --seed 1 --max_size 20
        """
    )
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (ba2, mutag, syn1, syn2, syn3, syn4)')
    parser.add_argument('--seed', type=int, required=True,
                       help='Random seed used for training')
    parser.add_argument('--causal', action='store_true',
                       help='Use causal results file')
    parser.add_argument('--max_size', type=int, default=20,
                       help='Maximum size for edge score normalization (default: 20)')
    parser.add_argument('--n_hop', type=int, default=3,
                       help='Number of hops for subgraph extraction in node tasks (default: 3)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Per-Graph AUC Analysis Tool")
    print("="*70)
    print(f"Dataset:      {args.dataset}")
    print(f"Seed:         {args.seed}")
    print(f"Causal mode:  {args.causal}")
    print("="*70)
    
    # 1. Determine file paths
    if args.causal:
        results_file = f'results/{args.dataset}_ppo_causal_seed_{args.seed}.txt'
        json_output = f'analysis/jsons/{args.dataset}_ppo_causal_seed_{args.seed}_auc_per_graph.json'
        plot_output = f'analysis/plots/{args.dataset}_ppo_causal_seed_{args.seed}_auc_scatter.png'
    else:
        results_file = f'results/{args.dataset}_ppo_seed_{args.seed}.txt'
        json_output = f'analysis/jsons/{args.dataset}_ppo_seed_{args.seed}_auc_per_graph.json'
        plot_output = f'analysis/plots/{args.dataset}_ppo_seed_{args.seed}_auc_scatter.png'
    
    # Check if results file exists
    if not os.path.exists(results_file):
        print(f"\n❌ Error: Results file not found: {results_file}")
        return
    
    print(f"\nResults file: {results_file}")
    
    # 2. Parse results file to get sgs from last epoch
    print("\n" + "-"*70)
    print("Step 1: Parsing results file...")
    print("-"*70)
    graphs_data = parse_results_file(results_file, args.dataset)
    print(f"✓ Parsed {len(graphs_data)} graphs from last epoch")
    
    # 3. Extract logged AUC from results file
    logged_auc = extract_logged_auc(results_file)
    if logged_auc is not None:
        print(f"✓ Logged AUC from file: {logged_auc:.6f}")
    else:
        print("⚠ Warning: Could not extract logged AUC from file")
    
    # 4. Load ground truth
    print("\n" + "-"*70)
    print("Step 2: Loading ground truth...")
    print("-"*70)
    gt_edge_indices, gt_edge_labels = load_ground_truth(args.dataset)
    
    is_node_task = args.dataset in ['syn1', 'syn2', 'syn3', 'syn4']
    if is_node_task:
        print(f"✓ Loaded ground truth for node classification")
    else:
        print(f"✓ Loaded ground truth for {len(gt_edge_labels)} graphs")
    
    # 5. Compute AUC for all graphs
    print("\n" + "-"*70)
    print("Step 3: Computing AUC for all graphs/nodes...")
    print("-"*70)
    print(f"Using max_size={args.max_size} for edge score normalization")
    if args.dataset in ['syn1', 'syn2', 'syn3', 'syn4']:
        print(f"Using n_hop={args.n_hop} for subgraph extraction (node task)")
    auc_scores = compute_all_graph_aucs(graphs_data, gt_edge_indices, 
                                        gt_edge_labels, args.dataset, args.max_size, args.n_hop)
    print(f"✓ Computed AUC for {len(auc_scores)} {'nodes' if args.dataset in ['syn1', 'syn2', 'syn3', 'syn4'] else 'graphs'}")
    
    # 6. Save results to JSON
    print("\n" + "-"*70)
    print("Step 4: Saving results...")
    print("-"*70)
    results = save_results_to_json(auc_scores, args.dataset, args.seed, 
                                   args.causal, logged_auc, json_output)
    
    # 7. Print statistics
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    stats = results['statistics']
    item_type = "nodes" if args.dataset in ['syn1', 'syn2', 'syn3', 'syn4'] else "graphs"
    print(f"Number of {item_type}:  {results['num_graphs']}")
    print(f"Mean AUC:          {stats['mean_auc']:.6f}")
    print(f"Std AUC:           {stats['std_auc']:.6f}")
    print(f"Min AUC:           {stats['min_auc']:.6f}")
    print(f"Max AUC:           {stats['max_auc']:.6f}")
    print(f"Median AUC:        {stats['median_auc']:.6f}")
    
    if logged_auc is not None:
        print(f"\nLogged AUC:        {stats['logged_auc']:.6f}")
        print(f"Mean vs Logged:    {stats['auc_difference']:.6f}")
        
        # Check if they match (within small tolerance)
        if stats['auc_difference'] < 1e-4:
            print("✓ Mean AUC matches logged AUC!")
        elif stats['auc_difference'] < 1e-3:
            print("⚠ Mean AUC approximately matches logged AUC (small difference)")
        else:
            print("⚠ Warning: Mean AUC differs from logged AUC")
    
    print(f"\nPerformance Distribution:")
    perf = results['performance_distribution']
    print(f"  High (≥0.8):      {perf['high_performance_count']} {item_type}")
    print(f"  Medium (0.5-0.8): {perf['medium_performance_count']} {item_type}")
    print(f"  Low (<0.5):       {perf['low_performance_count']} {item_type}")
    
    # 8. Create histogram plot
    print("\n" + "-"*70)
    print("Step 5: Creating histogram plot...")
    print("-"*70)
    create_scatter_plot(auc_scores, args.dataset, args.seed, args.causal, 
                       results, plot_output)
    
    print("\n" + "="*70)
    print("✓ Analysis complete!")
    print("="*70)


if __name__ == '__main__':
    main()

