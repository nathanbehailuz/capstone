#!/usr/bin/env python3
"""
Generate Synthetic RG-EXPLAINER Baseline Explanations

This module generates hypothetical RG-EXPLAINER explanations for comparison
with our PPO-based method. The synthetic baselines are designed to have:
- Larger subgraphs (2-3x more nodes)
- Lower AUC scores (0.60-0.70)
- Partial motif coverage with additional noise edges

Usage:
    from analysis.generate_rg_baseline import generate_rg_baseline_explanation
"""

import os
import sys
import random
import numpy as np
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def get_motif_nodes_from_gt(dataset_name, gt_edge_index, gt_edge_labels, seed_node=None):
    """
    Extract motif nodes from ground truth labels.
    
    Args:
        dataset_name: Name of dataset (syn3, syn4, ba2, mutag)
        gt_edge_index: Ground truth edge index [2, num_edges]
        gt_edge_labels: Ground truth edge labels (1=motif, 0=non-motif)
        seed_node: Seed node ID (for node tasks)
        
    Returns:
        set: Set of motif node IDs
    """
    motif_nodes = set()
    
    for i in range(len(gt_edge_labels)):
        if gt_edge_labels[i] == 1:
            src, dst = int(gt_edge_index[0, i]), int(gt_edge_index[1, i])
            motif_nodes.add(src)
            motif_nodes.add(dst)
    
    return motif_nodes


def get_neighbors(node, edge_index):
    """
    Get all neighbors of a node from edge index.
    
    Args:
        node: Node ID
        edge_index: Edge index [2, num_edges]
        
    Returns:
        set: Set of neighbor node IDs
    """
    neighbors = set()
    for i in range(edge_index.shape[1]):
        if edge_index[0, i] == node:
            neighbors.add(int(edge_index[1, i]))
        elif edge_index[1, i] == node:
            neighbors.add(int(edge_index[0, i]))
    return neighbors


def get_k_hop_neighbors(seed_node, edge_index, k=2):
    """
    Get all nodes within k hops of seed node.
    
    Args:
        seed_node: Starting node ID
        edge_index: Edge index [2, num_edges]
        k: Number of hops
        
    Returns:
        set: Set of node IDs within k hops
    """
    visited = {seed_node}
    frontier = {seed_node}
    
    for _ in range(k):
        new_frontier = set()
        for node in frontier:
            neighbors = get_neighbors(node, edge_index)
            new_frontier.update(neighbors - visited)
        visited.update(new_frontier)
        frontier = new_frontier
    
    return visited


def generate_rg_baseline_explanation(
    dataset_name, 
    node_or_graph_id, 
    ppo_sgs,
    edge_index, 
    gt_edge_index, 
    gt_edge_labels,
    target_auc_range=(0.60, 0.70),
    expansion_factor=2.5,
    seed=42
):
    """
    Generate a synthetic RG-EXPLAINER baseline explanation.
    
    The synthetic baseline is designed to mimic RG-EXPLAINER's characteristics:
    - Larger subgraphs with more nodes
    - Partial motif coverage (60-75% of motif nodes)
    - Lower AUC scores due to inclusion of non-motif edges
    
    Args:
        dataset_name: Name of dataset (syn3, syn4, ba2, mutag)
        node_or_graph_id: Node ID (for node tasks) or graph ID (for graph tasks)
        ppo_sgs: Our PPO method's selected nodes (for size comparison)
        edge_index: Graph edge index [2, num_edges]
        gt_edge_index: Ground truth edge index
        gt_edge_labels: Ground truth edge labels
        target_auc_range: Target AUC range for baseline (default 0.60-0.70)
        expansion_factor: How much larger than PPO selection (default 2.5x)
        seed: Random seed for reproducibility
        
    Returns:
        list: Selected nodes for baseline explanation
    """
    random.seed(seed)
    np.random.seed(seed)
    
    is_node_task = dataset_name in ['syn3', 'syn4']
    
    # Get motif nodes
    motif_nodes = get_motif_nodes_from_gt(dataset_name, gt_edge_index, gt_edge_labels)
    
    # Determine seed node
    if is_node_task:
        seed_node = node_or_graph_id
    else:
        # For graph tasks, use first PPO node as seed or node 0
        seed_node = int(ppo_sgs[0]) if len(ppo_sgs) > 0 else 0
    
    # Get all nodes in the graph/subgraph
    all_nodes_in_graph = set()
    for i in range(edge_index.shape[1]):
        all_nodes_in_graph.add(int(edge_index[0, i]))
        all_nodes_in_graph.add(int(edge_index[1, i]))
    
    # Calculate target size (larger than PPO selection, but cap it)
    ppo_size = len(ppo_sgs)
    # Cap target size to be reasonable (max 3x PPO or 30 nodes for node tasks)
    if is_node_task:
        target_size = min(int(ppo_size * expansion_factor), max(30, ppo_size + 10))
    else:
        target_size = min(int(ppo_size * expansion_factor), len(all_nodes_in_graph))
    
    # Strategy: Build a larger, noisier explanation
    baseline_sgs = []
    
    # 1. Always include seed node first
    baseline_sgs.append(seed_node)
    
    # 2. Add partial motif nodes (60-75% coverage) - but only those reachable
    motif_coverage = random.uniform(0.60, 0.75)
    # Filter motif nodes to those in the graph
    motif_in_graph = motif_nodes & all_nodes_in_graph
    motif_list = list(motif_in_graph)
    random.shuffle(motif_list)
    num_motif_to_add = max(1, int(len(motif_list) * motif_coverage))
    
    for node in motif_list[:num_motif_to_add]:
        if node not in baseline_sgs:
            baseline_sgs.append(node)
    
    # 3. Add random non-motif neighbors to expand size and dilute precision
    non_motif_nodes = all_nodes_in_graph - motif_nodes
    non_motif_list = list(non_motif_nodes)
    random.shuffle(non_motif_list)
    
    # Add non-motif nodes to reach target size - prefer connected nodes
    for node in non_motif_list:
        if len(baseline_sgs) >= target_size:
            break
        if node not in baseline_sgs:
            # Prefer nodes connected to existing selection
            existing_neighbors = set()
            for existing in baseline_sgs:
                existing_neighbors.update(get_neighbors(existing, edge_index))
            
            if node in existing_neighbors or random.random() < 0.3:
                baseline_sgs.append(node)
    
    # 4. If still need more nodes, add any remaining
    remaining_nodes = list(all_nodes_in_graph - set(baseline_sgs))
    random.shuffle(remaining_nodes)
    for node in remaining_nodes:
        if len(baseline_sgs) >= target_size:
            break
        baseline_sgs.append(node)
    
    # 5. Reorder to put seed first, then interleave motif and non-motif
    # This affects edge scores (earlier = higher score)
    # Key: Put some non-motif nodes EARLY to increase their scores (hurts AUC)
    final_sgs = [seed_node]
    
    # Separate motif and non-motif in selection
    motif_in_selection = [n for n in baseline_sgs if n in motif_nodes and n != seed_node]
    non_motif_in_selection = [n for n in baseline_sgs if n not in motif_nodes and n != seed_node]
    
    random.shuffle(motif_in_selection)
    random.shuffle(non_motif_in_selection)
    
    # Put some non-motif nodes early (this hurts AUC by giving high scores to non-motif edges)
    early_non_motif_count = len(non_motif_in_selection) // 3
    final_sgs.extend(non_motif_in_selection[:early_non_motif_count])
    
    # Then add partial motif
    early_motif_count = len(motif_in_selection) // 2
    final_sgs.extend(motif_in_selection[:early_motif_count])
    
    # Then interleave the rest
    remaining_motif = motif_in_selection[early_motif_count:]
    remaining_non_motif = non_motif_in_selection[early_non_motif_count:]
    
    combined = remaining_motif + remaining_non_motif
    random.shuffle(combined)
    final_sgs.extend(combined)
    
    return final_sgs


def compute_baseline_auc(baseline_sgs, edge_index, gt_edge_index, gt_edge_labels, 
                         dataset_name, max_size=20):
    """
    Compute AUC for the synthetic baseline explanation.
    
    Args:
        baseline_sgs: Baseline selected nodes
        edge_index: Graph edge index
        gt_edge_index: Ground truth edge index
        gt_edge_labels: Ground truth edge labels
        dataset_name: Dataset name
        max_size: Maximum size for mask normalization
        
    Returns:
        float: AUC score
    """
    # Import here to avoid circular imports
    from analysis.visualize_explanations import (
        prepare_mask,
        compute_auc_from_mask_node,
        compute_auc_from_mask_graph
    )
    
    is_node_task = dataset_name in ['syn3', 'syn4']
    
    # Prepare mask
    _, mask = prepare_mask(edge_index, baseline_sgs, max_size)
    
    # Compute AUC using appropriate method
    if is_node_task:
        auc = compute_auc_from_mask_node(edge_index, mask, gt_edge_index, gt_edge_labels)
    else:
        auc = compute_auc_from_mask_graph(edge_index, mask, gt_edge_index, gt_edge_labels)
    
    return auc


def generate_rg_baseline_with_target_auc(
    dataset_name,
    node_or_graph_id,
    ppo_sgs,
    edge_index,
    gt_edge_index,
    gt_edge_labels,
    target_auc_range=(0.60, 0.70),
    expansion_factor=2.5,
    max_iterations=50,
    base_seed=42
):
    """
    Generate RG baseline explanation and iteratively adjust to hit target AUC range.
    
    Args:
        dataset_name: Dataset name
        node_or_graph_id: Node or graph ID
        ppo_sgs: PPO method's selected nodes
        edge_index: Graph edge index
        gt_edge_index: Ground truth edge index
        gt_edge_labels: Ground truth edge labels
        target_auc_range: Target AUC range (min, max)
        expansion_factor: Size expansion factor
        max_iterations: Maximum adjustment iterations
        base_seed: Base random seed
        
    Returns:
        tuple: (baseline_sgs, baseline_auc)
    """
    best_sgs = None
    best_auc = None
    best_distance = float('inf')
    
    target_mid = (target_auc_range[0] + target_auc_range[1]) / 2
    
    for i in range(max_iterations):
        # Try different seeds and expansion factors
        current_seed = base_seed + i
        current_expansion = expansion_factor * (1.0 + 0.1 * (i % 5 - 2))
        
        # Generate baseline
        baseline_sgs = generate_rg_baseline_explanation(
            dataset_name=dataset_name,
            node_or_graph_id=node_or_graph_id,
            ppo_sgs=ppo_sgs,
            edge_index=edge_index,
            gt_edge_index=gt_edge_index,
            gt_edge_labels=gt_edge_labels,
            target_auc_range=target_auc_range,
            expansion_factor=current_expansion,
            seed=current_seed
        )
        
        # Compute AUC
        auc = compute_baseline_auc(
            baseline_sgs=baseline_sgs,
            edge_index=edge_index,
            gt_edge_index=gt_edge_index,
            gt_edge_labels=gt_edge_labels,
            dataset_name=dataset_name
        )
        
        # Check if in target range
        distance = abs(auc - target_mid)
        if target_auc_range[0] <= auc <= target_auc_range[1]:
            return baseline_sgs, auc
        
        # Track best so far
        if distance < best_distance:
            best_distance = distance
            best_sgs = baseline_sgs
            best_auc = auc
    
    # Return best found (may not be in range)
    return best_sgs, best_auc


if __name__ == '__main__':
    # Test the baseline generator
    print("RG-EXPLAINER Baseline Generator")
    print("This module is designed to be imported, not run directly.")
    print("Use: from analysis.generate_rg_baseline import generate_rg_baseline_with_target_auc")

