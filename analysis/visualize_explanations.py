#!/usr/bin/env python3
"""
Graph Explanation Visualization Tool
Visualizes RL-generated graph explanations compared to ground truth motifs.

Usage:
    python analysis/visualize_explanations.py --dataset ba2 --seed 4 [--graph_id 0]
"""

import os
# Limit OpenBLAS threads before importing numpy
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import re
import pickle as pkl
import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import networkx as nx
from collections import defaultdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets.ground_truth_loaders import load_ba2_ground_truth, load_mutag_ground_truth, _load_node_dataset_ground_truth
from datasets.dataset_loaders import load_graph_dataset, _load_node_dataset


# ============================================================
# Motif Layout Functions
# ============================================================

def get_house_layout(nodes):
    """
    Create house shape layout: triangle roof on top of a square base.
    5 nodes total.
    
    Structure:
         4
        / \
       2   3
       |   |
       0---1
    
    Args:
        nodes: List of 5 node IDs
        
    Returns:
        dict: {node_id: (x, y)} positions
    """
    if len(nodes) < 5:
        return {}
    
    nodes_list = list(nodes)[:5]
    
    # Traditional house: 2 bottom, 2 middle, 1 top
    base_width = 2.0
    base_height = 1.5
    roof_height = 1.5
    
    return {
        nodes_list[0]: (0.0, 0.0),                                  # Bottom left
        nodes_list[1]: (base_width, 0.0),                           # Bottom right
        nodes_list[2]: (0.0, base_height),                          # Middle left (wall)
        nodes_list[3]: (base_width, base_height),                   # Middle right (wall)
        nodes_list[4]: (base_width/2, base_height + roof_height),  # Top (roof peak)
    }


def get_hexagon_layout(nodes):
    """
    Create regular hexagon layout.
    6 nodes arranged in a hexagon.
    
    Args:
        nodes: List of 6 node IDs
        
    Returns:
        dict: {node_id: (x, y)} positions
    """
    if len(nodes) < 6:
        return {}
    
    import math
    nodes_list = list(nodes)[:6]
    pos = {}
    
    for i, node in enumerate(nodes_list):
        angle = i * math.pi / 3  # 60 degrees in radians
        x = math.cos(angle)
        y = math.sin(angle)
        pos[node] = (x, y)
    
    return pos


def get_grid_3x3_layout(nodes):
    """
    Create 3x3 grid layout.
    9 nodes arranged in a grid.
    
    Structure:
       6 7 8
       3 4 5
       0 1 2
    
    Args:
        nodes: List of 9 node IDs
        
    Returns:
        dict: {node_id: (x, y)} positions
    """
    if len(nodes) < 9:
        return {}
    
    nodes_list = list(nodes)[:9]
    pos = {}
    
    for i, node in enumerate(nodes_list):
        x = i % 3
        y = i // 3
        pos[node] = (float(x), float(y))
    
    return pos


def get_cycle_layout(nodes):
    """
    Create circular (cycle) layout.
    5 nodes arranged in a circle (pentagon).
    
    Args:
        nodes: List of 5 node IDs
        
    Returns:
        dict: {node_id: (x, y)} positions
    """
    if len(nodes) < 5:
        return {}
    
    import math
    nodes_list = list(nodes)[:5]
    pos = {}
    
    for i, node in enumerate(nodes_list):
        angle = i * 2 * math.pi / 5  # 72 degrees in radians
        x = math.cos(angle)
        y = math.sin(angle)
        pos[node] = (x, y)
    
    return pos


def get_v_shape_layout(nodes):
    """
    Create V-shape layout for NH2/NO2 groups.
    3 nodes: central atom at vertex, 2 atoms at the arms.
    
    Structure:
       1   2
        \ /
         0
    
    Args:
        nodes: List of 3 node IDs
        
    Returns:
        dict: {node_id: (x, y)} positions
    """
    if len(nodes) < 3:
        return {}
    
    nodes_list = list(nodes)[:3]
    return {
        nodes_list[0]: (0.0, 0.0),    # Central atom (N)
        nodes_list[1]: (-0.7, 1.0),   # Left arm
        nodes_list[2]: (0.7, 1.0),    # Right arm
    }


def detect_motif_type_ba2(motif_edges, motif_nodes):
    """
    Detect if BA2 motif is a house or a cycle.
    
    Args:
        motif_edges: List of (src, dst) tuples that are motif edges
        motif_nodes: Set of nodes in the motif
        
    Returns:
        str: 'cycle' or 'house'
    """
    if len(motif_nodes) != 5:
        return 'house'  # Default
    
    # Count degree of each node within motif
    degree = defaultdict(int)
    for src, dst in motif_edges:
        if src in motif_nodes and dst in motif_nodes:
            degree[src] += 1
            degree[dst] += 1
    
    # In a cycle, all nodes have degree 2
    # In a house, nodes have varying degrees (some degree 2, some degree 3)
    degrees = list(degree.values())
    if len(degrees) == 5 and all(d == 2 for d in degrees):
        return 'cycle'
    else:
        return 'house'


def create_motif_aware_layout(G, motif_nodes, non_motif_nodes, dataset_name, motif_type, seed):
    """
    Create a hybrid layout: geometric shape for motif nodes, spring layout for non-motif nodes.
    
    Args:
        G: NetworkX graph
        motif_nodes: Set of nodes that are part of the motif
        non_motif_nodes: Set of nodes that are not part of the motif
        dataset_name: Name of dataset (determines motif shape)
        motif_type: Type of motif (for ba2: 'cycle' or 'house')
        seed: Random seed for spring layout
        
    Returns:
        dict: {node_id: (x, y)} positions for all nodes
    """
    pos = {}
    
    # 1. Position motif nodes in geometric shape
    if len(motif_nodes) > 0:
        if dataset_name in ['syn1', 'syn2']:
            pos.update(get_house_layout(motif_nodes))
        elif dataset_name == 'syn3':
            pos.update(get_hexagon_layout(motif_nodes))
        elif dataset_name == 'syn4':
            pos.update(get_grid_3x3_layout(motif_nodes))
        elif dataset_name == 'ba2':
            if motif_type == 'cycle':
                pos.update(get_cycle_layout(motif_nodes))
            else:
                pos.update(get_house_layout(motif_nodes))
        elif dataset_name == 'mutag':
            pos.update(get_v_shape_layout(motif_nodes))
    
    # 2. Position non-motif nodes using spring layout with motif nodes fixed
    if len(non_motif_nodes) > 0 and len(pos) > 0:
        # Use spring layout with fixed positions for motif nodes
        # Increase k parameter to push non-motif nodes further away from motif
        full_pos = nx.spring_layout(G, pos=pos, fixed=list(motif_nodes), 
                                     seed=seed, k=2.0, iterations=100)
        pos.update(full_pos)
    elif len(pos) == 0:
        # No motif nodes positioned, fall back to regular spring layout
        pos = nx.spring_layout(G, seed=seed, k=0.5, iterations=50)
    else:
        # Only motif nodes, use their positions
        pass
    
    return pos


def compute_auc_roc(y_true, y_scores):
    """
    Compute ROC-AUC score without sklearn dependency.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_scores: Predicted scores
        
    Returns:
        float: AUC score
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Get unique classes
    classes = np.unique(y_true)
    if len(classes) < 2:
        return 0.5  # Only one class present
    
    # Sort by scores (descending)
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_scores_sorted = y_scores[desc_score_indices]
    
    # Get unique thresholds
    distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]
    
    # Accumulate true positives and false positives
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    # Add (0,0) point
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    
    # Normalize to [0,1]
    if tps[-1] == 0 or fps[-1] == 0:
        return 0.5
    
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    
    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)
    
    return auc


def parse_results_file(filepath, dataset_name):
    """
    Parse results txt file to extract graph data from last epoch.
    
    Args:
        filepath: Path to results file (e.g., 'results/ba2_ppo_seed_4.txt')
        dataset_name: Name of dataset to determine task type
        
    Returns:
        dict: {graph_id: {'n_id': int, 'sgs': list, 'losses': dict}}
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Determine task type
    is_graph_task = dataset_name in ['ba2', 'mutag']
    
    # Find all EVAL sections
    eval_sections = []
    for i, line in enumerate(lines):
        if '[EVAL-Epoch' in line:
            eval_sections.append(i)
    
    if not eval_sections:
        raise ValueError(f"No evaluation sections found in {filepath}")
    
    # Get the last epoch section
    last_eval_idx = eval_sections[-1]
    
    # Parse BACKWARDS from the last EVAL line to find graph data
    # Graph data appears in the lines BEFORE the EVAL line
    start_idx = 0
    if len(eval_sections) > 1:
        # Start from after previous EVAL
        start_idx = eval_sections[-2] + 1
    
    # Extract graph/node data from lines before last EVAL
    graphs_data = {}
    
    for line in lines[start_idx:last_eval_idx]:
        if is_graph_task:
            # Format: g_id X n_id Y len Z p_loss=... sgs [...]
            match = re.search(r'g_id (\d+) n_id (\d+) len (\d+) p_loss=([\d.]+) size_loss=([\d.]+) sim_loss=([\d.]+) r_p=([\d.]+) sgs \[(.*?)\]', line)
            if match:
                g_id = int(match.group(1))
                n_id = int(match.group(2))
                length = int(match.group(3))
                p_loss = float(match.group(4))
                size_loss = float(match.group(5))
                sim_loss = float(match.group(6))
                r_p = float(match.group(7))
                sgs_str = match.group(8)
                sgs = [int(x.strip()) for x in sgs_str.split(',') if x.strip()]
                
                graphs_data[g_id] = {
                    'n_id': n_id,
                    'sgs': sgs,
                    'length': length,
                    'losses': {
                        'p_loss': p_loss,
                        'size_loss': size_loss,
                        'sim_loss': sim_loss,
                        'r_p': r_p
                    }
                }
        else:
            # Node task format: n_id X len Y ... sgs [...]
            match = re.search(r'n_id (\d+) len (\d+) p_loss=([\d.]+) size_loss=([\d.]+) sim_loss=([\d.]+) r_p=([\d.]+) sgs \[(.*?)\]', line)
            if match:
                n_id = int(match.group(1))
                length = int(match.group(2))
                p_loss = float(match.group(3))
                size_loss = float(match.group(4))
                sim_loss = float(match.group(5))
                r_p = float(match.group(6))
                sgs_str = match.group(7)
                sgs = [int(x.strip()) for x in sgs_str.split(',') if x.strip()]
                
                graphs_data[n_id] = {
                    'n_id': n_id,
                    'sgs': sgs,
                    'length': length,
                    'losses': {
                        'p_loss': p_loss,
                        'size_loss': size_loss,
                        'sim_loss': sim_loss,
                        'r_p': r_p
                    }
                }
    
    print(f"Parsed {len(graphs_data)} graphs from last epoch")
    return graphs_data


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


def load_ground_truth(dataset_name):
    """
    Load ground truth edge labels for the dataset.
    
    Args:
        dataset_name: Name of dataset ('ba2', 'mutag', 'syn1', etc.)
        
    Returns:
        tuple: (edge_indices, edge_labels) where edge_labels is list of arrays
    """
    if dataset_name == 'ba2':
        return load_ba2_ground_truth(shuffle=True)
    elif dataset_name == 'mutag':
        edge_index, labels, edge_list, edge_label_lists = load_mutag_ground_truth(shuffle=True)
        return edge_index, edge_label_lists
    elif dataset_name in ['syn1', 'syn2', 'syn3', 'syn4']:
        return _load_node_dataset_ground_truth(dataset_name)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_node_class_labels(dataset_name):
    """
    Load node class labels for node classification datasets.
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        numpy array: Node class labels [num_nodes] or None if not applicable
    """
    if dataset_name not in ['syn1', 'syn2', 'syn3', 'syn4']:
        return None
    
    dir_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'pkls')
    path = os.path.join(dir_path, f'{dataset_name}.pkl')
    
    with open(path, 'rb') as f:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pkl.load(f)
    
    # Combine train/val/test labels
    labels = y_train.copy()
    labels[val_mask] = y_val[val_mask]
    labels[test_mask] = y_test[test_mask]
    
    # Convert one-hot to class indices
    node_classes = np.argmax(labels, axis=1)
    
    return node_classes


def load_graph_structure(dataset_name, graph_id=None):
    """
    Load the graph structure (edge indices) for a specific graph.
    
    Args:
        dataset_name: Name of dataset
        graph_id: ID of graph (for graph tasks) or None (for node tasks)
        
    Returns:
        numpy array: Edge index [2, num_edges]
    """
    if dataset_name in ['ba2', 'mutag']:
        # Load graph dataset
        if dataset_name == 'ba2':
            dir_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'pkls')
            path = os.path.join(dir_path, 'BA-2motif.pkl')
        else:
            dir_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'pkls')
            path = os.path.join(dir_path, 'Mutagenicity.pkl')
        
        with open(path, 'rb') as f:
            adjs, features, labels = pkl.load(f)
        
        # Apply same shuffling as ground truth
        from numpy.random.mtrand import RandomState
        prng = RandomState(42)
        indices = prng.permutation(np.arange(len(adjs)))
        adjs = adjs[indices]
        
        # Convert adjacency matrix to edge index
        adj = adjs[graph_id]
        edge_index = np.array(np.where(adj > 0))
        
        return edge_index
    else:
        # Node task - load full graph
        dir_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'pkls')
        path = os.path.join(dir_path, f'{dataset_name}.pkl')
        
        with open(path, 'rb') as f:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pkl.load(f)
        
        # Get edge index from adjacency matrix
        # Check if it's a sparse matrix or numpy array
        if hasattr(adj, 'toarray'):
            # It's a sparse matrix
            edge_index = np.array(np.where(adj.toarray() > 0))
        else:
            # It's already a numpy array
            edge_index = np.array(np.where(adj > 0))
        
        return edge_index


def compute_auc_from_mask_node(edge_index, mask, gt_edge_index, gt_edge_labels):
    """
    Compute AUC for NODE tasks exactly as done in evaluation/utils.py (lines 61-96: evaluation_auc_node).
    
    CRITICAL DIFFERENCE FROM GRAPH TASKS:
    - NODE tasks loop over EXPLANATION edges (lines 77-90)
    - GRAPH tasks loop over GROUND TRUTH edges (lines 47-55)
    
    Why? For node tasks, the explanation is a k-hop SUBGRAPH which only contains a subset
    of edges. Many ground truth edges won't be in the subgraph, so we can't loop over GT edges.
    Instead, we loop over explanation edges and check if each one is in the ground truth.
    
    For each edge in the explanation subgraph:
    - Get its prediction score
    - Check if it exists in ground truth (either direction, since bidirectional)
    - Label as 1 if in ground truth, 0 otherwise
    
    Args:
        edge_index: Subgraph edge index [2, num_subgraph_edges]
        mask: Edge scores [num_subgraph_edges]
        gt_edge_index: Full graph ground truth edge index [2, num_gt_edges]
        gt_edge_labels: Ground truth labels [num_gt_edges] (1=motif, 0=non-motif)
        
    Returns:
        float: AUC score
    """
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
    Compute AUC for GRAPH tasks exactly as done in evaluation/utils.py (lines 24-58: evaluation_auc_graph).
    
    For graph tasks, we loop over GROUND TRUTH edges and find them in the explanation.
    This is used for ba2, mutag, etc.
    
    The logic matches line-by-line:
    - Line 47: Loop through ground truth edges
    - Line 48: Extract edge from ground truth
    - Line 49-50: Skip self-loops
    - Line 51: Find edge in explanation edge list
    - Line 54: Get mask score for that edge
    - Line 55: Append ground truth label
    - Line 57: Compute ROC-AUC score
    
    Args:
        edge_index: Full graph edge index [2, num_edges] (called 'graph' in evaluation)
        mask: Edge scores [num_edges] (called 'mask' in evaluation)
        gt_edge_index: Ground truth edge index [2, num_gt_edges] (called 'edge_list' in evaluation)
        gt_edge_labels: Ground truth labels [num_gt_edges] (1=motif, 0=non-motif)
        
    Returns:
        float: AUC score
    """
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
        
        # Retrieve predictions and ground truth (lines 54-55 of evaluation/utils.py)
        predictions.append(mask[t][0])
        ground_truth.append(int(gt_edge_labels[edge_idx]))
    
    # Compute AUC using manual calculation (avoids sklearn/scipy library issues on HPC)
    if len(np.unique(ground_truth)) > 1:
        auc = compute_auc_roc(ground_truth, predictions)
    else:
        auc = 0.5  # Only one class present
    
    return auc


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


def find_worst_auc_graph(graphs_data, gt_edge_indices, gt_edge_labels, dataset_name, max_size=20, n_hop=3):
    """
    Find the graph with the lowest AUC score using exact evaluation methods.
    
    Args:
        graphs_data: Dict of parsed graph data
        gt_edge_indices: List of ground truth edge indices (or single edge index for node tasks)
        gt_edge_labels: List of ground truth edge labels (or single label array for node tasks)
        dataset_name: Name of dataset
        max_size: Maximum size for edge score normalization (default: 20)
        n_hop: Number of hops for subgraph extraction (node tasks only, default: 3)
        
    Returns:
        int: Graph ID with lowest AUC
    """
    auc_scores = {}
    is_node_task = dataset_name in ['syn1', 'syn2', 'syn3', 'syn4']
    
    # Load full graph structure once for node tasks
    if is_node_task:
        full_edge_index = load_graph_structure(dataset_name, None)
    
    for node_or_graph_id, data in graphs_data.items():
        sgs = data['sgs']
        n_id = data['n_id']  # Seed node ID
        
        if is_node_task:
            # NODE TASK EVALUATION
            seed_node = n_id
            
            # Convert relative node IDs to absolute
            selected_nodes = [seed_node + int(rel_id) for rel_id in sgs]
            
            # Extract k-hop subgraph around seed node
            subgraph_edge_index = get_khop_subgraph(seed_node, full_edge_index, n_hop)
            
            # Prepare mask using the subgraph edge index and absolute selected nodes
            _, mask = prepare_mask(subgraph_edge_index, selected_nodes, max_size)
            
            # For node tasks, use the same ground truth for all nodes
            gt_edge_idx = gt_edge_indices
            gt_labels = gt_edge_labels
            
            # Compute AUC using NODE task method (loops over explanation edges)
            auc = compute_auc_from_mask_node(subgraph_edge_index, mask, gt_edge_idx, gt_labels)
            
        else:
            # GRAPH TASK EVALUATION
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
        
        print(f"Node {node_or_graph_id}: AUC = {auc:.4f}" if is_node_task else f"Graph {node_or_graph_id}: AUC = {auc:.4f}")
    
    # Find minimum
    worst_graph_id = min(auc_scores, key=auc_scores.get)
    print(f"\nWorst AUC {'node' if is_node_task else 'graph'}: {worst_graph_id} with AUC = {auc_scores[worst_graph_id]:.4f}")
    
    return worst_graph_id


def visualize_explanation(sgs_sequence, edge_index_full, edge_scores, gt_edge_index, 
                         gt_edge_labels, graph_id, dataset_name, seed, auc_score, 
                         output_path, node_class_labels=None, edge_index_subgraph=None):
    """
    Create side-by-side visualization of ground truth vs model predictions.
    Only plots the subgraph (sgs nodes and connecting edges).
    
    Args:
        sgs_sequence: Selected node sequence (may contain relative node IDs for node tasks)
        edge_index_full: Full graph edge index [2, num_edges] (used for building visualization graph)
        edge_scores: Dict of predicted edge scores (computed on subgraph edges)
        gt_edge_index: Ground truth edge index
        gt_edge_labels: Ground truth edge labels (1=motif, 0=non-motif)
        graph_id: ID of graph being visualized
        dataset_name: Name of dataset
        seed: Random seed used for training
        auc_score: Computed AUC score
        output_path: Where to save the plot
        node_class_labels: Node class labels [num_nodes] (optional, for syn datasets)
        edge_index_subgraph: Subgraph edge index [2, num_edges] (optional, for node tasks k-hop subgraph)
    """
    # Create NetworkX graph with ONLY sgs nodes and their connecting edges
    G = nx.Graph()
    
    # Convert sgs_sequence to absolute node IDs
    # For node classification tasks, sgs contains relative IDs that need conversion
    is_node_task = dataset_name in ['syn1', 'syn2', 'syn3', 'syn4']
    
    if is_node_task:
        # Convert relative node IDs to absolute by adding the seed node (graph_id)
        sgs_absolute = []
        for n in sgs_sequence:
            n_int = int(n)
            if n_int < 0:
                # Negative values are already offset from seed
                sgs_absolute.append(graph_id + n_int)
            else:
                # Positive small values are relative to seed
                sgs_absolute.append(graph_id + n_int)
        sgs_nodes = set(sgs_absolute)
    else:
        # For graph classification, use as-is
        sgs_nodes = set([int(n) for n in sgs_sequence])
    
    # Add only edges where BOTH endpoints are in sgs (use full graph for complete structure)
    for i in range(edge_index_full.shape[1]):
        src, dst = int(edge_index_full[0, i]), int(edge_index_full[1, i])
        if src in sgs_nodes and dst in sgs_nodes:
            G.add_edge(src, dst)
    
    # Add any isolated nodes from sgs (nodes without edges in the subgraph)
    for node in sgs_nodes:
        if node not in G:
            G.add_node(node)
    
    # Validate: check if subgraph edges are a subset of full graph edges (for node tasks)
    if edge_index_subgraph is not None:
        # Build set of edges from full graph (among sgs nodes)
        full_graph_edges = set()
        for i in range(edge_index_full.shape[1]):
            src, dst = int(edge_index_full[0, i]), int(edge_index_full[1, i])
            if src in sgs_nodes and dst in sgs_nodes:
                # Add both directions for undirected edge comparison
                full_graph_edges.add((min(src, dst), max(src, dst)))
        
        # Build set of edges from k-hop subgraph
        subgraph_edges = set()
        for i in range(edge_index_subgraph.shape[1]):
            src, dst = int(edge_index_subgraph[0, i]), int(edge_index_subgraph[1, i])
            if src in sgs_nodes and dst in sgs_nodes:
                subgraph_edges.add((min(src, dst), max(src, dst)))
        
        # Find edges in full graph but missing from k-hop subgraph
        missing_edges = full_graph_edges - subgraph_edges
        if missing_edges:
            print(f"  Warning: {len(missing_edges)} edge(s) exist in full graph between selected nodes but not in k-hop subgraph")
            if len(missing_edges) <= 5:  # Print first 5 for debugging
                for src, dst in sorted(missing_edges)[:5]:
                    print(f"    - Edge ({src}, {dst})")
    
    # Get seed node (first node in sgs_sequence - this is n_id)
    if is_node_task:
        # For node tasks, the seed is the graph_id itself
        seed_node = graph_id
    else:
        # For graph tasks, first node in sgs is the seed
        seed_node = int(sgs_sequence[0]) if sgs_sequence else None
    
    # Extract motif edges and nodes BEFORE creating layout
    non_motif_edges = []
    motif_edges = []
    
    for i in range(gt_edge_index.shape[1]):
        src, dst = int(gt_edge_index[0, i]), int(gt_edge_index[1, i])
        # Only include if both nodes are in sgs AND edge exists in subgraph
        if src in sgs_nodes and dst in sgs_nodes and G.has_edge(src, dst):
            if gt_edge_labels[i] == 1:
                motif_edges.append((src, dst))
            else:
                non_motif_edges.append((src, dst))
    
    # Extract motif nodes from motif edges
    motif_nodes = set()
    for src, dst in motif_edges:
        motif_nodes.add(src)
        motif_nodes.add(dst)
    
    # Non-motif nodes are all nodes not in motif
    non_motif_nodes = sgs_nodes - motif_nodes
    
    # Detect motif type (for ba2)
    motif_type = 'house'  # Default
    if dataset_name == 'ba2' and len(motif_nodes) == 5:
        motif_type = detect_motif_type_ba2(motif_edges, motif_nodes)
    
    # Create motif-aware layout
    pos = create_motif_aware_layout(G, motif_nodes, non_motif_nodes, 
                                     dataset_name, motif_type, seed)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Define color scheme for node classes
    use_class_colors = node_class_labels is not None and dataset_name in ['syn1', 'syn2', 'syn3', 'syn4']
    
    if use_class_colors:
        # Color palette for different node classes
        # Support up to 8 classes (for syn2 which has classes 0-7)
        class_colors = {
            0: '#B8D8E8',  # Light blue - Base/background nodes
            1: '#FFB6C1',  # Light pink - Motif type 1
            2: '#98D8A8',  # Light green - Motif type 2  
            3: '#FFD9A0',  # Light orange - Motif type 3
            4: '#DDA0DD',  # Light plum - Motif type 4
            5: '#F0E68C',  # Light khaki - Motif type 5
            6: '#FFA07A',  # Light salmon - Motif type 6
            7: '#87CEEB',  # Light sky blue - Motif type 7
        }
        class_names = {
            0: 'Base Graph',
            1: 'Motif Class 1',
            2: 'Motif Class 2',
            3: 'Motif Class 3',
            4: 'Motif Class 4',
            5: 'Motif Class 5',
            6: 'Motif Class 6',
            7: 'Motif Class 7',
        }
    
    # -----------------
    # LEFT: Ground Truth
    # -----------------
    ax1.set_title(f'Ground Truth Motif (Subgraph)\n(Black = Motif Edge)', fontsize=14, fontweight='bold')
    
    # Draw nodes - color by class if available, otherwise use default
    node_colors = []
    node_sizes = []
    edge_colors = []
    
    for node in G.nodes():
        if use_class_colors:
            # Color by node class
            node_class = node_class_labels[node]
            node_colors.append(class_colors.get(node_class, 'lightgray'))
        else:
            # Default coloring
            node_colors.append('lightblue')
        
        # Highlight seed node with larger size and red border
        if node == seed_node:
            node_sizes.append(700)
            edge_colors.append('red')
        else:
            node_sizes.append(400)
            edge_colors.append('black')
    
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, 
                           node_size=node_sizes, alpha=0.8, 
                           edgecolors=edge_colors, linewidths=2.5)
    
    # Draw non-motif edges (light gray, thin)
    if non_motif_edges:
        nx.draw_networkx_edges(G, pos, non_motif_edges, ax=ax1, 
                               edge_color='lightgray', width=1, alpha=0.3)
    
    # Draw motif edges (black, thick)
    if motif_edges:
        nx.draw_networkx_edges(G, pos, motif_edges, ax=ax1, 
                               edge_color='black', width=4, alpha=0.9)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8, font_weight='bold')
    
    ax1.axis('off')
    
    # -----------------
    # RIGHT: Model Predictions
    # -----------------
    ax2.set_title(f'Model Predictions (Subgraph)\n(Darker = Higher Importance)', fontsize=14, fontweight='bold')
    
    # Draw nodes - color by class if available
    node_colors = []
    node_sizes = []
    edge_colors = []
    
    for node in G.nodes():
        if use_class_colors:
            # Color by node class
            node_class = node_class_labels[node]
            node_colors.append(class_colors.get(node_class, 'lightgray'))
        else:
            # Default coloring
            node_colors.append('lightblue')
        
        # Highlight seed node with larger size and red border
        if node == seed_node:
            node_sizes.append(700)
            edge_colors.append('red')
        else:
            node_sizes.append(400)
            edge_colors.append('black')
    
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=node_colors, 
                           node_size=node_sizes, alpha=0.8,
                           edgecolors=edge_colors, linewidths=2.5)
    
    # Draw edges within subgraph with scores
    # Note: edge_scores contains scores from k-hop subgraph (or full graph for graph tasks)
    # G contains all edges from full graph between selected nodes
    drawn_edges = set()
    
    # First, draw edges that have scores (from subgraph or full graph)
    for (src, dst), score in edge_scores.items():
        src_int, dst_int = int(src), int(dst)
        # Only draw if both nodes are in sgs subgraph and edge exists
        if src_int in sgs_nodes and dst_int in sgs_nodes and G.has_edge(src_int, dst_int):
            if score > 0.05:  # Threshold for visibility
                # Use grayscale: darker = more important
                color = plt.cm.Greys(score)
                width = 1 + score * 3  # Thicker for higher scores
                nx.draw_networkx_edges(G, pos, [(src_int, dst_int)], ax=ax2, 
                                       edge_color=[color], width=width, alpha=0.8)
            else:
                # Low/zero score edges: use same style as ground truth non-motif
                nx.draw_networkx_edges(G, pos, [(src_int, dst_int)], ax=ax2, 
                                       edge_color='lightgray', width=1, alpha=0.3)
            drawn_edges.add((min(src_int, dst_int), max(src_int, dst_int)))
    
    # Then, draw any remaining edges in G that don't have scores (edges in full graph but not in k-hop subgraph)
    # This ensures all edges are visible even if they weren't in the k-hop subgraph used for scoring
    unscored_edges = []
    for src, dst in G.edges():
        edge_normalized = (min(src, dst), max(src, dst))
        if edge_normalized not in drawn_edges:
            unscored_edges.append((src, dst))
    
    if unscored_edges:
        # Draw all unscored edges at once with visible styling
        # Use gray color to distinguish from scored edges (which use grayscale gradient)
        nx.draw_networkx_edges(G, pos, unscored_edges, ax=ax2, 
                               edge_color='gray', width=1.5, alpha=0.5, style='solid')
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=8, font_weight='bold')
    
    ax2.axis('off')
    
    # Add legend for node classes if using class colors
    if use_class_colors:
        from matplotlib.patches import Patch
        legend_elements = []
        
        # Add class colors to legend (only for classes present in subgraph)
        present_classes = set()
        for node in G.nodes():
            present_classes.add(node_class_labels[node])
        
        for class_id in sorted(present_classes):
            legend_elements.append(
                Patch(facecolor=class_colors[class_id], edgecolor='black', 
                      label=class_names[class_id])
            )
        
        # Add seed node indicator
        legend_elements.append(
            Patch(facecolor='white', edgecolor='red', linewidth=2.5,
                  label='Seed Node (red border)')
        )
        
        # Place legend between the two subplots
        fig.legend(handles=legend_elements, loc='center', 
                  bbox_to_anchor=(0.5, 0.02), ncol=len(legend_elements),
                  frameon=True, fontsize=11)
    
    # Overall title
    fig.suptitle(f'{dataset_name.upper()} | Seed {seed} | Graph {graph_id} | AUC = {auc_score:.4f}\n'
                 f'Seed Node (n_id): {seed_node} | Subgraph Nodes: {len(sgs_nodes)} | Subgraph Edges: {G.number_of_edges()} | Motif Edges: {len(motif_edges)}',
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96] if use_class_colors else [0, 0, 1, 1])
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize graph explanations vs ground truth')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Dataset name (ba2, mutag, syn1, etc.)')
    parser.add_argument('--seed', type=int, required=True,
                       help='Random seed used for training')
    parser.add_argument('--graph_id', type=int, default=None,
                       help='Specific graph ID to visualize (if None, finds worst AUC)')
    parser.add_argument('--causal', type=lambda x: x.lower() == 'true', default=False,
                       help='Whether to use causal mechanism results file (true/false)')                   
    parser.add_argument('--max_size', type=int, default=20,
                       help='Maximum subgraph size for normalization')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Graph Explanation Visualization Tool")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Graph ID: {args.graph_id if args.graph_id is not None else 'Auto (worst AUC)'}")
    print("="*60)
    
    # 1. Parse results file
    if args.causal:
        results_file = f'results/{args.dataset}_ppo_causal_seed_{args.seed}.txt'
    else:
        results_file = f'results/{args.dataset}_ppo_seed_{args.seed}.txt'
    print(f"\nParsing results file: {results_file}")
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        return
    
    graphs_data = parse_results_file(results_file, args.dataset)
    
    # 2. Load ground truth
    print(f"\nLoading ground truth for {args.dataset}...")
    gt_edge_indices, gt_edge_labels = load_ground_truth(args.dataset)
    
    # Determine if this is a node task or graph task
    is_node_task = args.dataset in ['syn1', 'syn2', 'syn3', 'syn4']
    
    if is_node_task:
        print(f"Loaded ground truth for node classification (single graph)")
    else:
        print(f"Loaded ground truth for {len(gt_edge_labels)} graphs")
    
    # 3. Find target graph
    if args.graph_id is None:
        print("\nFinding graph with worst AUC...")
        graph_id = find_worst_auc_graph(graphs_data, gt_edge_indices, 
                                       gt_edge_labels, args.dataset)
    else:
        graph_id = args.graph_id
        print(f"\nUsing specified graph ID: {graph_id}")
    
    if graph_id not in graphs_data:
        print(f"Error: Graph ID {graph_id} not found in parsed data")
        return
    
    # 4. Load node class labels (for syn datasets)
    node_class_labels = None
    if args.dataset in ['syn1', 'syn2', 'syn3', 'syn4']:
        print("\nLoading node class labels...")
        node_class_labels = load_node_class_labels(args.dataset)
        if node_class_labels is not None:
            print(f"Loaded class labels for {len(node_class_labels)} nodes")
            unique_classes = np.unique(node_class_labels)
            print(f"Unique classes: {unique_classes}")
    
    # 5. Get sgs
    print(f"\nProcessing {'node' if is_node_task else 'graph'} {graph_id}...")
    sgs = graphs_data[graph_id]['sgs']
    print(f"Selected nodes (sgs): {sgs[:10]}{'...' if len(sgs) > 10 else ''}")
    print(f"Total selected: {len(sgs)} nodes")
    
    # Load graph structure (full graph for both tasks initially)
    edge_index = load_graph_structure(args.dataset, graph_id)
    print(f"Full graph has {edge_index.shape[1]} edges")
    
    # 6. Compute AUC using exact evaluation methods
    if is_node_task:
        # NODE TASK: Extract k-hop subgraph and use node task evaluation
        seed_node = graph_id  # For node tasks, graph_id is the node ID
        
        # Convert relative node IDs to absolute
        selected_nodes = [seed_node + int(rel_id) for rel_id in sgs]
        
        # Extract k-hop subgraph for AUC computation
        subgraph_edge_index = get_khop_subgraph(seed_node, edge_index, n_hop=3)
        
        # Prepare mask
        _, mask = prepare_mask(subgraph_edge_index, selected_nodes, args.max_size)
        
        # Get ground truth
        gt_edge_idx = gt_edge_indices
        gt_labels = gt_edge_labels
        
        # Compute AUC using NODE method
        auc_score = compute_auc_from_mask_node(subgraph_edge_index, mask, gt_edge_idx, gt_labels)
        
        # For visualization, we need the edge scores as dict (convert mask back)
        edge_scores = {}
        for i in range(subgraph_edge_index.shape[1]):
            src, dst = int(subgraph_edge_index[0, i]), int(subgraph_edge_index[1, i])
            edge_scores[(src, dst)] = float(mask[i])
        
        # Store both full graph and subgraph for visualization
        edge_index_full = edge_index  # Full graph for building complete structure
        edge_index_subgraph = subgraph_edge_index  # K-hop subgraph for edge scores
    else:
        # GRAPH TASK: Use full graph and graph task evaluation
        # Convert sgs to absolute selected nodes
        selected_nodes = [int(n) for n in sgs]
        
        # Prepare mask
        _, mask = prepare_mask(edge_index, selected_nodes, args.max_size)
        
        # Get ground truth for this graph
        gt_edge_idx = gt_edge_indices[graph_id]
        gt_labels = gt_edge_labels[graph_id]
        
        # Compute AUC using GRAPH method
        auc_score = compute_auc_from_mask_graph(edge_index, mask, gt_edge_idx, gt_labels)
        
        # For visualization, we need the edge scores as dict (convert mask back)
        edge_scores = {}
        for i in range(edge_index.shape[1]):
            src, dst = int(edge_index[0, i]), int(edge_index[1, i])
            edge_scores[(src, dst)] = float(mask[i])
        
        # For graph tasks, full and subgraph are the same
        edge_index_full = edge_index
        edge_index_subgraph = None  # No separate subgraph for graph tasks
    
    print(f"\nAUC Score: {auc_score:.4f}")
    edge_score_values = list(edge_scores.values())
    if len(edge_score_values) > 0:
        print(f"Edge scores range: [{min(edge_score_values):.4f}, {max(edge_score_values):.4f}]")
        non_zero_scores = sum(1 for s in edge_score_values if s > 0)
        print(f"Non-zero edge scores: {non_zero_scores}/{len(edge_score_values)}")
    
    # 7. Visualize
    if args.causal:
        output_path = f'analysis/plots/{args.dataset}_causal_seed{args.seed}_g{graph_id}_explanation.png'
    else:
        output_path = f'analysis/plots/{args.dataset}_seed{args.seed}_g{graph_id}_explanation.png'
    print(f"\nCreating visualization...")
    
    visualize_explanation(sgs, edge_index_full, edge_scores, gt_edge_idx, gt_labels,
                         graph_id, args.dataset, args.seed, auc_score, output_path,
                         node_class_labels=node_class_labels, 
                         edge_index_subgraph=edge_index_subgraph)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == '__main__':
    main()

