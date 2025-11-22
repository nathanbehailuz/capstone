#!/usr/bin/env python3
"""
Script to verify that AUC calculations in visualize_explanations.py match evaluation/utils.py
and investigate why some graphs show no motif in ground truth.
"""
import os 
# Limit OpenBLAS threads before importing numpy
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


import sys
import numpy as np
import pickle as pkl
from numpy.random.mtrand import RandomState

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.visualize_explanations import (
    compute_auc_from_mask_graph, 
    compute_auc_roc,
    load_ground_truth, 
    load_graph_structure, 
    prepare_mask
)
from datasets.ground_truth_loaders import load_ba2_ground_truth


def evaluation_auc_graph_manual(explanations, explanation_labels, indices):
    """
    Manual implementation of evaluation_auc_graph that matches evaluation/utils.py
    but uses compute_auc_roc instead of sklearn to avoid import issues.
    
    This replicates the exact logic from evaluation/utils.py lines 24-58.
    """
    ground_truth = []
    predictions = []

    for idx, n in enumerate(indices):  # Use idx for explanation list and indices for ground truth list
        # Select explanation
        mask = explanations[idx][1].detach().cpu().numpy()
        graph = explanations[idx][0].detach().cpu().numpy()

        # Select ground truths
        edge_list = explanation_labels[0][n]
        edge_labels = explanation_labels[1][n]

        for edge_idx in range(0, edge_labels.shape[0]):  # Consider every edge in the ground truth
            edge_ = edge_list.T[edge_idx]
            if edge_[0] == edge_[1]:  # We dont consider self loops for our evaluation (Needed for Ba2Motif)
                continue
            t = np.where((graph.T == edge_.T).all(axis=1))  # Determine index of edge in graph

            # Retrieve predictions and ground truth
            # Note: evaluation/utils.py does mask[t][0] which assumes edge exists
            # If edge doesn't exist, t[0] will be empty and this will fail
            # But in practice, all GT edges should exist in the graph
            if len(t[0]) > 0:  # Edge found in graph
                predictions.append(mask[t[0][0]])
                ground_truth.append(edge_labels[edge_idx])
            # If edge not found, skip it (this shouldn't happen in practice)

    if len(ground_truth) == 0:
        return 0.5  # No edges to evaluate
    
    if len(np.unique(ground_truth)) > 1:
        score = compute_auc_roc(ground_truth, predictions)
    else:
        score = 0.5  # Only one class present
    
    return score


def test_auc_calculation_consistency():
    """Test that AUC calculations match between visualize_explanations.py and evaluation/utils.py"""
    print("="*60)
    print("Testing AUC Calculation Consistency")
    print("="*60)
    
    # Load ba2 data
    print("\n1. Loading ba2 ground truth...")
    gt_edge_indices, gt_edge_labels = load_ba2_ground_truth(shuffle=True)
    print(f"   Loaded {len(gt_edge_labels)} graphs")
    
    # Test on a few graphs
    test_graphs = [0, 4, 595]
    
    for graph_id in test_graphs:
        print(f"\n2. Testing graph {graph_id}...")
        
        # Load graph structure
        edge_index = load_graph_structure('ba2', graph_id)
        print(f"   Graph has {edge_index.shape[1]} edges")
        
        # Get ground truth
        gt_edge_idx = gt_edge_indices[graph_id]
        gt_labels = gt_edge_labels[graph_id]
        print(f"   Ground truth has {gt_labels.shape[0]} edges")
        print(f"   Motif edges (label=1): {np.sum(gt_labels)}")
        print(f"   Non-motif edges (label=0): {np.sum(gt_labels == 0)}")
        
        # Create a dummy explanation (all edges with random scores)
        # For testing, we'll use a simple mask
        np.random.seed(42)
        dummy_mask = np.random.rand(edge_index.shape[1])
        
        # Convert to torch format (as expected by evaluation)
        import torch
        explanation_torch = (
            torch.from_numpy(edge_index),
            torch.from_numpy(dummy_mask)
        )
        
        # Test evaluation/utils.py method (manual implementation)
        explanations_list = [explanation_torch]
        explanation_labels = (gt_edge_indices, gt_edge_labels)
        indices = [graph_id]
        
        try:
            auc_eval = evaluation_auc_graph_manual(
                explanations_list, 
                explanation_labels, 
                indices
            )
            print(f"   AUC (evaluation/utils.py logic): {auc_eval:.4f}")
        except Exception as e:
            print(f"   ERROR in evaluation/utils.py logic: {e}")
            import traceback
            traceback.print_exc()
            auc_eval = None
        
        # Test visualize_explanations.py method
        try:
            auc_viz = compute_auc_from_mask_graph(
                edge_index, 
                dummy_mask, 
                gt_edge_idx, 
                gt_labels
            )
            print(f"   AUC (visualize_explanations.py): {auc_viz:.4f}")
        except Exception as e:
            print(f"   ERROR in visualize_explanations.py: {e}")
            import traceback
            traceback.print_exc()
            auc_viz = None
        
        # Compare
        if auc_eval is not None and auc_viz is not None:
            diff = abs(auc_eval - auc_viz)
            if diff < 1e-6:
                print(f"   ✓ MATCH (difference: {diff:.2e})")
            else:
                print(f"   ✗ MISMATCH (difference: {diff:.4f})")
                print(f"     This indicates a bug in one of the implementations!")


def investigate_missing_motifs():
    """Investigate why some graphs show no motif in ground truth"""
    print("\n" + "="*60)
    print("Investigating Missing Motifs in Ground Truth")
    print("="*60)
    
    # Load ba2 data
    print("\n1. Loading ba2 dataset...")
    dir_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'pkls')
    path = os.path.join(dir_path, 'BA-2motif.pkl')
    
    with open(path, 'rb') as f:
        adjs, features, labels = pkl.load(f)
    
    print(f"   Total graphs: {len(labels)}")
    print(f"   Graphs with motif (label=1): {np.sum(labels == 1)}")
    print(f"   Graphs without motif (label=0): {np.sum(labels == 0)}")
    
    # Load ground truth
    gt_edge_indices, gt_edge_labels = load_ba2_ground_truth(shuffle=True)
    
    # Check specific graphs mentioned by user
    test_graphs = [0, 595]
    
    for graph_id in test_graphs:
        print(f"\n2. Analyzing graph {graph_id}...")
        
        # Get original label (before shuffling)
        # Need to find which original graph this is after shuffling
        prng = RandomState(42)
        n_graphs = len(labels)
        shuffled_indices = prng.permutation(np.arange(n_graphs))
        original_graph_id = np.where(shuffled_indices == graph_id)[0][0]
        
        print(f"   Original graph ID (before shuffle): {original_graph_id}")
        print(f"   Graph label (0=no motif, 1=has motif): {labels[original_graph_id]}")
        
        # Get ground truth edges
        gt_edge_idx = gt_edge_indices[graph_id]
        gt_labels = gt_edge_labels[graph_id]
        
        # Count motif edges
        motif_edges_count = np.sum(gt_labels == 1)
        print(f"   Ground truth motif edges (label=1): {motif_edges_count}")
        
        # Check which nodes are in motif region (20-24)
        motif_nodes = set()
        for i in range(gt_labels.shape[0]):
            if gt_labels[i] == 1:
                src, dst = gt_edge_idx[0, i], gt_edge_idx[1, i]
                motif_nodes.add(src)
                motif_nodes.add(dst)
        
        print(f"   Nodes in motif region: {sorted(motif_nodes)}")
        
        # Check if these nodes form a complete motif
        # For a house/cycle motif, we need 5 nodes
        if len(motif_nodes) == 0:
            print(f"   ⚠️  WARNING: No motif nodes found! This graph has no edges in motif region.")
        elif len(motif_nodes) < 5:
            print(f"   ⚠️  WARNING: Only {len(motif_nodes)} nodes in motif region (expected 5 for complete motif)")
        else:
            print(f"   ✓ Found {len(motif_nodes)} nodes in motif region")
        
        # Load actual graph structure
        edge_index = load_graph_structure('ba2', graph_id)
        
        # Check if edges in motif region actually exist in the graph
        motif_edges_in_graph = []
        for i in range(gt_labels.shape[0]):
            if gt_labels[i] == 1:
                src, dst = gt_edge_idx[0, i], gt_edge_idx[1, i]
                # Check if this edge exists in the graph
                edge_exists = np.any((edge_index[0] == src) & (edge_index[1] == dst)) or \
                             np.any((edge_index[0] == dst) & (edge_index[1] == src))
                if edge_exists:
                    motif_edges_in_graph.append((src, dst))
        
        print(f"   Motif edges that actually exist in graph: {len(motif_edges_in_graph)}")
        if len(motif_edges_in_graph) == 0:
            print(f"   ⚠️  WARNING: No motif edges exist in the actual graph!")
            print(f"   This explains why the visualization shows no motif - the ground truth")
            print(f"   labels edges in the motif region, but those edges don't exist in this graph.")


if __name__ == '__main__':
    test_auc_calculation_consistency()
    investigate_missing_motifs()

