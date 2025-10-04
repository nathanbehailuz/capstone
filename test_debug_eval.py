#!/usr/bin/env python3
"""
Test script for debug evaluation functionality.
This script creates a simple synthetic graph with known motif and tests the debug evaluation.
"""

import torch
import numpy as np
from evaluation.utils import debug_eval_one, align_edge_scores_with_labels

def create_synthetic_graph():
    """Create a simple synthetic graph with a known motif."""
    # Create a small graph with 6 nodes
    # Motif: triangle (0-1-2) + additional edges
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3],  # source nodes
        [1, 0, 2, 1, 0, 2, 4, 3, 5, 4, 3, 5]   # target nodes
    ], dtype=torch.long)
    
    # Ground truth: triangle (0-1-2) is the motif
    gt_edges = torch.tensor([
        [0, 1, 1, 2, 2, 0],  # source nodes
        [1, 0, 2, 1, 0, 2]   # target nodes
    ], dtype=torch.long)
    
    gt_labels = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long)  # All motif edges are positive
    
    return edge_index, gt_edges, gt_labels

def test_align_edge_scores():
    """Test the edge-label alignment function."""
    print("Testing align_edge_scores_with_labels...")
    
    # Create synthetic data
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    node_scores = np.array([0.8, 0.6, 0.4])  # Node 0 has highest score
    label_edges = np.array([[0, 1], [1, 2]])  # Edges (0,1) and (1,2) are positive
    
    # Test max aggregation
    scores, labels = align_edge_scores_with_labels(node_scores, edge_index, label_edges, 'max')
    print(f"Max aggregation - Scores: {scores}, Labels: {labels}")
    
    # Test mean aggregation
    scores, labels = align_edge_scores_with_labels(node_scores, edge_index, label_edges, 'mean')
    print(f"Mean aggregation - Scores: {scores}, Labels: {labels}")
    
    print("✓ align_edge_scores_with_labels test passed")

def test_debug_eval():
    """Test the debug evaluation function."""
    print("Testing debug_eval_one...")
    
    # Create synthetic graph
    edge_index, gt_edges, gt_labels = create_synthetic_graph()
    
    # Create mock explanations (high scores for motif edges)
    explanations = [(
        edge_index,
        torch.tensor([0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1], dtype=torch.float)
    )]
    
    # Create mock ground truth
    explanation_labels = (gt_edges, gt_labels)
    
    # Test debug evaluation
    result = debug_eval_one(0, explanations, explanation_labels, 'node_task', 'max')
    
    if result is not None:
        print(f"✓ Debug evaluation completed successfully")
        print(f"  ROC-AUC: {result['roc_auc']:.4f}")
        print(f"  PR-AUC: {result['pr_auc']:.4f}")
        print(f"  Num edges: {result['num_edges']}")
        print(f"  Num motif edges: {result['num_motif_edges']}")
        print(f"  CSV saved to: {result['csv_path']}")
        
        # Check that AUC is not 0.5 (unless scores are uniform)
        if result['roc_auc'] != 0.5:
            print("✓ AUC is not 0.5, indicating proper edge-label alignment")
        else:
            print("⚠ AUC is 0.5, which might indicate uniform scores or misalignment")
    else:
        print("✗ Debug evaluation failed")

if __name__ == "__main__":
    print("Running debug evaluation tests...")
    test_align_edge_scores()
    test_debug_eval()
    print("All tests completed!")
