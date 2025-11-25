#!/usr/bin/env python3
"""
Check if motif nodes are included in the selected subgraph (sgs) for specific graphs.
"""

import os
import sys
import re

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.visualize_explanations import parse_results_file, load_ground_truth


def check_motif_in_sgs(dataset_name, seed, graph_ids):
    """Check if motif nodes are in the selected subgraph"""
    
    # Parse results file
    results_file = f'results/{dataset_name}_ppo_seed_{seed}.txt'
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        return
    
    graphs_data = parse_results_file(results_file, dataset_name)
    
    # Load ground truth to get motif nodes
    gt_edge_indices, gt_edge_labels = load_ground_truth(dataset_name)
    
    # Motif nodes for ba2 are in range [20, 25)
    motif_node_range = set(range(20, 25))
    
    for graph_id in graph_ids:
        if graph_id not in graphs_data:
            print(f"Graph {graph_id} not found in results")
            continue
        
        data = graphs_data[graph_id]
        sgs = data['sgs']
        
        # Convert to set for easier checking
        sgs_nodes = set([int(n) for n in sgs])
        
        # Check overlap
        motif_in_sgs = motif_node_range.intersection(sgs_nodes)
        
        print(f"\nGraph {graph_id}:")
        print(f"  Selected nodes (sgs): {sorted(sgs_nodes)}")
        print(f"  Motif nodes (20-24): {sorted(motif_node_range)}")
        print(f"  Motif nodes in sgs: {sorted(motif_in_sgs)}")
        print(f"  All motif nodes selected: {len(motif_in_sgs) == 5}")
        
        # Get ground truth motif edges
        gt_edge_idx = gt_edge_indices[graph_id]
        gt_labels = gt_edge_labels[graph_id]
        
        motif_edges_gt = []
        for i in range(gt_labels.shape[0]):
            if gt_labels[i] == 1:
                src, dst = gt_edge_idx[0, i], gt_edge_idx[1, i]
                motif_edges_gt.append((src, dst))
        
        # Check which motif edges can be shown (both nodes in sgs)
        motif_edges_showable = []
        for src, dst in motif_edges_gt:
            if src in sgs_nodes and dst in sgs_nodes:
                motif_edges_showable.append((src, dst))
        
        print(f"  Total motif edges in GT: {len(motif_edges_gt)}")
        print(f"  Motif edges showable (both nodes in sgs): {len(motif_edges_showable)}")
        if len(motif_edges_showable) == 0:
            print(f"  ⚠️  WARNING: No motif edges can be shown because motif nodes are not in sgs!")


if __name__ == '__main__':
    # Check graphs mentioned by user
    check_motif_in_sgs('ba2', 4, [0, 595])

