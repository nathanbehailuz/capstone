#!/usr/bin/env python3
"""
Test the visualization logic to see if edges are being correctly identified.
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.visualize_explanations import (
    parse_results_file, 
    load_ground_truth, 
    load_graph_structure
)
import networkx as nx


def test_visualization_logic(dataset_name, seed, graph_id):
    """Test the exact logic used in visualize_explanation"""
    
    # Parse results
    results_file = f'results/{dataset_name}_ppo_seed_{seed}.txt'
    graphs_data = parse_results_file(results_file, dataset_name)
    
    if graph_id not in graphs_data:
        print(f"Graph {graph_id} not found")
        return
    
    data = graphs_data[graph_id]
    sgs = data['sgs']
    sgs_nodes = set([int(n) for n in sgs])
    
    print(f"Graph {graph_id}:")
    print(f"  Selected nodes (sgs): {sorted(sgs_nodes)}")
    
    # Load graph structure
    edge_index = load_graph_structure(dataset_name, graph_id)
    
    # Load ground truth
    gt_edge_indices, gt_edge_labels = load_ground_truth(dataset_name)
    gt_edge_idx = gt_edge_indices[graph_id]
    gt_labels = gt_edge_labels[graph_id]
    
    # Replicate the exact logic from visualize_explanation
    # Create NetworkX graph with ONLY sgs nodes and their connecting edges
    G = nx.Graph()
    
    # Add only edges where BOTH endpoints are in sgs
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        if src in sgs_nodes and dst in sgs_nodes:
            G.add_edge(src, dst)
    
    # Add any isolated nodes from sgs
    for node in sgs_nodes:
        if node not in G:
            G.add_node(node)
    
    print(f"  NetworkX graph G has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Extract motif edges and nodes (EXACT logic from visualize_explanation)
    non_motif_edges = []
    motif_edges = []
    
    for i in range(gt_edge_idx.shape[1]):
        src, dst = int(gt_edge_idx[0, i]), int(gt_edge_idx[1, i])
        # Only include if both nodes are in sgs AND edge exists in subgraph
        if src in sgs_nodes and dst in sgs_nodes and G.has_edge(src, dst):
            if gt_labels[i] == 1:
                motif_edges.append((src, dst))
            else:
                non_motif_edges.append((src, dst))
    
    print(f"\n  Motif edges found: {len(motif_edges)}")
    print(f"  Non-motif edges found: {len(non_motif_edges)}")
    
    # Check for duplicates (both directions)
    motif_edges_set = set()
    for src, dst in motif_edges:
        # Normalize to always have smaller node first
        edge = (min(src, dst), max(src, dst))
        motif_edges_set.add(edge)
    
    print(f"  Unique motif edges (deduplicated): {len(motif_edges_set)}")
    print(f"  Unique motif edges: {sorted(motif_edges_set)}")
    
    # Extract motif nodes from motif edges
    motif_nodes = set()
    for src, dst in motif_edges:
        motif_nodes.add(src)
        motif_nodes.add(dst)
    
    print(f"  Motif nodes: {sorted(motif_nodes)}")
    
    if len(motif_edges) == 0:
        print(f"  ⚠️  WARNING: No motif edges found! This explains why visualization shows no motif.")
    elif len(motif_edges_set) < 5:
        print(f"  ⚠️  WARNING: Only {len(motif_edges_set)} unique motif edges (expected at least 5 for a complete motif)")
    else:
        print(f"  ✓ Should show motif in visualization")


if __name__ == '__main__':
    print("="*60)
    print("Testing Graph 0:")
    print("="*60)
    test_visualization_logic('ba2', 4, 0)
    
    print("\n" + "="*60)
    print("Testing Graph 595:")
    print("="*60)
    test_visualization_logic('ba2', 4, 595)

