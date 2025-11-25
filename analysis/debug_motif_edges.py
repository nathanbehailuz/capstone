#!/usr/bin/env python3
"""
Debug why motif edges aren't showing in visualization for graph 595.
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


def debug_motif_edges(dataset_name, seed, graph_id):
    """Debug motif edge extraction for a specific graph"""
    
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
    print(f"  Full graph has {edge_index.shape[1]} edges")
    
    # Load ground truth
    gt_edge_indices, gt_edge_labels = load_ground_truth(dataset_name)
    gt_edge_idx = gt_edge_indices[graph_id]
    gt_labels = gt_edge_labels[graph_id]
    
    # Get all motif edges from ground truth
    motif_edges_gt = []
    for i in range(gt_labels.shape[0]):
        if gt_labels[i] == 1:
            src, dst = int(gt_edge_idx[0, i]), int(gt_edge_idx[1, i])
            motif_edges_gt.append((src, dst))
    
    print(f"\n  Ground truth motif edges ({len(motif_edges_gt)}):")
    for src, dst in motif_edges_gt:
        print(f"    ({src}, {dst})")
    
    # Check which edges exist in the full graph
    print(f"\n  Checking edge existence in full graph:")
    edges_in_graph = set()
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        edges_in_graph.add((src, dst))
        edges_in_graph.add((dst, src))  # Add reverse direction
    
    motif_edges_in_graph = []
    for src, dst in motif_edges_gt:
        exists = (src, dst) in edges_in_graph or (dst, src) in edges_in_graph
        print(f"    ({src}, {dst}): {'EXISTS' if exists else 'MISSING'}")
        if exists:
            motif_edges_in_graph.append((src, dst))
    
    print(f"\n  Motif edges in full graph: {len(motif_edges_in_graph)}/{len(motif_edges_gt)}")
    
    # Now check which edges are in the subgraph (both nodes in sgs)
    print(f"\n  Checking edges in subgraph (both nodes in sgs):")
    motif_edges_in_subgraph = []
    for src, dst in motif_edges_in_graph:
        both_in_sgs = src in sgs_nodes and dst in sgs_nodes
        print(f"    ({src}, {dst}): {'BOTH IN SGS' if both_in_sgs else 'NOT BOTH IN SGS'}")
        if both_in_sgs:
            motif_edges_in_subgraph.append((src, dst))
    
    print(f"\n  Motif edges in subgraph: {len(motif_edges_in_subgraph)}/{len(motif_edges_in_graph)}")
    
    # Build the NetworkX graph as done in visualization
    import networkx as nx
    G = nx.Graph()
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        if src in sgs_nodes and dst in sgs_nodes:
            G.add_edge(src, dst)
    
    print(f"\n  NetworkX graph G has {G.number_of_edges()} edges")
    
    # Check which motif edges are in G
    print(f"\n  Checking edges in NetworkX graph G:")
    motif_edges_in_G = []
    for src, dst in motif_edges_in_subgraph:
        in_G = G.has_edge(src, dst)
        print(f"    ({src}, {dst}): {'IN G' if in_G else 'NOT IN G'}")
        if in_G:
            motif_edges_in_G.append((src, dst))
    
    print(f"\n  Final motif edges that will be visualized: {len(motif_edges_in_G)}")
    if len(motif_edges_in_G) == 0:
        print(f"  ⚠️  WARNING: No motif edges will be shown in visualization!")
    else:
        print(f"  Motif edges to visualize:")
        for src, dst in motif_edges_in_G:
            print(f"    ({src}, {dst})")


if __name__ == '__main__':
    debug_motif_edges('ba2', 4, 595)

