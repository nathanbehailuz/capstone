"""
Edge dropout utility for structural data augmentation during training.

This module provides a single source of truth for edge dropping logic.
Edge dropout is applied at the environment level to modify world dynamics,
not policy behavior.
"""

import torch
import networkx as nx
import numpy as np


def apply_edge_dropout(edge_index, drop_prob, preserve_connectivity=False, device=None):
    """
    Apply edge dropout to a graph's edge_index.
    
    Randomly drops edges from the graph with probability `drop_prob`.
    Can optionally preserve graph connectivity by protecting spanning tree edges.
    
    Args:
        edge_index: torch.Tensor of shape [2, num_edges]
        drop_prob: float in [0, 1], probability of dropping each edge
        preserve_connectivity: bool, if True use spanning tree preservation
        device: torch device (if None, uses edge_index.device)
    
    Returns:
        torch.Tensor: edge_index with randomly dropped edges, shape [2, num_dropped_edges]
    
    Examples:
        >>> edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        >>> dropped = apply_edge_dropout(edge_index, drop_prob=0.3)
        >>> dropped.size(1) <= edge_index.size(1)  # Some edges dropped
        True
    """
    if drop_prob <= 0.0:
        return edge_index
    
    if device is None:
        device = edge_index.device
    
    if not preserve_connectivity:
        # Simple random dropping (fast)
        mask = (torch.rand(edge_index.size(1), device=device) > drop_prob)
        return edge_index[:, mask]
    
    # Connectivity-preserving edge dropping (slower but safer)
    # Strategy: Build a spanning tree first, then randomly drop from remaining edges
    
    # Convert to numpy for networkx
    edges_np = edge_index.cpu().numpy().T
    
    # Build undirected graph (remove duplicate edges for undirected)
    G = nx.Graph()
    G.add_edges_from(edges_np)
    
    # Get a spanning tree to maintain connectivity
    if nx.is_connected(G):
        spanning_tree = nx.minimum_spanning_tree(G)
        protected_edges = set(spanning_tree.edges())
        # Add reverse edges for undirected graph (make a copy first to avoid iteration error)
        protected_edges.update((v, u) for u, v in list(protected_edges))
    else:
        # Graph already disconnected, use connected components
        protected_edges = set()
        for component in nx.connected_components(G):
            if len(component) > 1:
                subgraph = G.subgraph(component)
                spanning_tree = nx.minimum_spanning_tree(subgraph)
                tree_edges = list(spanning_tree.edges())
                # Add both directions for undirected graph
                protected_edges.update(tree_edges)
                protected_edges.update((v, u) for u, v in tree_edges)
    
    # Build mask: keep protected edges, randomly drop others
    mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        edge = (u, v)
        
        if edge not in protected_edges:
            # This edge is not critical for connectivity, can drop it
            if torch.rand(1).item() < drop_prob:
                mask[i] = False
    
    return edge_index[:, mask]


