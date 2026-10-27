"""
Edge dropout utility for structural data augmentation during training.

This module provides a single source of truth for edge dropping logic.
Edge dropout is applied at the environment level to modify world dynamics,
not policy behavior.
"""

import torch
import networkx as nx


def apply_edge_dropout(
    edge_index,
    drop_prob,
    preserve_connectivity=False,
    device=None,
    generator=None,
):
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
    
    edges_np = edge_index.detach().cpu().numpy().T
    G = nx.Graph()
    G.add_edges_from(edges_np)
    protected_pairs = set()
    if preserve_connectivity:
        for component in nx.connected_components(G):
            if len(component) > 1:
                subgraph = G.subgraph(component)
                spanning_tree = nx.minimum_spanning_tree(subgraph)
                protected_pairs.update(
                    (min(int(u), int(v)), max(int(u), int(v)))
                    for u, v in spanning_tree.edges()
                )

    pair_to_indices = {}
    for index, (u, v) in enumerate(edges_np):
        pair = (min(int(u), int(v)), max(int(u), int(v)))
        pair_to_indices.setdefault(pair, []).append(index)

    mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
    for pair, indices in pair_to_indices.items():
        if pair in protected_pairs:
            continue
        draw = torch.rand((), device=device, generator=generator)
        if bool(draw < drop_prob):
            mask[torch.as_tensor(indices, device=device)] = False
    return edge_index[:, mask]


