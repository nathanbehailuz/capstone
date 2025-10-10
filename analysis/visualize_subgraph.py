#!/usr/bin/env python3
"""
Visualize generated subgraph vs ground truth subgraph for a given dataset, seed, epoch, and node ID.

Example usage:
    python visualize_subgraph.py --dataset syn3 --seed 0 --epoch 30 --n_id 805
"""

import argparse
import os
import pickle as pkl
import re
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score


def load_ground_truth_graph(dataset):
    """Load the ground truth graph and edge labels for a dataset."""
    dir_path = Path(__file__).parent / 'datasets' / 'pkls'
    path = dir_path / f'{dataset}.pkl'
    
    with open(path, 'rb') as fin:
        adj, _, _, _, _, _, _, _, edge_label_matrix = pkl.load(fin)
    
    # Convert adjacency matrix to edge list
    # adj is sparse, so convert to dense first
    if hasattr(adj, 'toarray'):
        adj_dense = adj.toarray()
    else:
        adj_dense = adj
    
    # Get edges (non-zero entries in adjacency matrix)
    edges = np.argwhere(adj_dense > 0)
    # Keep only upper triangle to avoid duplicates (undirected graph)
    edges = edges[edges[:, 0] < edges[:, 1]]
    
    edge_index = edges.T  # Shape: [2, num_edges]
    
    # Get edge labels
    edge_labels = []
    for edge in edges:
        edge_labels.append(edge_label_matrix[edge[0], edge[1]])
    edge_labels = np.array(edge_labels)
    
    return edge_index, edge_labels, edge_label_matrix, adj_dense


def extract_k_hop_subgraph(edge_index, node_id, k=3):
    """
    Extract k-hop subgraph around a node using BFS.
    Returns nodes in the subgraph and edges within those nodes.
    """
    # Convert edge_index to adjacency dict
    neighbors = {}
    edges_array = edge_index.T
    for u, v in edges_array:
        if u not in neighbors:
            neighbors[u] = set()
        if v not in neighbors:
            neighbors[v] = set()
        neighbors[u].add(v)
        neighbors[v].add(u)
    
    # BFS to find k-hop neighbors
    visited = {node_id}
    current_layer = {node_id}
    
    for _ in range(k):
        next_layer = set()
        for node in current_layer:
            if node in neighbors:
                next_layer.update(neighbors[node])
        next_layer -= visited
        visited.update(next_layer)
        current_layer = next_layer
    
    subgraph_nodes = sorted(visited)
    
    # Extract edges within subgraph
    subgraph_edges = []
    node_set = set(subgraph_nodes)
    for u, v in edges_array:
        if u in node_set and v in node_set:
            subgraph_edges.append([u, v])
    
    return subgraph_nodes, np.array(subgraph_edges)


def parse_result_file(dataset, seed, epoch, n_id):
    """
    Parse the result file to find the generated subgraph for a specific 
    dataset, seed, epoch, and n_id combination.
    """
    result_file = Path(__file__).parent / 'results' / f'{dataset}_ppo_seed_{seed}.txt'
    
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    
    with open(result_file, 'r') as f:
        lines = f.readlines()
    
    # Find the epoch section by looking for [EVAL-Epoch   X] marker
    # Then look backwards for n_id lines
    epoch_pattern = rf'\[EVAL-Epoch\s+{epoch}\]'
    
    # Find the line with the epoch marker
    epoch_line_idx = None
    for i, line in enumerate(lines):
        if re.search(epoch_pattern, line):
            epoch_line_idx = i
            break
    
    if epoch_line_idx is None:
        raise ValueError(f"Epoch {epoch} not found in result file")
    
    # Now search backwards from the epoch marker for n_id lines
    # Format: n_id 805 len 20 p_loss=0.40 size_loss=0.00 sim_loss=0.00 r_p=0.06 sgs [...]
    nid_pattern = rf'n_id\s+{n_id}\s+.*sgs\s+\[(.*?)\]'
    
    for i in range(epoch_line_idx - 1, -1, -1):
        line = lines[i]
        match = re.search(nid_pattern, line)
        if match:
            # Extract the subgraph nodes
            nodes_str = match.group(1)
            nodes = [int(x.strip()) for x in nodes_str.split(',')]
            return nodes
        # Stop if we hit the previous epoch or start of file
        if '[Epoch' in line or i == 0:
            break
    
    raise ValueError(f"No entry found for dataset={dataset}, seed={seed}, epoch={epoch}, n_id={n_id}")


def decode_subgraph_nodes(relative_nodes, center_node):
    """
    Decode subgraph node list from relative offsets to absolute node IDs.
    
    The output format in results is [x - center_node for x in subgraph_nodes],
    so we need to add back the center node to get absolute IDs.
    
    Args:
        relative_nodes: List of relative node offsets (e.g., [0, 1, -413, 5, ...])
        center_node: The center node ID (e.g., 805)
    
    Returns:
        List of absolute node IDs
    """
    return [offset + center_node for offset in relative_nodes]


def nodes_to_edges(nodes, adj_matrix):
    """
    Extract edges between nodes based on the adjacency matrix.
    
    Args:
        nodes: List of node IDs
        adj_matrix: Adjacency matrix indicating which edges exist (>0 means edge exists)
    
    Returns:
        Tuple of (nodes, edges array)
    """
    edges = []
    for i, u in enumerate(nodes):
        for v in nodes[i+1:]:
            # Check if edge exists in adjacency matrix
            if adj_matrix[u, v] > 0 or adj_matrix[v, u] > 0:
                edges.append([u, v])
    
    return nodes, np.array(edges) if edges else np.array([]).reshape(0, 2)


def calculate_auc(gt_nodes, gt_edges, gen_nodes, gen_edges, edge_label_matrix):
    """
    Calculate ROC AUC score for the generated subgraph using standard evaluation.
    
    For each edge in the ground truth k-hop subgraph:
    - Prediction score: 1.0 if edge is in generated subgraph, 0.0 otherwise
    - Ground truth label: 1 if motif edge, 0 if non-motif edge
    
    This matches the evaluation approach in evaluation/utils.py
    
    Args:
        gt_nodes: Ground truth k-hop subgraph nodes
        gt_edges: Ground truth k-hop subgraph edges [N, 2]
        gen_nodes: Generated subgraph nodes
        gen_edges: Generated subgraph edges [M, 2]
        edge_label_matrix: Matrix indicating motif edges
    
    Returns:
        tuple: (AUC score, precision, recall, F1) or (None, None, None, None) if cannot be calculated
    """
    if len(gt_edges) == 0:
        return None, None, None, None
    
    # Create set of generated edges for fast lookup
    gen_edge_set = set()
    if len(gen_edges) > 0:
        for u, v in gen_edges:
            gen_edge_set.add((min(u, v), max(u, v)))  # Undirected
    
    # For each edge in ground truth k-hop subgraph, get prediction and label
    predictions = []
    labels = []
    motif_edges_in_gt = []
    motif_edges_found = []
    
    for u, v in gt_edges:
        # Prediction: is this edge in the generated subgraph?
        edge_key = (min(u, v), max(u, v))
        prediction = 1.0 if edge_key in gen_edge_set else 0.0
        predictions.append(prediction)
        
        # Ground truth label: is this a motif edge?
        label = 1 if edge_label_matrix[u, v] == 1 else 0
        labels.append(label)
        
        if label == 1:
            motif_edges_in_gt.append(edge_key)
            if prediction == 1.0:
                motif_edges_found.append(edge_key)
    
    # Count motif edges in generated subgraph that are NOT in ground truth subgraph
    extra_motif_edges = []
    for u, v in gen_edges:
        edge_key = (min(u, v), max(u, v))
        # Check if this edge is in ground truth subgraph
        is_in_gt = any((min(gu, gv), max(gu, gv)) == edge_key for gu, gv in gt_edges)
        if not is_in_gt:
            # This edge is NOT in the ground truth k-hop subgraph
            # Check if it's a motif edge
            if edge_label_matrix[u, v] == 1 or edge_label_matrix[v, u] == 1:
                extra_motif_edges.append(edge_key)
    
    # Calculate metrics
    predictions_np = np.array(predictions)
    labels_np = np.array(labels)
    
    # True positives, false positives, false negatives, true negatives
    tp = np.sum((predictions_np == 1) & (labels_np == 1))
    fp = np.sum((predictions_np == 1) & (labels_np == 0))
    fn = np.sum((predictions_np == 0) & (labels_np == 1))
    tn = np.sum((predictions_np == 0) & (labels_np == 0))
    
    # Debug output
    print(f"  [Debug] GT k-hop subgraph: {len(gt_edges)} edges, {len(motif_edges_in_gt)} motif")
    print(f"  [Debug] Generated subgraph: {len(gen_edges)} edges")
    print(f"  [Debug] Motif edges found in GT subgraph: {len(motif_edges_found)}/{len(motif_edges_in_gt)}")
    num_extra_edges = len(gen_edges) - int(sum(predictions))
    print(f"  [Debug] Extra edges (not in GT k-hop subgraph): {num_extra_edges}")
    if extra_motif_edges:
        print(f"  [Debug] WARNING: {len(extra_motif_edges)} motif edges from outside GT neighborhood!")
    
    # IMPORTANT: Count extra edges (those outside GT k-hop subgraph) as false positives
    # These edges shouldn't be part of a local explanation
    fp_total = fp + num_extra_edges  # Original FP + out-of-scope edges
    
    # Recalculate precision, recall, F1 with corrected false positives
    precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate AUC if we have both positive and negative samples
    if len(set(labels)) < 2:
        # All edges are the same class, AUC is not well-defined
        return None, precision, recall, f1
    
    try:
        auc = roc_auc_score(labels, predictions)
        return auc, precision, recall, f1
    except Exception as e:
        print(f"Warning: Could not calculate AUC: {e}")
        return None, precision, recall, f1


def visualize_subgraphs(gt_nodes, gt_edges, gen_nodes, gen_edges, edge_label_matrix, 
                       dataset, seed, epoch, n_id, metrics=None, output_path=None):
    """
    Visualize ground truth subgraph and generated subgraph side by side.
    
    Args:
        gt_nodes: Ground truth subgraph nodes
        gt_edges: Ground truth subgraph edges
        gen_nodes: Generated subgraph nodes
        gen_edges: Generated subgraph edges
        edge_label_matrix: Matrix indicating motif edges
        dataset: Dataset name
        seed: Random seed
        epoch: Epoch number
        n_id: Center node ID
        auc_score: AUC score (optional)
        output_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Create NetworkX graphs
    G_gt = nx.Graph()
    G_gen = nx.Graph()
    
    # Add nodes and edges to ground truth graph
    G_gt.add_nodes_from(gt_nodes)
    if len(gt_edges) > 0:
        G_gt.add_edges_from(gt_edges)
    
    # Add nodes and edges to generated graph
    G_gen.add_nodes_from(gen_nodes)
    if len(gen_edges) > 0:
        G_gen.add_edges_from(gen_edges)
    
    # Get edge labels (1 = in motif, 0 = not in motif)
    gt_edge_colors = []
    for u, v in G_gt.edges():
        # Check both directions since matrix might be directional
        label = edge_label_matrix[u, v] if edge_label_matrix[u, v] > 0 else edge_label_matrix[v, u]
        if label == 1:
            gt_edge_colors.append('red')
        else:
            gt_edge_colors.append('lightgray')
    
    gen_edge_colors = []
    for u, v in G_gen.edges():
        # Check both directions since matrix might be directional
        label = edge_label_matrix[u, v] if edge_label_matrix[u, v] > 0 else edge_label_matrix[v, u]
        if label == 1:
            gen_edge_colors.append('red')
        else:
            gen_edge_colors.append('lightgray')
    
    # Node colors - highlight the center node and isolated nodes
    gt_node_colors = ['gold' if node == n_id else 'lightblue' for node in G_gt.nodes()]
    
    # For generated graph, highlight isolated nodes in red
    gen_node_colors = []
    for node in G_gen.nodes():
        if node == n_id:
            gen_node_colors.append('gold')
        elif G_gen.degree(node) == 0:
            gen_node_colors.append('red')  # Isolated nodes
        else:
            gen_node_colors.append('lightblue')
    
    # Draw ground truth subgraph
    pos_gt = nx.spring_layout(G_gt, seed=42, k=0.5, iterations=50)
    nx.draw(G_gt, pos_gt, ax=axes[0], 
            node_color=gt_node_colors,
            edge_color=gt_edge_colors if gt_edge_colors else 'lightgray',
            node_size=500,
            width=2,
            with_labels=True,
            font_size=8,
            font_weight='bold')
    axes[0].set_title(f'Ground Truth Subgraph\n(n_id={n_id}, nodes={len(gt_nodes)}, edges={len(G_gt.edges())})', 
                     fontsize=12, fontweight='bold')
    
    # Draw generated subgraph
    pos_gen = nx.spring_layout(G_gen, seed=42, k=0.5, iterations=50)
    nx.draw(G_gen, pos_gen, ax=axes[1],
            node_color=gen_node_colors,
            edge_color=gen_edge_colors if gen_edge_colors else 'lightgray',
            node_size=500,
            width=2,
            with_labels=True,
            font_size=8,
            font_weight='bold')
    axes[1].set_title(f'Generated Subgraph\n(epoch={epoch}, nodes={len(gen_nodes)}, edges={len(G_gen.edges())})', 
                     fontsize=12, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gold', label='Center Node'),
        Patch(facecolor='lightblue', label='Connected Nodes'),
        Patch(facecolor='red', label='Isolated Nodes (degree=0)'),
        Patch(facecolor='darkred', edgecolor='black', label='Motif Edges'),
        Patch(facecolor='lightgray', edgecolor='black', label='Non-Motif Edges')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, frameon=False, fontsize=9)
    
    # Overall title with metrics if available
    title = f'Subgraph Visualization: {dataset.upper()} (seed={seed})'
    if metrics is not None:
        auc, precision, recall, f1 = metrics
        if auc is not None:
            title += f' | AUC={auc:.3f}'
        title += f' P={precision:.3f} R={recall:.3f} F1={f1:.3f}'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print statistics
    print(f"\nGround Truth:")
    print(f"  Nodes: {len(gt_nodes)}")
    print(f"  Edges: {len(G_gt.edges())}")
    if len(gt_edges) > 0:
        motif_edges_gt = sum(1 for u, v in G_gt.edges() if (edge_label_matrix[u, v] == 1 or edge_label_matrix[v, u] == 1))
        print(f"  Motif Edges: {motif_edges_gt}")
    
    print(f"\nGenerated Subgraph:")
    print(f"  Nodes: {len(gen_nodes)}")
    print(f"  Edges: {len(G_gen.edges())}")
    if len(gen_edges) > 0:
        motif_edges_gen = sum(1 for u, v in G_gen.edges() if (edge_label_matrix[u, v] == 1 or edge_label_matrix[v, u] == 1))
        print(f"  Motif Edges: {motif_edges_gen}")
    
    # Connectivity analysis
    isolated_nodes = [node for node in G_gen.nodes() if G_gen.degree(node) == 0]
    if isolated_nodes:
        print(f"  ⚠️  Isolated Nodes: {len(isolated_nodes)} (shown in red)")
    
    # Count connected components
    if len(G_gen.nodes()) > 0:
        n_components = nx.number_connected_components(G_gen)
        if n_components > 1:
            print(f"  ⚠️  Connected Components: {n_components} (should be 1 for coherent explanation)")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize generated subgraph vs ground truth subgraph'
    )
    parser.add_argument('--dataset', type=str, default='syn3',
                       help='Dataset name (default: syn3)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--epoch', type=int, default=30,
                       help='Epoch number (default: 30)')
    parser.add_argument('--n_id', type=int, default=805,
                       help='Node ID (default: 805)')
    parser.add_argument('--k_hop', type=int, default=3,
                       help='K-hop for ground truth subgraph extraction (default: 3)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (if not specified, will save to visualization/<dataset>/)')
    
    args = parser.parse_args()
    
    print(f"Loading ground truth for {args.dataset}...")
    edge_index, edge_labels, edge_label_matrix, adj_matrix = load_ground_truth_graph(args.dataset)
    print(f"  Total nodes: {edge_label_matrix.shape[0]}")
    print(f"  Total edges: {len(edge_index[0])}")
    
    print(f"\nExtracting {args.k_hop}-hop subgraph around node {args.n_id}...")
    gt_nodes, gt_edges = extract_k_hop_subgraph(edge_index, args.n_id, k=args.k_hop)
    print(f"  Ground truth subgraph: {len(gt_nodes)} nodes, {len(gt_edges)} edges")
    
    print(f"\nParsing result file for seed={args.seed}, epoch={args.epoch}, n_id={args.n_id}...")
    gen_nodes_relative = parse_result_file(args.dataset, args.seed, args.epoch, args.n_id)
    print(f"  Generated nodes (relative offsets): {gen_nodes_relative}")
    
    # Decode relative offsets to absolute node IDs
    gen_nodes_abs = decode_subgraph_nodes(gen_nodes_relative, args.n_id)
    print(f"  Generated nodes (absolute IDs): {gen_nodes_abs}")
    
    # Extract edges between generated nodes using adjacency matrix
    gen_nodes, gen_edges = nodes_to_edges(gen_nodes_abs, adj_matrix)
    print(f"  Generated subgraph: {len(gen_nodes)} nodes, {len(gen_edges)} edges")
    
    # Calculate AUC and other metrics
    print(f"\nCalculating metrics...")
    auc, precision, recall, f1 = calculate_auc(gt_nodes, gt_edges, gen_nodes, gen_edges, edge_label_matrix)
    if auc is not None:
        print(f"  AUC: {auc:.4f}")
    else:
        print(f"  AUC: N/A (cannot calculate - need both positive and negative samples)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        viz_dir = Path(__file__).parent / 'visualization' / args.dataset
        viz_dir.mkdir(parents=True, exist_ok=True)
        output_path = viz_dir / f'subgraph_seed{args.seed}_epoch{args.epoch}_n{args.n_id}.png'
    
    print(f"\nGenerating visualization...")
    visualize_subgraphs(
        gt_nodes, gt_edges,
        gen_nodes, gen_edges,
        edge_label_matrix,
        args.dataset, args.seed, args.epoch, args.n_id,
        metrics=(auc, precision, recall, f1),
        output_path=output_path
    )
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()

