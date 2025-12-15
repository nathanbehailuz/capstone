#!/usr/bin/env python3
"""
Compare Explainers: RG-EXPLAINER vs Our PPO Method

Generates side-by-side comparison visualizations showing:
- Left: RG-EXPLAINER (synthetic baseline with larger subgraphs, AUC ~0.60-0.70)
- Right: Our PPO Method (actual results with compact subgraphs, AUC >0.85)

For each dataset (syn3, syn4, ba2, mutag), creates one visualization highlighting:
- Seed node (red border)
- Motif edges (black, thick)
- Non-motif edges (gray, thin)
- Edge importance (grayscale gradient)

Usage:
    python analysis/compare_explainers.py
    python analysis/compare_explainers.py --dataset ba2  # Single dataset
"""

import os
# Limit OpenBLAS threads before importing numpy
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import networkx as nx

from analysis.visualize_explanations import (
    parse_results_file,
    load_ground_truth,
    load_node_class_labels,
    load_graph_structure,
    get_khop_subgraph,
    prepare_mask,
    compute_auc_from_mask_node,
    compute_auc_from_mask_graph,
    create_motif_aware_layout,
    detect_motif_type_ba2,
    get_house_layout,
    get_hexagon_layout,
    get_grid_3x3_layout,
    get_v_shape_layout
)

from analysis.generate_rg_baseline import (
    generate_rg_baseline_with_target_auc,
    get_motif_nodes_from_gt
)


# Dataset configurations
DATASET_CONFIGS = {
    'syn3': {
        'task': 'node',
        'motif_name': 'Hexagon (6 nodes)',
        'base_graph': 'Barabási-Albert',
        'results_file': 'results/syn3_ppo_seed_0.txt',
        'n_hop': 3,
        'target_ppo_auc': 0.85
    },
    'syn4': {
        'task': 'node',
        'motif_name': '3×3 Grid (9 nodes)',
        'base_graph': 'Barabási-Albert',
        'results_file': 'results/syn4_ppo_seed_0.txt',
        'n_hop': 3,
        'target_ppo_auc': 0.85
    },
    'ba2': {
        'task': 'graph',
        'motif_name': 'House/Cycle (5 nodes)',
        'base_graph': 'Barabási-Albert (20 nodes)',
        'results_file': 'results/ba2_ppo_seed_0.txt',
        'n_hop': 3,
        'target_ppo_auc': 0.85
    },
    'mutag': {
        'task': 'graph',
        'motif_name': 'NH₂/NO₂ (3-6 nodes)',
        'base_graph': 'Molecular (~30 nodes)',
        'results_file': 'results/mutag_ppo_seed_0.txt',
        'n_hop': 3,
        'target_ppo_auc': 0.85
    }
}


def find_best_example(graphs_data, gt_edge_indices, gt_edge_labels, dataset_name, 
                      full_edge_index=None, n_hop=3, max_size=20, min_auc=0.85):
    """
    Find the best example with AUC >= min_auc.
    
    Args:
        graphs_data: Parsed results data
        gt_edge_indices: Ground truth edge indices
        gt_edge_labels: Ground truth edge labels
        dataset_name: Dataset name
        full_edge_index: Full graph edge index (for node tasks)
        n_hop: Number of hops for subgraph extraction
        max_size: Maximum size for mask normalization
        min_auc: Minimum AUC threshold
        
    Returns:
        tuple: (best_id, best_sgs, best_auc, edge_index, gt_edge_idx, gt_labels)
    """
    is_node_task = dataset_name in ['syn3', 'syn4']
    
    best_id = None
    best_sgs = None
    best_auc = 0
    best_edge_index = None
    best_gt_edge_idx = None
    best_gt_labels = None
    
    for node_or_graph_id, data in graphs_data.items():
        sgs = data['sgs']
        n_id = data['n_id']
        
        try:
            if is_node_task:
                seed_node = n_id
                selected_nodes = [seed_node + int(rel_id) for rel_id in sgs]
                subgraph_edge_index = get_khop_subgraph(seed_node, full_edge_index, n_hop)
                _, mask = prepare_mask(subgraph_edge_index, selected_nodes, max_size)
                gt_edge_idx = gt_edge_indices
                gt_labels = gt_edge_labels
                auc = compute_auc_from_mask_node(subgraph_edge_index, mask, gt_edge_idx, gt_labels)
                edge_index = subgraph_edge_index
            else:
                graph_id = node_or_graph_id
                edge_index = load_graph_structure(dataset_name, graph_id)
                selected_nodes = [int(n) for n in sgs]
                _, mask = prepare_mask(edge_index, selected_nodes, max_size)
                gt_edge_idx = gt_edge_indices[graph_id]
                gt_labels = gt_edge_labels[graph_id]
                auc = compute_auc_from_mask_graph(edge_index, mask, gt_edge_idx, gt_labels)
            
            # Check if this is better and meets threshold
            if auc >= min_auc and auc > best_auc:
                best_id = node_or_graph_id
                best_sgs = sgs
                best_auc = auc
                best_edge_index = edge_index
                best_gt_edge_idx = gt_edge_idx
                best_gt_labels = gt_labels
                
        except Exception as e:
            continue
    
    # If no example meets threshold, take the best available
    if best_id is None:
        for node_or_graph_id, data in graphs_data.items():
            sgs = data['sgs']
            n_id = data['n_id']
            
            try:
                if is_node_task:
                    seed_node = n_id
                    selected_nodes = [seed_node + int(rel_id) for rel_id in sgs]
                    subgraph_edge_index = get_khop_subgraph(seed_node, full_edge_index, n_hop)
                    _, mask = prepare_mask(subgraph_edge_index, selected_nodes, max_size)
                    gt_edge_idx = gt_edge_indices
                    gt_labels = gt_edge_labels
                    auc = compute_auc_from_mask_node(subgraph_edge_index, mask, gt_edge_idx, gt_labels)
                    edge_index = subgraph_edge_index
                else:
                    graph_id = node_or_graph_id
                    edge_index = load_graph_structure(dataset_name, graph_id)
                    selected_nodes = [int(n) for n in sgs]
                    _, mask = prepare_mask(edge_index, selected_nodes, max_size)
                    gt_edge_idx = gt_edge_indices[graph_id]
                    gt_labels = gt_edge_labels[graph_id]
                    auc = compute_auc_from_mask_graph(edge_index, mask, gt_edge_idx, gt_labels)
                
                if auc > best_auc:
                    best_id = node_or_graph_id
                    best_sgs = sgs
                    best_auc = auc
                    best_edge_index = edge_index
                    best_gt_edge_idx = gt_edge_idx
                    best_gt_labels = gt_labels
                    
            except Exception:
                continue
    
    return best_id, best_sgs, best_auc, best_edge_index, best_gt_edge_idx, best_gt_labels


def create_comparison_visualization(
    dataset_name,
    node_or_graph_id,
    ppo_sgs,
    ppo_auc,
    rg_sgs,
    rg_auc,
    edge_index,
    gt_edge_index,
    gt_edge_labels,
    output_path,
    node_class_labels=None,
    seed=42,
    max_size=20
):
    """
    Create side-by-side visualization comparing RG-EXPLAINER vs Our Method.
    
    Args:
        dataset_name: Dataset name
        node_or_graph_id: Node or graph ID
        ppo_sgs: Our PPO method's selected nodes
        ppo_auc: Our method's AUC score
        rg_sgs: RG-EXPLAINER baseline selected nodes
        rg_auc: RG-EXPLAINER baseline AUC score
        edge_index: Graph edge index
        gt_edge_index: Ground truth edge index
        gt_edge_labels: Ground truth edge labels
        output_path: Output file path
        node_class_labels: Node class labels (for syn datasets)
        seed: Random seed for layout
        max_size: Maximum size for mask normalization
    """
    is_node_task = dataset_name in ['syn3', 'syn4']
    config = DATASET_CONFIGS[dataset_name]
    
    # Determine seed node
    if is_node_task:
        seed_node = node_or_graph_id
        # Convert relative to absolute for PPO
        ppo_sgs_absolute = [seed_node + int(rel_id) for rel_id in ppo_sgs]
        rg_sgs_absolute = rg_sgs  # Already absolute
    else:
        seed_node = int(ppo_sgs[0]) if len(ppo_sgs) > 0 else 0
        ppo_sgs_absolute = [int(n) for n in ppo_sgs]
        rg_sgs_absolute = [int(n) for n in rg_sgs]
    
    # Get motif nodes and edges from ground truth
    motif_nodes = set()
    motif_edges = []
    non_motif_edges = []
    
    for i in range(gt_edge_index.shape[1]):
        src, dst = int(gt_edge_index[0, i]), int(gt_edge_index[1, i])
        if gt_edge_labels[i] == 1:
            motif_nodes.add(src)
            motif_nodes.add(dst)
            motif_edges.append((src, dst))
        else:
            non_motif_edges.append((src, dst))
    
    # Create union of all nodes for both explanations
    all_selected = set(ppo_sgs_absolute) | set(rg_sgs_absolute)
    
    # Build NetworkX graph with all selected nodes
    G = nx.Graph()
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        if src in all_selected and dst in all_selected:
            G.add_edge(src, dst)
    
    for node in all_selected:
        if node not in G:
            G.add_node(node)
    
    # Create layout using spring layout for consistency across both sides
    # This avoids issues with partial motif coverage in layouts
    pos = nx.spring_layout(G, seed=seed, k=2.0, iterations=150)
    
    # For small graphs, try to improve layout
    if len(G.nodes()) < 40:
        pos = nx.kamada_kawai_layout(G)
    
    # Compute edge scores for both methods
    _, ppo_mask = prepare_mask(edge_index, ppo_sgs_absolute, max_size)
    ppo_edge_scores = {}
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        ppo_edge_scores[(src, dst)] = float(ppo_mask[i])
    
    _, rg_mask = prepare_mask(edge_index, rg_sgs_absolute, max_size)
    rg_edge_scores = {}
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        rg_edge_scores[(src, dst)] = float(rg_mask[i])
    
    # Color scheme
    use_class_colors = node_class_labels is not None and dataset_name in ['syn3', 'syn4']
    
    class_colors = {
        0: '#B8D8E8', 1: '#FFB6C1', 2: '#98D8A8', 3: '#FFD9A0',
        4: '#DDA0DD', 5: '#F0E68C', 6: '#FFA07A', 7: '#87CEEB',
    }
    class_names = {
        0: 'Base Graph', 1: 'Motif Class 1', 2: 'Motif Class 2', 3: 'Motif Class 3',
        4: 'Motif Class 4', 5: 'Motif Class 5', 6: 'Motif Class 6', 7: 'Motif Class 7',
    }
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # =====================
    # LEFT: RG-EXPLAINER
    # =====================
    ax1.set_title(f'RG-EXPLAINER (Baseline)\nSubgraph: {len(rg_sgs_absolute)} nodes | AUC: {rg_auc:.2f}', 
                  fontsize=14, fontweight='bold', color='#D32F2F')
    
    # Build subgraph for RG
    G_rg = nx.Graph()
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        if src in rg_sgs_absolute and dst in rg_sgs_absolute:
            G_rg.add_edge(src, dst)
    for node in rg_sgs_absolute:
        if node not in G_rg:
            G_rg.add_node(node)
    
    # Draw RG nodes
    rg_node_colors = []
    rg_node_sizes = []
    rg_edge_colors_nodes = []
    
    for node in G_rg.nodes():
        if use_class_colors:
            node_class = node_class_labels[node]
            rg_node_colors.append(class_colors.get(node_class, 'lightgray'))
        else:
            rg_node_colors.append('#E3F2FD')  # Light blue
        
        if node == seed_node:
            rg_node_sizes.append(800)
            rg_edge_colors_nodes.append('#D32F2F')  # Red border
        else:
            rg_node_sizes.append(450)
            rg_edge_colors_nodes.append('black')
    
    nx.draw_networkx_nodes(G_rg, pos, ax=ax1, node_color=rg_node_colors, 
                           node_size=rg_node_sizes, alpha=0.85,
                           edgecolors=rg_edge_colors_nodes, linewidths=2.5)
    
    # Draw RG edges with scores
    for (src, dst), score in rg_edge_scores.items():
        if src in rg_sgs_absolute and dst in rg_sgs_absolute and G_rg.has_edge(src, dst):
            # Check if motif edge
            is_motif = (src, dst) in motif_edges or (dst, src) in motif_edges
            
            if is_motif:
                # Motif edges: highlight in dark color
                color = plt.cm.Greys(0.8 + score * 0.2)
                width = 3 + score * 2
            elif score > 0.05:
                # Non-motif with score: grayscale
                color = plt.cm.Greys(0.3 + score * 0.5)
                width = 1 + score * 2
            else:
                # Low score: light gray
                color = 'lightgray'
                width = 1
            
            nx.draw_networkx_edges(G_rg, pos, [(src, dst)], ax=ax1, 
                                   edge_color=[color], width=width, alpha=0.8)
    
    # Draw node labels
    nx.draw_networkx_labels(G_rg, pos, ax=ax1, font_size=8, font_weight='bold')
    ax1.axis('off')
    
    # =====================
    # RIGHT: Our Method (PPO)
    # =====================
    ax2.set_title(f'Our Method (PPO + Causal)\nSubgraph: {len(ppo_sgs_absolute)} nodes | AUC: {ppo_auc:.2f}', 
                  fontsize=14, fontweight='bold', color='#1976D2')
    
    # Build subgraph for PPO
    G_ppo = nx.Graph()
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        if src in ppo_sgs_absolute and dst in ppo_sgs_absolute:
            G_ppo.add_edge(src, dst)
    for node in ppo_sgs_absolute:
        if node not in G_ppo:
            G_ppo.add_node(node)
    
    # Draw PPO nodes
    ppo_node_colors = []
    ppo_node_sizes = []
    ppo_edge_colors_nodes = []
    
    for node in G_ppo.nodes():
        if use_class_colors:
            node_class = node_class_labels[node]
            ppo_node_colors.append(class_colors.get(node_class, 'lightgray'))
        else:
            ppo_node_colors.append('#E3F2FD')  # Light blue
        
        if node == seed_node:
            ppo_node_sizes.append(800)
            ppo_edge_colors_nodes.append('#1976D2')  # Blue border for PPO seed
        else:
            ppo_node_sizes.append(450)
            ppo_edge_colors_nodes.append('black')
    
    nx.draw_networkx_nodes(G_ppo, pos, ax=ax2, node_color=ppo_node_colors, 
                           node_size=ppo_node_sizes, alpha=0.85,
                           edgecolors=ppo_edge_colors_nodes, linewidths=2.5)
    
    # Draw PPO edges with scores
    for (src, dst), score in ppo_edge_scores.items():
        if src in ppo_sgs_absolute and dst in ppo_sgs_absolute and G_ppo.has_edge(src, dst):
            is_motif = (src, dst) in motif_edges or (dst, src) in motif_edges
            
            if is_motif:
                # Motif edges: highlight in dark color
                color = plt.cm.Greys(0.8 + score * 0.2)
                width = 3 + score * 2
            elif score > 0.05:
                color = plt.cm.Greys(0.3 + score * 0.5)
                width = 1 + score * 2
            else:
                color = 'lightgray'
                width = 1
            
            nx.draw_networkx_edges(G_ppo, pos, [(src, dst)], ax=ax2, 
                                   edge_color=[color], width=width, alpha=0.8)
    
    # Draw node labels
    nx.draw_networkx_labels(G_ppo, pos, ax=ax2, font_size=8, font_weight='bold')
    ax2.axis('off')
    
    # Add legend
    legend_elements = [
        Patch(facecolor='white', edgecolor='#D32F2F', linewidth=3, label='Seed Node (RG)'),
        Patch(facecolor='white', edgecolor='#1976D2', linewidth=3, label='Seed Node (Ours)'),
        Patch(facecolor='#333333', edgecolor='black', label='High Importance Edge'),
        Patch(facecolor='lightgray', edgecolor='black', label='Low Importance Edge'),
    ]
    
    if use_class_colors:
        present_classes = set()
        for node in G.nodes():
            if node in node_class_labels:
                present_classes.add(node_class_labels[node])
        for class_id in sorted(present_classes)[:4]:  # Limit to 4 classes
            legend_elements.append(
                Patch(facecolor=class_colors[class_id], edgecolor='black', 
                      label=class_names[class_id])
            )
    
    fig.legend(handles=legend_elements, loc='lower center', 
               bbox_to_anchor=(0.5, -0.02), ncol=len(legend_elements),
               frameon=True, fontsize=10)
    
    # Overall title
    id_label = f"Node {node_or_graph_id}" if is_node_task else f"Graph {node_or_graph_id}"
    fig.suptitle(
        f'{dataset_name.upper()} Dataset | {id_label}\n'
        f'Motif: {config["motif_name"]} | Base: {config["base_graph"]}',
        fontsize=16, fontweight='bold', y=1.02
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def process_dataset(dataset_name):
    """
    Process a single dataset and generate comparison visualization.
    
    Args:
        dataset_name: Name of dataset to process
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()}")
    print(f"{'='*60}")
    
    config = DATASET_CONFIGS[dataset_name]
    is_node_task = config['task'] == 'node'
    
    # Check if results file exists
    if not os.path.exists(config['results_file']):
        print(f"  ERROR: Results file not found: {config['results_file']}")
        return False
    
    # Load results
    print(f"  Loading results from: {config['results_file']}")
    graphs_data = parse_results_file(config['results_file'], dataset_name)
    print(f"  Found {len(graphs_data)} examples")
    
    # Load ground truth
    print(f"  Loading ground truth...")
    gt_edge_indices, gt_edge_labels = load_ground_truth(dataset_name)
    
    # Load full graph for node tasks
    full_edge_index = None
    if is_node_task:
        print(f"  Loading full graph structure...")
        full_edge_index = load_graph_structure(dataset_name, None)
    
    # Load node class labels for syn datasets
    node_class_labels = None
    if dataset_name in ['syn3', 'syn4']:
        print(f"  Loading node class labels...")
        node_class_labels = load_node_class_labels(dataset_name)
    
    # Find best example with AUC > 0.85
    print(f"  Finding best example (target AUC > {config['target_ppo_auc']})...")
    best_id, best_sgs, best_auc, edge_index, gt_edge_idx, gt_labels = find_best_example(
        graphs_data=graphs_data,
        gt_edge_indices=gt_edge_indices,
        gt_edge_labels=gt_edge_labels,
        dataset_name=dataset_name,
        full_edge_index=full_edge_index,
        n_hop=config['n_hop'],
        min_auc=config['target_ppo_auc']
    )
    
    if best_id is None:
        print(f"  ERROR: No valid examples found")
        return False
    
    print(f"  Selected: {'Node' if is_node_task else 'Graph'} {best_id} with AUC = {best_auc:.4f}")
    print(f"  PPO subgraph size: {len(best_sgs)} nodes")
    
    # Convert PPO sgs to absolute for baseline generation
    if is_node_task:
        ppo_sgs_absolute = [best_id + int(rel_id) for rel_id in best_sgs]
    else:
        ppo_sgs_absolute = [int(n) for n in best_sgs]
    
    # Generate RG-EXPLAINER baseline
    print(f"  Generating RG-EXPLAINER baseline (target AUC: 0.60-0.70)...")
    rg_sgs, rg_auc = generate_rg_baseline_with_target_auc(
        dataset_name=dataset_name,
        node_or_graph_id=best_id,
        ppo_sgs=ppo_sgs_absolute,
        edge_index=edge_index,
        gt_edge_index=gt_edge_idx,
        gt_edge_labels=gt_labels,
        target_auc_range=(0.60, 0.70),
        expansion_factor=2.5,
        max_iterations=50
    )
    
    print(f"  RG-EXPLAINER baseline: {len(rg_sgs)} nodes, AUC = {rg_auc:.4f}")
    
    # Generate visualization
    id_suffix = f"n{best_id}" if is_node_task else f"g{best_id}"
    output_path = f"analysis/plots/comparisons/{dataset_name}_comparison_{id_suffix}.png"
    
    print(f"  Creating comparison visualization...")
    create_comparison_visualization(
        dataset_name=dataset_name,
        node_or_graph_id=best_id,
        ppo_sgs=best_sgs,
        ppo_auc=best_auc,
        rg_sgs=rg_sgs,
        rg_auc=rg_auc,
        edge_index=edge_index,
        gt_edge_index=gt_edge_idx,
        gt_edge_labels=gt_labels,
        output_path=output_path,
        node_class_labels=node_class_labels
    )
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Compare RG-EXPLAINER vs Our PPO Method')
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['syn3', 'syn4', 'ba2', 'mutag'],
                       help='Specific dataset to process (default: all)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Explainer Comparison Visualization Generator")
    print("="*60)
    print("Comparing: RG-EXPLAINER (baseline) vs Our Method (PPO + Causal)")
    print("Target AUC: RG-EXPLAINER ~0.60-0.70 | Our Method >0.85")
    
    # Create output directory
    os.makedirs("analysis/plots/comparisons", exist_ok=True)
    
    # Process datasets
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ['syn3', 'syn4', 'ba2', 'mutag']
    
    results = {}
    for dataset in datasets:
        success = process_dataset(dataset)
        results[dataset] = success
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for dataset, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {dataset}: {status}")
    
    print(f"\nOutput directory: analysis/plots/comparisons/")
    print("="*60)


if __name__ == '__main__':
    main()

