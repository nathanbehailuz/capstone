#!/usr/bin/env python3
"""
Plot All Subgraphs from Last Epoch
Generates visualizations for all subgraphs extracted in the last epoch of training.

Usage:
    python analysis/plot_all_sgs.py --dataset syn1 --seed 0 [--causal]
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

# Import helper functions from visualize_explanations.py
from analysis.visualize_explanations import (
    parse_results_file,
    load_ground_truth,
    load_node_class_labels,
    load_graph_structure,
    get_khop_subgraph,
    prepare_mask,
    compute_auc_from_mask_node,
    compute_auc_from_mask_graph,
    visualize_explanation
)


def main():
    parser = argparse.ArgumentParser(description='Plot all subgraphs from last epoch')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (ba2, mutag, syn1, syn2, syn3, syn4)')
    parser.add_argument('--seed', type=int, required=True,
                       help='Random seed used for training')
    parser.add_argument('--causal', action='store_true',
                       help='Whether to use causal mechanism results file')
    parser.add_argument('--max_size', type=int, default=20,
                       help='Maximum subgraph size for normalization')
    parser.add_argument('--n_hop', type=int, default=3,
                       help='Number of hops for subgraph extraction (node tasks only)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Plot All Subgraphs from Last Epoch")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Causal: {args.causal}")
    print("="*80)
    
    # Determine if this is a node task or graph task
    is_node_task = args.dataset in ['syn1', 'syn2', 'syn3', 'syn4']
    
    # 1. Construct results filename
    if args.causal:
        results_file = f'results/{args.dataset}_ppo_causal_seed_{args.seed}.txt'
    else:
        results_file = f'results/{args.dataset}_ppo_seed_{args.seed}.txt'
    
    print(f"\nParsing results file: {results_file}")
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        return
    
    # 2. Parse results file to get all graphs from last epoch
    graphs_data = parse_results_file(results_file, args.dataset)
    print(f"Found {len(graphs_data)} graphs/nodes in last epoch")
    
    # 3. Load ground truth
    print(f"\nLoading ground truth for {args.dataset}...")
    gt_edge_indices, gt_edge_labels = load_ground_truth(args.dataset)
    
    if is_node_task:
        print(f"Loaded ground truth for node classification (single graph)")
    else:
        print(f"Loaded ground truth for {len(gt_edge_labels)} graphs")
    
    # 4. Load node class labels (for syn datasets)
    node_class_labels = None
    if args.dataset in ['syn1', 'syn2', 'syn3', 'syn4']:
        print("\nLoading node class labels...")
        node_class_labels = load_node_class_labels(args.dataset)
        if node_class_labels is not None:
            print(f"Loaded class labels for {len(node_class_labels)} nodes")
    
    # 5. For node tasks, load full graph structure once
    full_edge_index = None
    if is_node_task:
        print("\nLoading full graph structure...")
        full_edge_index = load_graph_structure(args.dataset, None)
        print(f"Full graph has {full_edge_index.shape[1]} edges")
    
    # 6. Process each graph/node
    print(f"\nProcessing {len(graphs_data)} {'nodes' if is_node_task else 'graphs'}...")
    print("="*80)
    
    for idx, (node_or_graph_id, data) in enumerate(graphs_data.items(), 1):
        print(f"\n[{idx}/{len(graphs_data)}] Processing {'node' if is_node_task else 'graph'} {node_or_graph_id}...")
        
        sgs = data['sgs']
        n_id = data['n_id']
        
        try:
            if is_node_task:
                # NODE TASK EVALUATION
                seed_node = n_id
                
                # Convert relative node IDs to absolute
                selected_nodes = [seed_node + int(rel_id) for rel_id in sgs]
                
                # Extract k-hop subgraph around seed node
                subgraph_edge_index = get_khop_subgraph(seed_node, full_edge_index, args.n_hop)
                
                # Prepare mask using the subgraph edge index and absolute selected nodes
                _, mask = prepare_mask(subgraph_edge_index, selected_nodes, args.max_size)
                
                # For node tasks, use the same ground truth for all nodes
                gt_edge_idx = gt_edge_indices
                gt_labels = gt_edge_labels
                
                # Compute AUC using NODE task method
                auc_score = compute_auc_from_mask_node(subgraph_edge_index, mask, gt_edge_idx, gt_labels)
                
                # Convert mask to edge scores dict
                edge_scores = {}
                for i in range(subgraph_edge_index.shape[1]):
                    src, dst = int(subgraph_edge_index[0, i]), int(subgraph_edge_index[1, i])
                    edge_scores[(src, dst)] = float(mask[i])
                
                # Store both full graph and subgraph for visualization
                edge_index_full = full_edge_index
                edge_index_subgraph = subgraph_edge_index
                graph_id = node_or_graph_id
                
            else:
                # GRAPH TASK EVALUATION
                graph_id = node_or_graph_id
                edge_index = load_graph_structure(args.dataset, graph_id)
                
                # selected_nodes are already in the correct coordinate system for this graph
                selected_nodes = [int(n) for n in sgs]
                
                # Prepare mask using exact method
                _, mask = prepare_mask(edge_index, selected_nodes, args.max_size)
                
                # Get ground truth for this specific graph
                gt_edge_idx = gt_edge_indices[graph_id]
                gt_labels = gt_edge_labels[graph_id]
                
                # Compute AUC using GRAPH task method
                auc_score = compute_auc_from_mask_graph(edge_index, mask, gt_edge_idx, gt_labels)
                
                # Convert mask to edge scores dict
                edge_scores = {}
                for i in range(edge_index.shape[1]):
                    src, dst = int(edge_index[0, i]), int(edge_index[1, i])
                    edge_scores[(src, dst)] = float(mask[i])
                
                # For graph tasks, full and subgraph are the same
                edge_index_full = edge_index
                edge_index_subgraph = None
            
            print(f"  AUC Score: {auc_score:.4f}")
            print(f"  Selected nodes: {len(sgs)}")
            
            # Define output path using n_id
            if args.causal:
                output_path = f'analysis/plots/{args.dataset}_causal_seed{args.seed}_n{n_id}_explanation.png'
            else:
                output_path = f'analysis/plots/{args.dataset}_seed{args.seed}_n{n_id}_explanation.png'
            
            # Generate visualization
            visualize_explanation(
                sgs, 
                edge_index_full, 
                edge_scores, 
                gt_edge_idx, 
                gt_labels,
                graph_id, 
                args.dataset, 
                args.seed, 
                auc_score, 
                output_path,
                node_class_labels=node_class_labels,
                edge_index_subgraph=edge_index_subgraph
            )
            
            print(f"  Saved: {output_path}")
            
        except Exception as e:
            print(f"  ERROR: Failed to process {'node' if is_node_task else 'graph'} {node_or_graph_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print(f"Complete! Generated {len(graphs_data)} visualizations.")
    print(f"Output directory: analysis/plots/")
    print("="*80)


if __name__ == '__main__':
    main()

