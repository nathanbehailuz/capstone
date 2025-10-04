import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import os


def evaluation_auc(task, explanations, explanation_labels, indices, edge_score_agg='max', debug_eval_graph=-1):
    """Determines based on the task which auc evaluation method should be called to determine the AUC score

    :param task: str either "node" or "graph".
    :param explanations: predicted labels.
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate. We ignore all others.
    :param edge_score_agg: How to aggregate node scores to edge scores ('max' or 'mean')
    :param debug_eval_graph: If >= 0, run debug evaluation for single graph
    :returns: area under curve score.
    """
    if task == 'graph_task':
        return evaluation_auc_graph(explanations, explanation_labels, indices, edge_score_agg, debug_eval_graph)
    elif task == 'node_task':
        return evaluation_auc_node(explanations, explanation_labels, edge_score_agg, debug_eval_graph)


def evaluation_auc_graph(explanations, explanation_labels, indices, edge_score_agg='max', debug_eval_graph=-1):
    """Evaluate the auc score given explaination and ground truth labels.

    :param explanations: predicted labels.
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate. We ignore all others.
    :param edge_score_agg: How to aggregate node scores to edge scores ('max' or 'mean')
    :param debug_eval_graph: If >= 0, run debug evaluation for single graph
    :returns: area under curve score.
    """
    ground_truth = []
    predictions = []

    for idx, n in enumerate(indices): # Use idx for explanation list and indices for ground truth list

        # Select explanation
        mask = explanations[idx][1].detach().cpu().numpy()
        graph = explanations[idx][0].detach().cpu().numpy()

        # Select ground truths
        edge_list = explanation_labels[0][n]
        edge_labels = explanation_labels[1][n]

        for edge_idx in range(0, edge_labels.shape[0]): # Consider every edge in the ground truth
            edge_ = edge_list.T[edge_idx]
            if edge_[0] == edge_[1]:  # We dont consider self loops for our evaluation (Needed for Ba2Motif)
                continue
            t = np.where((graph.T == edge_.T).all(axis=1)) # Determine index of edge in graph

            # Retrieve predictions and ground truth
            predictions.append(mask[t][0])
            ground_truth.append(edge_labels[edge_idx])

    score = roc_auc_score(ground_truth, predictions)
    return score


def evaluation_auc_node(explanations, explanation_labels, edge_score_agg='max', debug_eval_graph=-1):
    """Evaluate the auc score given explaination and ground truth labels.

    :param explanations: predicted labels.
    :param ground_truth: ground truth labels.
    :param edge_score_agg: How to aggregate node scores to edge scores ('max' or 'mean')
    :param debug_eval_graph: If >= 0, run debug evaluation for single graph
    :returns: area under curve score.
    """
    ground_truth = []
    predictions = []
    for expl in explanations: # Loop over the explanations for each node

        ground_truth_node = []
        prediction_node = []

        for i in range(0, expl[0].size(1)): # Loop over all edges in the explanation sub-graph
            prediction_node.append(expl[1][i].item())

            # Graphs are defined bidirectional, so we need to retrieve both edges
            pair = expl[0].T[i].numpy()
            idx_edge = np.where((explanation_labels[0].T == pair).all(axis=1))[0]
            idx_edge_rev = np.where((explanation_labels[0].T == [pair[1], pair[0]]).all(axis=1))[0]

            # If any of the edges is in the ground truth set, the edge should be in the explanation
            gt = explanation_labels[1][idx_edge] + explanation_labels[1][idx_edge_rev]
            if gt == 0:
                ground_truth_node.append(0)
            else:
                ground_truth_node.append(1)

        ground_truth.extend(ground_truth_node)
        predictions.extend(prediction_node)

    score = roc_auc_score(ground_truth, predictions)
    return score


def align_edge_scores_with_labels(subgraph_scores, full_graph_edge_index, label_edges, edge_score_agg='max'):
    """
    Align predicted edge scores with ground-truth labels on the same edge indexing.
    
    Args:
        subgraph_scores: Node scores from subgraph [num_nodes]
        full_graph_edge_index: Full graph edge index [2, num_edges]
        label_edges: Ground truth edge labels [num_label_edges]
        edge_score_agg: How to aggregate node scores to edge scores ('max' or 'mean')
        
    Returns:
        scores: Aligned edge scores
        labels: Aligned ground truth labels
    """
    # Convert node scores to edge scores
    edge_scores = []
    edge_labels = []
    
    for i in range(full_graph_edge_index.shape[1]):
        src, dst = full_graph_edge_index[0, i].item(), full_graph_edge_index[1, i].item()
        
        # Get node scores for this edge
        if src < len(subgraph_scores) and dst < len(subgraph_scores):
            if edge_score_agg == 'max':
                edge_score = max(subgraph_scores[src], subgraph_scores[dst])
            elif edge_score_agg == 'mean':
                edge_score = (subgraph_scores[src] + subgraph_scores[dst]) / 2
            else:
                raise ValueError(f"Unknown edge_score_agg: {edge_score_agg}")
            
            edge_scores.append(edge_score)
            
            # Find corresponding label
            edge_label = 0  # Default to 0 if not found
            for j, label_edge in enumerate(label_edges):
                if (label_edge[0] == src and label_edge[1] == dst) or \
                   (label_edge[0] == dst and label_edge[1] == src):
                    edge_label = 1
                    break
            edge_labels.append(edge_label)
    
    return np.array(edge_scores), np.array(edge_labels)


def debug_eval_one(graph_id, explanations, explanation_labels, task, edge_score_agg='max'):
    """
    Debug evaluation for a single graph.
    
    Args:
        graph_id: Graph ID to debug
        explanations: Predicted explanations
        explanation_labels: Ground truth labels
        task: Task type ('node_task' or 'graph_task')
        edge_score_agg: How to aggregate node scores to edge scores
        
    Returns:
        dict: Debug results including AUC scores
    """
    if graph_id >= len(explanations):
        print(f"Warning: graph_id {graph_id} >= {len(explanations)} explanations")
        return None
    
    expl = explanations[graph_id]
    graph_edges = expl[0].detach().cpu().numpy()  # [2, num_edges]
    edge_scores = expl[1].detach().cpu().numpy()  # [num_edges]
    
    print(f"Debug Graph {graph_id}:")
    print(f"  Graph edges shape: {graph_edges.shape}")
    print(f"  Edge scores shape: {edge_scores.shape}")
    
    # Get top-k predicted edges
    k = min(10, len(edge_scores))
    top_k_indices = np.argsort(edge_scores)[-k:][::-1]
    
    print(f"  Top-{k} predicted edges:")
    for i, idx in enumerate(top_k_indices):
        src, dst = graph_edges[0, idx], graph_edges[1, idx]
        score = edge_scores[idx]
        print(f"    {i+1}. Edge ({src}, {dst}) -> Score: {score:.4f}")
    
    # Get ground truth edges
    if task == 'node_task':
        gt_edges = explanation_labels[0]  # [2, num_gt_edges]
        gt_labels = explanation_labels[1]  # [num_gt_edges]
    else:
        gt_edges = explanation_labels[0][graph_id]  # [2, num_gt_edges]
        gt_labels = explanation_labels[1][graph_id]  # [num_gt_edges]
    
    print(f"  Ground truth edges shape: {gt_edges.shape}")
    print(f"  Ground truth labels shape: {gt_labels.shape}")
    
    # Find motif edges (positive labels)
    motif_edges = []
    for i in range(gt_edges.shape[1]):
        if gt_labels[i] == 1:
            src, dst = gt_edges[0, i], gt_edges[1, i]
            motif_edges.append((src, dst))
    
    print(f"  Motif edges ({len(motif_edges)}): {motif_edges[:5]}{'...' if len(motif_edges) > 5 else ''}")
    
    # Align scores and labels
    try:
        aligned_scores, aligned_labels = align_edge_scores_with_labels(
            edge_scores, graph_edges, gt_edges.T, edge_score_agg
        )
        
        # Compute AUC scores
        if len(np.unique(aligned_labels)) > 1:  # Need both positive and negative samples
            roc_auc = roc_auc_score(aligned_labels, aligned_scores)
            pr_auc = average_precision_score(aligned_labels, aligned_scores)
        else:
            roc_auc = 0.5  # Default for uniform labels
            pr_auc = 0.0
        
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  PR-AUC: {pr_auc:.4f}")
        
        # Save debug results to CSV
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        debug_data = {
            'edge_src': graph_edges[0],
            'edge_dst': graph_edges[1],
            'predicted_score': edge_scores,
            'ground_truth_label': aligned_labels
        }
        
        df = pd.DataFrame(debug_data)
        csv_path = f"{results_dir}/debug_eval_graph_{graph_id}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Debug results saved to: {csv_path}")
        
        return {
            'graph_id': graph_id,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'num_edges': len(edge_scores),
            'num_motif_edges': len(motif_edges),
            'csv_path': csv_path
        }
        
    except Exception as e:
        print(f"  Error in alignment: {e}")
        return None