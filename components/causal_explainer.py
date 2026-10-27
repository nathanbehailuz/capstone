"""
Causal Mechanism for Reinforced Graph Explainer

This module implements the causal mechanism described in the CAUSAL_MECHANISM_GUIDE.md
to enhance the existing reinforcement learning approach for subgraph generation.
The causal mechanism uses counterfactual reasoning to identify causal edges by comparing
factual (full graph) and counterfactual (selected edges only) GNN representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import copy


class NodeLevelCausalExplainer(nn.Module):
    """
    Causal explainer using node-level representation differences.
    Suitable for: Node classification, small-to-medium graphs
    
    This class implements the causal mechanism by:
    1. Computing factual node representations with all edges
    2. Computing counterfactual node representations with only selected edges
    3. Taking the difference to isolate causal effects
    4. Using this difference to score edge importance
    """
    
    def __init__(self, gnn_model, num_labels: int, hidden_size: int, use_edge_attr: bool = False):
        super().__init__()
        
        self.model = gnn_model
        self.model.eval()  # Keep GNN frozen
        
        # CRITICAL FIX: Freeze all parameters to prevent gradient leakage
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.use_edge_attr = use_edge_attr
        
        # Network to process edge representations
        # Input: concatenated node representations from both endpoints
        self.edge_action_rep_generator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        
        # Network to generate action probabilities
        # CRITICAL FIX: Input is hidden_size (not hidden_size * (1 + use_edge_attr))
        # because we don't actually concatenate edge features yet
        self.edge_action_prob_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels)
        )
    
    def forward(self, graph, state, target_y):
        """
        Compute edge importance scores using node-level causal reasoning.
        
        Args:
            graph: PyG Data object with x, edge_index, edge_attr, batch
            state: Boolean tensor [num_edges] indicating selected edges
            target_y: Target class label for explanation
            
        Returns:
            action_probs: Probability distribution over available edges [num_available_edges]
        """
        
        # ========== STEP 1: Separate edges into selected and available ==========
        
        # Selected/occupied edges (counterfactual graph)
        ocp_edge_index = graph.edge_index.T[state].T  # [2, num_selected_edges]
        ocp_edge_attr = graph.edge_attr[state] if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else None
        
        # Available edges (candidates for selection)
        ava_edge_index = graph.edge_index.T[~state].T  # [2, num_available_edges]
        ava_edge_attr = graph.edge_attr[~state] if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else None
        
        # CRITICAL FIX: Guard against no available edges
        if ava_edge_index.size(1) == 0:
            return torch.empty(0, device=graph.x.device)
        
        # ========== STEP 2: CAUSAL MECHANISM - Compute Factual Representations ==========
        
        # Factual: Node representations with ALL edges
        # CRITICAL FIX: Wrap in no_grad to prevent gradient leakage
        with torch.no_grad():
            ava_node_reps_0 = self.model.get_node_reps(
                graph.x,           # All node features
                graph.edge_index,  # ALL edges (factual)
                graph.edge_attr if hasattr(graph, 'edge_attr') else None,   # All edge features
                graph.batch if hasattr(graph, 'batch') else None
            )  # Shape: [num_nodes, hidden_dim]
        
        # ========== STEP 3: CAUSAL MECHANISM - Compute Counterfactual Representations ==========
        
        # Counterfactual: Node representations with ONLY selected edges
        if len(torch.where(state == True)[0]) == 0 or ocp_edge_index.size(1) == 0:
            ava_node_reps_1 = torch.zeros_like(ava_node_reps_0)
        else:
            # Add explicit edge count check
            if ocp_edge_index.size(1) < 1:
                print(f"[CAUSAL] Warning: Only {ocp_edge_index.size(1)} edges in counterfactual", flush=True)
                ava_node_reps_1 = torch.zeros_like(ava_node_reps_0)
            else:
                try:
                    with torch.no_grad():
                        ava_node_reps_1 = self.model.get_node_reps(
                            graph.x,
                            ocp_edge_index,
                            ocp_edge_attr,
                            graph.batch if hasattr(graph, 'batch') else None
                        )
                except (RuntimeError, IndexError) as e:
                    # More specific error logging
                    print(f"[CAUSAL] ERROR in get_node_reps: {type(e).__name__}: {str(e)}", flush=True)
                    print(f"[CAUSAL] Edge index shape: {ocp_edge_index.shape}, num_nodes: {graph.x.size(0)}", flush=True)
                    ava_node_reps_1 = torch.zeros_like(ava_node_reps_0)
        
        # ========== STEP 4: CAUSAL EFFECT - Compute the Difference ==========
        
        # This difference represents the CAUSAL CONTRIBUTION of unselected edges
        # Interpretation: "How much do the remaining edges change node representations?"
        ava_node_reps = ava_node_reps_0 - ava_node_reps_1  # [num_nodes, hidden_dim]
        
        # ========== STEP 5: Generate Edge Representations ==========
        
        # For each available edge, concatenate representations of its endpoint nodes
        # This captures the causal contribution at both endpoints
        ava_action_reps = torch.cat([
            ava_node_reps[ava_edge_index[0]],  # Source node causal features
            ava_node_reps[ava_edge_index[1]]   # Target node causal features
        ], dim=1)  # Shape: [num_available_edges, hidden_dim * 2]
        
        # Process edge representations
        ava_action_reps = self.edge_action_rep_generator(ava_action_reps)
        # Shape: [num_available_edges, hidden_dim]
        
        # ========== STEP 6: (Optional) Incorporate Edge Features ==========
        
        if self.use_edge_attr and ava_edge_attr is not None:
            # Note: This would require edge embedding layer in the GNN model
            # For now, we'll skip this as it's not implemented in the current models
            pass
        
        # ========== STEP 7: Generate Action Probabilities ==========
        
        # Predict importance for target class
        action_probs = self.edge_action_prob_generator(ava_action_reps)[:, target_y]
        action_probs = action_probs.reshape(-1)
        
        # Normalize to probability distribution
        action_probs = F.softmax(action_probs, dim=0)
        
        return action_probs


class GraphLevelCausalExplainer(nn.Module):
    """
    Causal explainer using graph-level representation differences.
    Suitable for: Graph classification, large graphs
    More efficient than node-level for graph-level predictions.
    """
    
    def __init__(self, gnn_model, num_labels: int, hidden_size: int, use_edge_attr: bool = False):
        super().__init__()
        
        self.model = gnn_model
        self.model.eval()
        
        # CRITICAL FIX: Freeze all parameters to prevent gradient leakage
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.use_edge_attr = use_edge_attr
        
        # Network to process edge representations
        # Input now includes graph-level context
        # For GraphGCN, hidden_size is 20, so input is 20*2=40
        self.edge_action_rep_generator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),  # 40 -> 80
            nn.ELU(),
            nn.Linear(hidden_size * 4, hidden_size * 2),  # 80 -> 40
            nn.ELU(),
            nn.Linear(hidden_size * 2, hidden_size)       # 40 -> 20
        )
        
        # Separate predictor for each class (more expressive)
        # Input size will be hidden_size * 2 (edge reps) + graph_rep_size (varies by model)
        # We'll make it flexible to handle different graph representation sizes
        self.edge_action_prob_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size * 2),  # Flexible input size
                nn.ELU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ELU(),
                nn.Linear(hidden_size, 1)
            ) for _ in range(num_labels)
        ])
    
    def forward(self, graph, state, train_flag=False):
        """
        Compute edge importance scores using graph-level causal reasoning.
        
        Args:
            graph: PyG Data object (can be batched)
            state: Boolean tensor [num_edges] indicating selected edges
            train_flag: Whether in training mode (for exploration)
            
        Returns:
            ava_action_probs: Edge importance scores [num_available_edges]
            selected_action_probs: Scores for selected actions [batch_size]
            selected_actions: Indices of selected actions [batch_size]
            unique_batch: Batch indices that have available actions
        """
        
        # ========== STEP 1: CAUSAL MECHANISM - Compute Factual Graph Representation ==========
        
        # Factual: Graph representation with ALL edges
        graph_rep = self.model.get_graph_rep(
            graph.x,
            graph.edge_index,  # ALL edges (factual)
            graph.edge_attr if hasattr(graph, 'edge_attr') else None,
            graph.batch if hasattr(graph, 'batch') else None
        )  # Shape: [batch_size, hidden_dim]
        
        # ========== STEP 2: CAUSAL MECHANISM - Compute Counterfactual Graph Representation ==========
        
        # Counterfactual: Graph representation with ONLY selected edges
        if len(torch.where(state == True)[0]) == 0:
            # No edges selected yet - zero representation
            subgraph_rep = torch.zeros(graph_rep.size()).to(graph_rep.device)
        else:
            # Create subgraph with only selected edges
            subgraph = self._relabel_graph(graph, state)
            subgraph_rep = self.model.get_graph_rep(
                subgraph.x,
                subgraph.edge_index,  # ONLY selected edges (counterfactual)
                subgraph.edge_attr if hasattr(subgraph, 'edge_attr') else None,
                subgraph.batch if hasattr(subgraph, 'batch') else None
            )  # Shape: [batch_size, hidden_dim]
        
        # ========== STEP 3: Get Available Edges ==========
        
        ava_edge_index = graph.edge_index.T[~state].T
        ava_edge_attr = graph.edge_attr[~state] if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else None
        
        # Get node representations for available edges
        # Note: We use only available edges to compute node representations
        # This is more efficient than computing all node reps
        ava_node_reps = self.model.get_node_reps(
            graph.x,
            ava_edge_index,
            ava_edge_attr,
            graph.batch if hasattr(graph, 'batch') else None
        )
        
        # ========== STEP 4: Generate Edge Representations ==========
        
        # Create edge representations by concatenating node representations
        ava_action_reps = torch.cat([
            ava_node_reps[ava_edge_index[0]],  # Source node
            ava_node_reps[ava_edge_index[1]]   # Target node
        ], dim=1)  # Shape: [num_available_edges, hidden_dim * 2]
        
        # Note: Edge attributes would be added here if use_edge_attr=True
        # but current models don't support edge attributes
        
        # Process edge representations
        ava_action_reps = self.edge_action_rep_generator(ava_action_reps)
        # Shape: [num_available_edges, hidden_dim]
        
        # ========== STEP 5: Identify which graph each edge belongs to ==========
        
        batch = graph.batch if hasattr(graph, 'batch') else torch.zeros(graph.x.size(0), dtype=torch.int64, device=graph.x.device)
        
        # Bounds checking for edge indices
        if ava_edge_index.size(1) == 0:
            # No available edges, return empty results
            return torch.tensor([], device=graph.x.device), torch.tensor([], device=graph.x.device), torch.tensor([], dtype=torch.long, device=graph.x.device), torch.tensor([], dtype=torch.long, device=graph.x.device)
        
        # Ensure edge indices are within bounds
        max_node_idx = graph.x.size(0) - 1
        if torch.any(ava_edge_index[0] > max_node_idx) or torch.any(ava_edge_index[1] > max_node_idx):
            print(f"[CAUSAL] Warning: Edge indices out of bounds. Max node idx: {max_node_idx}, Edge indices: {ava_edge_index.max()}")
            # Filter out invalid edges
            valid_mask = (ava_edge_index[0] <= max_node_idx) & (ava_edge_index[1] <= max_node_idx)
            ava_edge_index = ava_edge_index[:, valid_mask]
            ava_action_reps = ava_action_reps[valid_mask]
            if ava_edge_attr is not None:
                ava_edge_attr = ava_edge_attr[valid_mask]
        
        if ava_edge_index.size(1) == 0:
            # No valid edges after filtering
            return torch.tensor([], device=graph.x.device), torch.tensor([], device=graph.x.device), torch.tensor([], dtype=torch.long, device=graph.x.device), torch.tensor([], dtype=torch.long, device=graph.x.device)
        
        ava_action_batch = batch[ava_edge_index[0]]  # Batch assignment for each edge
        ava_y_batch = graph.y[ava_action_batch] if hasattr(graph, 'y') else torch.zeros(len(ava_action_batch), dtype=torch.int64, device=graph.x.device)
        
        # Handle case where some graphs have no available edges
        unique_batch, ava_action_batch = torch.unique(ava_action_batch, return_inverse=True)
        
        # ========== STEP 6: CAUSAL EFFECT - Incorporate Graph-Level Causal Information ==========
        
        # Causal effect at graph level: What's the difference between full and subgraph?
        action_graph_reps = graph_rep - subgraph_rep  # [batch_size, hidden_dim]
        
        # Broadcast to each edge based on its graph membership
        action_graph_reps = action_graph_reps[unique_batch]  # [num_active_graphs, hidden_dim]
        action_graph_reps = action_graph_reps[ava_action_batch]  # [num_available_edges, hidden_dim]
        
        # Concatenate edge-level and graph-level causal information
        # Ensure both tensors have compatible shapes
        if ava_action_reps.size(1) != action_graph_reps.size(1):
            # If dimensions don't match, we need to adjust
            # This happens when edge reps (hidden_size) != graph reps (hidden_size * 2)
            if ava_action_reps.size(1) < action_graph_reps.size(1):
                # Pad edge representations to match graph representation size
                padding = action_graph_reps.size(1) - ava_action_reps.size(1)
                ava_action_reps = torch.cat([ava_action_reps, torch.zeros(ava_action_reps.size(0), padding, device=ava_action_reps.device)], dim=1)
            else:
                # Truncate edge representations to match graph representation size
                ava_action_reps = ava_action_reps[:, :action_graph_reps.size(1)]
        
        action_graph_reps = torch.cat([ava_action_reps, action_graph_reps], dim=1)
        # Shape: [num_available_edges, hidden_dim * 2]
        
        # ========== STEP 7: Generate Action Probabilities ==========
        
        # Compute scores for each class using separate predictors
        action_probs = []
        for class_predictor in self.edge_action_prob_generators:
            class_action_probs = class_predictor(action_graph_reps)
            action_probs.append(class_action_probs)
        
        action_probs = torch.cat(action_probs, dim=1)  # [num_available_edges, num_labels]
        
        # Select scores for target class
        ava_y_batch_for_gathering = ava_y_batch[unique_batch[ava_action_batch]]
        action_probs = action_probs.gather(1, ava_y_batch_for_gathering.view(-1, 1))
        action_probs = action_probs.reshape(-1)  # [num_available_edges]
        
        # ========== STEP 8: Select Best Action per Graph ==========
        
        # Use scatter_max to find best edge for each graph in the batch
        try:
            from torch_scatter import scatter_max
            # Bounds checking for scatter operation
            if len(ava_action_batch) == 0 or len(action_probs) == 0:
                return torch.tensor([], device=graph.x.device), torch.tensor([], device=graph.x.device), torch.tensor([], dtype=torch.long, device=graph.x.device), torch.tensor([], dtype=torch.long, device=graph.x.device)
            
            # Ensure batch indices are valid
            max_batch_idx = unique_batch.max().item() if len(unique_batch) > 0 else 0
            if torch.any(ava_action_batch > max_batch_idx):
                print(f"[CAUSAL] Warning: Batch indices out of bounds. Max batch idx: {max_batch_idx}, Batch indices: {ava_action_batch.max()}")
                return torch.tensor([], device=graph.x.device), torch.tensor([], device=graph.x.device), torch.tensor([], dtype=torch.long, device=graph.x.device), torch.tensor([], dtype=torch.long, device=graph.x.device)
            
            selected_action_probs, selected_actions = scatter_max(
                action_probs,
                ava_action_batch
            )  # Returns: [num_active_graphs], [num_active_graphs]
        except ImportError:
            # Fallback implementation if torch_scatter is not available
            print("[WARNING] torch_scatter not available, using fallback implementation")
            selected_action_probs = []
            selected_actions = []
            for batch_id in torch.unique(ava_action_batch):
                mask = ava_action_batch == batch_id
                batch_probs = action_probs[mask]
                if len(batch_probs) > 0:
                    max_prob, max_idx = torch.max(batch_probs, dim=0)
                    selected_action_probs.append(max_prob)
                    # Find the original index in the full tensor
                    batch_indices = torch.where(mask)[0]
                    selected_actions.append(batch_indices[max_idx])
            selected_action_probs = torch.stack(selected_action_probs) if selected_action_probs else torch.tensor([])
            selected_actions = torch.stack(selected_actions) if selected_actions else torch.tensor([], dtype=torch.long)
        except Exception as e:
            print(f"[CAUSAL] Error in scatter_max: {e}")
            return torch.tensor([], device=graph.x.device), torch.tensor([], device=graph.x.device), torch.tensor([], dtype=torch.long, device=graph.x.device), torch.tensor([], dtype=torch.long, device=graph.x.device)
        
        if train_flag:
            # During training, add exploration with random selection
            rand_action_probs = torch.rand(action_probs.size()).to(action_probs.device)
            try:
                from torch_scatter import scatter_max
                _, rand_actions = scatter_max(rand_action_probs, ava_action_batch)
            except ImportError:
                # Fallback implementation
                rand_actions = []
                for batch_id in torch.unique(ava_action_batch):
                    mask = ava_action_batch == batch_id
                    batch_probs = rand_action_probs[mask]
                    if len(batch_probs) > 0:
                        _, max_idx = torch.max(batch_probs, dim=0)
                        batch_indices = torch.where(mask)[0]
                        rand_actions.append(batch_indices[max_idx])
                rand_actions = torch.stack(rand_actions) if rand_actions else torch.tensor([], dtype=torch.long)
            return action_probs, action_probs[rand_actions], rand_actions, unique_batch
        
        return action_probs, selected_action_probs, selected_actions, unique_batch
    
    def _relabel_graph(self, graph, selection):
        """
        Create a subgraph containing only selected edges.
        This creates the counterfactual graph for causal analysis.
        
        Args:
            graph: Original graph
            selection: Boolean mask of selected edges
            
        Returns:
            subgraph: Graph with only selected edges, nodes relabeled
        """
        subgraph = copy.deepcopy(graph)
        
        # Extract selected edges
        subgraph.edge_index = graph.edge_index.T[selection].T
        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            subgraph.edge_attr = graph.edge_attr[selection]
        
        # Get unique nodes in subgraph
        sub_nodes = torch.unique(subgraph.edge_index)
        
        # Extract node features
        subgraph.x = graph.x[sub_nodes]
        if hasattr(graph, 'batch') and graph.batch is not None:
            subgraph.batch = graph.batch[sub_nodes]
        
        # Relabel nodes to be contiguous [0, 1, 2, ...]
        row, col = graph.edge_index
        node_idx = row.new_full((graph.x.size(0),), -1)
        node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
        subgraph.edge_index = node_idx[subgraph.edge_index]
        
        return subgraph


class CausalRewardCalculator:
    """
    Calculates causal-based rewards for the reinforcement learning system.
    This integrates the causal mechanism into the existing reward function.
    """
    
    def __init__(
        self,
        gnn_model,
        causal_explainer,
        causal_weight: float = 0.5,
        predictor=None,
    ):
        """
        Args:
            gnn_model: The trained GNN model
            causal_explainer: Either NodeLevelCausalExplainer or GraphLevelCausalExplainer
            causal_weight: Weight for causal term in reward (0.0 to 1.0)
        """
        self.gnn_model = gnn_model
        self.causal_explainer = causal_explainer
        self.causal_weight = causal_weight
        self.predictor = predictor
        self.diagnostics = {
            "count": 0,
            "failed": 0,
            "saturated": 0,
            "last_raw": 0.0,
            "last_clipped": 0.0,
            "weight": causal_weight,
        }

    def set_weight(self, causal_weight):
        if not 0.0 <= causal_weight <= 1.0:
            raise ValueError("causal weight must be in [0, 1]")
        self.causal_weight = float(causal_weight)
        self.diagnostics["weight"] = self.causal_weight
    
    def compute_causal_reward(self, graph, selected_nodes, target_y):
        """
        Compute causal-based reward for a given subgraph using TRUE causal deltas.
        
        Args:
            graph: Original graph
            selected_nodes: List of selected nodes
            target_y: Target class label
            
        Returns:
            causal_reward: Scalar reward based on marginal causal effect
        """
        if self.predictor is not None:
            seed = int(selected_nodes[0])
            instance_id = (
                int(self.predictor.batch[seed].item())
                if self.predictor.task == "graph_task"
                else seed
            )
            try:
                selected = self.predictor.predict(
                    selected_nodes,
                    instance_id=instance_id,
                    target_node=seed,
                    target_class=target_y,
                )
                edgeless = self.predictor.predict(
                    [seed],
                    instance_id=instance_id,
                    target_node=seed,
                    target_class=target_y,
                )
                selected_logp = torch.log_softmax(selected.logits, -1)[target_y]
                edgeless_logp = torch.log_softmax(edgeless.logits, -1)[target_y]
                if not torch.isfinite(selected_logp) or not torch.isfinite(edgeless_logp):
                    raise FloatingPointError("non-finite causal log probability")
                raw = float((selected_logp - edgeless_logp).item())
                raw += 0.5 if int(selected.logits.argmax()) == target_y else -0.5
                clipped = max(-1.0, min(1.0, raw))
                self.diagnostics.update(
                    count=self.diagnostics["count"] + 1,
                    saturated=self.diagnostics["saturated"] + int(clipped != raw),
                    last_raw=raw,
                    last_clipped=clipped,
                )
                return clipped
            except Exception:
                self.diagnostics["failed"] += 1
                raise

        raise RuntimeError(
            "causal rewards require a PredictorAdapter; placeholder "
            "neutral values are not allowed"
        )
        edge_index = graph.edge_index
        device = edge_index.device
        
        # CRITICAL FIX: Vectorized edge mask (no Python loop)
        node_mask = torch.zeros(graph.x.size(0), dtype=torch.bool, device=device)
        selected_nodes_tensor = torch.as_tensor(selected_nodes, dtype=torch.long, device=device)
        node_mask[selected_nodes_tensor] = True
        state = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        
        # Edge cases
        if state.sum() == 0:
            # No edges in subgraph - return small constant (not punish, just neutral)
            return 0.0
        
        if (~state).sum() == 0:
            # All edges selected (full graph) - not a good explanation
            return 0.0
        
        # CRITICAL FIX: Compute TRUE causal effect using CE-difference
        # Compare: control (subgraph) vs counterfactual (no subgraph / empty graph)
        
        with torch.no_grad():
            # Control: prediction with current subgraph
            ctrl_edge_index = edge_index[:, state]
            if ctrl_edge_index.size(1) == 0:
                return 0.0
            
            try:
                # Get prediction for subgraph
                edge_weights_ctrl = torch.ones(ctrl_edge_index.size(1), device=device)
                ctrl_logits = self.gnn_model(graph.x, ctrl_edge_index, edge_weights_ctrl)
                
                # Handle node-level vs graph-level predictions
                if ctrl_logits.dim() > 1 and ctrl_logits.size(0) > 1:
                    # Node classification: use seed node (first selected node)
                    seed_node = selected_nodes[0]
                    if seed_node < ctrl_logits.size(0):
                        ctrl_logits = ctrl_logits[seed_node].unsqueeze(0)
                    else:
                        return 0.0
                elif ctrl_logits.dim() == 1:
                    ctrl_logits = ctrl_logits.unsqueeze(0)
                
                # Compute log probability for target class
                ctrl_logp = torch.log_softmax(ctrl_logits, dim=-1).squeeze()
                if ctrl_logp.dim() == 0:
                    ctrl_logp = ctrl_logp.unsqueeze(0)
                ctrl_logp_c = ctrl_logp[target_y]
                
                # Counterfactual: prediction with NO edges (baseline)
                # This measures how much the subgraph improves prediction vs nothing
                baseline_logits = self.gnn_model(graph.x, torch.empty((2, 0), dtype=torch.long, device=device), 
                                                 torch.empty(0, device=device))
                
                if baseline_logits.dim() > 1 and baseline_logits.size(0) > 1:
                    seed_node = selected_nodes[0]
                    if seed_node < baseline_logits.size(0):
                        baseline_logits = baseline_logits[seed_node].unsqueeze(0)
                    else:
                        return 0.0
                elif baseline_logits.dim() == 1:
                    baseline_logits = baseline_logits.unsqueeze(0)
                
                baseline_logp = torch.log_softmax(baseline_logits, dim=-1).squeeze()
                if baseline_logp.dim() == 0:
                    baseline_logp = baseline_logp.unsqueeze(0)
                baseline_logp_c = baseline_logp[target_y]
                
                # Issue 5 fix: Add guard for NaN/inf in log probabilities
                if not torch.isfinite(ctrl_logp_c) or not torch.isfinite(baseline_logp_c):
                    print(f"[CAUSAL REWARD] Warning: Non-finite log-prob detected, returning 0.0")
                    return 0.0
                
                # Causal effect: how much does subgraph improve log-prob for target class?
                causal_delta = (ctrl_logp_c - baseline_logp_c).item()
                
                # Add correctness bonus
                pred_class = ctrl_logits.argmax(dim=-1).item()
                correctness_bonus = 0.5 if pred_class == target_y else -0.5
                
                # Issue 7 fix: Clamp causal reward to prevent extreme values
                causal_reward = torch.clamp(
                    torch.tensor(causal_delta + correctness_bonus),
                    min=-1.0,
                    max=1.0
                ).item()
                
                # Commented out to reduce log spam (prints ~5000 times per epoch)
                # print(f"[CAUSAL REWARD] Subgraph: {len(selected_nodes)} nodes, {state.sum().item()} edges | "
                #       f"Delta: {causal_delta:.4f}, Correct: {pred_class==target_y}, Total: {causal_reward:.4f}")
                
                return causal_reward
                
            except Exception:
                self.diagnostics["failed"] += 1
                raise
    
    def compute_combined_reward(self, original_reward, causal_reward):
        """
        Combine original reward with causal reward.
        
        Args:
            original_reward: Reward from existing system
            causal_reward: Reward from causal mechanism
            
        Returns:
            combined_reward: Weighted combination of both rewards
        """
        return (1 - self.causal_weight) * original_reward + self.causal_weight * causal_reward
