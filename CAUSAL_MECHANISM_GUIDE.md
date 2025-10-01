# Causal Mechanism Implementation Guide
## RC-Explainer: Node-Level and Graph-Level Causal Reasoning

This document provides a detailed explanation of how the causal mechanism is implemented in the Reinforced Causal Explainer, suitable for integration into other projects.

---

## Overview of the Causal Approach

The RC-Explainer uses **counterfactual reasoning** to identify causal edges by comparing:
- **Factual**: GNN representations with the full/original graph
- **Counterfactual**: GNN representations with only selected edges
- **Causal Effect**: The difference between factual and counterfactual representations

This implements a form of **interventional causality** where we ask:
> *"What is the causal effect of including/excluding certain edges on the GNN's internal representations?"*

---

## Prerequisites

Your GNN model must implement these methods to enable causal analysis:

```python
class YourGNNModel(torch.nn.Module):
    def get_node_reps(self, x, edge_index, edge_attr, batch):
        """
        Extract node-level representations after message passing.
        
        Args:
            x: Node features [num_nodes, node_feat_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            node_representations: [num_nodes, hidden_dim]
        """
        # Apply your GNN layers (GCN, GAT, GIN, etc.)
        # Return node embeddings BEFORE pooling
        pass
    
    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        """
        Extract graph-level representations after pooling.
        
        Args:
            x: Node features [num_nodes, node_feat_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            graph_representations: [batch_size, hidden_dim]
        """
        # Get node reps, then apply global pooling (mean, sum, max, etc.)
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)  # or global_add_pool, etc.
        return graph_x
```

---

## Method 1: Node-Level Causal Reasoning

### Concept

Compare node representations when using:
1. **All edges** in the graph (factual state)
2. **Only selected edges** (counterfactual state)

The difference captures the causal contribution of the unselected edges.

### Implementation

```python
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

class NodeLevelCausalExplainer(torch.nn.Module):
    """
    Causal explainer using node-level representation differences.
    Suitable for: Node classification, small-to-medium graphs
    """
    
    def __init__(self, gnn_model, num_labels, hidden_size, use_edge_attr=False):
        super().__init__()
        
        self.model = gnn_model
        self.model.eval()  # Keep GNN frozen
        
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.use_edge_attr = use_edge_attr
        
        # Network to process edge representations
        # Input: concatenated node representations from both endpoints
        self.edge_action_rep_generator = Sequential(
            Linear(hidden_size * 2, hidden_size * 2),
            ReLU(),
            Linear(hidden_size * 2, hidden_size),
        )
        
        # Network to generate action probabilities
        self.edge_action_prob_generator = Sequential(
            Linear(hidden_size * (1 + use_edge_attr), hidden_size),
            ReLU(),
            Linear(hidden_size, num_labels)
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
        ocp_edge_attr = graph.edge_attr[state]         # [num_selected_edges, edge_feat_dim]
        
        # Available edges (candidates for selection)
        ava_edge_index = graph.edge_index.T[~state].T  # [2, num_available_edges]
        ava_edge_attr = graph.edge_attr[~state]         # [num_available_edges, edge_feat_dim]
        
        # ========== STEP 2: CAUSAL MECHANISM - Compute Factual Representations ==========
        
        # Factual: Node representations with ALL edges
        ava_node_reps_0 = self.model.get_node_reps(
            graph.x,           # All node features
            graph.edge_index,  # ALL edges (factual)
            graph.edge_attr,   # All edge features
            graph.batch
        )  # Shape: [num_nodes, hidden_dim]
        
        # ========== STEP 3: CAUSAL MECHANISM - Compute Counterfactual Representations ==========
        
        # Counterfactual: Node representations with ONLY selected edges
        ava_node_reps_1 = self.model.get_node_reps(
            graph.x,          # Same node features
            ocp_edge_index,   # ONLY selected edges (counterfactual)
            ocp_edge_attr,    # Only selected edge features
            graph.batch
        )  # Shape: [num_nodes, hidden_dim]
        
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
        
        if self.use_edge_attr:
            ava_edge_reps = self.model.edge_emb(ava_edge_attr)
            ava_action_reps = torch.cat([ava_action_reps, ava_edge_reps], dim=1)
        
        # ========== STEP 7: Generate Action Probabilities ==========
        
        # Predict importance for target class
        action_probs = self.edge_action_prob_generator(ava_action_reps)[:, target_y]
        action_probs = action_probs.reshape(-1)
        
        # Normalize to probability distribution
        action_probs = F.softmax(action_probs, dim=0)
        
        return action_probs


# ========== USAGE EXAMPLE ==========

def example_node_level_usage():
    """
    Example of using node-level causal explainer
    """
    # Assume you have a trained GNN and a graph
    gnn_model = YourTrainedGNN()
    graph = your_graph_data  # PyG Data object
    
    # Initialize explainer
    explainer = NodeLevelCausalExplainer(
        gnn_model=gnn_model,
        num_labels=3,
        hidden_size=64,
        use_edge_attr=False
    )
    
    # Start with no edges selected
    state = torch.zeros(graph.num_edges, dtype=torch.bool)
    
    # Get target prediction
    target_y = graph.y.item()
    
    # Iteratively select important edges
    important_edges = []
    for step in range(10):  # Select top 10 edges
        # Get edge importance scores
        edge_probs = explainer(graph, state, target_y)
        
        # Select most important edge
        best_edge_idx = torch.argmax(edge_probs)
        
        # Update state
        available_edges = torch.where(~state)[0]
        selected_edge = available_edges[best_edge_idx]
        state[selected_edge] = True
        
        important_edges.append(selected_edge.item())
    
    return important_edges
```

### When to Use Node-Level Causal Reasoning

✅ **Use when:**
- Working with node classification tasks
- Graphs are small to medium sized (< 10K nodes)
- You need fine-grained edge importance scores
- Computational resources are limited

❌ **Avoid when:**
- Working with very large graphs (expensive to compute all node reps)
- Only care about graph-level explanations

---

## Method 2: Graph-Level Causal Reasoning

### Concept

Compare graph-level representations when using:
1. **Full graph** with all edges (factual state)
2. **Current subgraph** with only selected edges (counterfactual state)

This captures the causal effect at the entire graph level, which is more efficient for graph classification.

### Implementation

```python
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, ELU
from torch_scatter import scatter_max

class GraphLevelCausalExplainer(torch.nn.Module):
    """
    Causal explainer using graph-level representation differences.
    Suitable for: Graph classification, large graphs
    More efficient than node-level for graph-level predictions.
    """
    
    def __init__(self, gnn_model, num_labels, hidden_size, use_edge_attr=False):
        super().__init__()
        
        self.model = gnn_model
        self.model.eval()
        
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.use_edge_attr = use_edge_attr
        
        # Network to process edge representations
        # Input now includes graph-level context
        self.edge_action_rep_generator = Sequential(
            Linear(hidden_size * (2 + use_edge_attr), hidden_size * 4),
            ELU(),
            Linear(hidden_size * 4, hidden_size * 2),
            ELU(),
            Linear(hidden_size * 2, hidden_size)
        )
        
        # Separate predictor for each class (more expressive)
        self.edge_action_prob_generators = torch.nn.ModuleList([
            Sequential(
                Linear(hidden_size * 2, hidden_size * 2),  # *2 because we concat with graph rep
                ELU(),
                Linear(hidden_size * 2, hidden_size),
                ELU(),
                Linear(hidden_size, 1)
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
            graph.edge_attr,
            graph.batch
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
                subgraph.edge_attr,
                subgraph.batch
            )  # Shape: [batch_size, hidden_dim]
        
        # ========== STEP 3: Get Available Edges ==========
        
        ava_edge_index = graph.edge_index.T[~state].T
        ava_edge_attr = graph.edge_attr[~state]
        
        # Get node representations for available edges
        # Note: We use only available edges to compute node representations
        # This is more efficient than computing all node reps
        ava_node_reps = self.model.get_node_reps(
            graph.x,
            ava_edge_index,
            ava_edge_attr,
            graph.batch
        )
        
        # ========== STEP 4: Generate Edge Representations ==========
        
        if self.use_edge_attr:
            ava_edge_reps = self.model.edge_emb(ava_edge_attr)
            ava_action_reps = torch.cat([
                ava_node_reps[ava_edge_index[0]],  # Source node
                ava_node_reps[ava_edge_index[1]],  # Target node
                ava_edge_reps                       # Edge features
            ], dim=1)
        else:
            ava_action_reps = torch.cat([
                ava_node_reps[ava_edge_index[0]],
                ava_node_reps[ava_edge_index[1]]
            ], dim=1)  # Shape: [num_available_edges, hidden_dim * 2]
        
        # Process edge representations
        ava_action_reps = self.edge_action_rep_generator(ava_action_reps)
        # Shape: [num_available_edges, hidden_dim]
        
        # ========== STEP 5: Identify which graph each edge belongs to ==========
        
        ava_action_batch = graph.batch[ava_edge_index[0]]  # Batch assignment for each edge
        ava_y_batch = graph.y[ava_action_batch]            # Target label for each edge
        
        # Handle case where some graphs have no available edges
        unique_batch, ava_action_batch = torch.unique(ava_action_batch, return_inverse=True)
        
        # ========== STEP 6: CAUSAL EFFECT - Incorporate Graph-Level Causal Information ==========
        
        # Causal effect at graph level: What's the difference between full and subgraph?
        action_graph_reps = graph_rep - subgraph_rep  # [batch_size, hidden_dim]
        
        # Broadcast to each edge based on its graph membership
        action_graph_reps = action_graph_reps[unique_batch]  # [num_active_graphs, hidden_dim]
        action_graph_reps = action_graph_reps[ava_action_batch]  # [num_available_edges, hidden_dim]
        
        # Concatenate edge-level and graph-level causal information
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
        selected_action_probs, selected_actions = scatter_max(
            action_probs,
            ava_action_batch
        )  # Returns: [num_active_graphs], [num_active_graphs]
        
        if train_flag:
            # During training, add exploration with random selection
            rand_action_probs = torch.rand(action_probs.size()).to(action_probs.device)
            _, rand_actions = scatter_max(rand_action_probs, ava_action_batch)
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
        import copy
        subgraph = copy.deepcopy(graph)
        
        # Extract selected edges
        subgraph.edge_index = graph.edge_index.T[selection].T
        subgraph.edge_attr = graph.edge_attr[selection]
        
        # Get unique nodes in subgraph
        sub_nodes = torch.unique(subgraph.edge_index)
        
        # Extract node features
        subgraph.x = graph.x[sub_nodes]
        subgraph.batch = graph.batch[sub_nodes]
        
        # Relabel nodes to be contiguous [0, 1, 2, ...]
        row, col = graph.edge_index
        node_idx = row.new_full((graph.num_nodes,), -1)
        node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
        subgraph.edge_index = node_idx[subgraph.edge_index]
        
        return subgraph


# ========== USAGE EXAMPLE ==========

def example_graph_level_usage():
    """
    Example of using graph-level causal explainer with batched graphs
    """
    from torch_geometric.data import DataLoader
    
    # Assume you have a trained GNN and a dataset
    gnn_model = YourTrainedGNN()
    dataset = your_dataset  # List of PyG Data objects
    
    # Initialize explainer
    explainer = GraphLevelCausalExplainer(
        gnn_model=gnn_model,
        num_labels=3,
        hidden_size=128,
        use_edge_attr=False
    )
    
    # Create batched graphs
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    for batch_graph in loader:
        # Start with no edges selected
        state = torch.zeros(batch_graph.num_edges, dtype=torch.bool)
        
        # Iteratively select important edges for each graph
        for step in range(10):
            # Get edge importance scores
            edge_probs, selected_probs, selected_actions, active_batch = \
                explainer(batch_graph, state, train_flag=False)
            
            # Update state: mark selected edges as occupied
            available_edges = torch.where(~state)[0]
            for action in selected_actions:
                state[available_edges[action]] = True
            
            # Optional: Check if all graphs have predictions
            if len(active_batch) == 0:
                break  # All graphs exhausted their edges
    
    return state  # Boolean mask of important edges
```

### When to Use Graph-Level Causal Reasoning

✅ **Use when:**
- Working with graph classification tasks
- Need to process batches of graphs efficiently
- Graphs are large (>10K nodes)
- Want to capture global causal structure
- Care about computational efficiency

❌ **Avoid when:**
- Need node-level or edge-level predictions
- Working with single small graphs

---

## Comparison: Node-Level vs Graph-Level

| **Aspect** | **Node-Level Causal** | **Graph-Level Causal** |
|------------|----------------------|------------------------|
| **Computation** | Computes node reps twice | Computes graph reps twice |
| **Complexity** | O(N × D) per forward | O(G × D) per forward |
| **Memory** | Higher (stores all node reps) | Lower (stores graph reps) |
| **Granularity** | Fine-grained edge scores | Holistic graph-level scores |
| **Batching** | Harder to batch | Native batching support |
| **Use Case** | Node classification, small graphs | Graph classification, large graphs |
| **Causal Information** | Local node-level effects | Global graph-level effects |

Where:
- N = number of nodes
- G = number of graphs in batch
- D = hidden dimension

---

## Key Mathematical Formulation

### Node-Level Causal Effect

For each edge $(u, v)$:

```
Causal_Effect(u,v) = [h_u^full - h_u^sub, h_v^full - h_v^sub]

where:
  h_u^full = GNN_node_rep(G_full)[u]      # Node u rep with all edges
  h_u^sub  = GNN_node_rep(G_selected)[u]  # Node u rep with selected edges
```

### Graph-Level Causal Effect

For each edge $(u, v)$ in graph $G_i$:

```
Causal_Effect(u,v) = [h_u, h_v, g_i^full - g_i^sub]

where:
  g_i^full = GNN_graph_rep(G_full)      # Graph rep with all edges
  g_i^sub  = GNN_graph_rep(G_selected)  # Graph rep with selected edges
  h_u, h_v = GNN_node_rep(available_edges)[u,v]
```

---

## Integration Checklist

To integrate this causal mechanism into your project:

- [ ] **Step 1**: Ensure your GNN implements `get_node_reps()` and `get_graph_rep()`
- [ ] **Step 2**: Choose node-level or graph-level based on your task
- [ ] **Step 3**: Copy the appropriate explainer class
- [ ] **Step 4**: Implement the `_relabel_graph()` helper function
- [ ] **Step 5**: Set up iterative edge selection loop
- [ ] **Step 6**: (Optional) Add reinforcement learning for training
- [ ] **Step 7**: Implement evaluation metrics (ACC-AUC, Precision@K, etc.)

---

## Advanced: Combining with Reinforcement Learning

The causal mechanism can be trained using policy gradients:

```python
def train_causal_explainer_with_rl(explainer, gnn_model, train_loader, num_episodes=30):
    """
    Train the causal explainer using REINFORCE algorithm.
    """
    optimizer = torch.optim.Adam(explainer.parameters(), lr=1e-4, weight_decay=1e-5)
    
    for episode in range(num_episodes):
        for graph in train_loader:
            state = torch.zeros(graph.num_edges, dtype=torch.bool)
            
            full_pred = F.softmax(gnn_model(graph.x, graph.edge_index, 
                                            graph.edge_attr, graph.batch), dim=1)
            
            loss = 0
            for step in range(10):  # Budget of 10 edges
                # Get action probabilities using CAUSAL mechanism
                action_probs, selected_probs, selected_actions, _ = \
                    explainer(graph, state, train_flag=True)
                
                # Update state
                available_edges = torch.where(~state)[0]
                state[available_edges[selected_actions]] = True
                
                # Get subgraph prediction
                subgraph = explainer._relabel_graph(graph, state)
                sub_pred = F.softmax(gnn_model(subgraph.x, subgraph.edge_index,
                                               subgraph.edge_attr, subgraph.batch), dim=1)
                
                # Compute reward (mutual information + correctness)
                reward = torch.sum(full_pred * torch.log(sub_pred + 1e-15), dim=1)
                reward += 2 * (graph.y == sub_pred.argmax(dim=1)).float() - 1
                
                # Policy gradient loss
                loss += -torch.log(selected_probs + 1e-15) * reward
            
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
    
    return explainer
```

---

## Citation

If you use this causal mechanism in your research, please cite:

```
Reinforced Causal Explainer for Graph Neural Networks
Xiang Wang, Yingxin Wu, An Zhang, Fuli Feng, Xiangnan He, Tat-Seng Chua
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
```

---

## Summary

The causal mechanism works by:

1. **Computing factual representations** with all edges
2. **Computing counterfactual representations** with only selected edges  
3. **Taking the difference** to isolate causal effects
4. **Using this difference** to score edge importance

This provides a principled way to identify which edges causally contribute to a GNN's prediction, rather than just correlational importance.

Choose **node-level** for fine-grained analysis on small graphs, or **graph-level** for efficient batch processing on large graphs.
