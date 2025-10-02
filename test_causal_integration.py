#!/usr/bin/env python3
"""
Test script for validating the causal mechanism integration.

This script tests the basic functionality of the causal mechanism
without running the full training pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data

# Import our components
from models.GNN_paper import NodeGCN, GraphGCN
from components.causal_explainer import NodeLevelCausalExplainer, GraphLevelCausalExplainer


def create_test_graph(num_nodes=10, num_edges=15):
    """Create a simple test graph."""
    # Create random edge index
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    # Remove self-loops
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    
    # Create node features
    x = torch.randn(num_nodes, 5)
    
    # Create batch (single graph)
    batch = torch.zeros(num_nodes, dtype=torch.int64)
    
    # Create target label
    y = torch.randint(0, 3, (1,))
    
    return Data(x=x, edge_index=edge_index, batch=batch, y=y)


def test_node_level_causal():
    """Test node-level causal explainer."""
    print("Testing NodeLevelCausalExplainer...")
    
    # Create test graph
    graph = create_test_graph(num_nodes=10, num_edges=15)
    
    # Create GNN model
    gnn_model = NodeGCN(num_features=5, num_classes=3)
    
    # Create causal explainer
    causal_explainer = NodeLevelCausalExplainer(
        gnn_model=gnn_model,
        num_labels=3,
        hidden_size=60,  # 20 * 3 from NodeGCN
        use_edge_attr=False
    )
    
    # Create state (no edges selected initially)
    state = torch.zeros(graph.edge_index.size(1), dtype=torch.bool)
    
    # Test forward pass
    try:
        action_probs = causal_explainer(graph, state, target_y=graph.y.item())
        print(f"✓ NodeLevelCausalExplainer forward pass successful")
        print(f"  Action probabilities shape: {action_probs.shape}")
        print(f"  Action probabilities sum: {action_probs.sum().item():.4f}")
        
        # Test with some edges selected
        state[:5] = True  # Select first 5 edges
        action_probs = causal_explainer(graph, state, target_y=graph.y.item())
        print(f"✓ NodeLevelCausalExplainer with selected edges successful")
        print(f"  Available edges: {torch.sum(~state).item()}")
        
    except Exception as e:
        print(f"✗ NodeLevelCausalExplainer failed: {e}")
        return False
    
    return True


def test_graph_level_causal():
    """Test graph-level causal explainer."""
    print("\nTesting GraphLevelCausalExplainer...")
    
    # Create test graph
    graph = create_test_graph(num_nodes=10, num_edges=15)
    
    # Create GNN model
    gnn_model = GraphGCN(num_features=5, num_classes=3)
    
    # Create causal explainer
    causal_explainer = GraphLevelCausalExplainer(
        gnn_model=gnn_model,
        num_labels=3,
        hidden_size=20,  # Hidden size from GraphGCN (not the concatenated size)
        use_edge_attr=False
    )
    
    # Create state (no edges selected initially)
    state = torch.zeros(graph.edge_index.size(1), dtype=torch.bool)
    
    # Test forward pass
    try:
        action_probs, selected_probs, selected_actions, unique_batch = causal_explainer(
            graph, state, train_flag=False
        )
        print(f"✓ GraphLevelCausalExplainer forward pass successful")
        print(f"  Action probabilities shape: {action_probs.shape}")
        print(f"  Selected actions shape: {selected_actions.shape}")
        
        # Test with some edges selected
        state[:5] = True  # Select first 5 edges
        action_probs, selected_probs, selected_actions, unique_batch = causal_explainer(
            graph, state, train_flag=False
        )
        print(f"✓ GraphLevelCausalExplainer with selected edges successful")
        
    except Exception as e:
        print(f"✗ GraphLevelCausalExplainer failed: {e}")
        return False
    
    return True


def test_gnn_methods():
    """Test that GNN models have the required methods."""
    print("\nTesting GNN model methods...")
    
    # Test NodeGCN
    node_gnn = NodeGCN(num_features=5, num_classes=3)
    graph = create_test_graph(num_nodes=10, num_edges=15)
    
    try:
        # Test get_node_reps
        node_reps = node_gnn.get_node_reps(graph.x, graph.edge_index, None, graph.batch)
        print(f"✓ NodeGCN.get_node_reps successful, shape: {node_reps.shape}")
        
    except Exception as e:
        print(f"✗ NodeGCN.get_node_reps failed: {e}")
        return False
    
    # Test GraphGCN
    graph_gnn = GraphGCN(num_features=5, num_classes=3)
    
    try:
        # Test get_node_reps
        node_reps = graph_gnn.get_node_reps(graph.x, graph.edge_index, None, graph.batch)
        print(f"✓ GraphGCN.get_node_reps successful, shape: {node_reps.shape}")
        
        # Test get_graph_rep
        graph_reps = graph_gnn.get_graph_rep(graph.x, graph.edge_index, None, graph.batch)
        print(f"✓ GraphGCN.get_graph_rep successful, shape: {graph_reps.shape}")
        
    except Exception as e:
        print(f"✗ GraphGCN methods failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("CAUSAL MECHANISM INTEGRATION TEST")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    tests = [
        ("GNN Methods", test_gnn_methods),
        ("Node-Level Causal", test_node_level_causal),
        ("Graph-Level Causal", test_graph_level_causal),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:.<40} {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Causal mechanism integration is working.")
        return True
    else:
        print(f"\n❌ {len(results) - passed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
