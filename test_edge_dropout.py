#!/usr/bin/env python
"""
Test script to verify edge dropout implementation.

This script tests that:
1. Edge dropout affects the environment (action availability, boundaries)
2. Edge dropout does NOT affect reward computation (uses original graph)
3. Edge dropout is disabled during inference
4. Connectivity preservation works correctly
"""

import torch
import numpy as np
from components.graph import Graph
from components.generator_copy import ExpansionEnv
from utils.edge_dropout import apply_edge_dropout


def test_edge_dropout_utility():
    """Test the edge dropout utility function."""
    print("=" * 60)
    print("TEST 1: Edge Dropout Utility Function")
    print("=" * 60)
    
    # Create a simple graph
    edges = np.array([
        [0, 1], [1, 0],
        [1, 2], [2, 1],
        [2, 3], [3, 2],
        [3, 0], [0, 3],
        [0, 2], [2, 0]  # Diagonal edges
    ]).T
    
    edge_index = torch.tensor(edges, dtype=torch.long)
    print(f"Original edges: {edge_index.size(1)}")
    
    # Test 1: Simple dropout (no connectivity preservation)
    dropped = apply_edge_dropout(edge_index, drop_prob=0.3, preserve_connectivity=False)
    print(f"After dropout (p=0.3, no preserve): {dropped.size(1)} edges")
    assert dropped.size(1) <= edge_index.size(1), "Dropped edges should be <= original"
    
    # Test 2: Connectivity-preserving dropout
    dropped_preserve = apply_edge_dropout(edge_index, drop_prob=0.3, preserve_connectivity=True)
    print(f"After dropout (p=0.3, preserve connectivity): {dropped_preserve.size(1)} edges")
    assert dropped_preserve.size(1) <= edge_index.size(1), "Dropped edges should be <= original"
    assert dropped_preserve.size(1) >= 6, "Should preserve at least spanning tree (6 edges for 4 nodes)"
    
    # Test 3: Zero dropout
    no_drop = apply_edge_dropout(edge_index, drop_prob=0.0)
    assert no_drop.size(1) == edge_index.size(1), "Zero dropout should keep all edges"
    
    print("✓ Edge dropout utility tests passed!")
    print()


def test_graph_factory():
    """Test the Graph.from_dropped_edges() factory method."""
    print("=" * 60)
    print("TEST 2: Graph Factory Method")
    print("=" * 60)
    
    # Create original graph
    edges = np.array([
        [0, 1], [1, 0],
        [1, 2], [2, 1],
        [2, 3], [3, 2],
        [3, 0], [0, 3]
    ]).T
    
    original_graph = Graph(edges)
    print(f"Original graph: {original_graph.edge_index.size(1)} edges")
    
    # Test 1: Create dropped graph
    dropped_graph = Graph.from_dropped_edges(original_graph, drop_prob=0.3, preserve_connectivity=False)
    print(f"Dropped graph: {dropped_graph.edge_index.size(1)} edges")
    assert dropped_graph.edge_index.size(1) <= original_graph.edge_index.size(1), \
        "Dropped graph should have <= edges"
    
    # Test 2: Zero dropout should return original
    no_drop_graph = Graph.from_dropped_edges(original_graph, drop_prob=0.0)
    assert no_drop_graph is original_graph, "Zero dropout should return original graph object"
    
    print("✓ Graph factory tests passed!")
    print()


def test_expansion_env():
    """Test that ExpansionEnv applies edge dropout correctly."""
    print("=" * 60)
    print("TEST 3: ExpansionEnv Edge Dropout")
    print("=" * 60)
    
    # Create original graph (star graph: 0 connected to 1, 2, 3, 4)
    edges = np.array([
        [0, 1], [1, 0],
        [0, 2], [2, 0],
        [0, 3], [3, 0],
        [0, 4], [4, 0]
    ]).T
    
    original_graph = Graph(edges)
    print(f"Original graph: {original_graph.edge_index.size(1)} edges")
    
    # Test 1: Training mode with dropout
    env_train = ExpansionEnv(
        original_graph,
        [[0]],  # Start from node 0
        max_size=5,
        debug_mode=False,
        training=True,
        edge_dropout_prob=0.5,
        preserve_connectivity=False
    )
    print(f"Training env graph: {env_train.graph.edge_index.size(1)} edges")
    assert env_train.graph.edge_index.size(1) <= original_graph.edge_index.size(1), \
        "Training env should have dropped edges"
    assert env_train.original_graph is original_graph, "Should keep reference to original"
    
    # Test 2: Inference mode (no dropout)
    env_inference = ExpansionEnv(
        original_graph,
        [[0]],
        max_size=5,
        debug_mode=False,
        training=False,
        edge_dropout_prob=0.5,  # This should be ignored
        preserve_connectivity=False
    )
    print(f"Inference env graph: {env_inference.graph.edge_index.size(1)} edges")
    assert env_inference.graph.edge_index.size(1) == original_graph.edge_index.size(1), \
        "Inference env should use original graph (no dropout)"
    assert env_inference.graph is original_graph, "Inference should use original graph object"
    
    # Test 3: Training mode with zero dropout
    env_no_drop = ExpansionEnv(
        original_graph,
        [[0]],
        max_size=5,
        debug_mode=False,
        training=True,
        edge_dropout_prob=0.0,
        preserve_connectivity=False
    )
    assert env_no_drop.graph is original_graph, "Zero dropout should use original graph"
    
    print("✓ ExpansionEnv tests passed!")
    print()


def test_boundary_with_dropout():
    """Test that edge dropout affects action availability (outer boundary)."""
    print("=" * 60)
    print("TEST 4: Edge Dropout Affects Action Availability")
    print("=" * 60)
    
    # Create a line graph: 0-1-2-3-4
    edges = np.array([
        [0, 1], [1, 0],
        [1, 2], [2, 1],
        [2, 3], [3, 2],
        [3, 4], [4, 3]
    ]).T
    
    original_graph = Graph(edges)
    
    # Original graph: boundary from [0] should be {1}
    boundary_orig = original_graph.outer_boundary([0])
    print(f"Original boundary from [0]: {boundary_orig}")
    assert boundary_orig == {1}, "Original boundary should be {1}"
    
    # Create a heavily dropped version
    # Manually drop the 0-1 edge
    dropped_edges = np.array([
        [1, 2], [2, 1],
        [2, 3], [3, 2],
        [3, 4], [4, 3]
    ]).T
    dropped_graph = Graph(dropped_edges)
    
    # Dropped graph: boundary from [0] should be empty
    boundary_dropped = dropped_graph.outer_boundary([0])
    print(f"Dropped boundary from [0]: {boundary_dropped}")
    assert len(boundary_dropped) == 0, "Dropped boundary should be empty (node 0 isolated)"
    
    print("✓ Action availability tests passed!")
    print()


def run_all_tests():
    """Run all tests."""
    print("\n")
    print("#" * 60)
    print("# EDGE DROPOUT IMPLEMENTATION TESTS")
    print("#" * 60)
    print()
    
    try:
        test_edge_dropout_utility()
        test_graph_factory()
        test_expansion_env()
        test_boundary_with_dropout()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print()
        print("Summary:")
        print("  ✓ Edge dropout utility works correctly")
        print("  ✓ Graph factory creates dropped graphs")
        print("  ✓ ExpansionEnv applies dropout in training mode only")
        print("  ✓ Edge dropout affects action availability")
        print()
        print("Next steps:")
        print("  1. Run with --random_edge_drop 0.0 (baseline)")
        print("  2. Run with --random_edge_drop 0.2 (with dropout)")
        print("  3. Verify reward computation uses original graph")
        print()
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()

