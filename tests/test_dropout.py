import numpy as np
import torch

from components.generator_core import Agent
from components.generator_copy import ExpansionEnv, Generator
from components.graph import Graph
from utils.edge_dropout import apply_edge_dropout


def test_undirected_pairs_are_dropped_together():
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    generator = torch.Generator().manual_seed(4)
    dropped = apply_edge_dropout(
        edges, 0.5, preserve_connectivity=False, generator=generator
    )
    retained = {tuple(edge) for edge in dropped.t().tolist()}
    assert ((0, 1) in retained) == ((1, 0) in retained)
    assert ((1, 2) in retained) == ((2, 1) in retained)


def test_node_universe_survives_total_dropout(isolated_high_index_graph):
    dropped = Graph.from_dropped_edges(
        isolated_high_index_graph, 1.0, preserve_connectivity=False
    )
    assert dropped.n_nodes == 4
    assert dropped.edge_index.shape == (2, 0)
    assert dropped.isolated_nodes == {0, 1, 2, 3}


def test_zero_dropout_returns_original(triangle_graph):
    assert Graph.from_dropped_edges(triangle_graph, 0.0) is triangle_graph


def test_removed_triangle_edge_is_not_a_direct_action(triangle_graph):
    chain = Graph(
        torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]).numpy(),
        n_nodes=3,
    )
    env = ExpansionEnv(chain, [[0]], max_size=3)
    env.reset()
    assert env.graphs[0].outer_boundary([0]) == {1}
    try:
        env.step([2], [0])
    except ValueError:
        pass
    else:
        raise AssertionError("environment accepted a node outside the boundary")


def test_list_pretraining_includes_stop(triangle_graph):
    model = Agent(8, with_attr=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = Generator(
        triangle_graph,
        model,
        optimizer,
        device=torch.device("cpu"),
        max_size=5,
        min_steps_before_stop=0,
        use_value_clip=False,
        adaptive_kl_penalty=False,
    )
    loss = generator.train_from_lists([[0, 1, "EOS"]])
    assert np.isfinite(loss)


def test_terminal_seed_does_not_crash_batch(isolated_high_index_graph):
    env = ExpansionEnv(isolated_high_index_graph, [[3], [0]], max_size=3)
    env.reset()
    assert env.dones == [True, False]
    assert env.valid_index == [1]
