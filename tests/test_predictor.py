import torch

from components.predictor import PredictorAdapter


def test_graph_prediction_ignores_unrelated_graph(
    stub_predictor, disjoint_graph_batch
):
    edge_index, batch = disjoint_graph_batch
    x = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    adapter = PredictorAdapter(
        stub_predictor, x, edge_index, "graph_task", batch=batch
    )

    batched = adapter.original(instance_id=0)
    isolated = PredictorAdapter(
        stub_predictor,
        x[:2],
        edge_index[:, :2],
        "graph_task",
        batch=torch.zeros(2, dtype=torch.long),
    ).original(instance_id=0)

    assert torch.equal(batched.logits, isolated.logits)
    assert torch.equal(batched.embedding, isolated.embedding)


def test_deleting_no_edges_matches_original(stub_predictor, disjoint_graph_batch):
    edge_index, batch = disjoint_graph_batch
    x = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    adapter = PredictorAdapter(
        stub_predictor, x, edge_index, "graph_task", batch=batch
    )
    original = adapter.original(instance_id=0)
    deleted = adapter.without_selected_edges(
        [0], instance_id=0, target_class=original.target_class
    )
    assert torch.equal(original.logits, deleted.logits)


def test_edgeless_prediction_keeps_class_dimension(stub_predictor):
    x = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    adapter = PredictorAdapter(
        stub_predictor,
        x,
        torch.empty((2, 0), dtype=torch.long),
        "node_task",
    )
    prediction = adapter.predict([0], instance_id=0, target_node=0)
    assert prediction.logits.shape == (stub_predictor.num_classes,)


def test_single_item_batch_keeps_class_dimension(stub_predictor, disjoint_graph_batch):
    edge_index, batch = disjoint_graph_batch
    x = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    adapter = PredictorAdapter(
        stub_predictor, x, edge_index, "graph_task", batch=batch
    )
    prediction = adapter.original(instance_id=1)
    assert prediction.logits.shape == (stub_predictor.num_classes,)
    assert prediction.embedding.numel() > 0


def test_predictor_is_frozen(stub_predictor):
    stub_predictor.register_parameter(
        "weight", torch.nn.Parameter(torch.tensor(1.0))
    )
    adapter = PredictorAdapter(
        stub_predictor,
        torch.ones((1, 5)),
        torch.empty((2, 0), dtype=torch.long),
        "node_task",
    )
    assert not adapter.model.training
    assert all(not parameter.requires_grad for parameter in adapter.model.parameters())
