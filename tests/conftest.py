import pytest
import torch

from components.graph import Graph


@pytest.fixture
def triangle_graph():
    edges = torch.tensor(
        [[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long
    )
    return Graph(edges.numpy(), n_nodes=3)


@pytest.fixture
def disjoint_graph_batch():
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    return edge_index, batch


@pytest.fixture
def isolated_high_index_graph():
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return Graph(edges.numpy(), n_nodes=4)


class StubPredictor(torch.nn.Module):
    def __init__(self, num_classes, embedding_dim=5):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

    def embedding(self, x, edge_index, edge_weights=None):
        return x[:, : self.embedding_dim]

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        result = []
        for graph_id in torch.unique(batch, sorted=True):
            result.append(self.embedding(x, edge_index)[batch == graph_id].mean(0))
        return torch.stack(result)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:
            base = x.sum(-1, keepdim=True)
        else:
            base = self.graph_embedding(x, edge_index, batch, edge_weights).sum(
                -1, keepdim=True
            )
        return torch.cat([base + i for i in range(self.num_classes)], dim=-1)


@pytest.fixture(params=[2, 4, 8])
def stub_predictor(request):
    return StubPredictor(request.param)
