import torch
from torch_geometric.data import Data

from components.causal_explainer import CausalRewardCalculator
from components.predictor import PredictorAdapter


def test_causal_zero_weight_removes_contribution(stub_predictor):
    x = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    edges = torch.tensor([[0, 1, 1, 0], [1, 0, 2, 2]])
    predictor = PredictorAdapter(
        stub_predictor, x, edges, task="node_task"
    )
    calculator = CausalRewardCalculator(
        stub_predictor, causal_explainer=None, predictor=predictor
    )
    calculator.set_weight(0.0)
    causal = calculator.compute_causal_reward(
        Data(x=x, edge_index=edges), [0, 1], target_y=0
    )
    assert calculator.compute_combined_reward(2.5, causal) == 2.5


def test_nonzero_causal_fixture_changes_reward(stub_predictor):
    x = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    edges = torch.tensor([[0, 1, 1, 0], [1, 0, 2, 2]])
    predictor = PredictorAdapter(
        stub_predictor, x, edges, task="node_task"
    )
    calculator = CausalRewardCalculator(
        stub_predictor, causal_explainer=None, predictor=predictor
    )
    calculator.set_weight(0.5)
    causal = calculator.compute_causal_reward(
        Data(x=x, edge_index=edges), [0, 1], target_y=0
    )
    assert calculator.compute_combined_reward(2.5, causal) != 2.5
