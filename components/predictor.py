from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass(frozen=True)
class Prediction:
    logits: torch.Tensor
    embedding: torch.Tensor
    target_class: int


class PredictorAdapter:
    """Task-aware, identity-preserving access to a frozen predictor."""

    def __init__(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        task: str,
        batch: Optional[torch.Tensor] = None,
    ):
        if task not in {"node_task", "graph_task"}:
            raise ValueError(f"unsupported task: {task}")
        if task == "graph_task" and batch is None:
            raise ValueError("graph tasks require an explicit node-to-graph batch")
        self.model = model
        self.x = x
        self.edge_index = edge_index
        self.task = task
        self.batch = batch
        self.freeze()

    def freeze(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _edge_mask(self, nodes: torch.Tensor) -> torch.Tensor:
        selected = torch.zeros(
            self.x.size(0), dtype=torch.bool, device=self.edge_index.device
        )
        selected[nodes] = True
        return selected[self.edge_index[0]] & selected[self.edge_index[1]]

    def _node_prediction(
        self, nodes: torch.Tensor, target_node: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = self.edge_index[:, self._edge_mask(nodes)]
        weights = torch.ones(edge_index.size(1), device=edge_index.device)
        logits = self.model(
            self.x, edge_index, edge_weights=weights
        )[target_node]
        embedding = self.model.embedding(
            self.x, edge_index, edge_weights=weights
        )[target_node]
        return logits, embedding

    def _graph_prediction(
        self, nodes: torch.Tensor, instance_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        belongs = self.batch[nodes] == instance_id
        nodes = nodes[belongs]
        if nodes.numel() == 0:
            raise ValueError(f"no selected nodes belong to graph {instance_id}")

        local = torch.full(
            (self.x.size(0),), -1, dtype=torch.long, device=self.x.device
        )
        local[nodes] = torch.arange(nodes.numel(), device=self.x.device)
        edge_mask = self._edge_mask(nodes)
        global_edges = self.edge_index[:, edge_mask]
        local_edges = local[global_edges]
        local_x = self.x[nodes]
        local_batch = torch.zeros(
            nodes.numel(), dtype=torch.long, device=self.x.device
        )
        weights = torch.ones(local_edges.size(1), device=self.x.device)
        logits = self.model(
            local_x,
            local_edges,
            batch=local_batch,
            edge_weights=weights,
        )[0]
        if hasattr(self.model, "graph_embedding"):
            embedding = self.model.graph_embedding(
                local_x,
                local_edges,
                batch=local_batch,
                edge_weights=weights,
            )[0]
        else:
            embedding = self.model.embedding(
                local_x, local_edges, edge_weights=weights
            ).mean(0)
        return logits, embedding

    @torch.no_grad()
    def predict(
        self,
        selected_nodes: Sequence[int],
        *,
        instance_id: int,
        target_node: Optional[int] = None,
        target_class: Optional[int] = None,
    ) -> Prediction:
        self.freeze()
        nodes = torch.as_tensor(
            selected_nodes, dtype=torch.long, device=self.x.device
        )
        if self.task == "node_task":
            if target_node is None:
                raise ValueError("node tasks require target_node")
            logits, embedding = self._node_prediction(nodes, target_node)
        else:
            logits, embedding = self._graph_prediction(nodes, instance_id)
        resolved_target = (
            int(logits.argmax().item()) if target_class is None else int(target_class)
        )
        if not 0 <= resolved_target < logits.size(-1):
            raise ValueError(
                f"target class {resolved_target} is invalid for "
                f"{logits.size(-1)} predictor classes"
            )
        return Prediction(logits, embedding, resolved_target)

    def original(
        self, *, instance_id: int, target_node: Optional[int] = None
    ) -> Prediction:
        if self.task == "node_task":
            nodes = torch.arange(self.x.size(0), device=self.x.device)
        else:
            nodes = torch.nonzero(
                self.batch == instance_id, as_tuple=False
            ).flatten()
        return self.predict(
            nodes, instance_id=instance_id, target_node=target_node
        )

    @torch.no_grad()
    def without_selected_edges(
        self,
        selected_nodes: Sequence[int],
        *,
        instance_id: int,
        target_node: Optional[int] = None,
        target_class: Optional[int] = None,
    ) -> Prediction:
        """Score the original instance after deleting the selected induced edges."""
        self.freeze()
        selected = torch.as_tensor(
            selected_nodes, dtype=torch.long, device=self.x.device
        )
        remaining_edges = self.edge_index[:, ~self._edge_mask(selected)]
        original_edges = self.edge_index
        self.edge_index = remaining_edges
        try:
            if self.task == "node_task":
                nodes = torch.arange(self.x.size(0), device=self.x.device)
            else:
                nodes = torch.nonzero(
                    self.batch == instance_id, as_tuple=False
                ).flatten()
            return self.predict(
                nodes.tolist(),
                instance_id=instance_id,
                target_node=target_node,
                target_class=target_class,
            )
        finally:
            self.edge_index = original_edges
