from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence

import torch
from torch.distributions import Categorical


class Termination(str, Enum):
    STOP = "stop"
    EXHAUSTED = "exhausted"
    SIZE_LIMIT = "size_limit"
    TRUNCATED = "truncated"


@dataclass
class Transition:
    observation: Any
    candidates: Sequence[int]
    valid_mask: torch.Tensor
    action: int
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    reward: float = 0.0
    reward_terms: Optional[dict] = None
    termination: Optional[Termination] = None

    @property
    def terminated(self) -> bool:
        return self.termination in {
            Termination.STOP,
            Termination.EXHAUSTED,
            Termination.SIZE_LIMIT,
        }

    @property
    def truncated(self) -> bool:
        return self.termination == Termination.TRUNCATED


def masked_categorical(logits: torch.Tensor, valid_mask: torch.Tensor) -> Categorical:
    """Build the one distribution used for both collection and PPO replay."""
    if logits.shape != valid_mask.shape:
        raise ValueError(
            f"logits and valid_mask differ: {logits.shape} != {valid_mask.shape}"
        )
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    if not bool(valid_mask.any()):
        raise ValueError("categorical distribution has no valid action")
    masked_logits = logits.masked_fill(~valid_mask, -torch.inf)
    return Categorical(logits=masked_logits)


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    valid_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE without allowing padded or terminal states to bootstrap."""
    for tensor in (values, next_values, terminated):
        if tensor.shape != rewards.shape:
            raise ValueError("all GAE tensors must have the same shape")
    if valid_mask is None:
        valid_mask = torch.ones_like(rewards, dtype=torch.bool)
    if valid_mask.shape != rewards.shape:
        raise ValueError("valid_mask must have the same shape as rewards")

    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros(rewards.shape[0], device=rewards.device)
    for step in reversed(range(rewards.shape[1])):
        valid = valid_mask[:, step]
        bootstrap = (~terminated[:, step]).to(rewards.dtype)
        delta = (
            rewards[:, step]
            + gamma * next_values[:, step] * bootstrap
            - values[:, step]
        )
        current = delta + gamma * gae_lambda * bootstrap * next_advantage
        next_advantage = torch.where(valid, current, torch.zeros_like(current))
        advantages[:, step] = torch.where(
            valid, next_advantage, torch.zeros_like(next_advantage)
        )
    returns = advantages + values
    returns = torch.where(valid_mask, returns, torch.zeros_like(returns))
    return advantages, returns


def transition_rewards_from_costs(
    costs: Sequence[float],
    reward_scale: float = 1.0,
    stop_reward: Optional[float] = None,
) -> list[float]:
    """Turn seed/next-state costs into one reward per expansion action."""
    if len(costs) < 1:
        raise ValueError("costs must include the initial seed state")
    rewards = [
        reward_scale * (float(previous) - float(current))
        for previous, current in zip(costs, costs[1:])
    ]
    if stop_reward is not None:
        rewards.append(float(stop_reward))
    return rewards
