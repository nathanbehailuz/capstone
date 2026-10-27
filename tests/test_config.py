from argparse import Namespace

import pytest
import torch

from components.generator_core import Agent
from components.generator_copy import Generator
from run_RG_explainer import validate_run_args


def test_reinforce_is_rejected():
    with pytest.raises(ValueError, match="PPO only"):
        validate_run_args(Namespace(policy="REINFORCE"))


def test_value_clip_is_rejected():
    with pytest.raises(ValueError, match="value clipping"):
        validate_run_args(Namespace(policy="PPO", use_value_clip=True))


def test_adaptive_kl_is_rejected():
    with pytest.raises(ValueError, match="adaptive KL"):
        validate_run_args(Namespace(policy="PPO", adaptive_kl_penalty=True))


def test_supported_ppo_args_are_accepted():
    args = validate_run_args(
        Namespace(policy="PPO", use_value_clip=False, adaptive_kl_penalty=False)
    )
    assert args.policy == "PPO"


def test_generator_rejects_value_clip(triangle_graph):
    model = Agent(8, with_attr=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="value clipping"):
        Generator(triangle_graph, model, optimizer, use_value_clip=True)
