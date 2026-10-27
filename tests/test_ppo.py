import torch

from components.ppo import (
    generalized_advantage_estimate,
    masked_categorical,
    transition_rewards_from_costs,
)


def test_unchanged_policy_has_unit_ratio_and_zero_kl():
    logits = torch.tensor([0.3, 0.2, 0.5])
    mask = torch.tensor([True, False, True])
    old = masked_categorical(logits, mask)
    action = torch.tensor(2)
    old_log_prob = old.log_prob(action).detach()

    current = masked_categorical(logits.clone().requires_grad_(), mask)
    ratio = torch.exp(current.log_prob(action) - old_log_prob)
    approx_kl = old_log_prob - current.log_prob(action)
    clip_fraction = ((ratio < 0.8) | (ratio > 1.2)).float()

    assert ratio.item() == 1.0
    assert approx_kl.item() == 0.0
    assert clip_fraction.item() == 0.0
    assert old.probs[1].item() == 0.0


def test_cost_improvements_align_with_expansion_and_stop_actions():
    assert transition_rewards_from_costs([10, 8, 5]) == [2.0, 3.0]
    assert transition_rewards_from_costs(
        [10, 8, 5], stop_reward=0.1
    ) == [2.0, 3.0, 0.1]


def test_hand_calculated_gae_with_terminal_and_padding():
    rewards = torch.tensor([[2.0, 3.0, 0.1], [4.0, 0.0, 0.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.zeros_like(rewards)
    terminated = torch.tensor(
        [[False, False, True], [True, False, False]]
    )
    valid = torch.tensor([[True, True, True], [True, False, False]])

    advantages, returns = generalized_advantage_estimate(
        rewards, values, next_values, terminated, gamma=1.0, gae_lambda=1.0,
        valid_mask=valid
    )

    assert torch.allclose(advantages[0], torch.tensor([5.1, 3.1, 0.1]))
    assert torch.allclose(advantages[1], torch.tensor([4.0, 0.0, 0.0]))
    assert torch.equal(advantages, returns)


def test_truncation_bootstraps():
    rewards = torch.tensor([[1.0]])
    values = torch.tensor([[0.0]])
    next_values = torch.tensor([[2.0]])
    terminated = torch.tensor([[False]])
    advantages, _ = generalized_advantage_estimate(
        rewards, values, next_values, terminated, gamma=0.5, gae_lambda=1.0
    )
    assert advantages.item() == 2.0


def test_real_optimizer_step_changes_policy_and_value_parameters():
    model = torch.nn.Linear(3, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    observation = torch.tensor([[1.0, 0.5, -1.0]])
    before = [parameter.detach().clone() for parameter in model.parameters()]

    output = model(observation)
    distribution = masked_categorical(
        output[0, :3], torch.tensor([True, True, True])
    )
    policy_loss = -distribution.log_prob(torch.tensor(1))
    value_loss = output[0, 3].square()
    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()

    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer.step()
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters())
    )
