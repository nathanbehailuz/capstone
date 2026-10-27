import torch

from components.locator_seed import Locator, locator_kl


def test_equal_locator_distributions_have_zero_kl():
    logits = torch.tensor([0.0, 0.0])
    target = torch.tensor([0.5, 0.5])
    assert abs(locator_kl(logits, target).item()) < 1e-7


def test_locator_embeddings_use_actual_node_ids():
    g_embeds = torch.zeros(1, 2)
    n_embeds = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    locator = Locator(
        g_embeds,
        n_embeds,
        [[0, 1, 2]],
        d_lr=1e-3,
        isolated_nodes={1},
    )
    expected = torch.stack(
        [
            torch.cat([g_embeds[0], n_embeds[0]]),
            torch.cat([g_embeds[0], n_embeds[2]]),
        ]
    )
    assert torch.equal(locator.logit_embeds.cpu(), expected)
