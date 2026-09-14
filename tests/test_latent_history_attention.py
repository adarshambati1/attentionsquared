from types import SimpleNamespace

import torch
from torch import nn

from src.models.latent_history_huginn import PerTokenHistoryAttention
from src.training.latent_history import answer_cross_entropy


def test_zero_output_projection_has_exactly_zero_initial_effect():
    torch.manual_seed(1)
    module = PerTokenHistoryAttention(12, projection_size=8, num_heads=2)
    current = torch.randn(2, 5, 12)
    history = [torch.randn_like(current), current]
    output = module(current, history)
    assert torch.count_nonzero(module.q_proj.weight) > 0
    assert torch.count_nonzero(module.k_proj.weight) > 0
    assert torch.count_nonzero(module.v_proj.weight) > 0
    assert torch.count_nonzero(module.out_proj.weight) == 0
    assert torch.equal(output, torch.zeros_like(output))


def test_history_attention_never_mixes_token_positions_and_masks_padding():
    torch.manual_seed(2)
    module = PerTokenHistoryAttention(12, projection_size=8, num_heads=2)
    nn.init.normal_(module.out_proj.weight)
    current = torch.randn(1, 4, 12)
    earlier = torch.randn_like(current)
    baseline = module(current, [earlier, current], token_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.bool))
    changed_current = current.clone(); changed_earlier = earlier.clone()
    changed_current[:, 2:] += 100; changed_earlier[:, 2:] -= 100
    changed = module(changed_current, [changed_earlier, changed_current], token_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.bool))
    assert torch.equal(baseline[:, :2], changed[:, :2])
    assert torch.equal(baseline[:, 3], torch.zeros_like(baseline[:, 3]))
    assert torch.equal(changed[:, 3], torch.zeros_like(changed[:, 3]))


def test_current_only_and_uniform_are_exact_declared_controls():
    torch.manual_seed(3)
    module = PerTokenHistoryAttention(12, projection_size=8, num_heads=2)
    nn.init.normal_(module.out_proj.weight)
    current = torch.randn(1, 3, 12)
    old = torch.randn_like(current)
    assert torch.equal(module(current, [old, current], mode="current_only"), module(current, [current], mode="learned"))
    uniform = module(current, [old, current], mode="uniform")
    values = module.v_proj(torch.stack([old, current], dim=2)).view(1, 3, 2, 2, 4).mean(dim=2)
    expected = module.out_proj(values.reshape(1, 3, 8))
    assert torch.equal(uniform, expected)


def test_raw_mean_is_exact_unprojected_completed_state_average():
    module = PerTokenHistoryAttention(12, projection_size=8, num_heads=2)
    current = torch.full((1, 3, 12), 3.0)
    old = torch.full_like(current, 1.0)
    output = module(current, [old, current], mode="raw_mean")
    assert torch.equal(output, torch.full_like(current, 2.0))


def test_answer_loss_includes_first_answer_token_and_excludes_prompt():
    logits = torch.full((1, 6, 10), -20.0)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    answer_start = 3
    logits[0, 2, 4] = 20
    logits[0, 3, 5] = 20
    logits[0, 4, 6] = 20
    loss, count = answer_cross_entropy(logits, ids, answer_start)
    assert count == 3
    assert loss < 1e-6
    logits[0, 1, 9] = 100
    unchanged, _ = answer_cross_entropy(logits, ids, answer_start)
    assert torch.equal(loss, unchanged)
