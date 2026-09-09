import pytest
import torch

from src.models.attention2 import Attention2Lite


def _model() -> Attention2Lite:
    torch.manual_seed(123)
    return Attention2Lite(hidden=8, depth=3, heads=2, rounds=2).eval()


def test_padded_values_cannot_change_any_real_token_output():
    model = _model()
    torch.manual_seed(456)
    h0 = torch.randn(2, 5, 8)
    x = torch.randn(2, 5, 8)
    mask = torch.tensor(
        [[True, False, True, True, False], [False, True, True, False, True]]
    )
    changed_h0 = h0.clone()
    changed_x = x.clone()
    changed_h0[~mask] = torch.randn_like(changed_h0[~mask]) * 10_000
    changed_x[~mask] = torch.randn_like(changed_x[~mask]) * 10_000

    observed, observed_states, _ = model(h0, x, token_mask=mask)
    changed, changed_states, _ = model(changed_h0, changed_x, token_mask=mask)

    expanded = mask[:, None, :, None].expand_as(observed)
    torch.testing.assert_close(
        observed[expanded], changed[expanded], rtol=0, atol=0
    )
    assert torch.count_nonzero(observed[~expanded]) == 0
    assert torch.count_nonzero(changed[~expanded]) == 0
    for state, changed_state in zip(observed_states, changed_states):
        assert torch.count_nonzero(state[~expanded]) == 0
        assert torch.count_nonzero(changed_state[~expanded]) == 0


def test_interspersed_padding_matches_compacted_no_padding_reference():
    model = _model()
    torch.manual_seed(654)
    h0 = torch.randn(1, 5, 8)
    x = torch.randn(1, 5, 8)
    mask = torch.tensor([[True, False, True, False, True]])

    padded, _, _ = model(h0, x, token_mask=mask)
    compact, _, _ = model(h0[:, mask[0]], x[:, mask[0]])

    torch.testing.assert_close(
        padded[:, :, mask[0]], compact, rtol=1e-6, atol=1e-6
    )


def test_exact_key_padding_mask_is_passed_on_every_round(monkeypatch):
    model = _model()
    mask = torch.tensor(
        [[True, False, True, True], [False, True, True, False]]
    )
    inputs = torch.randn(2, 4, 8)
    observed_masks = []
    original = model.token_attn.forward

    def wrapped(*args, **kwargs):
        observed_masks.append(kwargs["key_padding_mask"].detach().clone())
        return original(*args, **kwargs)

    monkeypatch.setattr(model.token_attn, "forward", wrapped)
    model(inputs, inputs, token_mask=mask)

    expected = (
        (~mask[:, None, :])
        .expand(2, model.depth, 4)
        .reshape(2 * model.depth, 4)
    )
    assert len(observed_masks) == model.rounds
    for observed_mask in observed_masks:
        assert torch.equal(observed_mask, expected)


def test_intermediate_padding_would_influence_later_token_without_key_mask():
    model = _model()
    torch.manual_seed(789)
    h0 = torch.randn(1, 4, 8)
    x = torch.randn(1, 4, 8)
    changed_h0 = h0.clone()
    changed_x = x.clone()
    changed_h0[:, 1] += 1_000
    changed_x[:, 1] -= 1_000
    mask = torch.tensor([[True, False, True, True]])

    masked, _, _ = model(h0, x, token_mask=mask)
    changed_masked, _, _ = model(changed_h0, changed_x, token_mask=mask)
    torch.testing.assert_close(masked[:, :, 2:], changed_masked[:, :, 2:], rtol=0, atol=0)

    unmasked, _, _ = model(h0, x)
    changed_unmasked, _, _ = model(changed_h0, changed_x)
    assert not torch.allclose(unmasked[:, :, 2], changed_unmasked[:, :, 2])


def test_causal_mask_blocks_future_real_token_values():
    model = _model()
    torch.manual_seed(321)
    h0 = torch.randn(1, 5, 8)
    x = torch.randn(1, 5, 8)
    changed_h0 = h0.clone()
    changed_x = x.clone()
    changed_h0[:, 4] += 5_000
    changed_x[:, 4] -= 5_000
    mask = torch.ones((1, 5), dtype=torch.bool)

    observed, _, _ = model(h0, x, token_mask=mask)
    changed, _, _ = model(changed_h0, changed_x, token_mask=mask)
    torch.testing.assert_close(observed[:, :, :4], changed[:, :, :4], rtol=0, atol=0)


def test_state_dict_and_positional_return_attention_remain_backward_compatible():
    original = _model()
    restored = _model()
    restored.load_state_dict(original.state_dict(), strict=True)
    inputs = torch.randn(1, 4, 8)

    output, states, attentions = restored(inputs, inputs, True)

    assert output.shape == (1, restored.depth, 4, restored.hidden)
    assert len(states) == restored.rounds
    assert len(attentions) == restored.rounds
    assert all("depth" in attention for attention in attentions)


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones((1, 4), dtype=torch.int64),
        torch.ones((1, 3), dtype=torch.bool),
        torch.zeros((1, 4), dtype=torch.bool),
    ],
)
def test_token_mask_rejects_wrong_dtype_shape_or_empty_sequence(mask):
    model = _model()
    inputs = torch.randn(1, 4, 8)
    with pytest.raises(ValueError):
        model(inputs, inputs, token_mask=mask)
