from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.evaluation.functional_autoregressive import (
    FullPrefixAttention2Evaluator,
    strictly_growing_full_prefix_lengths,
)


class FakeTokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {"<|end_text|>": 9, "<|end_turn|>": 8}[token]

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(token) for token in token_ids)


class RecordingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lengths = []

    def forward(self, state, frequencies, block_index, mask, cache):
        self.lengths.append(state.shape[1])
        assert frequencies.shape[1] == state.shape[1]
        assert mask is None and cache is None
        return state


class FakeHuginn(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_scale = 1
        self.prelude = RecordingBlock()
        self.coda = RecordingBlock()
        self.transformer = SimpleNamespace(
            wte=nn.Embedding(16, 4),
            prelude=[self.prelude],
            coda=[self.coda],
            ln_f=nn.Identity(),
        )
        self.lm_head = nn.Linear(4, 10, bias=False)
        nn.init.zeros_(self.lm_head.weight)
        self.freqs_cis = torch.zeros(1, 64, 2)

    def initialize_state(self, input_embeds, scale=1.0):
        return torch.randn_like(input_embeds) * scale


class RecordingA2(nn.Module):
    def __init__(self):
        super().__init__()
        self.lengths = []

    def forward(self, h0, x, token_mask=None):
        self.lengths.append(x.shape[1])
        assert h0.shape == x.shape
        assert token_mask.shape == x.shape[:2]
        assert token_mask.all()
        state = (h0 + x)[:, None].expand(-1, 16, -1, -1)
        return state, [], []


def test_generation_recomputes_entire_strictly_growing_prefix_each_step():
    huginn = FakeHuginn()
    a2 = RecordingA2()
    evaluator = FullPrefixAttention2Evaluator(
        huginn, a2, FakeTokenizer(), base_seed=3000, seed_index=0
    )
    prompt = torch.tensor([[2, 3]])

    result = evaluator.generate(prompt, example_id=7, max_new_tokens=3)

    assert result.prefix_lengths == (2, 3, 4)
    assert strictly_growing_full_prefix_lengths(result.prefix_lengths, 2)
    assert a2.lengths == [2, 3, 4]
    assert huginn.prelude.lengths == [2, 3, 4]
    assert huginn.coda.lengths == [2, 3, 4]
    assert len(result.token_ids) == 3
    assert result.hit_max_new_tokens is True
    assert result.ended_naturally is False


def test_full_prefix_a2_coda_does_not_normalize_already_normalized_z16_twice():
    class TimesTwo(nn.Module):
        def forward(self, state):
            return state * 2

    class AddThree(nn.Module):
        def forward(self, state, frequencies, block_index, mask, cache):
            return state + 3

    huginn = FakeHuginn()
    huginn.transformer.ln_f = TimesTwo()
    huginn.transformer.coda = [AddThree()]
    huginn.lm_head = nn.Identity()
    evaluator = FullPrefixAttention2Evaluator(
        huginn, RecordingA2(), FakeTokenizer()
    )
    normalized_z16 = torch.tensor([[[2.0]]])
    logits = evaluator._coda_last_logits(
        normalized_z16, torch.zeros(1, 1, 1)
    )

    assert torch.equal(logits, torch.tensor([[10.0]]))
    assert not torch.equal(logits, torch.tensor([[14.0]]))  # double-ln_f result


def test_next_token_logits_have_one_vocabulary_vector_not_sequence_logits():
    evaluator = FullPrefixAttention2Evaluator(
        FakeHuginn(), RecordingA2(), FakeTokenizer()
    )
    logits = evaluator.next_token_logits(torch.tensor([[1, 2, 3, 4]]), example_id=0)
    assert logits.shape == (1, 10)


def test_full_prefix_validation_rejects_empty_or_non_singleton_batch():
    evaluator = FullPrefixAttention2Evaluator(
        FakeHuginn(), RecordingA2(), FakeTokenizer()
    )
    with pytest.raises(ValueError):
        evaluator.next_token_logits(torch.empty((1, 0), dtype=torch.long), example_id=0)
    with pytest.raises(ValueError):
        evaluator.generate(torch.ones((2, 3), dtype=torch.long), example_id=0, max_new_tokens=2)
    with pytest.raises(ValueError):
        evaluator.generate(torch.ones((1, 3), dtype=torch.long), example_id=0, max_new_tokens=0)
