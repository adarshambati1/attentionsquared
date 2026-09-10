import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts import run_004_phase13_evaluator_controls as phase13
from src.evaluation.functional_autoregressive import (
    FullPrefixHuginnD16Evaluator,
    frozen_coda_logits_from_normalized_state,
)


class FakeTokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {"<|end_text|>": 9, "<|end_turn|>": 8}[token]

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(token) for token in token_ids)


class RecordingHuginn(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_scale = 1
        self.transformer = SimpleNamespace(
            wte=nn.Embedding(16, 4), prelude=[], coda=[], ln_f=nn.Identity()
        )
        self.freqs_cis = torch.zeros(1, 32, 2)
        self.received_h0 = []

    def initialize_state(self, input_embeds, scale=1.0):
        return torch.randn_like(input_embeds) * scale

    def forward(self, *, input_ids, input_states, **kwargs):
        self.received_h0.append(input_states)
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 10)
        logits[..., 9] = 1
        return SimpleNamespace(logits=logits)


def test_phase13_config_locks_preregistered_ids_seed_bounds_and_roles():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/004_functional_v2_phase13.json").read_text())
    phase13.validate_config(config)
    assert config["dataset_split"] == "train"
    assert config["example_id_range_inclusive"] == [2250, 2499]
    assert config["h0_base_seed"] == 3000
    assert config["h0_seed_index"] == 0
    assert config["cache_control_example_ids"] == list(range(2250, 2258))
    assert config["cached_live_tolerances"] == {
        "hidden_max_abs": 0.003,
        "logit_mean_abs": 0.005,
        "logit_max_abs": 0.06,
        "kl_per_token": 0.0005,
        "require_all_next_token_argmax_equal": True,
    }
    for key, value in [
        ("dataset_split", "test"),
        ("example_id_range_inclusive", [0, 249]),
        ("h0_base_seed", 0),
        ("cache_control_example_ids", list(range(8))),
    ]:
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError, match="exactly match"):
            phase13.validate_config(changed)


def test_13b_literally_reuses_one_h0_object_in_both_independent_forwards():
    model = RecordingHuginn()
    tokenizer = FakeTokenizer()
    evaluator = FullPrefixHuginnD16Evaluator(
        model, tokenizer, base_seed=3000, seed_index=0
    )
    result = phase13.compare_live_routes(
        model,
        tokenizer,
        evaluator,
        torch.tensor([[1, 2]]),
        example_id=2250,
        gold_text="#### 9",
        max_new_tokens=4,
    )
    assert result["passed"]
    assert result["trusted_live_model_forward"]["generated_token_ids"] == [9]
    assert result["full_prefix_huginn_evaluator"]["generated_token_ids"] == [9]
    assert len(model.received_h0) == 2
    assert model.received_h0[0] is model.received_h0[1]
    assert result["prefixes"][0]["same_materialized_h0_object_and_value_reused"] is True


def test_scorer_golden_controls_distinguish_historical_and_corrected_cap_fallback():
    controls = phase13.scorer_golden_controls()
    historical = controls["historical_cap_fallback_emulation"]
    corrected = controls["authoritative_corrected_cap_scoring"]
    assert historical["fallback_allowed"] is True
    assert historical["correct"] is True
    assert corrected["fallback_allowed"] is False
    assert corrected["predicted_answer"] is None
    assert corrected["correct"] is False


class TimesTwo(nn.Module):
    def forward(self, value):
        return value * 2


class AddThreeCoda(nn.Module):
    def forward(self, state, frequencies, block_index, mask, cache):
        assert mask is None and cache is None
        return state + 3


class CodaModel:
    def __init__(self):
        self.transformer = SimpleNamespace(
            ln_f=TimesTwo(), coda=[AddThreeCoda()]
        )
        self.lm_head = nn.Identity()


def test_authoritative_coda_accepts_normalized_state_without_double_normalization():
    model = CodaModel()
    raw_recurrent_state = torch.tensor([[[1.0], [2.0]]])
    normalized_h16 = model.transformer.ln_f(raw_recurrent_state)
    frequencies = torch.zeros(1, 2, 1)

    logits = frozen_coda_logits_from_normalized_state(
        model, normalized_h16, frequencies
    )
    expected = model.transformer.ln_f(normalized_h16 + 3)
    assert torch.equal(logits, expected)

    # This literal regression assertion fails if the helper adds an initial
    # ln_f to cache-v2's already-normalized h16.
    double_normalized = model.transformer.ln_f(
        model.transformer.ln_f(normalized_h16) + 3
    )
    assert not torch.equal(logits, double_normalized)


def test_phase13_publication_is_atomic_read_only_and_no_replace(tmp_path):
    attempt = tmp_path / ".phase13.attempt-1"
    attempt.mkdir()
    evaluation = attempt / "evaluation.json"
    evaluation.write_text('{"status":"fail"}\n')
    evaluation.chmod(0o444)
    final = tmp_path / "phase13"

    phase13.publish_attempt(attempt, final)

    assert not attempt.exists()
    assert final.stat().st_mode & 0o777 == 0o555
    assert (final / "evaluation.json").stat().st_mode & 0o777 == 0o444
    second = tmp_path / ".phase13.attempt-2"
    second.mkdir()
    second_file = second / "evaluation.json"
    second_file.write_text("preserved")
    second_file.chmod(0o444)
    with pytest.raises(FileExistsError):
        phase13.publish_attempt(second, final)
    assert second.exists()


def test_phase13_main_is_hard_blocked_pending_corrected_phase11_and_phase12():
    with pytest.raises(RuntimeError, match="hard-blocked"):
        phase13.main(phase13.CONFIG, phase13.OUTPUT_ROOT)


def test_phase13_runner_persists_no_logits_and_has_no_paid_execution():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/run_004_phase13_evaluator_controls.py").read_text()
    assert '"full_vocabulary_logits_persisted": False' in source
    assert '"training_performed": False' in source
    assert "optimizer" not in source.lower()
    assert "teacher_prefix_answer_offsets" not in source
    assert 'positions = slice(start, end)' in source
    assert "TokenKLAggregator" in source
