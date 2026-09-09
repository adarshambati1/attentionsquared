import json
import random
import time

import numpy as np
import pytest
import torch

import src.evaluation.correctness as correctness
from src.evaluation.correctness import (
    STOP_STRINGS,
    aligned_answer_logits_and_targets,
    answer_bounds,
    answer_prediction_mask,
    build_chat_prompt,
    degeneration_metadata,
    ensure_audit_rerun_output,
    ensure_audit_rerun_root,
    ensure_new_artifact_root,
    ensure_new_output_path,
    find_repetition_onset,
    TokenKLAggregator,
    corrected_v2_protocol_metadata,
    generation_config_kwargs,
    generation_status,
    guard_corrected_v2_artifact_root,
    numpy_generator_for_example,
    per_example_seed,
    prepare_corrected_v2_output,
    require_invalid_v1_opt_in,
    score_generation,
    seed_for_example,
    sha256_file,
    stop_token_ids,
    synchronized_cuda_timer,
    token_kl_sum_and_count,
    token_normalized_kl,
    tokenize_prompt,
    torch_generator_for_example,
    write_json_exclusive,
)


class FakeTokenizer:
    eos_token_id = 9
    pad_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return f"S:{messages[0]['content']}|U:{messages[1]['content']}|A:"

    def __call__(self, text, **kwargs):
        assert text.endswith("|A:")
        assert kwargs["return_tensors"] == "pt"
        assert kwargs["add_special_tokens"] is False
        return {"input_ids": torch.tensor([[1, 2]]), "token_type_ids": torch.ones(1, 2)}


def test_shared_prompt_tokenization_and_generation_settings_are_literal():
    tokenizer = FakeTokenizer()
    prompt = build_chat_prompt(tokenizer, "question", "system")
    assert prompt == "S:system|U:question|A:"
    assert set(tokenize_prompt(tokenizer, prompt)) == {"input_ids"}
    settings = generation_config_kwargs(tokenizer, 123)
    assert settings["stop_strings"] == list(STOP_STRINGS)
    assert settings["max_new_tokens"] == 123
    assert settings["do_sample"] is False
    assert settings["pad_token_id"] == tokenizer.eos_token_id


def test_answer_bounds_select_causal_predecessors_and_only_valid_positions():
    bounds = answer_bounds(answer_start=3, valid_end=6, sequence_length=8)
    logits = torch.arange(8 * 2).reshape(8, 2)
    ids = torch.arange(100, 108)
    selected_logits, targets = aligned_answer_logits_and_targets(logits, ids, bounds)
    assert torch.equal(selected_logits, logits[2:5])
    assert torch.equal(targets, ids[3:6])
    assert answer_prediction_mask(bounds).tolist() == [False, False, True, True, True, False, False, False]
    assert bounds.token_count == 3


def test_answer_bounds_bounds_reject_invalid_empty_or_mismatched_sequences():
    with pytest.raises(ValueError):
        answer_bounds(answer_start=0, valid_end=2, sequence_length=3)
    with pytest.raises(ValueError):
        answer_bounds(answer_start=2, valid_end=2, sequence_length=3)
    bounds = answer_bounds(1, 3, 2)
    with pytest.raises(ValueError):
        aligned_answer_logits_and_targets(torch.zeros(4, 2), torch.zeros(3), bounds)


def test_authoritative_scorer_preserves_exact_exp1_48_character_window():
    assert score_generation(
        "The answer is" + "x" * 48 + "42", "#### 42", hit_max_new_tokens=True
    ).correct
    assert not score_generation(
        "The answer is" + "x" * 49 + "42", "#### 42", hit_max_new_tokens=True
    ).correct
    assert score_generation(
        "Therefore," + "x" * 48 + "42", "#### 42", hit_max_new_tokens=True
    ).correct


def test_authoritative_scorer_disables_unsafe_fallback_on_cap():
    capped = score_generation("intermediate value 42", "#### 42", hit_max_new_tokens=True)
    assert capped.predicted_answer is None
    assert not capped.correct
    legacy = score_generation(
        "intermediate value 42",
        "#### 42",
        hit_max_new_tokens=True,
        allow_fallback_on_cap=True,
    )
    assert legacy.correct
    explicit = score_generation("Therefore, the answer is 42", "#### 42", hit_max_new_tokens=True)
    assert explicit.correct


def test_generation_status_does_not_treat_cap_as_eos():
    with pytest.raises(ValueError):
        generation_status([], max_new_tokens=0, stop_token_ids=[9])
    assert generation_status([1, 2, 3], max_new_tokens=3, stop_token_ids=[9]).hit_max_new_tokens
    natural = generation_status([1, 9], max_new_tokens=3, stop_token_ids=[9, None, -1])
    assert natural.ended_naturally
    assert not natural.hit_max_new_tokens
    natural_at_cap = generation_status([1, 2, 9], max_new_tokens=3, stop_token_ids=[9])
    assert natural_at_cap.ended_naturally
    assert not natural_at_cap.hit_max_new_tokens
    empty = generation_status([], max_new_tokens=3, stop_token_ids=[9])
    assert not empty.ended_naturally and not empty.hit_max_new_tokens


def test_stop_token_ids_match_shared_stop_policy():
    tokenizer = FakeTokenizer()
    tokenizer.convert_tokens_to_ids = lambda token: {STOP_STRINGS[0]: 10, STOP_STRINGS[1]: 11}[token]
    assert stop_token_ids(tokenizer) == (9, 10, 11)


def test_degeneration_metadata_requires_explicit_shortening_decision():
    bounds = answer_bounds(5, 20)
    repeated = list(range(12)) * 3
    assert find_repetition_onset([99, *repeated], chunk_size=12, repetitions=3) == 1
    with pytest.raises(ValueError):
        degeneration_metadata(bounds, hit_max_new_tokens=True, reviewed_valid_end=12)
    metadata = degeneration_metadata(
        bounds,
        hit_max_new_tokens=True,
        degeneration_onset=7,
        reviewed_valid_end=12,
        review_decision="manual_repetition_onset",
    )
    assert metadata.to_dict() == {
        "truncated": True,
        "pathological": True,
        "degeneration_onset": 7,
        "valid_end": 12,
        "review_decision": "manual_repetition_onset",
    }


def test_private_seed_generators_are_repeatable_without_mutating_global_rngs():
    torch.manual_seed(123)
    expected_global = torch.rand(2)
    torch.manual_seed(123)
    first = torch.rand(2, generator=torch_generator_for_example(100, 7, step=2))
    second = torch.rand(2, generator=torch_generator_for_example(100, 7, step=2))
    assert torch.equal(first, second)
    assert torch.equal(torch.rand(2), expected_global)
    assert np.array_equal(
        numpy_generator_for_example(100, 7).random(3),
        numpy_generator_for_example(100, 7).random(3),
    )


def test_per_example_seeding_is_paired_and_repeatable():
    assert per_example_seed(100, 7) == per_example_seed(100, 7)
    seed_for_example(100, 7, step=2)
    values1 = (random.random(), np.random.rand(), torch.rand(2))
    seed_for_example(100, 7, step=2)
    values2 = (random.random(), np.random.rand(), torch.rand(2))
    assert values1[0] == values2[0]
    assert values1[1] == values2[1]
    assert torch.equal(values1[2], values2[2])
    assert per_example_seed(100, 7, step=1) != per_example_seed(100, 8, step=1)
    assert per_example_seed(100, 7, seed_index=1) != per_example_seed(100, 7)
    assert per_example_seed(100, 7) != per_example_seed(101, 6)
    values = {
        per_example_seed(100 + base, 7 + example, step=step, seed_index=index)
        for base in range(3)
        for example in range(4)
        for step in range(3)
        for index in range(2)
    }
    assert len(values) == 3 * 4 * 3 * 2
    with pytest.raises(ValueError):
        per_example_seed(1 << 24, 0)
    with pytest.raises(ValueError):
        per_example_seed(0, 0, step=1 << 7)
    # Regression for the collision in the prior additive implementation.
    assert per_example_seed(0, 11, seed_index=1) != per_example_seed(0, 0, step=10)
    coordinates = {
        per_example_seed(3000, example, step, seed_index=seed_index)
        for example in range(100)
        for step in range(4)
        for seed_index in range(3)
    }
    assert len(coordinates) == 100 * 4 * 3
    with pytest.raises(ValueError):
        per_example_seed(100, 7, seed_index=-1)


def test_token_normalized_kl_uses_only_masked_tokens_and_self_is_zero():
    teacher = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    student = torch.tensor([[[0.0, 2.0], [0.0, 2.0], [2.0, 0.0]]])
    mask = torch.tensor([[False, True, True]])
    per_token = torch.nn.functional.kl_div(
        torch.log_softmax(student, -1), torch.softmax(teacher, -1), reduction="none"
    ).sum(-1)
    assert torch.allclose(token_normalized_kl(student, teacher, mask), per_token[mask].mean())
    assert token_normalized_kl(teacher, teacher).abs().item() < 1e-6
    with pytest.raises(ValueError):
        token_normalized_kl(student, teacher, torch.zeros_like(mask))


def test_kl_global_aggregation_uses_total_numerator_and_count():
    teacher_short = torch.tensor([[[2.0, 0.0]]])
    student_short = torch.tensor([[[0.0, 2.0]]])
    teacher_long = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]])
    student_long = teacher_long.clone()
    short_sum, short_count = token_kl_sum_and_count(student_short, teacher_short)
    long_sum, long_count = token_kl_sum_and_count(student_long, teacher_long)
    aggregate = TokenKLAggregator()
    aggregate.update(short_sum, short_count)
    aggregate.update(long_sum, long_count)
    assert aggregate.count == 4
    assert aggregate.mean == pytest.approx((short_sum + long_sum).item() / 4)
    assert aggregate.mean != pytest.approx(
        (token_normalized_kl(student_short, teacher_short).item()
         + token_normalized_kl(student_long, teacher_long).item()) / 2
    )


def test_invalid_v1_execution_requires_explicit_audit_opt_in():
    with pytest.raises(RuntimeError, match="quarantined"):
        require_invalid_v1_opt_in(False, "legacy.py")
    require_invalid_v1_opt_in(True, "legacy.py")


def test_in_place_degeneration_annotator_is_frozen():
    from scripts.annotate_functional_sequences import DEPRECATION_MESSAGE, main

    with pytest.raises(RuntimeError, match="v1 artifacts are immutable"):
        main()
    assert "Phase 7" in DEPRECATION_MESSAGE


def test_corrected_v2_guard_rejects_v1_and_mismatched_resumes(tmp_path):
    with pytest.raises(ValueError, match="separate corrected-v2 path"):
        guard_corrected_v2_artifact_root(tmp_path / "training_cache")
    root = tmp_path / "corrected-v2"
    root.mkdir()
    guard_corrected_v2_artifact_root(root)
    (root / "existing.npz").write_bytes(b"preserved")
    with pytest.raises(ValueError, match="requires protocol metadata"):
        guard_corrected_v2_artifact_root(root)
    metadata = corrected_v2_protocol_metadata()
    guard_corrected_v2_artifact_root(root, metadata)
    with pytest.raises(ValueError, match="protocol mismatch"):
        guard_corrected_v2_artifact_root(root, {**metadata, "scorer_protocol": "v1"})
    assert (root / "existing.npz").read_bytes() == b"preserved"


def test_sha256_file_records_artifact_identity(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"attention-squared")
    assert sha256_file(artifact) == "e86aec92fe8a9b73cd51a3f666a52c7f3038005f8f6d01b7663f8e519431a058"


def test_new_output_and_root_guards_never_overwrite(tmp_path):
    output = tmp_path / "nested" / "result.json"
    ensure_new_output_path(output)
    assert output.parent.is_dir()
    write_json_exclusive(output, {"preserved": True})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ensure_new_output_path(output)
    with pytest.raises(FileExistsError):
        write_json_exclusive(output, {"preserved": False})
    assert output.read_text().startswith('{\n  "preserved": true')

    root = tmp_path / "fresh-root"
    ensure_new_artifact_root(root)
    with pytest.raises(FileExistsError):
        ensure_new_artifact_root(root)

    with pytest.raises(ValueError, match="audit-rerun"):
        ensure_audit_rerun_root(tmp_path / "legacy-models")
    audit_root = tmp_path / "audit-rerun-models"
    ensure_audit_rerun_root(audit_root)
    with pytest.raises(FileExistsError):
        ensure_audit_rerun_root(audit_root)
    with pytest.raises(ValueError, match="audit-rerun"):
        ensure_audit_rerun_output(tmp_path / "legacy.json")


def test_active_seed_provenance_call_sites_use_shared_protocol():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    call_sites = [
        root / "scripts/capture_003_dataset.py",
        root / "scripts/capture_004a_trajectories.py",
        root / "scripts/capture_004b_dataset.py",
        root / "scripts/evaluate_003.py",
        root / "src/evaluation/gsm8k.py",
        root / "src/analysis/huginn_trajectory.py",
    ]
    for path in call_sites:
        source = path.read_text()
        assert "SEED_PROTOCOL" in source, path
        assert "seed_for_example" in source, path


def test_provenance_output_guards_are_fail_closed(tmp_path):
    existing = tmp_path / "existing.json"
    existing.write_text("original")
    with pytest.raises(FileExistsError):
        ensure_new_output_path(existing)
    assert existing.read_text() == "original"


def test_corrected_v2_output_persists_and_validates_protocol(tmp_path):
    output = tmp_path / "corrected-v2" / "result.json"
    prepare_corrected_v2_output(output)
    manifest = output.parent / "protocol.json"
    assert json.loads(manifest.read_text()) == corrected_v2_protocol_metadata()
    write_json_exclusive(output, {"ok": True})
    with pytest.raises(FileExistsError):
        prepare_corrected_v2_output(output)
    manifest.write_text('{"protocol":"wrong"}')
    with pytest.raises(ValueError, match="manifest mismatch"):
        prepare_corrected_v2_output(output.parent / "other.json")


def test_synchronized_timer_records_completed_block():
    with synchronized_cuda_timer("cpu") as timing:
        time.sleep(0.001)
    assert timing.seconds >= 0.001


def test_synchronized_timer_calls_both_boundaries(monkeypatch):
    events = []
    monkeypatch.setattr(correctness, "synchronize_cuda", lambda device=None: events.append(("sync", device)))
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(correctness.time, "perf_counter", lambda: next(ticks))
    with correctness.synchronized_cuda_timer("cuda") as timing:
        events.append(("body", None))
    assert events == [("sync", "cuda"), ("body", None), ("sync", "cuda")]
    assert timing.seconds == 2.5


def test_measure_synchronized_excludes_warmup():
    calls = []
    result, seconds = correctness.measure_synchronized(
        lambda: calls.append(len(calls)) or len(calls), device="cpu", warmup=2
    )
    assert calls == [0, 1, 2]
    assert result == 3
    assert seconds >= 0
