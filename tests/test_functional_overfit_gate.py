import copy
import json
from pathlib import Path
import time

import pytest
import torch
from torch import nn

from scripts import preflight_004_functional_overfit_gate as preflight_script
from scripts import recover_004_functional_overfit_publication as recovery_script
from scripts import run_004_functional_overfit_gate as overfit_script
from src.training.overfit_gate import gate_passed, quantitative_overfit_checks


def test_quantitative_gate_requires_every_prespecified_improvement():
    initial = {
        "kl_per_token": 10.0,
        "first_token_distribution_kl": 4.0,
        "mean_first_target_probability": 0.01,
        "first_target_top1_count": 0,
    }
    final = {
        "kl_per_token": 0.4,
        "first_token_distribution_kl": 0.4,
        "mean_first_target_probability": 0.6,
        "first_target_top1_count": 6,
    }
    checks = quantitative_overfit_checks(
        initial,
        final,
        maximum_kl_ratio=0.2,
        maximum_final_kl=0.5,
        maximum_first_distribution_kl_ratio=0.5,
        maximum_final_first_distribution_kl=0.5,
        minimum_final_first_target_probability=0.5,
        minimum_final_first_target_top1_count=6,
    )
    assert gate_passed(checks)
    for key in checks:
        failed = dict(checks)
        failed[key] = False
        assert not gate_passed(failed)
    with pytest.raises(ValueError):
        gate_passed({"dramatic_global_kl_decrease": True})


def test_overfit_config_is_exact_and_has_no_trajectory_loss():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/004_functional_v2_overfit.json").read_text())
    overfit_script.validate_config(config)
    assert config["example_ids"] == list(range(8))
    assert config["h0_seed_indices"] == [0]
    assert config["trajectory_loss_weight"] == 0.0
    assert config["generation"]["full_prefix_recomputation"] is True
    assert config["generation"]["use_cache"] is False
    for section, key, value in [
        (None, "example_ids", list(range(1, 9))),
        (None, "trajectory_loss_weight", 1.0),
        ("pass_criteria", "maximum_final_over_initial_kl_ratio", 0.9),
        ("generation", "use_cache", True),
        ("generation", "minimum_teacher_answer_match_count", 1),
        ("optimizer", "learning_rate", 2e-4),
    ]:
        changed = copy.deepcopy(config)
        if section is None:
            changed[key] = value
        else:
            changed[section][key] = value
        with pytest.raises(ValueError, match="exactly match"):
            overfit_script.validate_config(changed)


def test_model_publication_is_atomic_read_only_and_contains_no_logits(tmp_path):
    model = nn.Linear(4, 4)
    output = tmp_path / "model.pt"
    metadata = {
        "status": "pass",
        "full_vocabulary_logits_persisted": False,
    }
    overfit_script.save_model_atomic(output, model, metadata)
    assert output.stat().st_mode & 0o777 == 0o444
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    assert set(checkpoint) == {"protocol", "state_dict", "metadata"}
    assert checkpoint["metadata"] == metadata
    assert all("logit" not in key for key in checkpoint["state_dict"])
    with pytest.raises(FileExistsError):
        overfit_script.save_model_atomic(output, model, metadata)


class FakeTokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {"<|end_text|>": 9, "<|end_turn|>": 8}[token]

    def decode(self, token_ids, skip_special_tokens=False):
        if skip_special_tokens:
            token_ids = [token for token in token_ids if token not in {8, 9}]
        return " ".join(str(token) for token in token_ids)


class FakeTransformer:
    ln_f = nn.Identity()


class FakeHuginn:
    def __init__(self):
        self.transformer = FakeTransformer()
        self.lengths = []
        self.iterate_forward = self.original_iterate_forward

    def original_iterate_forward(self, *args, **kwargs):
        raise AssertionError("original recurrence must be patched")

    def initialize_state(self, input_embeds, scale=1.0):
        return torch.randn_like(input_embeds) * scale

    def next_token(self):
        return min(len(self.lengths), 9)

    def __call__(self, *, input_ids, attention_mask, num_steps, use_cache, return_dict):
        assert use_cache is False
        assert attention_mask.all()
        self.lengths.append(input_ids.shape[1])
        embeds = torch.zeros((*input_ids.shape, 4), dtype=torch.float32)
        state = self.iterate_forward(
            embeds, None, None, torch.tensor(0), None, num_steps=num_steps
        )[0]
        logits = torch.zeros((*input_ids.shape, 10)) + state[..., :1] * 0
        logits[:, -1, self.next_token()] = 10
        return type("Output", (), {"logits": logits})()


class FakeA2(nn.Module):
    def forward(self, h0, x, token_mask=None):
        assert token_mask.shape == h0.shape[:2]
        output = h0[:, None].expand(-1, 16, -1, -1)
        return output, [], []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_training_loop_globally_counts_all_eight_and_never_stores_teacher_logits():
    class TinyA2(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(4, 4)

        def forward(self, h0, x, token_mask=None):
            state = self.projection(h0 + x)
            return state[:, None].expand(-1, 16, -1, -1), [], []

    class TinyHuginn(nn.Module):
        def __init__(self):
            super().__init__()
            self.coda = nn.Linear(4, 7)
            self.calls = 0

        def forward(self, *, input_states, **kwargs):
            self.calls += 1
            return type("Output", (), {"logits": self.coda(input_states)})()

    device = torch.device("cuda")
    a2 = TinyA2().to(device)
    huginn = TinyHuginn().to(device).eval().requires_grad_(False)
    examples = []
    for example_id in range(8):
        attention = torch.ones((1, 3), dtype=torch.bool, device=device)
        examples.append(
            {
                "example_id": example_id,
                "input_ids": torch.tensor([[0, 1, 2]], device=device),
                "attention_mask": attention,
                "h0": torch.randn(1, 3, 4, device=device),
                "x": torch.randn(1, 3, 4, device=device),
                "h16_teacher": torch.randn(1, 3, 4, device=device),
                "answer_start": 1,
                "valid_end": 3,
                "loss_mask": overfit_script.causal_answer_loss_mask(
                    attention, [1], [3]
                ),
            }
        )
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/004_functional_v2_overfit.json").read_text())
    config["training"] = {
        "minimum_updates": 1,
        "maximum_updates": 1,
        "evaluate_every_updates": 1,
        "data_order_seed": 9100,
    }

    trace, _ = overfit_script.train(a2, huginn, examples, config)

    assert [row["valid_token_count"] for row in trace] == [16, 16]
    assert huginn.calls == 8 * 2 * 3  # initial metrics, update, final metrics
    assert all("teacher_logits" not in example for example in examples)
    assert all(parameter.grad is None for parameter in huginn.parameters())


def test_short_generation_cannot_inflate_fixed_width_prefix_agreement():
    teacher = list(range(32))
    assert overfit_script.fixed_width_teacher_prefix_match([0], teacher) == 1 / 32
    assert overfit_script.fixed_width_teacher_prefix_match(teacher, teacher) == 1.0


def test_generation_recomputes_the_complete_prefix_each_token():
    huginn = FakeHuginn()
    example = {
        "example_id": 0,
        "input_ids": torch.tensor([[7, 7, 1, 2, 3, 4]]),
        "answer_start": 2,
        "valid_end": 6,
    }
    generated = overfit_script.generate_same_examples(
        FakeA2(), huginn, FakeTokenizer(), [example], max_new=4
    )
    assert huginn.lengths == [2, 3, 4, 5]
    assert generated[0]["generated_tokens"] == 4
    assert generated[0]["nonempty_text"] is True
    assert generated[0]["hit_max_new_tokens"] is True
    assert generated[0]["ended_naturally"] is False
    assert generated[0]["fallback_allowed"] is False
    assert generated[0]["teacher_answer_match"] is False


def test_stop_only_generation_is_not_substantive():
    class StopOnlyHuginn(FakeHuginn):
        def next_token(self):
            return 9

    example = {
        "example_id": 0,
        "input_ids": torch.tensor([[7, 7, 9, 2]]),
        "answer_start": 2,
        "valid_end": 4,
    }
    generated = overfit_script.generate_same_examples(
        FakeA2(), StopOnlyHuginn(), FakeTokenizer(), [example], max_new=4
    )
    assert generated[0]["ended_naturally"] is True
    assert generated[0]["hit_max_new_tokens"] is False
    assert generated[0]["generated_tokens"] == 1
    assert generated[0]["nonempty_text"] is False
    assert generated[0]["substantive_text"] == ""


def test_prelaunch_attestation_records_absent_output_and_process(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "FROZEN.json").write_text("frozen")
    initialization = tmp_path / "init.pt"
    initialization.write_text("init")
    output = tmp_path / "final"
    attempts = tmp_path / "final_attempts"
    monkeypatch.setattr(preflight_script, "CACHE_ROOT", cache)
    monkeypatch.setattr(preflight_script, "INITIALIZATION", initialization)
    monkeypatch.setattr(preflight_script, "OUTPUT_ROOT", output)
    monkeypatch.setattr(preflight_script, "ATTEMPTS_ROOT", attempts)
    monkeypatch.setattr(preflight_script, "clean_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(preflight_script, "matching_training_processes", lambda: [])

    path = preflight_script.main()

    value = json.loads(path.read_text())
    assert value["final_output_absent"] is True
    assert value["matching_training_processes"] == []
    assert path.stat().st_mode & 0o777 == 0o444


def test_training_requires_recent_matching_prelaunch_attestation(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "FROZEN.json").write_text("frozen")
    initialization = tmp_path / "init.pt"
    initialization.write_text("init")
    output = tmp_path / "final"
    attempts = tmp_path / "final_attempts"
    attempts.mkdir()
    record = attempts / "prelaunch-test.json"
    commit = "a" * 40
    monkeypatch.setattr(overfit_script, "CACHE_ROOT", cache)
    monkeypatch.setattr(overfit_script, "INITIALIZATION", initialization)
    monkeypatch.setattr(overfit_script, "OUTPUT_ROOT", output)
    value = {
        "protocol": overfit_script.PRELAUNCH_PROTOCOL,
        "status": "pass",
        "created_unix_seconds": time.time(),
        "git_commit": commit,
        "final_output": str(output),
        "final_output_absent": True,
        "matching_training_processes": [],
        "cache_freeze_sha256": overfit_script.sha256_file(cache / "FROZEN.json"),
        "initialization_sha256": overfit_script.sha256_file(initialization),
    }
    overfit_script.write_json_exclusive_fsync(record, value)

    assert overfit_script.validate_prelaunch_attestation(record, commit) == value
    stale = dict(value)
    stale["created_unix_seconds"] = time.time() - 301
    stale_record = attempts / "prelaunch-stale.json"
    overfit_script.write_json_exclusive_fsync(stale_record, stale)
    with pytest.raises(ValueError, match="not recent"):
        overfit_script.validate_prelaunch_attestation(stale_record, commit)


def test_complete_publication_recovery_preserves_source_and_bytes(tmp_path, monkeypatch):
    source = tmp_path / "attempts" / "attempt-preserved"
    source.mkdir(parents=True)
    model = source / "model.pt"
    model.write_bytes(b"unchanged-model")
    model_hash = recovery_script.sha256_file(model)
    result = source / "result.json"
    result.write_text(
        json.dumps(
            {
                "status": "pending_scientific_review",
                "model_sha256": model_hash,
            }
        )
    )
    result_hash = recovery_script.sha256_file(result)
    model.chmod(0o444)
    result.chmod(0o444)
    final = tmp_path / "final"
    record = tmp_path / "attempts" / "publication.json"
    monkeypatch.setattr(recovery_script, "SOURCE", source)
    monkeypatch.setattr(recovery_script, "FINAL", final)
    monkeypatch.setattr(recovery_script, "RECORD", record)
    monkeypatch.setattr(recovery_script, "EXPECTED_RESULT_SHA256", result_hash)
    monkeypatch.setattr(recovery_script, "clean_git_commit", lambda: "a" * 40)

    observed = recovery_script.main()

    assert source.is_dir()
    assert (source / "model.pt").read_bytes() == b"unchanged-model"
    assert recovery_script.sha256_file(final / "model.pt") == model_hash
    assert recovery_script.sha256_file(final / "result.json") == result_hash
    assert observed["retraining_performed"] is False
    assert observed["model_or_result_bytes_changed"] is False
    assert record.stat().st_mode & 0o777 == 0o444


def test_publication_recovery_copies_bytes_read_only_without_mutation(tmp_path):
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"preserved-scientific-result")
    source.chmod(0o444)

    recovery_script.copy_fsync_read_only(source, destination)

    assert source.read_bytes() == destination.read_bytes()
    assert source.stat().st_mode & 0o777 == 0o444
    assert destination.stat().st_mode & 0o777 == 0o444


def test_attempt_directory_publication_is_atomic_and_no_replace(tmp_path):
    attempt = tmp_path / ".final.attempt-1"
    attempt.mkdir()
    for name in ("model.pt", "result.json"):
        path = attempt / name
        path.write_text(name)
        path.chmod(0o444)
    output = tmp_path / "final"

    overfit_script.atomic_publish_attempt(attempt, output)

    assert not attempt.exists()
    assert output.is_dir()
    assert (output / "model.pt").read_text() == "model.pt"
    second = tmp_path / ".final.attempt-2"
    second.mkdir()
    for name in ("model.pt", "result.json"):
        path = second / name
        path.write_text(name)
        path.chmod(0o444)
    with pytest.raises(FileExistsError):
        overfit_script.atomic_publish_attempt(second, output)
    assert second.exists()
