import copy
import json
from pathlib import Path

import pytest

from scripts import run_004_phase12_full_prefix_evaluation as phase12


def test_phase12_config_locks_slow_full_prefix_semantics():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/004_functional_v2_phase12.json").read_text())
    phase12.validate_config(config)
    assert config["example_ids"] == list(range(8))
    assert config["use_cache"] is False
    assert config["attention2_kv_cache"] is False
    assert config["recompute_complete_prefix_every_token"] is True
    for key, value in [
        ("use_cache", True),
        ("attention2_kv_cache", True),
        ("recompute_complete_prefix_every_token", False),
        ("example_ids", [0]),
        ("phase11_model_sha256", "wrong"),
    ]:
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError, match="exactly match"):
            phase12.validate_config(changed)


def test_phase12_publication_is_atomic_read_only_and_no_replace(tmp_path):
    attempt = tmp_path / ".phase12.attempt-1"
    attempt.mkdir()
    evaluation = attempt / "evaluation.json"
    evaluation.write_text('{"status":"pass"}\n')
    evaluation.chmod(0o444)
    final = tmp_path / "phase12"

    phase12.publish_attempt(attempt, final)

    assert not attempt.exists()
    assert final.stat().st_mode & 0o777 == 0o555
    assert (final / "evaluation.json").stat().st_mode & 0o777 == 0o444
    second = tmp_path / ".phase12.attempt-2"
    second.mkdir()
    second_evaluation = second / "evaluation.json"
    second_evaluation.write_text("preserved")
    second_evaluation.chmod(0o444)
    with pytest.raises(FileExistsError):
        phase12.publish_attempt(second, final)
    assert second.exists()


def test_phase12_runner_does_not_use_incremental_generate_or_persist_logits():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/run_004_phase12_full_prefix_evaluation.py").read_text()
    evaluator = (root / "src/evaluation/functional_autoregressive.py").read_text()
    assert ".generate(" not in source.replace("evaluator.generate(", "")
    assert "past_key_values" not in evaluator
    assert '"full_vocabulary_logits_persisted": False' in source
    assert "lm_head(state[:, -1:])" in evaluator
