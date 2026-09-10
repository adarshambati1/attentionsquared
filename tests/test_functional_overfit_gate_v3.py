import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts import run_004_functional_overfit_gate_v3 as runner
import src.evaluation.functional_autoregressive as autoregressive


ROOT = Path(__file__).resolve().parents[1]


def test_v3_config_preserves_overfit_protocol_and_locks_phase10():
    config = json.loads((ROOT / "configs/004_functional_v3_overfit.json").read_text())
    runner.validate_config(config)
    assert config["example_ids"] == list(range(8))
    assert config["K"] == 4
    assert config["h0_seed_indices"] == [0]
    assert config["trajectory_loss_weight"] == 0.0
    assert config["training"]["data_order_seed"] == 9100
    assert config["generation"]["maximum_new_tokens"] == 384
    assert config["phase10_artifact_sha256"] == runner.PHASE10_ARTIFACT_SHA256
    assert config["phase10_attestation_sha256"] == runner.PHASE10_ATTESTATION_SHA256

    changed = copy.deepcopy(config)
    changed["output"] = "/workspace/functional_overfit_v2"
    with pytest.raises(ValueError, match="exactly match"):
        runner.validate_config(changed)


def test_runner_can_only_target_new_no_replace_v3_output():
    assert runner.CONFIG == Path("configs/004_functional_v3_overfit.json")
    assert runner.OUTPUT_ROOT == Path("/workspace/functional_overfit_v3")
    source = (ROOT / "scripts/run_004_functional_overfit_gate_v3.py").read_text()
    assert "/workspace/functional_overfit_v2" not in source
    assert "atomic_publish_attempt(attempt, output_root)" in source
    assert "RENAME_NOREPLACE" in source
    with pytest.raises(ValueError, match="exact production paths"):
        runner.main(runner.CONFIG, Path("/workspace/functional_overfit_v2"), Path("unused"))


def test_phase10_artifact_and_attestation_are_both_immutable_prerequisites(tmp_path, monkeypatch):
    artifact = tmp_path / "correctness_gate_coda_v3.json"
    artifact.write_text(json.dumps({"protocol": "functional-correctness-gate-coda-v3", "status": "pass"}))
    artifact_hash = runner.sha256_file(artifact)
    attestation = tmp_path / "attestation.json"
    attestation.write_text(json.dumps({
        "protocol": "functional-correctness-gate-coda-v3-runtime-attestation-v1",
        "status": "pass",
        "artifact": str(artifact),
        "artifact_sha256": artifact_hash,
        "git_commit": "f521b2b",
        "is_symlink": False,
        "lstat_type": "regular-file",
        "mode": "0444",
    }))
    monkeypatch.setattr(runner, "PHASE10_ARTIFACT", artifact)
    monkeypatch.setattr(runner, "PHASE10_ATTESTATION", attestation)
    monkeypatch.setattr(runner, "PHASE10_ARTIFACT_SHA256", artifact_hash)
    monkeypatch.setattr(runner, "PHASE10_ATTESTATION_SHA256", runner.sha256_file(attestation))

    assert runner.validate_phase10_prerequisites()["git_commit"] == "f521b2b"
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        runner.validate_phase10_prerequisites()


class AddThree(nn.Module):
    def forward(self, state, frequencies, block_index, mask, cache):
        return state + 3


class TimesTwo(nn.Module):
    def forward(self, state):
        return state * 2


class TinyHuginn(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_scale = 1
        self.transformer = SimpleNamespace(
            wte=nn.Embedding(16, 1),
            prelude=[],
            coda=[AddThree()],
            ln_f=TimesTwo(),
        )
        self.lm_head = nn.Identity()
        self.freqs_cis = torch.zeros(1, 32, 1)

    def initialize_state(self, input_embeds, scale=1.0):
        return torch.zeros_like(input_embeds)


class ConstantA2(nn.Module):
    def forward(self, h0, x, token_mask=None):
        state = torch.full_like(x, 2.0)
        return state[:, None].expand(-1, 16, -1, -1), [], []


def test_teacher_student_and_generation_share_single_authoritative_coda_helper(monkeypatch):
    calls = []
    authoritative = autoregressive.frozen_coda_logits_from_normalized_state

    def spy(huginn, normalized_state, frequencies, *, last_only=False):
        calls.append((normalized_state.detach().clone(), last_only))
        return authoritative(huginn, normalized_state, frequencies, last_only=last_only)

    # Phase 11 teacher/student calls use this imported authoritative helper.
    monkeypatch.setattr(runner, "normalized_state_coda_logits", spy)
    huginn = TinyHuginn()
    example = {"input_ids": torch.tensor([[1, 2]])}
    teacher = runner.coda_logits(huginn, example, torch.tensor([[[2.0], [2.0]]]))
    student = runner.coda_logits(huginn, example, torch.tensor([[[4.0], [4.0]]]))
    assert torch.equal(teacher, torch.tensor([[[10.0], [10.0]]]))
    assert torch.equal(student, torch.tensor([[[14.0], [14.0]]]))

    # Correct full-prefix generation resolves the same module-level helper.
    monkeypatch.setattr(autoregressive, "frozen_coda_logits_from_normalized_state", spy)
    evaluator = runner.FullPrefixAttention2Evaluator(huginn, ConstantA2(), SimpleNamespace())
    generation_logits = evaluator.next_token_logits(torch.tensor([[1, 2]]), example_id=0)
    assert torch.equal(generation_logits, torch.tensor([[10.0]]))
    assert [last_only for _, last_only in calls] == [False, False, True]


def test_preflight_process_scan_ignores_parent_shell_text_and_matches_exact_python_script(tmp_path):
    shell = tmp_path / "100"
    shell.mkdir()
    (shell / "cmdline").write_bytes(
        b"bash\0-c\0python scripts/run_004_functional_overfit_gate_v3.py\0"
    )
    runner_process = tmp_path / "101"
    runner_process.mkdir()
    (runner_process / "cmdline").write_bytes(
        b"python\0scripts/run_004_functional_overfit_gate_v3.py\0"
    )
    unrelated = tmp_path / "102"
    unrelated.mkdir()
    (unrelated / "cmdline").write_bytes(b"python\0other.py\0")

    from scripts import preflight_004_functional_overfit_gate_v3 as preflight

    assert preflight.matching_training_processes(tmp_path) == [
        {
            "pid": 101,
            "argv": ["python", "scripts/run_004_functional_overfit_gate_v3.py"],
        }
    ]


def test_superseded_v3_runner_and_preflight_are_hard_blocked(monkeypatch):
    with pytest.raises(RuntimeError, match="prefix-stable"):
        runner.main(runner.CONFIG, runner.OUTPUT_ROOT, Path("unused"))
    from scripts import preflight_004_functional_overfit_gate_v3 as preflight
    with pytest.raises(RuntimeError, match="prefix-stable"):
        preflight.main()


def test_v3_has_no_surrogate_patch_or_extra_initial_ln_f_and_persists_no_logits():
    source = (ROOT / "scripts/run_004_functional_overfit_gate_v3.py").read_text()
    assert "patched_attention2_recurrence" not in source
    assert "iterate_forward" not in source
    assert "transformer.ln_f(output[:, 15])" not in source
    assert "FullPrefixAttention2Evaluator" in source
    assert '"full_vocabulary_logits_persisted": False' in source
    assert '"teacher_logits"' not in source
