import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data.functional_extraction import capture_huginn_functional_states
from scripts.build_004_functional_cache_v2 import (
    _git_commit,
    _load_source,
    _validate_config,
)
from src.data.functional_cache import (
    CACHE_ITEM_KEYS,
    FROZEN_VALID_END_MANIFEST_SHA256,
)
from src.evaluation.correctness import SEED_PROTOCOL
from src.evaluation.splits import load_split_manifest


ROOT = Path(__file__).resolve().parents[1]


class FakeHuginn:
    def core_block_forward(
        self,
        x,
        input_embeds,
        freqs_cis,
        mask,
        past_key_values,
        block_idx,
        current_step,
    ):
        return x + input_embeds, block_idx + 1

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        num_steps,
        use_cache,
        output_details,
        input_states=None,
    ):
        assert use_cache is False
        assert output_details == {
            "return_logits": False,
            "return_latents": True,
            "return_head": False,
            "return_stats": False,
        }
        batch, tokens = input_ids.shape
        h0 = (
            torch.zeros((batch, tokens, 3), device=input_ids.device)
            if input_states is None
            else input_states
        )
        recurrent_input = torch.ones_like(h0)
        state = h0
        block = torch.tensor(-1)
        for step in range(num_steps):
            state, block = self.core_block_forward(
                state, recurrent_input, None, None, None, block, step
            )
        return SimpleNamespace(latent_states=state)


def test_phase6_config_is_one_shared_compact_d16_cache():
    config = json.loads((ROOT / "configs/004_functional_v2.json").read_text())
    assert config["teacher_depth"] == 16
    assert config["model_compute_dtype"] == "bfloat16"
    assert config["cache_state_dtype"] == "float16"
    assert config["seed_policy"] == SEED_PROTOCOL
    assert config["cache_output"] == "/workspace/functional_cache_v2"
    assert not any("refinement" in key or key == "K" for key in config)
    sidecar = ROOT / config["valid_end_manifest"]
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == (
        FROZEN_VALID_END_MANIFEST_SHA256
    )


def test_builder_rejects_pin_changes_and_wrong_source_hash(tmp_path):
    config = json.loads((ROOT / "configs/004_functional_v2.json").read_text())
    split_manifest = load_split_manifest(ROOT / config["split_manifest"])
    _validate_config(config, split_manifest)
    changed = copy.deepcopy(config)
    changed["model_revision"] = "0" * 40
    with pytest.raises(ValueError, match="model_revision"):
        _validate_config(changed, split_manifest)

    source = tmp_path / "source.npz"
    np.savez(source, input_ids=np.arange(4, dtype=np.int32))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert np.array_equal(_load_source(source, digest), np.arange(4, dtype=np.int32))
    with pytest.raises(ValueError, match="source hash"):
        _load_source(source, "0" * 64)


def test_git_provenance_rejects_untracked_files(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True
    )
    (tmp_path / "tracked.txt").write_text("frozen\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "frozen"], cwd=tmp_path, check=True
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert _git_commit(tmp_path) == commit

    (tmp_path / "untracked.py").write_text("print('not committed')\n")
    with pytest.raises(RuntimeError, match="completely clean"):
        _git_commit(tmp_path)


def test_compact_schema_excludes_logits_and_builder_adopts_atomic_writer():
    assert "teacher_logits" not in CACHE_ITEM_KEYS
    assert "student_logits" not in CACHE_ITEM_KEYS
    assert {"h0_full", "x_full", "h16_teacher"} <= CACHE_ITEM_KEYS
    builder = (ROOT / "scripts/build_004_functional_cache_v2.py").read_text()
    assert "write_cache_item_atomic(item_path, arrays, manifest)" in builder


def test_capture_extracts_h0_x_and_exact_d16_coda_input_and_restores_hook():
    model = FakeHuginn()
    original = model.core_block_forward
    input_ids = torch.arange(5).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    states = capture_huginn_functional_states(
        model, input_ids, attention_mask, depth=16
    )

    assert model.core_block_forward == original
    assert set(states) == {"h0_full", "x_full", "h16_teacher"}
    assert all(value.shape == (5, 3) for value in states.values())
    assert all(value.dtype == np.float16 for value in states.values())
    assert np.all(states["h0_full"] == 0)
    assert np.all(states["x_full"] == 1)
    assert np.all(states["h16_teacher"] == 16)


def test_capture_optionally_injects_and_returns_exact_explicit_h0():
    model = FakeHuginn()
    input_ids = torch.arange(5).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    injected = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)

    states = capture_huginn_functional_states(
        model,
        input_ids,
        attention_mask,
        depth=16,
        input_states=injected,
    )

    assert np.array_equal(states["h0_full"], injected.numpy()[0].astype(np.float16))
    assert np.array_equal(states["h16_teacher"], states["h0_full"] + np.float16(16))


def test_capture_rejects_noncanonical_depth_before_model_execution():
    model = FakeHuginn()
    input_ids = torch.arange(5).unsqueeze(0)
    with pytest.raises(ValueError, match="frozen to Huginn D=16"):
        capture_huginn_functional_states(
            model,
            input_ids,
            torch.ones_like(input_ids, dtype=torch.bool),
            depth=15,
        )


def test_capture_restores_original_hook_when_model_raises():
    class FailingHuginn(FakeHuginn):
        def __call__(self, **kwargs):
            self.core_block_forward(
                torch.zeros((1, 2, 3)),
                torch.ones((1, 2, 3)),
                None,
                None,
                None,
                torch.tensor(-1),
                0,
            )
            raise RuntimeError("injected model failure")

    model = FailingHuginn()
    original = model.core_block_forward
    with pytest.raises(RuntimeError, match="injected model failure"):
        capture_huginn_functional_states(
            model,
            torch.arange(2).unsqueeze(0),
            torch.ones((1, 2), dtype=torch.bool),
        )
    assert model.core_block_forward == original
