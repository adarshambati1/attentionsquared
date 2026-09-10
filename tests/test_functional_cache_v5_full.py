import json
import random
from pathlib import Path

import numpy as np

from scripts import build_004_functional_cache_v5_full as builder
from scripts import freeze_004_functional_cache_v5_full as freezer
from src.data.functional_cache_v5_full import (
    CACHE_V5_ITEM_KEYS,
    CACHE_V5_ITEM_PROTOCOL,
    FULL_EXAMPLE_IDS,
    build_manifest,
    manifest_sha256,
    validate_item,
    write_item_atomic,
)
from src.data.functional_extraction_v3 import NORMALIZED_H16_SEMANTICS
from src.evaluation.correctness import SEED_PROTOCOL, per_example_seed
from src.training.functional_protocol import FIXED_H0_SCHEDULE_PROTOCOL

ROOT = Path(__file__).resolve().parents[1]


def make_manifest():
    hashes = {i: "a" * 64 for i in FULL_EXAMPLE_IDS}
    return build_manifest(
        model_revision="b" * 40,
        tokenizer_id="tomg-group-umd/huginn-0125",
        tokenizer_revision="b" * 40,
        tokenizer_template="template",
        git_commit="c" * 40,
        extraction_location="canonical native extraction",
        source_continuations="results/004_functional/teacher_sequences",
        source_sha256_by_id=hashes,
        base_seed=3000,
        seed_index=0,
        cache_output="/workspace/functional_cache_v5_full",
        split_manifest_sha256="d" * 64,
        source_cache_v2_manifest_sha256="e" * 64,
        source_cache_v2_freeze_sha256="f" * 64,
        estimated_uncompressed_item_bytes=1,
    )


def make_arrays(manifest, example_id=0, length=3, hidden=5280):
    return {
        "input_ids": np.arange(length, dtype=np.int32),
        "attention_mask": np.ones(length, dtype=np.bool_),
        "answer_start": np.array(1, dtype=np.int32),
        "valid_end": np.array(length, dtype=np.int32),
        "h0": np.zeros((length, hidden), dtype=np.uint16),
        "x": np.ones((length, hidden), dtype=np.float32),
        "h16": np.full((length, hidden), 2, dtype=np.float32),
        "example_id": np.array(example_id, dtype=np.int64),
        "dataset_split": np.array("train"),
        "base_seed": np.array(3000, dtype=np.int64),
        "seed_index": np.array(0, dtype=np.int64),
        "derived_seed": np.array(per_example_seed(3000, example_id, seed_index=0), dtype=np.int64),
        "seed_protocol": np.array(SEED_PROTOCOL),
        "schedule_length": np.array(2048, dtype=np.int64),
        "schedule_protocol": np.array(FIXED_H0_SCHEDULE_PROTOCOL),
        "full_schedule_sha256": np.array("1" * 64),
        "source_sha256": np.array("a" * 64),
        "h16_semantics": np.array(NORMALIZED_H16_SEMANTICS),
        "cache_protocol": np.array(CACHE_V5_ITEM_PROTOCOL),
        "manifest_sha256": np.array(manifest_sha256(manifest)),
    }


def test_full_config_and_locked_partition():
    config = json.loads((ROOT / "configs/004_functional_cache_v5_full.json").read_text())
    builder.validate_config(config)
    rows = builder.locked_rows(config)
    assert len(rows) == 2500
    assert [row["example_id"] for row in rows] == list(range(2500))
    assert all(row["phase3_role"] == "gradient_update" for row in rows[:2250])
    assert all(row["phase3_role"] == "checkpoint_selection_validation" for row in rows[2250:])


def test_replay_sample_is_prospectively_seeded_plus_boundaries():
    config = json.loads((ROOT / "configs/004_functional_cache_v5_full.json").read_text())
    random_ids = sorted(random.Random(config["validation_replay_selection_seed"]).sample(range(2500), 16))
    assert config["validation_replay_ids"] == sorted({0, 2249, 2250, 2499, *random_ids})


def test_full_native_item_schema_has_no_logits(tmp_path):
    manifest = make_manifest()
    arrays = make_arrays(manifest)
    path = tmp_path / "00000.npz"
    assert write_item_atomic(path, arrays, manifest) == "created"
    assert validate_item(path, manifest)["example_id"] == 0
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == CACHE_V5_ITEM_KEYS
        assert not any("logit" in key.lower() for key in archive.files)
        assert archive["h0"].dtype == np.uint16
        assert archive["x"].dtype == archive["h16"].dtype == np.float32


def test_build_is_resumable_and_finalization_hashes_complete_tree():
    source = (ROOT / "scripts/build_004_functional_cache_v5_full.py").read_text()
    assert 'f".{OUTPUT.name}.building"' in source
    assert "status=validated_existing" in source
    assert "publish_directory_no_replace" in source
    assert "quarantine evidence preserved outside cache" in source
    validator = (ROOT / "scripts/validate_004_functional_cache_v5_full.py").read_text()
    assert "all_items_readable_and_schema_exact" in validator
    assert "coda_logits_exact" in validator
    assert "full_schedule_hash_exact" in validator
    assert "globally_token_normalized_kl" in validator


def test_tree_hash_includes_hidden_lock_file(tmp_path):
    (tmp_path / "manifest.json").write_bytes(b"manifest")
    (tmp_path / ".functional-cache-v5.lock").write_bytes(b"")
    digest, hashes = freezer.tree_sha256(tmp_path)
    assert len(digest) == 64
    assert set(hashes) == {"manifest.json", ".functional-cache-v5.lock"}
