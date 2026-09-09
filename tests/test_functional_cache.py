import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from src.data.functional_cache import (
    CACHE_ITEM_KEYS,
    CACHE_ITEM_PROTOCOL,
    HUGINN_MODEL_ID,
    build_cache_manifest,
    cache_manifest_sha256,
    validate_cache_item,
    validate_cache_manifest,
    write_cache_item_atomic,
    write_manifest_atomic,
)
from src.evaluation.correctness import SEED_PROTOCOL, per_example_seed
from src.evaluation.splits import (
    GSM8K_DATASET_CONFIG,
    GSM8K_DATASET_ID,
    GSM8K_DATASET_REVISION,
    load_split_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "b" * 40
GIT_COMMIT = "a" * 40


def cache_manifest():
    split_manifest = load_split_manifest(
        ROOT / "configs" / "004_functional_v2_splits.json"
    )
    return build_cache_manifest(
        model_revision=MODEL_REVISION,
        tokenizer_template="native apply_chat_template(add_generation_prompt=True)",
        git_commit=GIT_COMMIT,
        extraction_location="h0 before core; x input embedding; h16 before ln_f/coda",
        split_manifest=split_manifest,
        seed_policy=SEED_PROTOCOL,
        base_seed=3000,
        cap_degeneration_policy="cap-safe-manual-valid-end-v1",
    )


def cache_arrays(sequence_length=6, hidden_size=4, manifest=None):
    manifest = cache_manifest() if manifest is None else manifest
    return {
        "input_ids": np.arange(sequence_length, dtype=np.int32),
        "attention_mask": np.ones(sequence_length, dtype=np.bool_),
        "answer_start": np.array(2, dtype=np.int32),
        "valid_end": np.array(5, dtype=np.int32),
        "h0_full": np.zeros((sequence_length, hidden_size), dtype=np.float16),
        "x_full": np.ones((sequence_length, hidden_size), dtype=np.float16),
        "h16_teacher": np.full(
            (sequence_length, hidden_size), 2, dtype=np.float16
        ),
        "example_id": np.array(7, dtype=np.int64),
        "dataset_split": np.array("train"),
        "base_seed": np.array(3000, dtype=np.int64),
        "derived_seed": np.array(per_example_seed(3000, 7), dtype=np.int64),
        "seed_protocol": np.array(SEED_PROTOCOL),
        "cache_protocol": np.array(CACHE_ITEM_PROTOCOL),
        "manifest_sha256": np.array(cache_manifest_sha256(manifest)),
    }


def write_npz(path, arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def test_cache_item_write_close_validate_and_atomic_publish(tmp_path):
    path = tmp_path / "00007.npz"
    arrays = cache_arrays()
    manifest = cache_manifest()
    assert write_cache_item_atomic(path, arrays, manifest) == "created"
    assert path.exists()
    assert not list(tmp_path.glob(".00007.npz.*.tmp"))
    metadata = validate_cache_item(path, manifest)
    assert metadata["sequence_length"] == 6
    assert metadata["hidden_size"] == 4
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == CACHE_ITEM_KEYS


def test_resume_rejects_empty_or_malformed_existing_file(tmp_path):
    for payload in (b"", b"not an npz"):
        path = tmp_path / f"bad-{len(payload)}.npz"
        path.write_bytes(payload)
        with pytest.raises(ValueError, match="invalid cache item"):
            write_cache_item_atomic(path, cache_arrays(), cache_manifest())
        assert path.read_bytes() == payload


def test_resume_promotes_only_matching_valid_temporary_item(tmp_path):
    path = tmp_path / "00007.npz"
    temporary = tmp_path / ".00007.npz.interrupted.tmp"
    arrays = cache_arrays()
    write_npz(temporary, arrays)
    assert write_cache_item_atomic(path, arrays, cache_manifest()) == "resumed_tmp"
    assert path.exists() and not temporary.exists()


def test_resume_rejects_wrong_item_and_preserves_temporary_file(tmp_path):
    path = tmp_path / "00007.npz"
    temporary = tmp_path / ".00007.npz.interrupted.tmp"
    wrong = cache_arrays()
    wrong["input_ids"] = wrong["input_ids"].copy()
    wrong["input_ids"][0] = 99
    write_npz(temporary, wrong)
    with pytest.raises(ValueError, match="does not match requested item"):
        write_cache_item_atomic(path, cache_arrays(), cache_manifest())
    assert temporary.exists() and not path.exists()


def test_existing_schema_valid_item_must_match_requested_content(tmp_path):
    path = tmp_path / "00007.npz"
    arrays = cache_arrays()
    assert write_cache_item_atomic(path, arrays, cache_manifest()) == "created"
    changed = cache_arrays()
    changed["x_full"] = np.full_like(changed["x_full"], 3)
    with pytest.raises(ValueError, match="does not match requested item"):
        write_cache_item_atomic(path, changed, cache_manifest())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda arrays: arrays.pop("valid_end"),
        lambda arrays: arrays.__setitem__(
            "input_ids", arrays["input_ids"].astype(np.int64)
        ),
        lambda arrays: arrays.__setitem__(
            "attention_mask", arrays["attention_mask"].astype(np.int8)
        ),
        lambda arrays: arrays.__setitem__(
            "answer_start", np.array(0, dtype=np.int32)
        ),
        lambda arrays: arrays.__setitem__(
            "valid_end", np.array(7, dtype=np.int32)
        ),
        lambda arrays: arrays.__setitem__(
            "h16_teacher", arrays["h16_teacher"][:-1]
        ),
        lambda arrays: arrays.__setitem__(
            "h0_full", arrays["h0_full"].astype(np.float32)
        ),
        lambda arrays: arrays["h0_full"].__setitem__((0, 0), np.inf),
        lambda arrays: arrays.__setitem__(
            "base_seed", np.array(3000, dtype=np.int32)
        ),
    ],
)
def test_cache_item_validation_rejects_bad_schema_bounds_dtypes_shapes_finiteness(
    tmp_path, mutation
):
    arrays = cache_arrays()
    mutation(arrays)
    path = tmp_path / "bad.npz"
    write_npz(path, arrays)
    with pytest.raises(ValueError):
        validate_cache_item(path, cache_manifest())


def test_item_manifest_binding_rejects_wrong_split_and_seed(tmp_path):
    for field, value in (
        ("dataset_split", np.array("validation")),
        ("derived_seed", np.array(1, dtype=np.int64)),
        ("seed_protocol", np.array("wrong")),
    ):
        arrays = cache_arrays()
        arrays[field] = value
        path = tmp_path / f"{field}.npz"
        write_npz(path, arrays)
        with pytest.raises(ValueError):
            validate_cache_item(path, cache_manifest())


def test_concurrent_compliant_writers_serialize_without_clobbering(tmp_path):
    path = tmp_path / "00007.npz"
    manifest = cache_manifest()
    arrays = cache_arrays(manifest=manifest)

    def write():
        return write_cache_item_atomic(path, arrays, manifest)

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = list(executor.map(lambda _: write(), range(2)))
    assert sorted(statuses) == ["created", "validated_existing"]
    with np.load(path, allow_pickle=False) as archive:
        assert np.array_equal(archive["x_full"], arrays["x_full"])


def test_item_rejects_a_different_valid_manifest_digest(tmp_path):
    original_manifest = cache_manifest()
    arrays = cache_arrays(manifest=original_manifest)
    changed_manifest = copy.deepcopy(original_manifest)
    changed_manifest["git_commit"] = "c" * 40
    path = tmp_path / "00007.npz"
    write_npz(path, arrays)
    with pytest.raises(ValueError, match="exact cache manifest"):
        validate_cache_item(path, changed_manifest)


def test_manifest_contains_all_required_phase5_provenance(tmp_path):
    manifest = cache_manifest()
    assert manifest["huginn_model_id"] == HUGINN_MODEL_ID
    assert manifest["huginn_revision"] == MODEL_REVISION
    assert manifest["dataset_id"] == GSM8K_DATASET_ID
    assert manifest["dataset_config"] == GSM8K_DATASET_CONFIG
    assert manifest["dataset_revision"] == GSM8K_DATASET_REVISION
    assert manifest["tokenizer_id"] == HUGINN_MODEL_ID
    assert manifest["tokenizer_revision"] == MODEL_REVISION
    assert manifest["state_dtype"] == "float16"
    assert manifest["sequence_count"] == 2500
    assert manifest["train_ids"] == list(range(2250))
    assert manifest["validation_ids"] == list(range(2250, 2500))
    assert len(manifest["split_manifest_sha256"]) == 64
    assert manifest["git_commit"] == GIT_COMMIT
    assert manifest["extraction_location"]
    assert manifest["seed_policy"] == SEED_PROTOCOL
    assert manifest["cap_degeneration_policy"]

    path = tmp_path / "manifest.json"
    assert write_manifest_atomic(path, manifest) == "created"
    assert write_manifest_atomic(path, manifest) == "validated_existing"
    changed = copy.deepcopy(manifest)
    changed["git_commit"] = "c" * 40
    with pytest.raises(ValueError, match="does not match"):
        write_manifest_atomic(path, changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("huginn_model_id", "wrong"),
        ("dataset_revision", "0" * 40),
        ("sequence_count", 2499),
        ("train_ids", list(range(1, 2251))),
        ("validation_ids", list(range(2249, 2499))),
        ("split_manifest_sha256", "bad"),
        ("git_commit", "abc123"),
        ("seed_policy", ""),
        ("cap_degeneration_policy", ""),
    ],
)
def test_manifest_validation_rejects_provenance_or_split_tampering(
    tmp_path, field, value
):
    manifest = cache_manifest()
    manifest[field] = value
    with pytest.raises(ValueError):
        validate_cache_manifest(manifest)
    with pytest.raises(ValueError):
        write_manifest_atomic(tmp_path / "manifest.json", manifest)
