import copy
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import src.data.functional_cache as functional_cache
from src.data.functional_cache import (
    CACHE_ITEM_KEYS,
    CACHE_ITEM_PROTOCOL,
    FROZEN_VALID_END_MANIFEST_SHA256,
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
        cap_degeneration_policy="reviewed-valid-end-manifest-v1",
        valid_end_manifest_sha256=FROZEN_VALID_END_MANIFEST_SHA256,
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


def test_resume_quarantines_empty_or_malformed_final_and_regenerates(tmp_path):
    for payload in (b"", b"not an npz"):
        path = tmp_path / f"bad-{len(payload)}.npz"
        path.write_bytes(payload)
        assert (
            write_cache_item_atomic(path, cache_arrays(), cache_manifest())
            == "created_after_quarantine"
        )
        validate_cache_item(path, cache_manifest())
        quarantined = list(
            (tmp_path / "quarantine").glob(f"{path.name}.*.quarantine")
        )
        assert len(quarantined) == 1
        assert quarantined[0].read_bytes() == payload
        event = json.loads(
            quarantined[0].with_suffix(quarantined[0].suffix + ".json").read_text()
        )
        assert event["category"] == "invalid_final"
        assert event["original_name"] == path.name
        assert event["sha256"] == hashlib.sha256(payload).hexdigest()


def test_resume_promotes_only_matching_valid_temporary_item(tmp_path):
    path = tmp_path / "00007.npz"
    temporary = tmp_path / ".00007.npz.interrupted.tmp"
    arrays = cache_arrays()
    write_npz(temporary, arrays)
    assert write_cache_item_atomic(path, arrays, cache_manifest()) == "resumed_tmp"
    assert path.exists() and not temporary.exists()


def test_resume_quarantines_wrong_temporary_and_regenerates(tmp_path):
    path = tmp_path / "00007.npz"
    temporary = tmp_path / ".00007.npz.interrupted.tmp"
    wrong = cache_arrays()
    wrong["input_ids"] = wrong["input_ids"].copy()
    wrong["input_ids"][0] = 99
    write_npz(temporary, wrong)
    assert (
        write_cache_item_atomic(path, cache_arrays(), cache_manifest())
        == "created_after_quarantine"
    )
    assert path.exists() and not temporary.exists()
    quarantined = list(
        (tmp_path / "quarantine").glob("*.invalid_temporary.*.quarantine")
    )
    assert len(quarantined) == 1
    with np.load(quarantined[0], allow_pickle=False) as archive:
        assert archive["input_ids"][0] == 99


def test_existing_mismatched_item_is_quarantined_before_regeneration(tmp_path):
    path = tmp_path / "00007.npz"
    arrays = cache_arrays()
    assert write_cache_item_atomic(path, arrays, cache_manifest()) == "created"
    original = path.read_bytes()
    changed = cache_arrays()
    changed["x_full"] = np.full_like(changed["x_full"], 3)
    assert (
        write_cache_item_atomic(path, changed, cache_manifest())
        == "created_after_quarantine"
    )
    with np.load(path, allow_pickle=False) as archive:
        assert np.all(archive["x_full"] == 3)
    quarantined = list(
        (tmp_path / "quarantine").glob("*.invalid_final.*.quarantine")
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == original


def test_mixed_temporary_items_quarantine_invalid_and_promote_valid(tmp_path):
    path = tmp_path / "00007.npz"
    arrays = cache_arrays()
    valid_temporary = tmp_path / ".00007.npz.complete.tmp"
    invalid_temporary = tmp_path / ".00007.npz.partial.tmp"
    write_npz(valid_temporary, arrays)
    invalid_temporary.write_bytes(b"partial")

    assert (
        write_cache_item_atomic(path, arrays, cache_manifest())
        == "resumed_tmp_after_quarantine"
    )
    validate_cache_item(path, cache_manifest())
    assert not valid_temporary.exists() and not invalid_temporary.exists()
    quarantined = list(
        (tmp_path / "quarantine").glob("*.invalid_temporary.*.quarantine")
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"partial"


def test_multiple_valid_and_invalid_temporaries_preserve_all_extras(tmp_path):
    path = tmp_path / "00007.npz"
    arrays = cache_arrays()
    for suffix in ("complete-a", "complete-b"):
        write_npz(tmp_path / f".00007.npz.{suffix}.tmp", arrays)
    for suffix, payload in (("partial-a", b"a"), ("partial-b", b"b")):
        (tmp_path / f".00007.npz.{suffix}.tmp").write_bytes(payload)

    assert (
        write_cache_item_atomic(path, arrays, cache_manifest())
        == "resumed_tmp_after_quarantine"
    )
    validate_cache_item(path, cache_manifest())
    quarantined = list((tmp_path / "quarantine").glob("*.quarantine"))
    assert len(quarantined) == 3
    categories = {
        json.loads(item.with_suffix(".quarantine.json").read_text())["category"]
        for item in quarantined
    }
    assert categories == {"invalid_temporary", "redundant_valid_temporary"}


def test_atomic_no_replace_recovers_from_concurrent_invalid_final(
    tmp_path, monkeypatch
):
    path = tmp_path / "00007.npz"
    arrays = cache_arrays()
    real_link = os.link
    injected = False

    def racing_link(source, destination):
        nonlocal injected
        if Path(destination) == path and not injected:
            injected = True
            path.write_bytes(b"concurrent-invalid-final")
            raise FileExistsError
        return real_link(source, destination)

    monkeypatch.setattr(functional_cache.os, "link", racing_link)
    assert write_cache_item_atomic(path, arrays, cache_manifest()) == "created"
    validate_cache_item(path, cache_manifest())
    quarantined = list(
        (tmp_path / "quarantine").glob("*.invalid_concurrent_final.*.quarantine")
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"concurrent-invalid-final"


def test_valid_final_quarantines_leftover_invalid_temporary(tmp_path):
    path = tmp_path / "00007.npz"
    arrays = cache_arrays()
    assert write_cache_item_atomic(path, arrays, cache_manifest()) == "created"
    temporary = tmp_path / ".00007.npz.leftover.tmp"
    temporary.write_bytes(b"partial")

    assert write_cache_item_atomic(path, arrays, cache_manifest()) == "validated_existing"
    assert not temporary.exists()
    assert len(list((tmp_path / "quarantine").glob("*.quarantine"))) == 1


def test_crash_before_quarantine_rename_leaves_final_recoverable(
    tmp_path, monkeypatch
):
    path = tmp_path / "00007.npz"
    path.write_bytes(b"corrupt-final")
    real_rename = os.rename
    injected = False

    def crash_on_quarantine_rename(source, destination):
        nonlocal injected
        if Path(source) == path and not injected:
            injected = True
            raise RuntimeError("injected crash before quarantine rename")
        return real_rename(source, destination)

    monkeypatch.setattr(functional_cache.os, "rename", crash_on_quarantine_rename)
    with pytest.raises(RuntimeError, match="injected crash"):
        write_cache_item_atomic(path, cache_arrays(), cache_manifest())
    assert path.read_bytes() == b"corrupt-final"

    monkeypatch.setattr(functional_cache.os, "rename", real_rename)
    assert (
        write_cache_item_atomic(path, cache_arrays(), cache_manifest())
        == "created_after_quarantine"
    )
    validate_cache_item(path, cache_manifest())
    quarantined = list((tmp_path / "quarantine").glob("*.quarantine"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"corrupt-final"


def test_recovery_repairs_missing_quarantine_audit_record(tmp_path, monkeypatch):
    path = tmp_path / "00007.npz"
    path.write_bytes(b"corrupt-final")
    original_writer = functional_cache._write_quarantine_record

    def crash_before_record(*args, **kwargs):
        raise RuntimeError("injected crash before quarantine record publication")

    monkeypatch.setattr(
        functional_cache, "_write_quarantine_record", crash_before_record
    )
    with pytest.raises(RuntimeError, match="injected crash"):
        write_cache_item_atomic(path, cache_arrays(), cache_manifest())

    quarantined = list((tmp_path / "quarantine").glob("*.quarantine"))
    assert len(quarantined) == 1
    assert not quarantined[0].with_suffix(".quarantine.json").exists()

    monkeypatch.setattr(functional_cache, "_write_quarantine_record", original_writer)
    assert write_cache_item_atomic(path, cache_arrays(), cache_manifest()) == "created"
    record = quarantined[0].with_suffix(".quarantine.json")
    event = json.loads(record.read_text())
    assert event["sha256"] == hashlib.sha256(b"corrupt-final").hexdigest()
    assert "recovered" in event["error"]


def test_recovery_after_link_before_temporary_unlink(tmp_path):
    path = tmp_path / "00007.npz"
    temporary = tmp_path / ".00007.npz.published-before-crash.tmp"
    arrays = cache_arrays()
    write_npz(temporary, arrays)
    os.link(temporary, path)

    assert write_cache_item_atomic(path, arrays, cache_manifest()) == "validated_existing"
    validate_cache_item(path, cache_manifest())
    assert not temporary.exists()
    quarantined = list(
        (tmp_path / "quarantine").glob("*.redundant_valid_temporary.*.quarantine")
    )
    assert len(quarantined) == 1


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
    assert (
        manifest["valid_end_manifest_sha256"]
        == FROZEN_VALID_END_MANIFEST_SHA256
    )
    assert manifest["valid_end_review_protocol"]

    path = tmp_path / "manifest.json"
    assert write_manifest_atomic(path, manifest) == "created"
    assert write_manifest_atomic(path, manifest) == "validated_existing"
    changed = copy.deepcopy(manifest)
    changed["git_commit"] = "c" * 40
    assert write_manifest_atomic(path, changed) == "created_after_quarantine"
    assert json.loads(path.read_text()) == changed
    quarantined = list(
        (tmp_path / "quarantine").glob("manifest.json.invalid_final.*.quarantine")
    )
    assert len(quarantined) == 1
    assert json.loads(quarantined[0].read_text()) == manifest


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
        ("valid_end_manifest_sha256", "f" * 64),
        ("valid_end_review_protocol", "unreviewed"),
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


def test_manifest_invalid_final_is_quarantined_and_regenerated(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_bytes(b"not-json")
    manifest = cache_manifest()

    assert write_manifest_atomic(path, manifest) == "created_after_quarantine"
    assert json.loads(path.read_text()) == manifest
    quarantined = list(
        (tmp_path / "quarantine").glob("manifest.json.invalid_final.*.quarantine")
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"not-json"


def test_manifest_mixed_temporaries_promote_valid_and_quarantine_invalid(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = cache_manifest()
    assert write_manifest_atomic(path, manifest) == "created"
    valid_temporary = tmp_path / ".manifest.json.complete.tmp"
    path.rename(valid_temporary)
    invalid_temporary = tmp_path / ".manifest.json.partial.tmp"
    invalid_temporary.write_bytes(b"partial")

    assert (
        write_manifest_atomic(path, manifest)
        == "resumed_tmp_after_quarantine"
    )
    assert json.loads(path.read_text()) == manifest
    assert not valid_temporary.exists() and not invalid_temporary.exists()
    quarantined = list(
        (tmp_path / "quarantine").glob("*.invalid_temporary.*.quarantine")
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"partial"


def test_concurrent_manifest_writers_are_serialized(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = cache_manifest()
    with ThreadPoolExecutor(max_workers=4) as executor:
        statuses = list(
            executor.map(lambda _: write_manifest_atomic(path, manifest), range(4))
        )
    assert statuses.count("created") == 1
    assert statuses.count("validated_existing") == 3
    assert json.loads(path.read_text()) == manifest
