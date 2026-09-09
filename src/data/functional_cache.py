"""Crash-safe storage for corrected functional-v2 cache items and manifests."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.evaluation.correctness import SEED_PROTOCOL, per_example_seed
from src.evaluation.splits import (
    GSM8K_DATASET_CONFIG,
    GSM8K_DATASET_ID,
    GSM8K_DATASET_REVISION,
    SPLIT_GENERATION_POLICY,
    SPLIT_PROTOCOL,
    validate_split_manifest,
)


CACHE_ITEM_PROTOCOL = "functional-cache-item-v2"
CACHE_MANIFEST_PROTOCOL = "functional-cache-manifest-v2"
HUGINN_MODEL_ID = "tomg-group-umd/huginn-0125"
STATE_DTYPE = np.dtype(np.float16)
FROZEN_SPLIT_MANIFEST_SHA256 = (
    "f8202a1327e65168ae33b6417813969436ed2d6743e9923e7d4bb58d3c50a139"
)

CACHE_ITEM_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "answer_start",
        "valid_end",
        "h0_full",
        "x_full",
        "h16_teacher",
        "example_id",
        "dataset_split",
        "base_seed",
        "derived_seed",
        "seed_protocol",
        "cache_protocol",
        "manifest_sha256",
    }
)

CACHE_MANIFEST_KEYS = frozenset(
    {
        "protocol",
        "huginn_model_id",
        "huginn_revision",
        "dataset_id",
        "dataset_config",
        "dataset_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "tokenizer_template",
        "git_commit",
        "state_dtype",
        "extraction_location",
        "sequence_count",
        "train_ids",
        "validation_ids",
        "split_protocol",
        "split_generation_policy",
        "split_manifest_sha256",
        "seed_policy",
        "base_seed",
        "cap_degeneration_policy",
    }
)


def validate_cache_item(
    path: str | Path, manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Reopen and validate one item; optionally bind it to its manifest."""
    item_path = Path(path)
    try:
        with np.load(item_path, allow_pickle=False) as archive:
            if set(archive.files) != CACHE_ITEM_KEYS:
                raise ValueError(
                    f"keys {sorted(archive.files)}, expected {sorted(CACHE_ITEM_KEYS)}"
                )
            arrays = {key: archive[key] for key in archive.files}
    except Exception as error:
        raise ValueError(f"invalid cache item {item_path}: {error}") from error

    input_ids = arrays["input_ids"]
    attention_mask = arrays["attention_mask"]
    states = (arrays["h0_full"], arrays["x_full"], arrays["h16_teacher"])
    if input_ids.ndim != 1 or input_ids.dtype != np.int32:
        raise ValueError(f"{item_path}: input_ids must be rank-1 int32")
    if attention_mask.shape != input_ids.shape or attention_mask.dtype != np.bool_:
        raise ValueError(f"{item_path}: attention_mask must be rank-1 bool")
    if not attention_mask.all():
        raise ValueError(f"{item_path}: unpadded attention_mask must be all true")

    token_count = input_ids.shape[0]
    if any(state.ndim != 2 or state.shape[0] != token_count for state in states):
        raise ValueError(f"{item_path}: state tensors must have shape [T,H]")
    if not (states[0].shape == states[1].shape == states[2].shape):
        raise ValueError(f"{item_path}: h0, x, and h16 shapes must match")
    if any(state.dtype != STATE_DTYPE for state in states):
        raise ValueError(f"{item_path}: state tensors must be float16")
    if not all(np.isfinite(state).all() for state in states):
        raise ValueError(f"{item_path}: state tensors must be finite")

    answer_start = _scalar_int(arrays, "answer_start", np.int32, item_path)
    valid_end = _scalar_int(arrays, "valid_end", np.int32, item_path)
    if not 1 <= answer_start < valid_end <= token_count:
        raise ValueError(f"{item_path}: invalid answer_start/valid_end bounds")
    metadata = {
        "sequence_length": token_count,
        "hidden_size": states[0].shape[1],
        "answer_start": answer_start,
        "valid_end": valid_end,
        "example_id": _scalar_int(arrays, "example_id", np.int64, item_path),
        "dataset_split": _scalar_string(arrays, "dataset_split", item_path),
        "base_seed": _scalar_int(arrays, "base_seed", np.int64, item_path),
        "derived_seed": _scalar_int(arrays, "derived_seed", np.int64, item_path),
        "seed_protocol": _scalar_string(arrays, "seed_protocol", item_path),
        "manifest_sha256": _scalar_string(arrays, "manifest_sha256", item_path),
    }
    if _scalar_string(arrays, "cache_protocol", item_path) != CACHE_ITEM_PROTOCOL:
        raise ValueError(f"{item_path}: unexpected cache item protocol")
    if metadata["dataset_split"] not in {"train", "validation"}:
        raise ValueError(f"{item_path}: invalid training-side split")
    if manifest is not None:
        _validate_item_against_manifest(metadata, manifest, item_path)
    return metadata


def write_cache_item_atomic(
    path: str | Path,
    arrays: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    """Write, close, reopen-validate, then atomically rename an NPZ item.

    A directory lock serializes compliant writers, so interrupted temporary
    files can only belong to a crashed prior lock holder. Existing, temporary,
    and concurrent final files are accepted only when their exact arrays and
    manifest-bound provenance match the requested item.
    """
    validate_cache_manifest(manifest)
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with _writer_lock(final_path.parent):
        return _write_cache_item_locked(final_path, arrays, manifest)


def _write_cache_item_locked(
    final_path: Path, arrays: Mapping[str, Any], manifest: Mapping[str, Any]
) -> str:
    if final_path.exists():
        _validate_expected_item(final_path, arrays, manifest)
        return "validated_existing"

    stale = _temporary_candidates(final_path)
    if stale:
        if len(stale) != 1:
            raise ValueError(f"multiple interrupted temporary files for {final_path}")
        _validate_expected_item(stale[0], arrays, manifest)
        return _publish_by_rename(
            stale[0], final_path, arrays, manifest, "resumed_tmp"
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.", suffix=".tmp", dir=final_path.parent
    )
    temporary_path = Path(temporary_name)
    with os.fdopen(descriptor, "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    _validate_expected_item(temporary_path, arrays, manifest)
    return _publish_by_rename(
        temporary_path, final_path, arrays, manifest, "created"
    )


def build_cache_manifest(
    *,
    model_revision: str,
    tokenizer_template: str,
    git_commit: str,
    extraction_location: str,
    split_manifest: dict[str, Any],
    seed_policy: str,
    base_seed: int,
    cap_degeneration_policy: str,
    tokenizer_id: str = HUGINN_MODEL_ID,
    tokenizer_revision: str | None = None,
) -> dict[str, Any]:
    """Build mandatory cache provenance from the frozen Phase 3 manifest."""
    validate_split_manifest(split_manifest)
    training = split_manifest["training_side"]
    manifest = {
        "protocol": CACHE_MANIFEST_PROTOCOL,
        "huginn_model_id": HUGINN_MODEL_ID,
        "huginn_revision": model_revision,
        "dataset_id": GSM8K_DATASET_ID,
        "dataset_config": GSM8K_DATASET_CONFIG,
        "dataset_revision": GSM8K_DATASET_REVISION,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision or model_revision,
        "tokenizer_template": tokenizer_template,
        "git_commit": git_commit,
        "state_dtype": STATE_DTYPE.name,
        "extraction_location": extraction_location,
        "sequence_count": 2500,
        "train_ids": training["gradient_update_ids"],
        "validation_ids": training["checkpoint_selection_ids"],
        "split_protocol": SPLIT_PROTOCOL,
        "split_generation_policy": SPLIT_GENERATION_POLICY,
        "split_manifest_sha256": _mapping_sha256(split_manifest),
        "seed_policy": seed_policy,
        "base_seed": base_seed,
        "cap_degeneration_policy": cap_degeneration_policy,
    }
    validate_cache_manifest(manifest)
    return manifest


def validate_cache_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate all required cache-level provenance and frozen split IDs."""
    if set(manifest) != CACHE_MANIFEST_KEYS:
        raise ValueError("cache manifest keys do not match the required schema")
    fixed = {
        "protocol": CACHE_MANIFEST_PROTOCOL,
        "huginn_model_id": HUGINN_MODEL_ID,
        "dataset_id": GSM8K_DATASET_ID,
        "dataset_config": GSM8K_DATASET_CONFIG,
        "dataset_revision": GSM8K_DATASET_REVISION,
        "state_dtype": STATE_DTYPE.name,
        "sequence_count": 2500,
        "train_ids": list(range(2250)),
        "validation_ids": list(range(2250, 2500)),
        "split_protocol": SPLIT_PROTOCOL,
        "split_generation_policy": SPLIT_GENERATION_POLICY,
    }
    for key, expected in fixed.items():
        if manifest[key] != expected:
            raise ValueError(f"cache manifest {key} does not match frozen protocol")
    if manifest["seed_policy"] != SEED_PROTOCOL:
        raise ValueError("cache seed policy is not the authoritative protocol")
    for key in (
        "huginn_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "tokenizer_template",
        "extraction_location",
        "seed_policy",
        "cap_degeneration_policy",
    ):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ValueError(f"cache manifest {key} must be a nonempty string")
    for key in ("huginn_revision", "tokenizer_revision", "git_commit"):
        if not re.fullmatch(r"[0-9a-f]{40}", manifest[key]):
            raise ValueError(f"cache manifest {key} must be a 40-character commit")
    if not isinstance(manifest["base_seed"], int) or manifest["base_seed"] < 0:
        raise ValueError("cache manifest base_seed must be a nonnegative integer")
    if manifest["split_manifest_sha256"] != FROZEN_SPLIT_MANIFEST_SHA256:
        raise ValueError("cache split manifest checksum is not the frozen manifest")


def write_manifest_atomic(path: str | Path, manifest: Mapping[str, Any]) -> str:
    """Write, reopen, validate, and atomically rename the cache manifest."""
    validate_cache_manifest(manifest)
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n"
    with _writer_lock(final_path.parent):
        return _write_manifest_locked(final_path, manifest, payload)


def _write_manifest_locked(
    final_path: Path, manifest: Mapping[str, Any], payload: str
) -> str:
    if final_path.exists():
        if final_path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"existing manifest does not match: {final_path}")
        return "validated_existing"

    stale = _temporary_candidates(final_path)
    if stale:
        if len(stale) != 1 or stale[0].read_text(encoding="utf-8") != payload:
            raise ValueError(f"interrupted temporary manifest does not match: {final_path}")
        return _publish_text_by_rename(stale[0], final_path, payload, "resumed_tmp")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.", suffix=".tmp", dir=final_path.parent
    )
    temporary_path = Path(temporary_name)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if json.loads(temporary_path.read_text(encoding="utf-8")) != dict(manifest):
        raise ValueError("temporary manifest failed reopen validation")
    return _publish_text_by_rename(temporary_path, final_path, payload, "created")


def _validate_expected_item(
    path: Path, arrays: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    validate_cache_item(path, manifest)
    if set(arrays) != CACHE_ITEM_KEYS:
        raise ValueError("requested cache arrays do not match the required schema")
    with np.load(path, allow_pickle=False) as archive:
        for key in CACHE_ITEM_KEYS:
            expected = np.asarray(arrays[key])
            if archive[key].dtype != expected.dtype or not np.array_equal(
                archive[key], expected
            ):
                raise ValueError(f"{path}: stored {key} does not match requested item")


def _validate_item_against_manifest(
    metadata: Mapping[str, Any], manifest: Mapping[str, Any], path: Path
) -> None:
    validate_cache_manifest(manifest)
    example_id = metadata["example_id"]
    expected_split = (
        "train"
        if example_id in set(manifest["train_ids"])
        else "validation"
        if example_id in set(manifest["validation_ids"])
        else None
    )
    if metadata["dataset_split"] != expected_split:
        raise ValueError(f"{path}: example ID does not match its frozen split role")
    if metadata["base_seed"] != manifest["base_seed"]:
        raise ValueError(f"{path}: base seed does not match manifest")
    if metadata["seed_protocol"] != manifest["seed_policy"]:
        raise ValueError(f"{path}: seed protocol does not match manifest")
    if metadata["manifest_sha256"] != cache_manifest_sha256(manifest):
        raise ValueError(f"{path}: item is not bound to the exact cache manifest")
    if metadata["derived_seed"] != per_example_seed(
        metadata["base_seed"], example_id
    ):
        raise ValueError(f"{path}: derived seed is invalid")


def _publish_by_rename(
    temporary: Path,
    final: Path,
    arrays: Mapping[str, Any],
    manifest: Mapping[str, Any],
    success_status: str,
) -> str:
    if final.exists():
        _validate_expected_item(final, arrays, manifest)
        return "validated_existing"
    os.rename(temporary, final)
    _fsync_directory(final.parent)
    return success_status


def _publish_text_by_rename(
    temporary: Path, final: Path, payload: str, success_status: str
) -> str:
    if final.exists():
        if final.read_text(encoding="utf-8") != payload:
            raise ValueError(f"concurrent manifest does not match: {final}")
        return "validated_existing"
    os.rename(temporary, final)
    _fsync_directory(final.parent)
    return success_status


@contextmanager
def _writer_lock(directory: Path):
    lock_path = directory / ".functional-cache.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _temporary_candidates(final_path: Path) -> list[Path]:
    candidates = list(final_path.parent.glob(f".{final_path.name}.*.tmp"))
    legacy = Path(f"{final_path}.tmp")
    if legacy.exists():
        candidates.append(legacy)
    return sorted(set(candidates))


def cache_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Return the canonical digest embedded in every cache item."""
    validate_cache_manifest(manifest)
    return _mapping_sha256(manifest)


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _scalar_int(
    arrays: Mapping[str, Any], key: str, dtype: np.dtype[Any], path: Path
) -> int:
    value = arrays[key]
    if value.shape != () or value.dtype != np.dtype(dtype):
        raise ValueError(f"{path}: {key} must be a scalar {np.dtype(dtype).name}")
    return int(value)


def _scalar_string(arrays: Mapping[str, Any], key: str, path: Path) -> str:
    value = arrays[key]
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"{path}: {key} must be a scalar string")
    return str(value)
