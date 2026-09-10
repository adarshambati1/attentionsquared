"""Crash-safe schema and storage for the Phase 9R eight-item cache-v3 smoke."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from src.data.functional_cache import (
    FROZEN_VALID_END_MANIFEST_SHA256,
    HUGINN_MODEL_ID,
    _fsync_directory,
    _publish_no_replace,
    _quarantine,
    _repair_quarantine_records,
    _temporary_candidates,
)
from src.evaluation.correctness import SEED_PROTOCOL, per_example_seed
from src.data.functional_extraction_v3 import NORMALIZED_H16_SEMANTICS
from src.training.functional_protocol import (
    FIXED_H0_SCHEDULE_LENGTH,
    FIXED_H0_SCHEDULE_PROTOCOL,
)


SMOKE_EXAMPLE_IDS = tuple(range(8))
CACHE_V3_ITEM_PROTOCOL = "functional-cache-v3-smoke-item-v1"
CACHE_V3_MANIFEST_PROTOCOL = "functional-cache-v3-smoke-manifest-v1"
STATE_DTYPE = np.dtype(np.float16)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

CACHE_V3_ITEM_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "answer_start",
        "valid_end",
        "h0",
        "x",
        "h16",
        "example_id",
        "dataset_split",
        "base_seed",
        "seed_index",
        "derived_seed",
        "seed_protocol",
        "schedule_length",
        "schedule_protocol",
        "full_schedule_sha256",
        "source_sha256",
        "h16_semantics",
        "cache_protocol",
        "manifest_sha256",
    }
)

CACHE_V3_MANIFEST_KEYS = frozenset(
    {
        "protocol",
        "huginn_model_id",
        "huginn_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "tokenizer_template_sha256",
        "git_commit",
        "model_compute_dtype",
        "state_dtype",
        "hidden_size",
        "extraction_location",
        "example_ids",
        "sequence_count",
        "source_continuations",
        "source_sha256_by_id",
        "valid_end_manifest_sha256",
        "split_manifest_sha256",
        "source_cache_v2_manifest_sha256",
        "source_cache_v2_freeze_sha256",
        "seed_policy",
        "base_seed",
        "seed_index",
        "schedule_length",
        "schedule_protocol",
        "generation_cap_tokens",
        "maximum_context_tokens",
        "h16_semantics",
        "full_vocabulary_logits_persisted",
        "estimated_uncompressed_item_bytes",
        "cache_output",
    }
)


def schedule_sha256(schedule: Any) -> str:
    """Hash all canonical BF16 bytes of the transient complete schedule."""
    import torch

    if not isinstance(schedule, torch.Tensor):
        raise TypeError("schedule must be a torch Tensor")
    if tuple(schedule.shape[:2]) != (1, FIXED_H0_SCHEDULE_LENGTH) or schedule.ndim != 3:
        raise ValueError("schedule must have shape [1,2048,H]")
    if schedule.dtype != torch.bfloat16:
        raise ValueError("schedule must be BF16")
    tensor = schedule.detach().cpu().contiguous()
    return hashlib.sha256(
        tensor.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    validate_manifest(manifest)
    payload = json.dumps(dict(manifest), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_manifest(
    *,
    model_revision: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    tokenizer_template: str,
    git_commit: str,
    extraction_location: str,
    source_continuations: str,
    source_sha256_by_id: Mapping[int, str],
    base_seed: int,
    seed_index: int,
    cache_output: str,
    split_manifest_sha256: str,
    source_cache_v2_manifest_sha256: str,
    source_cache_v2_freeze_sha256: str,
    estimated_uncompressed_item_bytes: int,
) -> dict[str, Any]:
    hashes = {str(key): value for key, value in source_sha256_by_id.items()}
    manifest = {
        "protocol": CACHE_V3_MANIFEST_PROTOCOL,
        "huginn_model_id": HUGINN_MODEL_ID,
        "huginn_revision": model_revision,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_template_sha256": hashlib.sha256(tokenizer_template.encode()).hexdigest(),
        "git_commit": git_commit,
        "model_compute_dtype": "bfloat16",
        "state_dtype": STATE_DTYPE.name,
        "hidden_size": 5280,
        "extraction_location": extraction_location,
        "example_ids": list(SMOKE_EXAMPLE_IDS),
        "sequence_count": len(SMOKE_EXAMPLE_IDS),
        "source_continuations": source_continuations,
        "source_sha256_by_id": hashes,
        "valid_end_manifest_sha256": FROZEN_VALID_END_MANIFEST_SHA256,
        "split_manifest_sha256": split_manifest_sha256,
        "source_cache_v2_manifest_sha256": source_cache_v2_manifest_sha256,
        "source_cache_v2_freeze_sha256": source_cache_v2_freeze_sha256,
        "seed_policy": SEED_PROTOCOL,
        "base_seed": base_seed,
        "seed_index": seed_index,
        "schedule_length": FIXED_H0_SCHEDULE_LENGTH,
        "schedule_protocol": FIXED_H0_SCHEDULE_PROTOCOL,
        "generation_cap_tokens": 1024,
        "maximum_context_tokens": 2048,
        "h16_semantics": NORMALIZED_H16_SEMANTICS,
        "full_vocabulary_logits_persisted": False,
        "estimated_uncompressed_item_bytes": estimated_uncompressed_item_bytes,
        "cache_output": cache_output,
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if set(manifest) != CACHE_V3_MANIFEST_KEYS:
        raise ValueError("cache-v3 manifest keys do not match the required schema")
    fixed = {
        "protocol": CACHE_V3_MANIFEST_PROTOCOL,
        "huginn_model_id": HUGINN_MODEL_ID,
        "model_compute_dtype": "bfloat16",
        "state_dtype": "float16",
        "hidden_size": 5280,
        "example_ids": list(SMOKE_EXAMPLE_IDS),
        "sequence_count": 8,
        "valid_end_manifest_sha256": FROZEN_VALID_END_MANIFEST_SHA256,
        "seed_policy": SEED_PROTOCOL,
        "seed_index": 0,
        "schedule_length": FIXED_H0_SCHEDULE_LENGTH,
        "schedule_protocol": FIXED_H0_SCHEDULE_PROTOCOL,
        "generation_cap_tokens": 1024,
        "maximum_context_tokens": 2048,
        "h16_semantics": NORMALIZED_H16_SEMANTICS,
        "full_vocabulary_logits_persisted": False,
        "cache_output": "/workspace/functional_cache_v3_smoke",
    }
    for key, expected in fixed.items():
        if manifest[key] != expected:
            raise ValueError(f"cache-v3 manifest {key} does not match the frozen smoke protocol")
    for key in ("huginn_revision", "tokenizer_revision", "git_commit"):
        if not isinstance(manifest[key], str) or not re.fullmatch(r"[0-9a-f]{40}", manifest[key]):
            raise ValueError(f"cache-v3 manifest {key} must be a 40-character commit")
    for key in ("tokenizer_id", "extraction_location", "source_continuations"):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ValueError(f"cache-v3 manifest {key} must be nonempty")
    if not SHA256_PATTERN.fullmatch(str(manifest["tokenizer_template_sha256"])):
        raise ValueError("invalid tokenizer template hash")
    hashes = manifest["source_sha256_by_id"]
    if not isinstance(hashes, dict) or set(hashes) != {str(i) for i in SMOKE_EXAMPLE_IDS}:
        raise ValueError("cache-v3 manifest must contain source hashes for exactly IDs 0-7")
    if not all(isinstance(value, str) and SHA256_PATTERN.fullmatch(value) for value in hashes.values()):
        raise ValueError("invalid source hash in cache-v3 manifest")
    if not isinstance(manifest["base_seed"], int) or manifest["base_seed"] < 0:
        raise ValueError("cache-v3 base seed must be nonnegative")
    for key in ("split_manifest_sha256", "source_cache_v2_manifest_sha256", "source_cache_v2_freeze_sha256"):
        if not SHA256_PATTERN.fullmatch(str(manifest[key])):
            raise ValueError(f"invalid locked source hash: {key}")
    if not isinstance(manifest["estimated_uncompressed_item_bytes"], int) or manifest["estimated_uncompressed_item_bytes"] <= 0:
        raise ValueError("invalid preflight byte estimate")


def validate_item(path: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    validate_manifest(manifest)
    item_path = Path(path)
    try:
        with np.load(item_path, allow_pickle=False) as archive:
            if set(archive.files) != CACHE_V3_ITEM_KEYS:
                raise ValueError("item keys do not match cache-v3 smoke schema")
            arrays = {key: archive[key] for key in archive.files}
    except Exception as error:
        raise ValueError(f"invalid cache-v3 item {item_path}: {error}") from error

    ids = arrays["input_ids"]
    mask = arrays["attention_mask"]
    if ids.ndim != 1 or ids.dtype != np.int32 or not 1 <= len(ids) <= FIXED_H0_SCHEDULE_LENGTH:
        raise ValueError(f"{item_path}: input_ids must be rank-1 int32 with T<=2048")
    if mask.shape != ids.shape or mask.dtype != np.bool_ or not mask.all():
        raise ValueError(f"{item_path}: attention_mask must be all-true rank-1 bool")
    states = [arrays[key] for key in ("h0", "x", "h16")]
    if any(value.ndim != 2 or value.shape[0] != len(ids) for value in states):
        raise ValueError(f"{item_path}: state slices must have shape [T,H]")
    if not (states[0].shape == states[1].shape == states[2].shape):
        raise ValueError(f"{item_path}: h0/x/h16 shapes differ")
    if states[0].shape[1] != manifest["hidden_size"]:
        raise ValueError(f"{item_path}: hidden size differs from frozen Huginn")
    if any(value.dtype != STATE_DTYPE for value in states):
        raise ValueError(f"{item_path}: state slices must be float16")
    if not all(np.isfinite(value).all() for value in states):
        raise ValueError(f"{item_path}: state slices must be finite")

    answer_start = _scalar_int(arrays, "answer_start", np.int32, item_path)
    valid_end = _scalar_int(arrays, "valid_end", np.int32, item_path)
    if not 1 <= answer_start < valid_end <= len(ids):
        raise ValueError(f"{item_path}: invalid answer bounds")
    if answer_start + int(manifest["generation_cap_tokens"]) > FIXED_H0_SCHEDULE_LENGTH:
        raise ValueError(f"{item_path}: rendered prompt plus cap exceeds schedule")
    example_id = _scalar_int(arrays, "example_id", np.int64, item_path)
    if example_id not in SMOKE_EXAMPLE_IDS:
        raise ValueError(f"{item_path}: example ID is outside exact smoke IDs 0-7")
    expected = {
        "base_seed": manifest["base_seed"],
        "seed_index": manifest["seed_index"],
        "derived_seed": per_example_seed(manifest["base_seed"], example_id, seed_index=0),
        "schedule_length": FIXED_H0_SCHEDULE_LENGTH,
    }
    for key, value in expected.items():
        if _scalar_int(arrays, key, np.int64, item_path) != value:
            raise ValueError(f"{item_path}: invalid {key}")
    strings = {
        "dataset_split": "train",
        "seed_protocol": SEED_PROTOCOL,
        "schedule_protocol": FIXED_H0_SCHEDULE_PROTOCOL,
        "source_sha256": manifest["source_sha256_by_id"][str(example_id)],
        "h16_semantics": NORMALIZED_H16_SEMANTICS,
        "cache_protocol": CACHE_V3_ITEM_PROTOCOL,
        "manifest_sha256": manifest_sha256(manifest),
    }
    for key, value in strings.items():
        if _scalar_string(arrays, key, item_path) != value:
            raise ValueError(f"{item_path}: invalid {key}")
    full_hash = _scalar_string(arrays, "full_schedule_sha256", item_path)
    if not SHA256_PATTERN.fullmatch(full_hash):
        raise ValueError(f"{item_path}: invalid full schedule hash")
    return {
        "example_id": example_id,
        "sequence_length": len(ids),
        "hidden_size": states[0].shape[1],
        "answer_start": answer_start,
        "valid_end": valid_end,
        "full_schedule_sha256": full_hash,
    }


def write_item_atomic(path: str | Path, arrays: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    """Write unique temporary, fsync, validate, and publish without replacement."""
    validate_manifest(manifest)
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    with _writer_lock(final.parent):
        _repair_quarantine_records(final.parent)
        validate_expected = lambda candidate: _validate_expected(candidate, arrays, manifest)
        quarantined = False
        if final.exists():
            try:
                validate_expected(final)
            except Exception as error:
                _quarantine(final, "invalid_final", error)
                quarantined = True
            else:
                _quarantine_temporaries(final, validate_expected)
                _require_or_make_read_only(final)
                return "validated_existing"
        valid_temporaries = []
        for candidate in _temporary_candidates(final):
            try:
                validate_expected(candidate)
            except Exception as error:
                _quarantine(candidate, "invalid_temporary", error)
                quarantined = True
            else:
                valid_temporaries.append(candidate)
        if valid_temporaries:
            temporary = valid_temporaries.pop(0)
            for redundant in valid_temporaries:
                _quarantine(redundant, "redundant_valid_temporary", "another valid temporary was selected")
                quarantined = True
            status = "resumed_tmp_after_quarantine" if quarantined else "resumed_tmp"
        else:
            descriptor, name = tempfile.mkstemp(prefix=f".{final.name}.", suffix=".tmp", dir=final.parent)
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            validate_expected(temporary)
            status = "created_after_quarantine" if quarantined else "created"
        published = _publish_no_replace(
            temporary,
            final,
            validate_expected,
            invalid_category="invalid_concurrent_final",
            redundant_reason="a matching final cache-v3 item appeared before promotion",
        )
        _require_or_make_read_only(final)
        return status if published else "validated_existing"


def write_manifest_atomic(path: str | Path, manifest: Mapping[str, Any]) -> str:
    validate_manifest(manifest)
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n").encode()
    with _writer_lock(final.parent):
        _repair_quarantine_records(final.parent)
        def validate_expected(candidate: Path) -> None:
            parsed = json.loads(candidate.read_text(encoding="utf-8"))
            validate_manifest(parsed)
            if candidate.read_bytes() != payload:
                raise ValueError("manifest differs from canonical requested payload")
        quarantined = False
        if final.exists():
            try:
                validate_expected(final)
            except Exception as error:
                _quarantine(final, "invalid_final", error)
                quarantined = True
            else:
                _quarantine_temporaries(final, validate_expected)
                _require_or_make_read_only(final)
                return "validated_existing"
        valid_temporaries = []
        for candidate in _temporary_candidates(final):
            try:
                validate_expected(candidate)
            except Exception as error:
                _quarantine(candidate, "invalid_temporary", error)
                quarantined = True
            else:
                valid_temporaries.append(candidate)
        if valid_temporaries:
            temporary = valid_temporaries.pop(0)
            for redundant in valid_temporaries:
                _quarantine(redundant, "redundant_valid_temporary", "another valid temporary was selected")
                quarantined = True
            status = "resumed_tmp_after_quarantine" if quarantined else "resumed_tmp"
        else:
            descriptor, name = tempfile.mkstemp(prefix=f".{final.name}.", suffix=".tmp", dir=final.parent)
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            validate_expected(temporary)
            status = "created_after_quarantine" if quarantined else "created"
        published = _publish_no_replace(
            temporary, final, validate_expected,
            invalid_category="invalid_concurrent_final",
            redundant_reason="a matching cache-v3 manifest appeared before promotion",
        )
        _require_or_make_read_only(final)
        return status if published else "validated_existing"


def freeze_cache_directory(path: str | Path) -> None:
    directory = Path(path)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("cache-v3 root must be a real directory")
    for payload in [directory / "manifest.json", *(directory / f"{i:05d}.npz" for i in SMOKE_EXAMPLE_IDS)]:
        if payload.is_symlink() or not payload.is_file() or stat.S_IMODE(payload.stat().st_mode) != 0o444:
            raise ValueError(f"cache-v3 payload is not immutable: {payload}")
    lock = directory / ".functional-cache-v3.lock"
    if lock.exists():
        if lock.is_symlink() or not lock.is_file():
            raise ValueError("cache-v3 writer lock is not a regular file")
        os.chmod(lock, 0o444)
    _fsync_directory(directory)
    os.chmod(directory, 0o555)
    _fsync_directory(directory.parent)


def _validate_expected(path: Path, arrays: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    validate_item(path, manifest)
    if set(arrays) != CACHE_V3_ITEM_KEYS:
        raise ValueError("requested arrays do not match cache-v3 schema")
    with np.load(path, allow_pickle=False) as archive:
        for key in CACHE_V3_ITEM_KEYS:
            expected = np.asarray(arrays[key])
            if archive[key].dtype != expected.dtype or not np.array_equal(archive[key], expected):
                raise ValueError(f"{path}: stored {key} differs from requested item")


def _quarantine_temporaries(final: Path, validator: Callable[[Path], None]) -> None:
    for candidate in _temporary_candidates(final):
        try:
            validator(candidate)
        except Exception as error:
            _quarantine(candidate, "invalid_temporary", error)
        else:
            _quarantine(candidate, "redundant_valid_temporary", "a valid final already exists")


def _require_or_make_read_only(path: Path) -> None:
    os.chmod(path, 0o444)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise RuntimeError(f"failed to make payload read-only: {path}")


@contextmanager
def _writer_lock(directory: Path):
    lock_path = directory / ".functional-cache-v3.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _scalar_int(arrays: Mapping[str, Any], key: str, dtype: Any, path: Path) -> int:
    value = arrays[key]
    if value.shape != () or value.dtype != np.dtype(dtype):
        raise ValueError(f"{path}: {key} must be scalar {np.dtype(dtype).name}")
    return int(value)


def _scalar_string(arrays: Mapping[str, Any], key: str, path: Path) -> str:
    value = arrays[key]
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"{path}: {key} must be a scalar string")
    return str(value)
