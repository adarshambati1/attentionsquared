"""Build and validate the immutable Experiment 004 valid-end sidecar."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.splits import (
    GSM8K_DATASET_CONFIG,
    GSM8K_DATASET_ID,
    GSM8K_DATASET_REVISION,
)

VALID_END_SCHEMA = "functional-valid-end-item-v1"
REVIEW_PROTOCOL = "dual-independent-conservative-valid-end-v1"
SOURCE_KEYS = {
    "input_ids",
    "answer_start",
    "generated_tokens",
    "truncated",
    "pathological",
    "valid_end",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(archive: Any, key: str, expected_dtype: np.dtype[Any]) -> Any:
    value = archive[key]
    if value.shape != () or value.dtype != expected_dtype:
        raise ValueError(f"{key} must be a scalar with dtype {expected_dtype}")
    return value.item()


def _phase3_role(example_id: int) -> str:
    if 0 <= example_id < 2250:
        return "gradient_update"
    if 2250 <= example_id < 2500:
        return "checkpoint_selection_validation"
    raise ValueError(f"example ID outside frozen training-side range: {example_id}")


def _load_reviews(path: str | Path) -> dict[int, dict[str, Any]]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("unexpected capped-review schema")
    if document.get("review_protocol") != REVIEW_PROTOCOL:
        raise ValueError("unexpected capped-review protocol")
    items = document.get("items")
    if not isinstance(items, list):
        raise ValueError("capped-review items must be a list")
    reviews: dict[int, dict[str, Any]] = {}
    for item in items:
        example_id = item.get("example_id")
        if type(example_id) is not int or example_id in reviews:
            raise ValueError("capped-review IDs must be unique integers")
        reviews[example_id] = item
    return reviews


def build_valid_end_records(
    source_root: str | Path,
    capped_review_path: str | Path,
    *,
    expected_count: int = 2500,
    expected_capped_count: int = 27,
) -> list[dict[str, Any]]:
    """Validate every preserved source and return deterministic sidecar rows."""
    root = Path(source_root)
    reviews = _load_reviews(capped_review_path)
    paths = list(root.glob("*/*.npz"))
    if len(paths) != expected_count:
        raise ValueError(f"expected {expected_count} source files, found {len(paths)}")

    records: dict[int, dict[str, Any]] = {}
    capped_ids: set[int] = set()
    for path in paths:
        try:
            example_id = int(path.stem)
        except ValueError as error:
            raise ValueError(f"non-integer source filename: {path}") from error
        if example_id in records:
            raise ValueError(f"duplicate source example ID: {example_id}")
        role = _phase3_role(example_id)
        source_sha256 = _sha256(path)

        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != SOURCE_KEYS:
                raise ValueError(f"unexpected source keys in {path}")
            input_ids = archive["input_ids"]
            if input_ids.ndim != 1 or input_ids.dtype != np.dtype("int32"):
                raise ValueError(f"input_ids must be rank-1 int32 in {path}")
            answer_start = int(_scalar(archive, "answer_start", np.dtype("int32")))
            generated_tokens = int(
                _scalar(archive, "generated_tokens", np.dtype("int32"))
            )
            truncated = bool(_scalar(archive, "truncated", np.dtype("bool")))
            pathological = bool(_scalar(archive, "pathological", np.dtype("bool")))
            original_valid_end = int(
                _scalar(archive, "valid_end", np.dtype("int32"))
            )

        sequence_length = int(input_ids.shape[0])
        if not (0 < answer_start <= original_valid_end <= sequence_length):
            raise ValueError(f"invalid answer/valid-end bounds in {path}")
        if generated_tokens != sequence_length - answer_start:
            raise ValueError(f"generated-token count mismatch in {path}")

        decision = "uncapped_original_valid_end"
        reviewed_valid_end = original_valid_end
        reviewed_degeneration = pathological
        review_protocol: str | None = None
        if truncated:
            capped_ids.add(example_id)
            review = reviews.get(example_id)
            if review is None:
                raise ValueError(f"capped example lacks review: {example_id}")
            expected_values = {
                "source_relative_path": str(path.relative_to(root)),
                "source_sha256": source_sha256,
                "answer_start": answer_start,
                "sequence_length": sequence_length,
                "original_valid_end": original_valid_end,
                "source_truncated": True,
                "source_pathological": pathological,
                "phase3_role": (
                    "train"
                    if role == "gradient_update"
                    else "validation"
                ),
            }
            for key, expected in expected_values.items():
                if review.get(key) != expected:
                    raise ValueError(f"review/source mismatch for {example_id}: {key}")
            reviewed_valid_end = review.get("final_reviewed_valid_end")
            onset = review.get("final_degeneration_onset_relative")
            reviewer_1_onset = review.get("reviewer_1_onset_relative")
            reviewer_2_onset = review.get("reviewer_2_onset_relative")
            if not all(
                type(value) is int
                for value in (
                    reviewed_valid_end,
                    onset,
                    reviewer_1_onset,
                    reviewer_2_onset,
                )
            ):
                raise ValueError(f"invalid reviewed boundary for {example_id}")
            if onset != max(reviewer_1_onset, reviewer_2_onset):
                raise ValueError(f"final onset is not the conservative later review for {example_id}")
            if review.get("adjudication_policy") != (
                "choose_later_proposed_onset_to_preserve_valid_prefix"
            ):
                raise ValueError(f"unexpected adjudication policy for {example_id}")
            if reviewed_valid_end != answer_start + onset:
                raise ValueError(f"reviewed onset mismatch for {example_id}")
            if not answer_start <= reviewed_valid_end <= sequence_length:
                raise ValueError(f"reviewed boundary out of range for {example_id}")
            if review.get("final_decision") != "truncate_clear_repetition":
                raise ValueError(f"unexpected capped decision for {example_id}")
            decision = "truncate_clear_repetition"
            reviewed_degeneration = True
            review_protocol = REVIEW_PROTOCOL

        records[example_id] = {
            "schema": VALID_END_SCHEMA,
            "example_id": example_id,
            "dataset_id": GSM8K_DATASET_ID,
            "dataset_config": GSM8K_DATASET_CONFIG,
            "dataset_revision": GSM8K_DATASET_REVISION,
            "dataset_split": "train",
            "phase3_role": role,
            "historical_source_partition": path.parent.name,
            "source_relative_path": str(path.relative_to(root)),
            "source_sha256": source_sha256,
            "source_sequence_length": sequence_length,
            "answer_start": answer_start,
            "generated_tokens": generated_tokens,
            "generation_cap_tokens": 1024,
            "source_truncated_at_cap": truncated,
            "source_pathological": pathological,
            "original_valid_end": original_valid_end,
            "reviewed_valid_end": reviewed_valid_end,
            "reviewed_degeneration": reviewed_degeneration,
            "decision": decision,
            "review_protocol": review_protocol,
        }

    expected_ids = set(range(expected_count))
    if set(records) != expected_ids:
        raise ValueError("source IDs do not exactly match the expected contiguous range")
    if len(capped_ids) != expected_capped_count:
        raise ValueError(
            f"expected {expected_capped_count} capped sources, found {len(capped_ids)}"
        )
    if set(reviews) != capped_ids:
        raise ValueError("review IDs do not exactly match capped source IDs")
    return [records[index] for index in range(expected_count)]


def serialize_records(records: list[dict[str, Any]]) -> bytes:
    """Serialize one canonical JSON object per source file."""
    return ("".join(json.dumps(row, sort_keys=True) + "\n" for row in records)).encode()


def write_immutable_manifest(path: str | Path, records: list[dict[str, Any]]) -> str:
    """Create a manifest once, or verify an existing byte-identical manifest."""
    destination = Path(path)
    payload = serialize_records(records)
    digest = hashlib.sha256(payload).hexdigest()
    if destination.exists():
        if destination.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite non-identical manifest: {destination}")
        return digest

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != payload:
                raise FileExistsError(
                    f"refusing to overwrite non-identical manifest: {destination}"
                ) from None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def load_and_validate_manifest(
    path: str | Path, *, expected_count: int = 2500
) -> list[dict[str, Any]]:
    """Load a canonical sidecar and reject malformed ordering or bounds."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} manifest rows, found {len(rows)}")
    for expected_id, row in enumerate(rows):
        if row.get("schema") != VALID_END_SCHEMA:
            raise ValueError("unexpected valid-end item schema")
        if row.get("example_id") != expected_id:
            raise ValueError("manifest rows are not in exact example-ID order")
        start = row.get("answer_start")
        end = row.get("reviewed_valid_end")
        length = row.get("source_sequence_length")
        if not all(type(value) is int for value in (start, end, length)):
            raise ValueError("manifest bounds must be integers")
        if not 0 < start <= end <= length:
            raise ValueError("manifest contains invalid reviewed bounds")
        digest = row.get("source_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("manifest contains invalid source hash")
    return rows
