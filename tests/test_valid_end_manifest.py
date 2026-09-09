import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.valid_end_manifest import (
    REVIEW_PROTOCOL,
    build_valid_end_records,
    load_and_validate_manifest,
    write_immutable_manifest,
)


def _write_source(
    root: Path,
    example_id: int,
    *,
    truncated: bool = False,
    pathological: bool = False,
) -> Path:
    partition = "train" if example_id < 2 else "test"
    path = root / partition / f"{example_id:06d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    input_ids = np.arange(7, dtype=np.int32)
    np.savez(
        path,
        input_ids=input_ids,
        answer_start=np.array(3, dtype=np.int32),
        generated_tokens=np.array(4, dtype=np.int32),
        truncated=np.array(truncated, dtype=np.bool_),
        pathological=np.array(pathological, dtype=np.bool_),
        valid_end=np.array(7, dtype=np.int32),
    )
    return path


def _review(path: Path, source: Path, *, final_end: int = 5) -> None:
    item = {
        "example_id": 1,
        "source_relative_path": "train/000001.npz",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "phase3_role": "train",
        "answer_start": 3,
        "sequence_length": 7,
        "original_valid_end": 7,
        "source_truncated": True,
        "source_pathological": False,
        "reviewer_1_onset_relative": final_end - 3,
        "reviewer_2_onset_relative": final_end - 3,
        "final_reviewed_valid_end": final_end,
        "final_degeneration_onset_relative": final_end - 3,
        "final_decision": "truncate_clear_repetition",
        "adjudication_policy": "choose_later_proposed_onset_to_preserve_valid_prefix",
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_protocol": REVIEW_PROTOCOL,
                "items": [item],
            }
        )
    )


def test_builds_exact_ordered_manifest_and_does_not_mutate_sources(tmp_path):
    source_root = tmp_path / "sources"
    sources = [_write_source(source_root, index, truncated=index == 1) for index in range(3)]
    before = [path.read_bytes() for path in sources]
    review = tmp_path / "review.json"
    _review(review, sources[1])

    records = build_valid_end_records(
        source_root, review, expected_count=3, expected_capped_count=1
    )

    assert [row["example_id"] for row in records] == [0, 1, 2]
    assert records[0]["decision"] == "uncapped_original_valid_end"
    assert records[1]["reviewed_valid_end"] == 5
    assert records[1]["decision"] == "truncate_clear_repetition"
    assert records[2]["phase3_role"] == "gradient_update"
    assert [path.read_bytes() for path in sources] == before


def test_review_is_bound_to_exact_source_hash_and_metadata(tmp_path):
    source_root = tmp_path / "sources"
    sources = [_write_source(source_root, index, truncated=index == 1) for index in range(3)]
    review = tmp_path / "review.json"
    _review(review, sources[1])
    document = json.loads(review.read_text())
    document["items"][0]["source_sha256"] = "0" * 64
    review.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="review/source mismatch"):
        build_valid_end_records(
            source_root, review, expected_count=3, expected_capped_count=1
        )


def test_final_onset_must_use_later_independent_review(tmp_path):
    source_root = tmp_path / "sources"
    sources = [_write_source(source_root, index, truncated=index == 1) for index in range(3)]
    review = tmp_path / "review.json"
    _review(review, sources[1])
    document = json.loads(review.read_text())
    document["items"][0]["reviewer_2_onset_relative"] = 3
    review.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="conservative later review"):
        build_valid_end_records(
            source_root, review, expected_count=3, expected_capped_count=1
        )


def test_reviewed_boundary_must_be_causally_valid(tmp_path):
    source_root = tmp_path / "sources"
    sources = [_write_source(source_root, index, truncated=index == 1) for index in range(3)]
    review = tmp_path / "review.json"
    _review(review, sources[1], final_end=2)

    with pytest.raises(ValueError, match="out of range"):
        build_valid_end_records(
            source_root, review, expected_count=3, expected_capped_count=1
        )


def test_immutable_writer_reuses_identical_and_rejects_changes(tmp_path):
    destination = tmp_path / "manifest.jsonl"
    rows = [
        {
            "schema": "functional-valid-end-item-v1",
            "example_id": 0,
            "answer_start": 2,
            "reviewed_valid_end": 3,
            "source_sequence_length": 3,
            "source_sha256": "a" * 64,
        }
    ]
    first_digest = write_immutable_manifest(destination, rows)
    first_bytes = destination.read_bytes()
    assert write_immutable_manifest(destination, rows) == first_digest
    assert destination.read_bytes() == first_bytes

    changed = [dict(rows[0], reviewed_valid_end=2)]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_immutable_manifest(destination, changed)


def test_manifest_loader_rejects_wrong_order(tmp_path):
    destination = tmp_path / "manifest.jsonl"
    rows = [
        {
            "schema": "functional-valid-end-item-v1",
            "example_id": 1,
            "answer_start": 2,
            "reviewed_valid_end": 3,
            "source_sequence_length": 3,
            "source_sha256": "a" * 64,
        }
    ]
    write_immutable_manifest(destination, rows)
    with pytest.raises(ValueError, match="exact example-ID order"):
        load_and_validate_manifest(destination, expected_count=1)
