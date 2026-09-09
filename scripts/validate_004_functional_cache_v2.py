#!/usr/bin/env python3
"""Validate, replay, checksum, and freeze the durable functional cache v2."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from scripts.build_004_functional_cache_v2 import _git_commit
from src.data.functional_cache import (
    FROZEN_VALID_END_MANIFEST_SHA256,
    validate_cache_item,
    validate_cache_manifest,
)
from src.data.functional_extraction import capture_huginn_functional_states
from src.data.functional_validation import (
    CHECKSUM_PROTOCOL,
    arrays_byte_identical,
    checksum_inventory_bytes,
    file_sha256,
    require_identical_regular_files,
    validate_runpod_volume_attestation,
    validate_mount_output,
    verify_checksum_inventory,
    write_bytes_immutable,
)
from src.data.valid_end_manifest import load_and_validate_manifest
from src.evaluation.correctness import seed_for_example

CACHE_ROOT = Path("/workspace/functional_cache_v2")
SOURCE_ROOT = ROOT / "results/004_functional/teacher_sequences"
VALID_END_PATH = ROOT / "results/004_functional/valid_end_manifest.jsonl"
COMMITTED_MANIFEST_PATH = ROOT / "results/004_functional/cache_v2_manifest.json"
ATTESTATION_PATH = ROOT / "results/004_functional/cache_v2_runtime_attestation.json"
BUILDER_COMMIT = "01439eee033ec6b86c5a53806b16fd1b8c0100a1"
MODEL_ID = "tomg-group-umd/huginn-0125"
MODEL_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
CACHE_MANIFEST_SHA256 = "32db951f6cfbf76220b1c7126e23149f087eba58934ab691b588aeb3a43535bf"
VOLUME_ID = "jggambl3qv"
DATA_CENTER_ID = "US-GA-2"
MOUNT_ROOT = Path("/workspace")
POD_ID = "jwc04iw92bdebm"
REPLAY_IDS = (0, 22, 817, 1671, 2249, 2250, 2389, 2499)


def _expected_item_path(example_id: int) -> Path:
    split = "train" if example_id < 2250 else "validation"
    return CACHE_ROOT / split / f"{example_id:05d}.npz"


def _validate_file_set() -> list[Path]:
    for directory in (CACHE_ROOT, CACHE_ROOT / "train", CACHE_ROOT / "validation"):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"cache directory must be a real directory: {directory}")
    symlinks = [path for path in CACHE_ROOT.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"cache contains symlinks: {symlinks}")
    items = [_expected_item_path(example_id) for example_id in range(2500)]
    expected_payload = {CACHE_ROOT / "manifest.json", *items}
    allowed_auxiliary = {
        CACHE_ROOT / ".functional-cache.lock",
        CACHE_ROOT / "train/.functional-cache.lock",
        CACHE_ROOT / "validation/.functional-cache.lock",
        CACHE_ROOT / "SHA256SUMS.tsv",
        CACHE_ROOT / "validation_report.json",
        CACHE_ROOT / "FROZEN.json",
    }
    actual = {path for path in CACHE_ROOT.rglob("*") if path.is_file()}
    unexpected = actual - expected_payload - allowed_auxiliary
    missing = expected_payload - actual
    if unexpected or missing:
        raise ValueError(
            f"cache file set mismatch: missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    temporary = [path for path in CACHE_ROOT.rglob("*.tmp") if path.is_file()]
    quarantine = CACHE_ROOT / "quarantine"
    quarantined = list(quarantine.rglob("*")) if quarantine.exists() else []
    if temporary or any(path.is_file() for path in quarantined):
        raise ValueError("cache contains temporary or quarantined files")
    return items


def _validate_all_items(
    item_paths: list[Path], manifest: dict, rows: list[dict]
) -> dict:
    hidden_sizes: set[int] = set()
    total_tokens = 0
    total_valid_answer_tokens = 0
    for example_id, (path, row) in enumerate(zip(item_paths, rows)):
        metadata = validate_cache_item(path, manifest)
        if metadata["example_id"] != example_id:
            raise ValueError(f"cache example ID mismatch: {path}")
        if metadata["sequence_length"] != row["source_sequence_length"]:
            raise ValueError(f"cache/source sequence length mismatch: {example_id}")
        if metadata["answer_start"] != row["answer_start"]:
            raise ValueError(f"cache/source answer_start mismatch: {example_id}")
        if metadata["valid_end"] != row["reviewed_valid_end"]:
            raise ValueError(f"cache/reviewed valid_end mismatch: {example_id}")
        source = SOURCE_ROOT / row["source_relative_path"]
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"preserved source must be a regular non-symlink: {example_id}")
        if file_sha256(source) != row["source_sha256"]:
            raise ValueError(f"preserved source hash mismatch: {example_id}")
        with np.load(source, allow_pickle=False) as source_archive, np.load(
            path, allow_pickle=False
        ) as cache_archive:
            if not np.array_equal(source_archive["input_ids"], cache_archive["input_ids"]):
                raise ValueError(f"cached token IDs differ from source: {example_id}")
        hidden_sizes.add(metadata["hidden_size"])
        total_tokens += metadata["sequence_length"]
        total_valid_answer_tokens += metadata["valid_end"] - metadata["answer_start"]
    if hidden_sizes != {5280}:
        raise ValueError(f"unexpected hidden sizes: {hidden_sizes}")
    return {
        "item_count": len(item_paths),
        "train_count": 2250,
        "validation_count": 250,
        "total_tokens": total_tokens,
        "total_valid_answer_tokens": total_valid_answer_tokens,
        "hidden_size": 5280,
    }


def _replay_states(manifest: dict) -> list[dict]:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).eval().cuda()
    model.requires_grad_(False)
    results = []
    for example_id in REPLAY_IDS:
        path = _expected_item_path(example_id)
        with np.load(path, allow_pickle=False) as archive:
            input_ids_array = archive["input_ids"].copy()
            expected = {
                key: archive[key].copy()
                for key in ("h0_full", "x_full", "h16_teacher")
            }
        seed_for_example(manifest["base_seed"], example_id)
        input_ids = torch.from_numpy(input_ids_array).unsqueeze(0).cuda()
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        observed = capture_huginn_functional_states(
            model, input_ids, attention_mask, depth=16
        )
        state_result = {}
        for key in expected:
            equal = arrays_byte_identical(observed[key], expected[key])
            max_abs = float(
                np.max(
                    np.abs(
                        observed[key].astype(np.float32)
                        - expected[key].astype(np.float32)
                    )
                )
            )
            if not equal:
                raise ValueError(
                    f"live replay differs for example {example_id} {key}: {max_abs}"
                )
            state_result[key] = {"byte_identical": True, "max_abs_error": max_abs}
        results.append({"example_id": example_id, "states": state_result})
        del input_ids, attention_mask, observed, expected
        torch.cuda.empty_cache()
    return results


def main() -> None:
    validation_commit = _git_commit()
    manifest_path = CACHE_ROOT / "manifest.json"
    require_identical_regular_files(manifest_path, COMMITTED_MANIFEST_PATH)
    if file_sha256(manifest_path) != CACHE_MANIFEST_SHA256:
        raise ValueError("cache manifest does not match the committed frozen digest")
    if ATTESTATION_PATH.is_symlink() or not ATTESTATION_PATH.is_file():
        raise ValueError("runtime attestation must be a committed regular file")
    attestation = json.loads(ATTESTATION_PATH.read_text(encoding="utf-8"))
    validate_runpod_volume_attestation(
        attestation,
        pod_id=POD_ID,
        volume_id=VOLUME_ID,
        data_center_id=DATA_CENTER_ID,
        mount_path=str(MOUNT_ROOT),
        minimum_size_gb=100,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_cache_manifest(manifest)
    if manifest["git_commit"] != BUILDER_COMMIT:
        raise ValueError("cache manifest does not name the frozen builder commit")
    if manifest["valid_end_manifest_sha256"] != FROZEN_VALID_END_MANIFEST_SHA256:
        raise ValueError("cache manifest does not name the frozen Phase 4 sidecar")
    if file_sha256(VALID_END_PATH) != FROZEN_VALID_END_MANIFEST_SHA256:
        raise ValueError("local Phase 4 sidecar checksum changed")

    mount = subprocess.run(
        ["findmnt", "-T", str(CACHE_ROOT), "-n", "-o", "SOURCE,FSTYPE,TARGET"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    validate_mount_output(
        CACHE_ROOT,
        mount,
        expected_mount=MOUNT_ROOT,
        expected_data_center_id=DATA_CENTER_ID,
        expected_volume_id=VOLUME_ID,
    )

    rows = load_and_validate_manifest(VALID_END_PATH)
    item_paths = _validate_file_set()
    summary = _validate_all_items(item_paths, manifest, rows)
    replay = _replay_states(manifest)

    payload_paths = [manifest_path, *item_paths]
    inventory_payload = checksum_inventory_bytes(CACHE_ROOT, payload_paths)
    verified_files = verify_checksum_inventory(CACHE_ROOT, inventory_payload)
    if verified_files != 2501:
        raise ValueError(f"expected 2501 checksummed payload files, got {verified_files}")
    inventory_path = CACHE_ROOT / "SHA256SUMS.tsv"
    inventory_sha256 = write_bytes_immutable(inventory_path, inventory_payload)

    report = {
        "protocol": "functional-cache-v2-validation-v1",
        "status": "pass",
        "cache_manifest_sha256": file_sha256(manifest_path),
        "cache_manifest_git_commit": manifest["git_commit"],
        "validation_git_commit": validation_commit,
        "valid_end_manifest_sha256": FROZEN_VALID_END_MANIFEST_SHA256,
        "checksum_protocol": CHECKSUM_PROTOCOL,
        "inventory_sha256": inventory_sha256,
        "inventory_file_count": verified_files,
        "durable_volume_id": VOLUME_ID,
        "validation_pod_id": POD_ID,
        "runtime_attestation_sha256": file_sha256(ATTESTATION_PATH),
        "mount": mount,
        "summary": summary,
        "replay_ids": list(REPLAY_IDS),
        "replay": replay,
    }
    report_payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    report_path = CACHE_ROOT / "validation_report.json"
    report_sha256 = write_bytes_immutable(report_path, report_payload)

    freeze = {
        "protocol": "functional-cache-v2-freeze-v1",
        "status": "frozen",
        "cache_item_count": 2500,
        "payload_inventory_file_count": verified_files,
        "inventory_sha256": inventory_sha256,
        "validation_report_sha256": report_sha256,
        "cache_manifest_sha256": report["cache_manifest_sha256"],
        "builder_git_commit": BUILDER_COMMIT,
        "validation_git_commit": validation_commit,
        "durable_volume_id": VOLUME_ID,
        "validation_pod_id": POD_ID,
        "runtime_attestation_sha256": report["runtime_attestation_sha256"],
    }
    freeze_payload = (json.dumps(freeze, indent=2, sort_keys=True) + "\n").encode()
    freeze_path = CACHE_ROOT / "FROZEN.json"
    freeze_sha256 = write_bytes_immutable(freeze_path, freeze_payload)
    directory_fd = os.open(CACHE_ROOT, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

    print(json.dumps({**summary, "inventory_sha256": inventory_sha256,
                      "validation_report_sha256": report_sha256,
                      "freeze_sha256": freeze_sha256,
                      "replay_ids": list(REPLAY_IDS)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
