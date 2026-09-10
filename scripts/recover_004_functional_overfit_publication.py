#!/usr/bin/env python3
"""Publish the preserved complete Phase 11 attempt without retraining or mutation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.create_004_a2_initialization import write_json_exclusive_fsync
from scripts.run_004_functional_overfit_gate import atomic_publish_attempt


SOURCE = Path(
    "/workspace/functional_overfit_v2_attempts/"
    "attempt-5896a4f4ba174c2e8231c310ef0404dc"
)
FINAL = Path("/workspace/functional_overfit_v2")
RECORD = Path(
    "/workspace/functional_overfit_v2_attempts/"
    "publication-recovery-5896a4f4ba174c2e8231c310ef0404dc.json"
)
EXPECTED_RESULT_SHA256 = (
    "0d84ac8353ca412c9e4b557d3fc42805e65585e4ec4e1f06b987f20340dbcd91"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_git_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if status:
        raise RuntimeError("publication recovery requires a clean committed checkout")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def copy_fsync_read_only(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"invalid preserved source file: {source}")
    if stat.S_IMODE(source.stat().st_mode) != 0o444:
        raise ValueError(f"preserved source is not read-only: {source}")
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    if sha256_file(destination) != sha256_file(source):
        raise RuntimeError(f"publication copy checksum mismatch: {source}")
    os.chmod(destination, 0o444)


def main() -> dict:
    commit = clean_git_commit()
    if FINAL.exists() or FINAL.is_symlink():
        raise FileExistsError(f"refusing to replace final output: {FINAL}")
    if RECORD.exists() or RECORD.is_symlink():
        raise FileExistsError(f"refusing to replace recovery record: {RECORD}")
    result_source = SOURCE / "result.json"
    model_source = SOURCE / "model.pt"
    if sha256_file(result_source) != EXPECTED_RESULT_SHA256:
        raise ValueError("preserved result checksum mismatch")
    result = json.loads(result_source.read_text(encoding="utf-8"))
    if result.get("status") != "pending_scientific_review":
        raise ValueError("preserved result has unexpected status")
    if sha256_file(model_source) != result.get("model_sha256"):
        raise ValueError("preserved model checksum mismatch")

    staging = FINAL.parent / f".{FINAL.name}.publication-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    print(f"publication staging preserved on failure: {staging}", flush=True)
    copy_fsync_read_only(model_source, staging / "model.pt")
    copy_fsync_read_only(result_source, staging / "result.json")
    atomic_publish_attempt(staging, FINAL)

    record = {
        "protocol": "functional-overfit-publication-recovery-v1",
        "status": "pass",
        "git_commit": commit,
        "source_attempt": str(SOURCE),
        "source_attempt_preserved": SOURCE.is_dir(),
        "final_output": str(FINAL),
        "result_sha256": sha256_file(FINAL / "result.json"),
        "model_sha256": sha256_file(FINAL / "model.pt"),
        "result_status": result["status"],
        "retraining_performed": False,
        "model_or_result_bytes_changed": False,
        "publication": "same-parent-renameat2-RENAME_NOREPLACE",
    }
    write_json_exclusive_fsync(RECORD, record)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return record


if __name__ == "__main__":
    main()
