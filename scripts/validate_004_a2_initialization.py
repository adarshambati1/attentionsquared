#!/usr/bin/env python3
"""Attest the immutable production Attention² initialization and file modes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.create_004_a2_initialization import (
    clean_git_commit,
    write_json_exclusive_fsync,
)
from src.training.functional_protocol import (
    Attention2Architecture,
    validate_shared_initialization,
)


PROTOCOL = "attention2-shared-initialization-runtime-attestation-v1"
PRODUCTION_ARTIFACT = Path("/workspace/functional_protocol/a2_init_seed_0.pt")
PRODUCTION_RECORD = Path("/workspace/functional_protocol/a2_init_seed_0.json")
PRODUCTION_VALIDATION_LOG = Path(
    "/workspace/functional_protocol/a2_init_seed_0_validation.log"
)
PRODUCTION_ATTESTATION = Path(
    "/workspace/functional_protocol/a2_init_seed_0_runtime_attestation.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_regular_read_only(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular non-symlink file: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o444:
        raise ValueError(f"expected mode 0444 for {path}, observed {mode:04o}")
    return f"{mode:04o}"


def attest(
    artifact: Path,
    record_path: Path,
    validation_log: Path,
    *,
    validation_commit: str,
    architecture: Attention2Architecture = Attention2Architecture(),
) -> dict:
    modes = {
        "artifact": require_regular_read_only(artifact),
        "builder_record": require_regular_read_only(record_path),
        "validation_log": require_regular_read_only(validation_log),
    }
    record = json.loads(record_path.read_text(encoding="utf-8"))
    metadata = validate_shared_initialization(artifact, architecture=architecture)
    artifact_sha256 = sha256_file(artifact)
    if artifact_sha256 != record["artifact_sha256"]:
        raise ValueError("artifact file checksum does not match builder record")
    if artifact.stat().st_size != record["artifact_bytes"]:
        raise ValueError("artifact size does not match builder record")
    if metadata["state_dict_sha256"] != record["state_dict_sha256"]:
        raise ValueError("state-dict checksum does not match builder record")
    return {
        "protocol": PROTOCOL,
        "status": "pass",
        "artifact": str(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": artifact_sha256,
        "state_dict_sha256": metadata["state_dict_sha256"],
        "builder_git_commit": record["git_commit"],
        "builder_record_sha256": sha256_file(record_path),
        "prior_validation_log_sha256": sha256_file(validation_log),
        "validation_git_commit": validation_commit,
        "modes": modes,
    }


def main(
    artifact: Path,
    record: Path,
    validation_log: Path,
    output: Path,
) -> dict:
    expected = (
        PRODUCTION_ARTIFACT,
        PRODUCTION_RECORD,
        PRODUCTION_VALIDATION_LOG,
        PRODUCTION_ATTESTATION,
    )
    if (artifact, record, validation_log, output) != expected:
        raise ValueError("runtime attestation requires exact production paths")
    validation_commit = clean_git_commit()
    result = attest(
        artifact,
        record,
        validation_log,
        validation_commit=validation_commit,
    )
    write_json_exclusive_fsync(output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=PRODUCTION_ARTIFACT)
    parser.add_argument("--record", type=Path, default=PRODUCTION_RECORD)
    parser.add_argument(
        "--validation-log", type=Path, default=PRODUCTION_VALIDATION_LOG
    )
    parser.add_argument("--output", type=Path, default=PRODUCTION_ATTESTATION)
    arguments = parser.parse_args()
    main(
        arguments.artifact,
        arguments.record,
        arguments.validation_log,
        arguments.output,
    )
