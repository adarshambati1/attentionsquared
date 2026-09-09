#!/usr/bin/env python3
"""Create the one immutable shared Attention² initialization for K=1/2/4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.functional_protocol import (
    A2_INITIALIZATION_SEED,
    PAIRED_DATA_ORDER_SEED,
    PAIRED_RANDOMNESS_PROTOCOL,
    Attention2Architecture,
    OptimizerSpec,
    create_shared_initialization,
    initialization_metadata,
    validate_shared_initialization,
)
from src.evaluation.correctness import SEED_PROTOCOL


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
        raise RuntimeError("shared initialization requires a clean committed checkout")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def validate_config(config: dict, output: Path) -> Attention2Architecture:
    optimizer = OptimizerSpec()
    expected = {
        "protocol": PAIRED_RANDOMNESS_PROTOCOL,
        "cache_freeze_sha256": "94417eab65fd04a5827bdef9aadc9a5b8b66c266700b6cfc7ab56d39f949d3f0",
        "cache_manifest_sha256": "32db951f6cfbf76220b1c7126e23149f087eba58934ab691b588aeb3a43535bf",
        "gradient_update_ids": [0, 2249],
        "checkpoint_validation_ids": [2250, 2499],
        "scientific_test_ids": [0, 249],
        "scientific_test_dataset_split": "test",
        "h0": {
            "base_seed": 3000,
            "seed_protocol": SEED_PROTOCOL,
            "smoke_seed_indices": [0],
            "final_seed_indices": [0, 1, 2],
            "step": 0,
        },
        "attention2": {
            "hidden": 5280,
            "depth": 16,
            "heads": 16,
            "mlp_ratio": 1,
            "depth_scale": 16.0,
            "film": False,
            "paired_K": [1, 2, 4],
            "initialization_seed": A2_INITIALIZATION_SEED,
            "initialization_artifact": "/workspace/functional_protocol/a2_init_seed_0.pt",
        },
        "data_order": {
            "protocol": "paired-epoch-permutation-v1",
            "seed": PAIRED_DATA_ORDER_SEED,
        },
        "optimizer": {
            "name": optimizer.name,
            "learning_rate": optimizer.learning_rate,
            "betas": list(optimizer.betas),
            "epsilon": optimizer.epsilon,
            "weight_decay": optimizer.weight_decay,
        },
        "preprocessing": {
            "cache_state_dtype": "float16",
            "training_state_dtype": "float32",
            "model_compute_dtype": "bfloat16",
            "padding_mask": "boolean-real-token-mask-v1",
            "causal_alignment": "answer-token-t-is-predicted-by-logit-t-minus-1-v1",
        },
        "loss": {
            "name": "teacher-to-student-KL",
            "normalization": "global-sum-over-valid-answer-token-count",
            "mask": "answer_start:valid_end-only",
            "trajectory_loss_weight": 0.0,
        },
    }
    if config != expected:
        raise ValueError("training configuration does not exactly match the frozen protocol")
    expected_path = Path(expected["attention2"]["initialization_artifact"])
    if output != expected_path:
        raise ValueError("output must be the exact configured initialization artifact")
    return Attention2Architecture(
        **{
            key: expected["attention2"][key]
            for key in ("hidden", "depth", "heads", "mlp_ratio", "depth_scale", "film")
        }
    )


def write_json_exclusive_fsync(path: Path, value: dict) -> None:
    """Publish complete JSON atomically without ever replacing a final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite metadata record: {path}")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _preflight_new_paths(output: Path, record: Path) -> None:
    if record != output.with_suffix(".json"):
        raise ValueError("metadata record must be adjacent to the configured artifact")
    for path in (output, record):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite immutable output: {path}")
        if any(parent.is_symlink() for parent in path.parents if parent.exists()):
            raise ValueError(f"immutable output may not traverse symlinks: {path}")


def main(config_path: Path, output: Path, record: Path) -> dict:
    _preflight_new_paths(output, record)
    commit = clean_git_commit()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    architecture = validate_config(config, output)
    expected_metadata = initialization_metadata(architecture)
    metadata = create_shared_initialization(output, architecture=architecture)
    if any(metadata[key] != value for key, value in expected_metadata.items()):
        raise RuntimeError("created initialization metadata changed unexpectedly")
    validate_shared_initialization(output, architecture=architecture)
    result = {
        **metadata,
        "artifact": str(output),
        "artifact_bytes": output.stat().st_size,
        "artifact_sha256": sha256_file(output),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "git_commit": commit,
    }
    write_json_exclusive_fsync(record, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/004_functional_v2_training.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/workspace/functional_protocol/a2_init_seed_0.pt")
    )
    parser.add_argument(
        "--record", type=Path, default=Path("/workspace/functional_protocol/a2_init_seed_0.json")
    )
    arguments = parser.parse_args()
    main(arguments.config, arguments.output, arguments.record)
