#!/usr/bin/env python3
"""Build the one shared compact functional cache v2 from preserved sequences."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.functional_cache import (
    CACHE_ITEM_PROTOCOL,
    FROZEN_VALID_END_MANIFEST_SHA256,
    build_cache_manifest,
    cache_manifest_sha256,
    write_cache_item_atomic,
    write_manifest_atomic,
)
from src.data.functional_extraction import (
    EXTRACTION_LOCATION,
    capture_huginn_functional_states,
)
from src.data.valid_end_manifest import load_and_validate_manifest
from src.evaluation.correctness import SEED_PROTOCOL, per_example_seed, seed_for_example
from src.evaluation.splits import load_split_manifest, validate_split_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repository_root: Path = ROOT) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("working tree must be completely clean before cache extraction")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_config(config: dict[str, Any], split_manifest: dict[str, Any]) -> None:
    validate_split_config(config, split_manifest)
    expected = {
        "model_id": "tomg-group-umd/huginn-0125",
        "model_revision": "bb6621b65e90b6a4b9b29ef88dc83866d450470c",
        "teacher_depth": 16,
        "model_compute_dtype": "bfloat16",
        "cache_state_dtype": "float16",
        "base_seed": 3000,
        "seed_policy": SEED_PROTOCOL,
        "source_continuations": "results/004_functional/teacher_sequences",
        "valid_end_manifest": "results/004_functional/valid_end_manifest.jsonl",
        "cache_output": "/workspace/functional_cache_v2",
        "cap_degeneration_policy": (
            "dual-independent-conservative-valid-end-v1; "
            "generation cap is truncation, never EOS"
        ),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"config {key} must equal {value!r}")
    for key in (
        "model_id",
        "model_revision",
        "source_continuations",
        "valid_end_manifest",
        "cache_output",
        "cap_degeneration_policy",
    ):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"config {key} must be a nonempty string")


def _load_source(path: Path, expected_sha256: str) -> np.ndarray:
    if _sha256(path) != expected_sha256:
        raise ValueError(f"source hash no longer matches reviewed input: {path}")
    with np.load(path, allow_pickle=False) as archive:
        input_ids = archive["input_ids"].copy()
    if input_ids.ndim != 1 or input_ids.dtype != np.int32:
        raise ValueError(f"source input_ids must be rank-1 int32: {path}")
    return input_ids


def main(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_path = ROOT / config["split_manifest"]
    split_manifest = load_split_manifest(split_path)
    _validate_config(config, split_manifest)

    valid_end_path = ROOT / config["valid_end_manifest"]
    if _sha256(valid_end_path) != FROZEN_VALID_END_MANIFEST_SHA256:
        raise ValueError("valid-end manifest is not the frozen reviewed Phase 4 sidecar")
    valid_end_rows = load_and_validate_manifest(valid_end_path)
    source_root = ROOT / config["source_continuations"]
    output_root = Path(config["cache_output"])
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        local_files_only=True,
    )
    tokenizer_template = tokenizer.chat_template
    if not isinstance(tokenizer_template, str) or not tokenizer_template:
        raise ValueError("pinned tokenizer does not expose its exact chat template")

    manifest = build_cache_manifest(
        model_revision=config["model_revision"],
        tokenizer_template=tokenizer_template,
        git_commit=_git_commit(),
        extraction_location=EXTRACTION_LOCATION,
        split_manifest=split_manifest,
        seed_policy=config["seed_policy"],
        base_seed=config["base_seed"],
        cap_degeneration_policy=config["cap_degeneration_policy"],
        valid_end_manifest_sha256=FROZEN_VALID_END_MANIFEST_SHA256,
        tokenizer_id=config["model_id"],
        tokenizer_revision=config["model_revision"],
    )
    manifest_status = write_manifest_atomic(output_root / "manifest.json", manifest)
    manifest_digest = cache_manifest_sha256(manifest)
    print(f"manifest: {manifest_status} sha256={manifest_digest}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).eval().cuda()
    model.requires_grad_(False)

    counts: dict[str, int] = {}
    for row in valid_end_rows:
        example_id = row["example_id"]
        source_path = source_root / row["source_relative_path"]
        input_ids_array = _load_source(source_path, row["source_sha256"])
        if len(input_ids_array) != row["source_sequence_length"]:
            raise ValueError(f"source length mismatch for example {example_id}")
        dataset_split = (
            "train" if row["phase3_role"] == "gradient_update" else "validation"
        )
        item_path = output_root / dataset_split / f"{example_id:05d}.npz"

        derived_seed = seed_for_example(config["base_seed"], example_id)
        if derived_seed != per_example_seed(config["base_seed"], example_id):
            raise RuntimeError("authoritative seed derivation disagrees with seed setup")
        input_ids = torch.from_numpy(input_ids_array).unsqueeze(0).cuda()
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        states = capture_huginn_functional_states(
            model,
            input_ids,
            attention_mask,
            depth=config["teacher_depth"],
        )
        arrays = {
            "input_ids": input_ids_array,
            "attention_mask": np.ones(input_ids_array.shape, dtype=np.bool_),
            "answer_start": np.array(row["answer_start"], dtype=np.int32),
            "valid_end": np.array(row["reviewed_valid_end"], dtype=np.int32),
            **states,
            "example_id": np.array(example_id, dtype=np.int64),
            "dataset_split": np.array(dataset_split),
            "base_seed": np.array(config["base_seed"], dtype=np.int64),
            "derived_seed": np.array(derived_seed, dtype=np.int64),
            "seed_protocol": np.array(SEED_PROTOCOL),
            "cache_protocol": np.array(CACHE_ITEM_PROTOCOL),
            "manifest_sha256": np.array(manifest_digest),
        }
        status = write_cache_item_atomic(item_path, arrays, manifest)
        counts[status] = counts.get(status, 0) + 1
        print(
            f"{example_id:04d}/2499 {dataset_split} T={len(input_ids_array)} "
            f"valid_end={row['reviewed_valid_end']} status={status}",
            flush=True,
        )
        del input_ids, attention_mask, states, arrays
        torch.cuda.empty_cache()
        gc.collect()

    print(f"complete: {json.dumps(counts, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/004_functional_v2.json")
    )
    arguments = parser.parse_args()
    main(arguments.config)
