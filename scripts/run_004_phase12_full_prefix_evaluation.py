#!/usr/bin/env python3
"""Regenerate the Phase 11 eight examples with the canonical full-prefix evaluator."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.create_004_a2_initialization import write_json_exclusive_fsync
from src.data.functional_cache import validate_cache_item, validate_cache_manifest
from src.evaluation.correctness import score_generation
from src.evaluation.functional_autoregressive import (
    FULL_PREFIX_PROTOCOL,
    FullPrefixAttention2Evaluator,
    strictly_growing_full_prefix_lengths,
)
from src.training.functional_objective import freeze_module
from src.training.functional_protocol import Attention2Architecture
from src.training.overfit_gate import OVERFIT_PROTOCOL


CONFIG = Path("configs/004_functional_v2_phase12.json")
OUTPUT_ROOT = Path("/workspace/functional_phase12_v2")
MODEL_ID = "tomg-group-umd/huginn-0125"
MODEL_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"


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
        raise RuntimeError("Phase 12 requires a clean committed checkout")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def validate_config(config: dict) -> None:
    expected = {
        "protocol": FULL_PREFIX_PROTOCOL,
        "phase11_model": "/workspace/functional_overfit_v2/model.pt",
        "phase11_model_sha256": "036a6578774d4352dd778a29a6c6214d1474ffb3b1e2d5e2db92e8267777777a",
        "phase11_result": "/workspace/functional_overfit_v2/result.json",
        "phase11_result_sha256": "0d84ac8353ca412c9e4b557d3fc42805e65585e4ec4e1f06b987f20340dbcd91",
        "cache_root": "/workspace/functional_cache_v2",
        "cache_freeze_sha256": "94417eab65fd04a5827bdef9aadc9a5b8b66c266700b6cfc7ab56d39f949d3f0",
        "example_ids": list(range(8)),
        "K": 4,
        "h0_base_seed": 3000,
        "h0_seed_index": 0,
        "max_new_tokens": 384,
        "greedy": True,
        "use_cache": False,
        "attention2_kv_cache": False,
        "recompute_complete_prefix_every_token": True,
        "output": str(OUTPUT_ROOT),
    }
    if config != expected:
        raise ValueError("Phase 12 config does not exactly match the frozen protocol")


def load_attention2(path: Path, expected_hash: str, device: torch.device):
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_hash:
        raise ValueError("Phase 11 model identity mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        set(checkpoint) != {"protocol", "state_dict", "metadata"}
        or checkpoint["protocol"] != OVERFIT_PROTOCOL
        or checkpoint["metadata"].get("status") != "pending_scientific_review"
        or checkpoint["metadata"].get("full_vocabulary_logits_persisted") is not False
    ):
        raise ValueError("unexpected Phase 11 checkpoint schema or provenance")
    model = Attention2Architecture().build(rounds=4)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return freeze_module(model.to(device))


def load_inputs(cache_root: Path, device: torch.device) -> list[dict]:
    manifest = json.loads((cache_root / "manifest.json").read_text())
    validate_cache_manifest(manifest)
    examples = []
    for example_id in range(8):
        path = cache_root / "train" / f"{example_id:05d}.npz"
        validate_cache_item(path, manifest)
        with np.load(path, allow_pickle=False) as archive:
            ids = torch.from_numpy(archive["input_ids"].astype(np.int64))[None].to(device)
            start = int(archive["answer_start"])
            end = int(archive["valid_end"])
        examples.append(
            {
                "example_id": example_id,
                "input_ids": ids,
                "answer_start": start,
                "valid_end": end,
            }
        )
    return examples


def publish_attempt(attempt: Path, final: Path) -> None:
    evaluation = attempt / "evaluation.json"
    if (
        evaluation.is_symlink()
        or not evaluation.is_file()
        or stat.S_IMODE(evaluation.stat().st_mode) != 0o444
    ):
        raise RuntimeError("Phase 12 evaluation is not a read-only regular file")
    directory = os.open(attempt, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    os.chmod(attempt, 0o555)
    libc = ctypes.CDLL(None, use_errno=True)
    status = libc.renameat2(
        -100, os.fsencode(attempt), -100, os.fsencode(final), 1
    )
    if status != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(final))
    parent = os.open(final.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def run(config: dict, commit: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    model_path = Path(config["phase11_model"])
    prior_result_path = Path(config["phase11_result"])
    cache_root = Path(config["cache_root"])
    if sha256_file(prior_result_path) != config["phase11_result_sha256"]:
        raise ValueError("Phase 11 result identity mismatch")
    if sha256_file(cache_root / "FROZEN.json") != config["cache_freeze_sha256"]:
        raise ValueError("cache freeze identity mismatch")
    prior = json.loads(prior_result_path.read_text())
    prior_by_id = {item["example_id"]: item for item in prior["generations"]}

    huginn = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    freeze_module(huginn)
    attention2 = load_attention2(
        model_path, config["phase11_model_sha256"], device
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    evaluator = FullPrefixAttention2Evaluator(
        huginn,
        attention2,
        tokenizer,
        base_seed=config["h0_base_seed"],
        seed_index=config["h0_seed_index"],
    )

    generations = []
    for example in load_inputs(cache_root, device):
        prompt = example["input_ids"][:, : example["answer_start"]]
        generated = evaluator.generate(
            prompt,
            example_id=example["example_id"],
            max_new_tokens=config["max_new_tokens"],
        )
        teacher_tokens = example["input_ids"][
            0, example["answer_start"] : example["valid_end"]
        ].tolist()
        teacher_text = tokenizer.decode(teacher_tokens, skip_special_tokens=False)
        score = score_generation(
            generated.text,
            teacher_text,
            hit_max_new_tokens=generated.hit_max_new_tokens,
        )
        prior_item = prior_by_id[example["example_id"]]
        generations.append(
            {
                "example_id": example["example_id"],
                "prompt_tokens": int(prompt.shape[1]),
                "generated_token_ids": list(generated.token_ids),
                "generated_tokens": len(generated.token_ids),
                "prefix_lengths": list(generated.prefix_lengths),
                "strict_full_prefix_growth": strictly_growing_full_prefix_lengths(
                    generated.prefix_lengths, int(prompt.shape[1])
                ),
                "ended_naturally": generated.ended_naturally,
                "hit_max_new_tokens": generated.hit_max_new_tokens,
                "teacher_answer": score.gold_answer,
                "generated_answer": score.predicted_answer,
                "teacher_answer_match": score.correct,
                "text": generated.text,
                "teacher_text": teacher_text,
                "phase11_text_exact_match": generated.text == prior_item["text"],
            }
        )
        print(
            f"example={example['example_id']} generated={len(generated.token_ids)} "
            f"natural={generated.ended_naturally}",
            flush=True,
        )
    if not all(item["strict_full_prefix_growth"] for item in generations):
        raise RuntimeError("canonical evaluator did not use complete growing prefixes")
    return {
        "protocol": FULL_PREFIX_PROTOCOL,
        "status": "pass",
        "git_commit": commit,
        "phase": 12,
        "example_ids": config["example_ids"],
        "K": config["K"],
        "use_cache": False,
        "attention2_kv_cache": False,
        "recompute_complete_prefix_every_token": True,
        "phase11_model_sha256": config["phase11_model_sha256"],
        "phase11_result_sha256": config["phase11_result_sha256"],
        "cache_freeze_sha256": config["cache_freeze_sha256"],
        "all_strict_full_prefix_growth": True,
        "phase11_text_exact_match_count": sum(
            item["phase11_text_exact_match"] for item in generations
        ),
        "teacher_answer_match_count": sum(
            item["teacher_answer_match"] for item in generations
        ),
        "natural_stop_count": sum(item["ended_naturally"] for item in generations),
        "cap_hit_count": sum(item["hit_max_new_tokens"] for item in generations),
        "full_vocabulary_logits_persisted": False,
        "generations": generations,
    }


def main(config_path: Path, output_root: Path) -> dict:
    if config_path != CONFIG or output_root != OUTPUT_ROOT:
        raise ValueError("Phase 12 requires exact production paths")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to replace Phase 12 output: {output_root}")
    commit = clean_git_commit()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)
    attempt = output_root.parent / f".{output_root.name}.attempt-{uuid.uuid4().hex}"
    attempt.mkdir(exist_ok=False)
    print(f"Phase 12 attempt preserved on failure: {attempt}", flush=True)
    result = run(config, commit)
    result["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    write_json_exclusive_fsync(attempt / "evaluation.json", result)
    result["evaluation_sha256"] = sha256_file(attempt / "evaluation.json")
    # The immutable file cannot include its own hash; report it on stdout and
    # preserve it in STATUS after copying the artifact.
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    publish_attempt(attempt, output_root)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args()
    main(arguments.config, arguments.output)
