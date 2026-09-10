#!/usr/bin/env python3
"""Independently validate and live-replay the full Phase 14 native-state cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import MethodType

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.functional_cache_v5_full import (
    CACHE_V5_ITEM_KEYS,
    CACHE_V5_ITEM_PROTOCOL,
    FULL_EXAMPLE_IDS,
    manifest_sha256,
    validate_item,
    validate_manifest,
)
from src.data.functional_extraction_v5 import uint16_to_bfloat16_tensor
from src.data.valid_end_manifest import load_and_validate_manifest
from src.evaluation.correctness import TokenKLAggregator, token_kl_sum_and_count
from src.evaluation.functional_autoregressive import (
    FullPrefixHuginnD16Evaluator,
    frozen_coda_logits_from_normalized_state,
)

CONFIG = Path("configs/004_functional_cache_v5_full.json")
CACHE_ROOT = Path("/workspace/functional_cache_v5_full")
MODEL_ID = "tomg-group-umd/huginn-0125"
MODEL_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> tuple[str, dict[str, str]]:
    hashes: dict[str, str] = {}
    for path in sorted(p for p in root.iterdir() if p.is_file()):
        hashes[path.name] = sha256_file(path)
    digest = hashlib.sha256()
    for name, value in hashes.items():
        digest.update(name.encode() + b"\0" + value.encode() + b"\n")
    return digest.hexdigest(), hashes


def validate_static(cache_root: Path, expected_builder_commit: str) -> tuple[dict, list[dict], dict]:
    config = json.loads((ROOT / CONFIG).read_text())
    rows = load_and_validate_manifest(ROOT / config["valid_end_manifest"])
    if [row["example_id"] for row in rows] != list(FULL_EXAMPLE_IDS):
        raise ValueError("valid-end manifest does not contain exact ordered IDs 0-2499")
    if cache_root.is_symlink() or not cache_root.is_dir() or stat.S_IMODE(cache_root.stat().st_mode) != 0o555:
        raise ValueError("cache root must be a real mode-0555 directory")
    expected_names = {"manifest.json", ".functional-cache-v5.lock", *(f"{i:05d}.npz" for i in FULL_EXAMPLE_IDS)}
    if {path.name for path in cache_root.iterdir()} != expected_names or any(path.is_dir() or path.is_symlink() for path in cache_root.iterdir()):
        raise ValueError("cache must contain exactly manifest, lock, and 2,500 regular item files")
    if any(stat.S_IMODE(path.stat().st_mode) != 0o444 for path in cache_root.iterdir()):
        raise ValueError("every cache file must be mode 0444")
    manifest = json.loads((cache_root / "manifest.json").read_text())
    validate_manifest(manifest)
    if manifest["git_commit"] != expected_builder_commit:
        raise ValueError("cache builder commit differs from authorized commit")
    if manifest["source_sha256_by_id"] != {str(row["example_id"]): row["source_sha256"] for row in rows}:
        raise ValueError("manifest source hashes differ from frozen valid-end manifest")
    manifest_digest = manifest_sha256(manifest)
    total_tokens = 0
    valid_tokens = 0
    role_counts = {"training": 0, "validation": 0}
    for row in rows:
        example_id = row["example_id"]
        path = cache_root / f"{example_id:05d}.npz"
        metadata = validate_item(path, manifest)
        source = ROOT / config["source_continuations"] / row["source_relative_path"]
        if sha256_file(source) != row["source_sha256"]:
            raise ValueError(f"source hash mismatch ID={example_id}")
        with np.load(source, allow_pickle=False) as raw_archive, np.load(path, allow_pickle=False) as cache:
            raw_ids = raw_archive["input_ids"]
            if set(cache.files) != CACHE_V5_ITEM_KEYS or any("logit" in key.lower() for key in cache.files):
                raise ValueError(f"invalid cache schema or persisted logits ID={example_id}")
            if not np.array_equal(cache["input_ids"], raw_ids):
                raise ValueError(f"cached input IDs differ from source ID={example_id}")
            if int(cache["answer_start"]) != row["answer_start"] or int(cache["valid_end"]) != row["reviewed_valid_end"]:
                raise ValueError(f"cached boundaries differ from reviewed manifest ID={example_id}")
            if str(cache["source_sha256"]) != row["source_sha256"] or str(cache["manifest_sha256"]) != manifest_digest:
                raise ValueError(f"cached provenance differs ID={example_id}")
            if str(cache["cache_protocol"]) != CACHE_V5_ITEM_PROTOCOL:
                raise ValueError(f"cached item protocol differs ID={example_id}")
        if metadata["sequence_length"] != row["source_sequence_length"]:
            raise ValueError(f"sequence length differs ID={example_id}")
        total_tokens += metadata["sequence_length"]
        valid_tokens += metadata["valid_end"] - metadata["answer_start"]
        role_counts["training" if example_id < 2250 else "validation"] += 1
    if role_counts != {"training": 2250, "validation": 250}:
        raise ValueError("train/validation partition count differs")
    return config, rows, {
        "item_count": 2500,
        "training_items": 2250,
        "validation_items": 250,
        "total_tokens": total_tokens,
        "valid_answer_tokens": valid_tokens,
        "manifest_sha256": sha256_file(cache_root / "manifest.json"),
        "manifest_canonical_sha256": manifest_digest,
        "no_full_vocabulary_logits": True,
        "all_items_readable_and_schema_exact": True,
    }


def live_replay(cache_root: Path, config: dict, rows: list[dict], manifest: dict) -> dict:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16,
        trust_remote_code=True, local_files_only=True,
    ).eval().cuda().requires_grad_(False)
    evaluator = FullPrefixHuginnD16Evaluator(model, None, base_seed=3000, seed_index=0)
    row_by_id = {row["example_id"]: row for row in rows}
    records = []
    aggregate_kl = TokenKLAggregator()
    total_argmax = 0
    total_positions = 0
    for example_id in config["validation_replay_ids"]:
        row = row_by_id[example_id]
        with np.load(cache_root / f"{example_id:05d}.npz", allow_pickle=False) as cache:
            ids = torch.from_numpy(cache["input_ids"].astype(np.int64))[None].cuda()
            h0 = uint16_to_bfloat16_tensor(cache["h0"].copy(), device="cuda")[None]
            cached_x = torch.from_numpy(cache["x"].copy())[None].cuda()
            cached_h16 = torch.from_numpy(cache["h16"].copy())[None].cuda()
            expected_schedule_hash = str(cache["full_schedule_sha256"])
        schedule = evaluator.materialize_generation_schedule(ids, example_id=example_id)
        live_h0 = schedule[:, :ids.shape[1]]
        actual_schedule_hash = hashlib.sha256(schedule.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        captured = {}
        calls = 0
        original = model.core_block_forward
        def wrapped(this, state, recurrent_input, *args, **kwargs):
            nonlocal calls
            if calls == 0:
                captured["x"] = recurrent_input.detach().clone()
            calls += 1
            return original(state, recurrent_input, *args, **kwargs)
        model.core_block_forward = MethodType(wrapped, model)
        try:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                live = model(input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool), input_states=live_h0, num_steps=16, use_cache=False, return_dict=True, output_details={"return_logits": True, "return_latents": True, "return_head": False, "return_stats": False})
                cached_logits = frozen_coda_logits_from_normalized_state(model, cached_h16, model.freqs_cis[:, :ids.shape[1]])
        finally:
            model.core_block_forward = original
        represented = slice(row["answer_start"] - 1, row["reviewed_valid_end"] - 1)
        live_logits = live.logits[:, represented].float()
        replay_logits = cached_logits[:, represented].float()
        kl_sum, count = token_kl_sum_and_count(replay_logits, live_logits)
        n = int(count.item())
        argmax = int((live_logits.argmax(-1) == replay_logits.argmax(-1)).sum())
        gates = {
            "full_schedule_hash_exact": actual_schedule_hash == expected_schedule_hash,
            "h0_bfloat16_bits_exact": torch.equal(live_h0, h0),
            "x_float32_bits_exact": calls == 16 and torch.equal(captured["x"], cached_x),
            "h16_float32_bits_exact": torch.equal(live.latent_states, cached_h16),
            "coda_logits_exact": torch.equal(live_logits, replay_logits),
            "next_token_argmax_exact": argmax == n,
        }
        if not all(gates.values()):
            raise ValueError(f"live replay exactness failed ID={example_id}: {gates}")
        aggregate_kl.update(kl_sum, count)
        total_argmax += argmax
        total_positions += n
        records.append({"example_id": example_id, "sequence_length": int(ids.shape[1]), "represented_teacher_positions": n, **gates})
        del ids, h0, cached_x, cached_h16, schedule, live_h0, live, cached_logits, live_logits, replay_logits
        torch.cuda.empty_cache()
    return {
        "selection_seed": config["validation_replay_selection_seed"],
        "example_ids": config["validation_replay_ids"],
        "records": records,
        "represented_teacher_positions": total_positions,
        "globally_token_normalized_kl": aggregate_kl.mean,
        "kl_absolute_tolerance": 1e-8,
        "argmax_equal": total_argmax,
        "argmax_total": total_positions,
        "passed": abs(aggregate_kl.mean) <= 1e-8 and total_argmax == total_positions,
    }


def write_exclusive(path: Path, payload: dict) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--expected-builder-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, rows, static = validate_static(args.cache_root, args.expected_builder_commit)
    manifest = json.loads((args.cache_root / "manifest.json").read_text())
    replay = live_replay(args.cache_root, config, rows, manifest)
    tree_digest, file_hashes = tree_sha256(args.cache_root)
    result = {
        "protocol": "functional-cache-v5-full-independent-validation-v1",
        "status": "pass" if replay["passed"] else "fail",
        "cache_root": str(args.cache_root),
        "builder_git_commit": args.expected_builder_commit,
        "static_validation": static,
        "live_replay": replay,
        "cache_tree_sha256": tree_digest,
        "cache_file_count_hashed": len(file_hashes),
    }
    write_exclusive(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
