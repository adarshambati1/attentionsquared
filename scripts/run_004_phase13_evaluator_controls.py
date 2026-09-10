#!/usr/bin/env python3
"""Importable Phase 13 control helpers; production main is remediation-blocked."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
import hashlib
import json
import math
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
from src.evaluation.correctness import (
    SCORER_PROTOCOL,
    SEED_PROTOCOL,
    STOP_STRINGS,
    TokenKLAggregator,
    build_chat_prompt,
    extract_answer,
    generation_status,
    per_example_seed,
    score_generation,
    stop_token_ids,
    tokenize_prompt,
    token_kl_sum_and_count,
)
from src.evaluation.functional_autoregressive import (
    HUGINN_D16_FULL_PREFIX_PROTOCOL,
    FullPrefixHuginnD16Evaluator,
    frozen_coda_logits_from_normalized_state,
)
from src.training.functional_objective import freeze_module


CONFIG = Path("configs/004_functional_v2_phase13.json")
OUTPUT_ROOT = Path("/workspace/functional_phase13_controls_v2")
PROTOCOL = "huginn-d16-phase13-evaluator-controls-v2"
MODEL_ID = "tomg-group-umd/huginn-0125"
MODEL_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
HISTORICAL_SEED_LABEL = (
    "prospective deterministic reproduction seed; historical 40.4% RNG provenance unknown"
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
        cwd=ROOT, check=True, text=True, capture_output=True,
    ).stdout
    if status:
        raise RuntimeError("Phase 13 requires a clean committed checkout")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True,
        capture_output=True,
    ).stdout.strip()


def expected_config() -> dict:
    return {
        "protocol": PROTOCOL,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": "openai/gsm8k",
        "dataset_config": "main",
        "dataset_revision": DATASET_REVISION,
        "dataset_split": "train",
        "example_id_range_inclusive": [2250, 2499],
        "depth": 16,
        "dtype": "bfloat16",
        "h0_base_seed": 3000,
        "h0_seed_index": 0,
        "seed_provenance": HISTORICAL_SEED_LABEL,
        "max_new_tokens": 1024,
        "greedy": True,
        "use_cache": False,
        "recompute_complete_prefix_every_token": True,
        "system_instruction": "You are a helpful assistant that can assist users with mathematical reasoning.",
        "historical_reference_correct": 101,
        "historical_reference_total": 250,
        "accuracy_anomaly_band": [0.34, 0.47],
        "accuracy_band_role": "broad anomaly check only; not deterministic route equivalence",
        "report_historical_emulation_and_corrected_scores": True,
        "cache_root": "/workspace/functional_cache_v2",
        "cache_freeze_sha256": "94417eab65fd04a5827bdef9aadc9a5b8b66c266700b6cfc7ab56d39f949d3f0",
        "cache_manifest_sha256": "32db951f6cfbf76220b1c7126e23149f087eba58934ab691b588aeb3a43535bf",
        "cache_control_split": "validation",
        "cache_control_example_ids": list(range(2250, 2258)),
        "cached_live_tolerances": {
            "hidden_max_abs": 0.003,
            "logit_mean_abs": 0.005,
            "logit_max_abs": 0.06,
            "kl_per_token": 0.0005,
            "require_all_next_token_argmax_equal": True,
        },
        "float16_hidden_bitwise_identity_is_diagnostic_only": True,
        "live_route_exact_requirements": [
            "generated_token_ids", "stop_reason", "hit_max_new_tokens",
            "extracted_answer", "correctness",
        ],
        "cache_equivalence_scope": "represented teacher-continuation prefixes only",
        "persist_logits": False,
        "training": False,
        "attention2_generation": False,
        "timing_or_speed_claims": False,
        "output": str(OUTPUT_ROOT),
    }


def validate_config(config: dict) -> None:
    if config != expected_config():
        raise ValueError("Phase 13 config does not exactly match the frozen protocol")


def scorer_golden_controls() -> dict:
    """Literal historical/corrected cap-fallback and extraction controls."""
    extraction_cases = [
        ("work\n#### 1,234", True, "1234"),
        ("The answer is" + "x" * 48 + "42", False, "42"),
        ("The answer is" + "x" * 49 + "42", False, None),
        ("intermediate 7 then 42", True, "42"),
    ]
    for text, fallback, expected in extraction_cases:
        if extract_answer(text, allow_fallback=fallback) != expected:
            raise RuntimeError("answer-extraction golden control failed")

    text = "unfinished reasoning with intermediate 42"
    historical = score_generation(
        text, "#### 42", hit_max_new_tokens=True, allow_fallback_on_cap=True
    )
    corrected = score_generation(text, "#### 42", hit_max_new_tokens=True)
    explicit = score_generation("Therefore, the answer is 42", "#### 42", hit_max_new_tokens=True)
    if not historical.correct or corrected.correct or corrected.predicted_answer is not None:
        raise RuntimeError("historical versus corrected cap-fallback golden control failed")
    if not explicit.correct:
        raise RuntimeError("explicit answer at cap golden control failed")

    statuses = {
        "cap": generation_status([1, 2, 3], max_new_tokens=3, stop_token_ids=[9]),
        "stop": generation_status([1, 9], max_new_tokens=3, stop_token_ids=[9]),
        "stop_at_cap": generation_status([1, 2, 9], max_new_tokens=3, stop_token_ids=[9]),
    }
    if not statuses["cap"].hit_max_new_tokens or statuses["cap"].ended_naturally:
        raise RuntimeError("cap-status golden control failed")
    if any(not statuses[key].ended_naturally or statuses[key].hit_max_new_tokens for key in ("stop", "stop_at_cap")):
        raise RuntimeError("stop-status golden control failed")
    if STOP_STRINGS != ("<|end_text|>", "<|end_turn|>"):
        raise RuntimeError("stop strings changed")
    return {
        "passed": True,
        "scorer_protocol": SCORER_PROTOCOL,
        "stop_strings": list(STOP_STRINGS),
        "historical_cap_fallback_emulation": asdict(historical),
        "authoritative_corrected_cap_scoring": asdict(corrected),
        "explicit_answer_allowed_at_cap": asdict(explicit),
        "statuses": {key: asdict(value) for key, value in statuses.items()},
    }


def trusted_live_next_token_logits(huginn, prefix: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """Independent trusted model-forward reference; the caller owns h0."""
    with torch.inference_mode(), torch.autocast(
        device_type=prefix.device.type, dtype=torch.bfloat16,
        enabled=prefix.device.type == "cuda",
    ):
        output = huginn(
            input_ids=prefix,
            attention_mask=torch.ones_like(prefix, dtype=torch.bool),
            input_states=h0,
            num_steps=16,
            use_cache=False,
            return_dict=True,
        )
        return output.logits[:, -1].float().clone()


def _stop_reason(token_ids: list[int], tokenizer, cap: int, *, route_mismatch: bool) -> tuple[str, bool]:
    status = generation_status(token_ids, max_new_tokens=cap, stop_token_ids=stop_token_ids(tokenizer))
    if route_mismatch and not status.ended_naturally:
        return "route_mismatch", status.hit_max_new_tokens
    if status.ended_naturally:
        return "stop_token", status.hit_max_new_tokens
    if status.hit_max_new_tokens:
        return "max_new_tokens", True
    return "incomplete", False


def compare_live_routes(
    huginn, tokenizer, evaluator: FullPrefixHuginnD16Evaluator,
    prompt: torch.Tensor, *, example_id: int, gold_text: str, max_new_tokens: int,
) -> dict:
    """Generate routes lockstep while literally reusing each materialized h0 object."""
    current = prompt.clone()
    trusted_tokens: list[int] = []
    evaluator_tokens: list[int] = []
    prefix_records = []
    route_mismatch = False
    stops = {int(x) for x in stop_token_ids(tokenizer) if x is not None and int(x) >= 0}

    for step in range(max_new_tokens):
        h0, derived_seed = evaluator.materialize_h0(current, example_id=example_id)
        # These are deliberately two independent forwards receiving the exact same object/value.
        trusted_logits = trusted_live_next_token_logits(huginn, current, h0)
        evaluator_logits = evaluator.next_token_logits_with_h0(current, h0)
        trusted_token = int(trusted_logits[0].argmax().item())
        evaluator_token = int(evaluator_logits[0].argmax().item())
        prefix_records.append({
            "step": step,
            "prefix_length": int(current.shape[1]),
            "prefix_token_ids_sha256": hashlib.sha256(
                current.detach().cpu().to(torch.int64).numpy().tobytes()
            ).hexdigest(),
            "derived_h0_seed": derived_seed,
            "same_materialized_h0_object_and_value_reused": True,
            "next_token_equal": trusted_token == evaluator_token,
        })
        del trusted_logits, evaluator_logits, h0
        trusted_tokens.append(trusted_token)
        evaluator_tokens.append(evaluator_token)
        if trusted_token != evaluator_token:
            route_mismatch = True
            break
        current = torch.cat((current, torch.tensor([[trusted_token]], device=current.device, dtype=current.dtype)), dim=1)
        if trusted_token in stops:
            break

    trusted_text = tokenizer.decode(trusted_tokens, skip_special_tokens=False)
    evaluator_text = tokenizer.decode(evaluator_tokens, skip_special_tokens=False)
    trusted_reason, trusted_cap = _stop_reason(trusted_tokens, tokenizer, max_new_tokens, route_mismatch=route_mismatch)
    evaluator_reason, evaluator_cap = _stop_reason(evaluator_tokens, tokenizer, max_new_tokens, route_mismatch=route_mismatch)

    def route(text: str, tokens: list[int], reason: str, hit_cap: bool) -> dict:
        historical = score_generation(text, gold_text, hit_max_new_tokens=hit_cap, allow_fallback_on_cap=True)
        corrected = score_generation(text, gold_text, hit_max_new_tokens=hit_cap)
        return {
            "generated_token_ids": tokens,
            "generated_tokens": len(tokens),
            "text": text,
            "stop_reason": reason,
            "hit_max_new_tokens": hit_cap,
            "historical_cap_fallback_emulation": asdict(historical),
            "authoritative_corrected_scoring": asdict(corrected),
        }

    trusted = route(trusted_text, trusted_tokens, trusted_reason, trusted_cap)
    evaluated = route(evaluator_text, evaluator_tokens, evaluator_reason, evaluator_cap)
    categorical = {
        "generated_token_ids": trusted_tokens == evaluator_tokens,
        "stop_reason": trusted_reason == evaluator_reason,
        "hit_max_new_tokens": trusted_cap == evaluator_cap,
        "extracted_answer": (
            trusted["authoritative_corrected_scoring"]["predicted_answer"]
            == evaluated["authoritative_corrected_scoring"]["predicted_answer"]
        ),
        "correctness": (
            trusted["authoritative_corrected_scoring"]["correct"]
            == evaluated["authoritative_corrected_scoring"]["correct"]
        ),
    }
    return {
        "example_id": example_id,
        "prompt_tokens": int(prompt.shape[1]),
        "prefixes": prefix_records,
        "trusted_live_model_forward": trusted,
        "full_prefix_huginn_evaluator": evaluated,
        "categorical_agreement": categorical,
        "passed": all(categorical.values()),
    }


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError("non-finite Phase 13 metric")
    return value


def run_cache_controls(config: dict, huginn, tokenizer, device: torch.device) -> dict:
    """Compare every valid teacher-prefix prediction position for IDs 2250--2257."""
    del tokenizer
    cache_root = Path(config["cache_root"])
    if sha256_file(cache_root / "FROZEN.json") != config["cache_freeze_sha256"]:
        raise ValueError("cache freeze identity mismatch")
    if sha256_file(cache_root / "manifest.json") != config["cache_manifest_sha256"]:
        raise ValueError("cache manifest file identity mismatch")
    manifest = json.loads((cache_root / "manifest.json").read_text())
    validate_cache_manifest(manifest)
    evaluator = FullPrefixHuginnD16Evaluator(
        huginn, None, base_seed=manifest["base_seed"], seed_index=0
    )

    hidden_max = 0.0
    hidden_dot = hidden_live_sq = hidden_cached_sq = hidden_diff_sq = 0.0
    logit_abs_sum = logit_max = 0.0
    logit_elements = 0
    argmax_equal = compared_tokens = 0
    bitwise_equal = bitwise_elements = 0
    kl = TokenKLAggregator()
    records = []

    for example_id in config["cache_control_example_ids"]:
        path = cache_root / config["cache_control_split"] / f"{example_id:05d}.npz"
        metadata = validate_cache_item(path, manifest)
        with np.load(path, allow_pickle=False) as archive:
            ids = torch.from_numpy(archive["input_ids"].astype(np.int64))[None].to(device)
            cached_fp16 = torch.from_numpy(archive["h16_teacher"].copy())[None].to(device)
        start = metadata["answer_start"] - 1
        end = metadata["valid_end"] - 1
        positions = slice(start, end)
        position_count = end - start
        if position_count <= 0:
            raise RuntimeError("cache control has no represented teacher-prefix prediction positions")

        h0, derived_seed = evaluator.materialize_h0(ids, example_id=example_id)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            live_output = huginn(
                input_ids=ids,
                attention_mask=torch.ones_like(ids, dtype=torch.bool),
                input_states=h0,
                num_steps=16,
                use_cache=False,
                return_dict=True,
                output_details={
                    "return_logits": True, "return_latents": True,
                    "return_head": False, "return_stats": False,
                },
            )
            live_h16 = live_output.latent_states
            live_logits = live_output.logits.float()
            frequencies = huginn.freqs_cis[:, : ids.shape[1]]
            cached_logits = frozen_coda_logits_from_normalized_state(
                huginn, cached_fp16.to(torch.bfloat16), frequencies
            )

        live_selected = live_h16[:, positions].float()
        cached_selected = cached_fp16[:, positions].float()
        hidden_diff = live_selected - cached_selected
        item_hidden_max = float(hidden_diff.abs().max().item())
        item_dot = float((live_selected * cached_selected).sum().item())
        item_live_sq = float(live_selected.square().sum().item())
        item_cached_sq = float(cached_selected.square().sum().item())
        item_diff_sq = float(hidden_diff.square().sum().item())
        item_cosine = item_dot / math.sqrt(item_live_sq * item_cached_sq)
        item_relative_l2 = math.sqrt(item_diff_sq / item_live_sq)
        item_bitwise = live_selected.to(torch.float16) == cached_fp16[:, positions]

        live_logit_selected = live_logits[:, positions]
        cached_logit_selected = cached_logits[:, positions]
        logit_diff = (live_logit_selected - cached_logit_selected).abs()
        item_logit_mean = float(logit_diff.mean().item())
        item_logit_max = float(logit_diff.max().item())
        item_kl_sum, item_kl_count = token_kl_sum_and_count(
            cached_logit_selected, live_logit_selected
        )
        item_kl = float((item_kl_sum / item_kl_count).item())
        item_argmax = live_logit_selected.argmax(-1) == cached_logit_selected.argmax(-1)

        hidden_max = max(hidden_max, item_hidden_max)
        hidden_dot += item_dot
        hidden_live_sq += item_live_sq
        hidden_cached_sq += item_cached_sq
        hidden_diff_sq += item_diff_sq
        logit_abs_sum += float(logit_diff.sum().item())
        logit_max = max(logit_max, item_logit_max)
        logit_elements += logit_diff.numel()
        argmax_equal += int(item_argmax.sum().item())
        compared_tokens += position_count
        bitwise_equal += int(item_bitwise.sum().item())
        bitwise_elements += item_bitwise.numel()
        kl.update(item_kl_sum, item_kl_count)
        records.append({
            "example_id": example_id,
            "derived_h0_seed": derived_seed,
            "prediction_position_start_inclusive": start,
            "prediction_position_end_exclusive": end,
            "compared_tokens": position_count,
            "hidden_max_abs": _finite(item_hidden_max),
            "hidden_cosine": _finite(item_cosine),
            "hidden_relative_l2": _finite(item_relative_l2),
            "logit_mean_abs": _finite(item_logit_mean),
            "logit_max_abs": _finite(item_logit_max),
            "kl_live_to_cached_per_token": _finite(item_kl),
            "next_token_argmax_equal": int(item_argmax.sum().item()),
            "float16_hidden_bitwise_equal_fraction_diagnostic": float(item_bitwise.float().mean().item()),
            "cached_coda_input_is_already_normalized": True,
        })
        del (
            h0, live_output, live_h16, live_logits, cached_logits, live_selected,
            cached_selected, hidden_diff, item_bitwise, live_logit_selected,
            cached_logit_selected, logit_diff, item_argmax,
        )

    aggregate = {
        "hidden_max_abs": _finite(hidden_max),
        "hidden_cosine": _finite(hidden_dot / math.sqrt(hidden_live_sq * hidden_cached_sq)),
        "hidden_relative_l2": _finite(math.sqrt(hidden_diff_sq / hidden_live_sq)),
        "logit_mean_abs": _finite(logit_abs_sum / logit_elements),
        "logit_max_abs": _finite(logit_max),
        "globally_token_normalized_kl_live_to_cached": _finite(kl.mean),
        "next_token_argmax_equal": argmax_equal,
        "compared_tokens": compared_tokens,
        "next_token_argmax_agreement": argmax_equal / compared_tokens,
        "float16_hidden_bitwise_equal": bitwise_equal,
        "float16_hidden_elements": bitwise_elements,
        "float16_hidden_bitwise_equal_fraction_diagnostic": bitwise_equal / bitwise_elements,
    }
    tolerances = config["cached_live_tolerances"]
    bounds = {
        "hidden_max_abs": aggregate["hidden_max_abs"] <= tolerances["hidden_max_abs"],
        "logit_mean_abs": aggregate["logit_mean_abs"] <= tolerances["logit_mean_abs"],
        "logit_max_abs": aggregate["logit_max_abs"] <= tolerances["logit_max_abs"],
        "globally_token_normalized_kl": (
            aggregate["globally_token_normalized_kl_live_to_cached"] <= tolerances["kl_per_token"]
        ),
        "all_next_token_argmax_equal": argmax_equal == compared_tokens,
    }
    return {
        "passed": all(bounds.values()),
        "scope": config["cache_equivalence_scope"],
        "cache_manifest_sha256": config["cache_manifest_sha256"],
        "cache_freeze_sha256": config["cache_freeze_sha256"],
        "tolerances": tolerances,
        "bounds_passed": bounds,
        "float16_hidden_bitwise_identity_is_diagnostic_only": True,
        "aggregate": aggregate,
        "records": records,
    }


def run(config: dict, commit: str) -> dict:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 13 canonical controls require CUDA")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"], local_files_only=True
    )
    huginn = AutoModelForCausalLM.from_pretrained(
        config["model_id"], revision=config["model_revision"],
        torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True,
    ).to(device).eval()
    freeze_module(huginn)

    golden = scorer_golden_controls()
    # Collect 13C even if it fails its frozen bounds; scientific failures still publish.
    cache_controls = run_cache_controls(config, huginn, tokenizer, device)
    dataset = load_dataset(
        config["dataset_id"], config["dataset_config"], split=config["dataset_split"],
        revision=config["dataset_revision"],
    )
    evaluator = FullPrefixHuginnD16Evaluator(
        huginn, tokenizer, base_seed=config["h0_base_seed"],
        seed_index=config["h0_seed_index"],
    )
    route_records = []
    first_id, last_id = config["example_id_range_inclusive"]
    for example_id in range(first_id, last_id + 1):
        example = dataset[example_id]
        prompt_text = build_chat_prompt(tokenizer, example["question"], config["system_instruction"])
        prompt = tokenize_prompt(tokenizer, prompt_text)["input_ids"].to(device)
        record = compare_live_routes(
            huginn, tokenizer, evaluator, prompt, example_id=example_id,
            gold_text=example["answer"], max_new_tokens=config["max_new_tokens"],
        )
        route_records.append(record)
        print(
            f"example={example_id} route_pass={record['passed']} "
            f"correct={record['full_prefix_huginn_evaluator']['authoritative_corrected_scoring']['correct']}",
            flush=True,
        )

    generations = [record["full_prefix_huginn_evaluator"] for record in route_records]
    historical_correct = sum(x["historical_cap_fallback_emulation"]["correct"] for x in generations)
    corrected_correct = sum(x["authoritative_corrected_scoring"]["correct"] for x in generations)
    total = len(generations)
    historical_accuracy = historical_correct / total
    corrected_accuracy = corrected_correct / total
    disagreements = [
        {
            "example_id": record["example_id"],
            "generated_token_ids": record["full_prefix_huginn_evaluator"]["generated_token_ids"],
            "text": record["full_prefix_huginn_evaluator"]["text"],
            "stop_reason": record["full_prefix_huginn_evaluator"]["stop_reason"],
            "hit_max_new_tokens": record["full_prefix_huginn_evaluator"]["hit_max_new_tokens"],
            "historical_cap_fallback_emulation": record["full_prefix_huginn_evaluator"]["historical_cap_fallback_emulation"],
            "authoritative_corrected_scoring": record["full_prefix_huginn_evaluator"]["authoritative_corrected_scoring"],
        }
        for record in route_records
        if record["full_prefix_huginn_evaluator"]["historical_cap_fallback_emulation"]
        != record["full_prefix_huginn_evaluator"]["authoritative_corrected_scoring"]
    ]
    low, high = config["accuracy_anomaly_band"]
    historical_control = {
        "passed_broad_anomaly_check": low <= historical_accuracy <= high,
        "seed_provenance": config["seed_provenance"],
        "historical_reference": {
            "correct": config["historical_reference_correct"],
            "total": config["historical_reference_total"],
            "accuracy": config["historical_reference_correct"] / config["historical_reference_total"],
            "role": "historical reference only; not an exact deterministic gate",
        },
        "accuracy_anomaly_band": config["accuracy_anomaly_band"],
        "accuracy_band_role": config["accuracy_band_role"],
        "historical_cap_fallback_emulation": {
            "correct": historical_correct, "total": total, "accuracy": historical_accuracy,
        },
        "authoritative_corrected_scoring": {
            "correct": corrected_correct, "total": total, "accuracy": corrected_accuracy,
        },
        "scoring_disagreement_count": len(disagreements),
        "scoring_disagreement_audit": disagreements,
    }
    route_control = {
        "passed": all(record["passed"] for record in route_records),
        "same_materialized_h0_object_required": True,
        "requirements": config["live_route_exact_requirements"],
        "records": route_records,
    }
    gates = {
        "13A_historical_broad_anomaly": historical_control["passed_broad_anomaly_check"],
        "13B_exact_live_route_equivalence": route_control["passed"],
        "13C_cached_live_bounds": cache_controls["passed"],
    }
    return {
        "protocol": PROTOCOL,
        "generation_protocol": HUGINN_D16_FULL_PREFIX_PROTOCOL,
        "phase": 13,
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "git_commit": commit,
        "model_revision": config["model_revision"],
        "dataset_revision": config["dataset_revision"],
        "example_ids": list(range(first_id, last_id + 1)),
        "depth": 16,
        "dtype": "bfloat16",
        "h0_base_seed": 3000,
        "h0_seed_index": 0,
        "seed_protocol": SEED_PROTOCOL,
        "max_new_tokens": 1024,
        "scorer_golden_controls": golden,
        "phase13A_historical_control": historical_control,
        "phase13B_live_route_control": route_control,
        "phase13C_cache_control": cache_controls,
        "full_vocabulary_logits_persisted": False,
        "training_performed": False,
        "attention2_generation_performed": False,
        "timing_or_speed_claims": False,
    }


def publish_attempt(attempt: Path, final: Path) -> None:
    expected = {attempt / "evaluation.json"}
    if {path for path in attempt.iterdir()} != expected:
        raise RuntimeError("Phase 13 output must contain only evaluation.json")
    evaluation = attempt / "evaluation.json"
    if evaluation.is_symlink() or not evaluation.is_file() or stat.S_IMODE(evaluation.stat().st_mode) != 0o444:
        raise RuntimeError("Phase 13 evaluation is not a read-only regular file")
    directory = os.open(attempt, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    os.chmod(attempt, 0o555)
    libc = ctypes.CDLL(None, use_errno=True)
    status = libc.renameat2(-100, os.fsencode(attempt), -100, os.fsencode(final), 1)
    if status != 0:
        error = ctypes.get_errno()
        if error == 17:
            raise FileExistsError(error, os.strerror(error), str(final))
        raise OSError(error, os.strerror(error), str(final))
    parent = os.open(final.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def main(config_path: Path, output_root: Path) -> dict:
    raise RuntimeError(
        "Phase 13 is hard-blocked pending independently reviewed corrected "
        "Phase 10 and new corrected Phase 11/12 artifacts"
    )
    if config_path != CONFIG or output_root != OUTPUT_ROOT:  # pragma: no cover
        raise ValueError("Phase 13 requires exact production paths")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to replace Phase 13 output: {output_root}")
    commit = clean_git_commit()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)
    attempt = output_root.parent / f".{output_root.name}.attempt-{uuid.uuid4().hex}"
    attempt.mkdir(exist_ok=False)
    print(f"Phase 13 attempt preserved on execution error: {attempt}", flush=True)
    result = run(config, commit)
    result["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    write_json_exclusive_fsync(attempt / "evaluation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    publish_attempt(attempt, output_root)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args()
    main(arguments.config, arguments.output)
