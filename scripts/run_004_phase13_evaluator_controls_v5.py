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
from types import MethodType
import subprocess
import sys
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.create_004_a2_initialization import write_json_exclusive_fsync
from src.data.functional_cache_v5 import validate_item, validate_manifest
from src.data.functional_extraction_v5 import uint16_to_bfloat16_tensor
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


CONFIG = Path("configs/004_functional_v5_phase13.json")
OUTPUT_ROOT = Path("/workspace/functional_phase13_controls_v5_native")
PROTOCOL = "huginn-d16-phase13-evaluator-controls-native-v5"
CACHE_VALIDATION = ROOT / "results/004_functional/cache_v5_smoke_validation_v1.json"
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
        "phase12_evaluation": "/workspace/functional_phase12_v5_native/evaluation.json",
        "phase12_evaluation_sha256": "c0009a81d932772039b0c4b4c90905a838836c99298f8d36924e6edfbadfa33a",
        "cache_root": "/workspace/functional_cache_v5_smoke",
        "cache_manifest_sha256": "55a5cc5cc77bf969ef200b49dae7d50d63ef97a88e608978902c6e50f5040583",
        "cache_validation_sha256": "65dd76fca9cba8df435b52c36f282ab8420fb742d8c4d65f61b813904e4af006",
        "cache_control_example_ids": [2277, 2473, 2452, 2485, 2415, 2264, 2317, 2439],
        "native_cache_exact_requirements": [
            "h0_bfloat16_bits", "x_float32_bits", "h16_float32_bits",
            "coda_logits", "next_token_argmax",
        ],
        "native_cache_kl_absolute_tolerance": 1e-8,
        "closure_audit": "results/004_functional/phase12_v5_upstream_closure_audit.json",
        "closure_audit_sha256": "80d692c9d583d0bb522fe653ae799c7d179f4bdc6b3343e1052942f52fe5e800",
        "live_route_exact_requirements": [
            "generated_token_ids", "generated_text", "stop_reason",
            "hit_max_new_tokens", "extracted_answer", "correctness",
        ],
        "cache_equivalence_scope": "eight preregistered length-stratified teacher-continuation items with exact native BF16/FP32 state storage",
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


def independent_decomposed_next_token_logits(
    huginn, prefix: torch.Tensor, h0: torch.Tensor
) -> torch.Tensor:
    """Literal prelude + 16 core calls + normalized-state coda reference."""
    with torch.inference_mode(), torch.autocast(
        device_type=prefix.device.type, dtype=torch.bfloat16,
        enabled=prefix.device.type == "cuda",
    ):
        frequencies = huginn.freqs_cis[:, : prefix.shape[1]]
        recurrent_input = huginn.transformer.wte(prefix)
        if huginn.emb_scale != 1:
            recurrent_input = recurrent_input * huginn.emb_scale
        block_index = torch.tensor(-1, device="cpu", dtype=torch.long)
        for block in huginn.transformer.prelude:
            block_index += 1
            recurrent_input = block(
                recurrent_input, frequencies, block_index, None, None
            )
        state = h0
        for step in range(16):
            state, block_index = huginn.core_block_forward(
                state, recurrent_input, frequencies, None, None, block_index, step
            )
        normalized = huginn.transformer.ln_f(state)
        logits = frozen_coda_logits_from_normalized_state(
            huginn, normalized, frequencies, last_only=True
        )
        return logits.clone()


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

    h0_schedule = evaluator.materialize_generation_schedule(prompt, example_id=example_id)
    derived_seed = per_example_seed(evaluator.base_seed, example_id, seed_index=evaluator.seed_index)
    for step in range(max_new_tokens):
        h0 = h0_schedule[:, : current.shape[1]]
        # These are deliberately two independent forwards receiving the exact same object/value.
        trusted_logits = independent_decomposed_next_token_logits(
            huginn, current, h0
        )
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
        "generated_text": trusted_text == evaluator_text,
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
    """Exact native-state replay on the eight preregistered Phase-13C items."""
    del tokenizer
    cache_root = Path(config["cache_root"])
    if sha256_file(cache_root / "manifest.json") != config["cache_manifest_sha256"]:
        raise ValueError("cache manifest file identity mismatch")
    if sha256_file(CACHE_VALIDATION) != config["cache_validation_sha256"]:
        raise ValueError("cache independent-validation identity mismatch")
    manifest = json.loads((cache_root / "manifest.json").read_text())
    validate_manifest(manifest)
    evaluator = FullPrefixHuginnD16Evaluator(
        huginn, None, base_seed=manifest["base_seed"], seed_index=0
    )
    records = []
    kl = TokenKLAggregator()
    compared_tokens = 0
    argmax_equal = 0
    all_state_exact = True
    all_logits_exact = True

    for example_id in config["cache_control_example_ids"]:
        path = cache_root / f"{example_id:05d}.npz"
        metadata = validate_item(path, manifest)
        with np.load(path, allow_pickle=False) as archive:
            ids = torch.from_numpy(archive["input_ids"].astype(np.int64))[None].to(device)
            cached_h0 = uint16_to_bfloat16_tensor(archive["h0"].copy(), device=device)[None]
            cached_x = torch.from_numpy(archive["x"].copy())[None].to(device)
            cached_h16 = torch.from_numpy(archive["h16"].copy())[None].to(device)
        schedule = evaluator.materialize_generation_schedule(ids, example_id=example_id)
        live_h0 = schedule[:, :ids.shape[1]]
        h0_exact = torch.equal(live_h0, cached_h0)
        captured = {}
        calls = 0
        original = huginn.core_block_forward
        def wrapped(this, state, recurrent_input, *args, **kwargs):
            nonlocal calls
            if calls == 0:
                captured["x"] = recurrent_input.detach().clone()
            calls += 1
            return original(state, recurrent_input, *args, **kwargs)
        huginn.core_block_forward = MethodType(wrapped, huginn)
        try:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                live_output = huginn(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids, dtype=torch.bool),
                    input_states=live_h0,
                    num_steps=16,
                    use_cache=False,
                    return_dict=True,
                    output_details={"return_logits": True, "return_latents": True, "return_head": False, "return_stats": False},
                )
                cached_logits = frozen_coda_logits_from_normalized_state(
                    huginn, cached_h16, huginn.freqs_cis[:, :ids.shape[1]]
                )
        finally:
            huginn.core_block_forward = original
        if calls != 16 or set(captured) != {"x"}:
            raise RuntimeError(f"Phase 13C expected exactly 16 core calls ID={example_id}")
        x_exact = torch.equal(captured["x"], cached_x)
        h16_exact = torch.equal(live_output.latent_states, cached_h16)
        start = metadata["answer_start"] - 1
        end = metadata["valid_end"] - 1
        live_logits = live_output.logits[:, start:end].float()
        replay_logits = cached_logits[:, start:end].float()
        logits_exact = torch.equal(live_logits, replay_logits)
        item_sum, item_count = token_kl_sum_and_count(replay_logits, live_logits)
        item_argmax = int((live_logits.argmax(-1) == replay_logits.argmax(-1)).sum())
        count = int(item_count.item())
        state_exact = h0_exact and x_exact and h16_exact
        all_state_exact &= state_exact
        all_logits_exact &= logits_exact
        compared_tokens += count
        argmax_equal += item_argmax
        kl.update(item_sum, item_count)
        records.append({
            "example_id": example_id,
            "sequence_length": int(ids.shape[1]),
            "represented_teacher_tokens": count,
            "h0_bfloat16_bits_exact": h0_exact,
            "x_float32_bits_exact": x_exact,
            "h16_float32_bits_exact": h16_exact,
            "coda_logits_exact": logits_exact,
            "kl_live_to_cached_per_token": _finite(float((item_sum / item_count).item())),
            "next_token_argmax_equal": item_argmax,
            "next_token_argmax_total": count,
            "recurrent_calls": calls,
        })
        del schedule, live_h0, live_output, cached_logits, live_logits, replay_logits
        torch.cuda.empty_cache()

    exact_gates = {
        "all_native_states_bitwise_equal": all_state_exact,
        "all_coda_logits_exact": all_logits_exact,
        "all_next_token_argmax_equal": argmax_equal == compared_tokens,
        "globally_token_normalized_kl_is_numerically_zero": (
            abs(kl.mean) <= config["native_cache_kl_absolute_tolerance"]
        ),
    }
    return {
        "passed": all(exact_gates.values()),
        "scope": config["cache_equivalence_scope"],
        "cache_manifest_sha256": config["cache_manifest_sha256"],
        "cache_validation_sha256": config["cache_validation_sha256"],
        "requirements": config["native_cache_exact_requirements"],
        "kl_absolute_tolerance": config["native_cache_kl_absolute_tolerance"],
        "exact_gates": exact_gates,
        "aggregate": {
            "globally_token_normalized_kl_live_to_cached": _finite(kl.mean),
            "next_token_argmax_equal": argmax_equal,
            "compared_tokens": compared_tokens,
            "next_token_argmax_agreement": argmax_equal / compared_tokens,
        },
        "records": records,
    }

def run(config: dict, commit: str) -> dict:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 13 canonical controls require CUDA")
    phase12_path = Path(config["phase12_evaluation"])
    if sha256_file(phase12_path) != config["phase12_evaluation_sha256"]:
        raise ValueError("corrected Phase 12 identity mismatch")
    phase12 = json.loads(phase12_path.read_text())
    if (
        phase12.get("status") != "pass"
        or phase12.get("phase11_model_sha256")
        != "caf5cd7d56a6491660ccdd7a0722407c2175ba5641faf7f462fac0cde2446731"
        or phase12.get("repetition_detected_count") != 0
        or phase12.get("phase11_text_exact_match_count") != 8
    ):
        raise ValueError("corrected Phase 12 is not authoritative")
    closure_path = ROOT / config["closure_audit"]
    if sha256_file(closure_path) != config["closure_audit_sha256"]:
        raise ValueError("upstream closure-audit identity mismatch")
    closure = json.loads(closure_path.read_text())
    if closure.get("status") != "pass" or not all(
        value.get("status") == "closed"
        for value in closure.get("closed_invariants", {}).values()
    ):
        raise ValueError("upstream closure audit is not complete")
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
    # Collect 13C even if it fails its frozen exact-equivalence gate; scientific failures still publish.
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
        "13C_native_cache_exact_equivalence": cache_controls["passed"],
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
        "phase13D_inference_limit": {
            "cache_scope": config["cache_equivalence_scope"],
            "attention2_judgment_route": "live fixed-2048-schedule full-prefix evaluation",
            "claim": "cache equivalence is limited to represented cached teacher positions and does not establish equivalence for arbitrary generated prefixes",
        },
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
    if config_path != CONFIG or output_root != OUTPUT_ROOT:
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
