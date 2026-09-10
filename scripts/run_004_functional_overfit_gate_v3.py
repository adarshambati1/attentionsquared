#!/usr/bin/env python3
"""Run corrected Phase 11 coda-v3 overfit gate, then stop."""

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
import tempfile
import time
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.create_004_a2_initialization import write_json_exclusive_fsync
from src.data.functional_cache import validate_cache_item, validate_cache_manifest
from src.evaluation.correctness import (
    find_repetition_onset,
    score_generation,
)
from src.evaluation.functional_autoregressive import (
    FullPrefixAttention2Evaluator,
    frozen_coda_logits_from_normalized_state as normalized_state_coda_logits,
)
from src.training.functional_objective import (
    any_parameter_gradient,
    causal_answer_loss_mask,
    freeze_module,
    functional_kl_loss,
)
from src.training.functional_protocol import (
    Attention2Architecture,
    OptimizerSpec,
    load_shared_initialization,
    paired_epoch_order,
)
from src.training.overfit_gate import gate_passed, quantitative_overfit_checks


CACHE_ROOT = Path("/workspace/functional_cache_v2")
INITIALIZATION = Path("/workspace/functional_protocol/a2_init_seed_0.pt")
CONFIG = Path("configs/004_functional_v3_overfit.json")
OUTPUT_ROOT = Path("/workspace/functional_overfit_v3")
PHASE10_ARTIFACT = Path("/workspace/functional_protocol/correctness_gate_coda_v3.json")
PHASE10_ATTESTATION = ROOT / "results/004_functional/correctness_gate_coda_v3_runtime_attestation.json"
PHASE10_ARTIFACT_SHA256 = "a83561b945f88607554087ce79f4b690e79f6094afd3b05d25c07c5c1c539788"
PHASE10_ATTESTATION_SHA256 = "3c80f8d27d96542f50ef221d7ff35e06d9a9ebb2b9d08ad611de9a42a2e424be"
MODEL_ID = "tomg-group-umd/huginn-0125"
MODEL_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
OVERFIT_PROTOCOL = "functional-eight-example-overfit-coda-v3"
PRELAUNCH_PROTOCOL = "functional-eight-example-overfit-coda-v3-prelaunch-v1"


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
        raise RuntimeError("overfit gate requires a clean committed checkout")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def validate_config(config: dict) -> None:
    if set(config) != {
        "protocol",
        "training_protocol",
        "cache_freeze_sha256",
        "initialization_sha256",
        "phase10_artifact_sha256",
        "phase10_attestation_sha256",
        "example_ids",
        "K",
        "h0_seed_indices",
        "trajectory_loss_weight",
        "objective",
        "optimizer",
        "training",
        "pass_criteria",
        "generation",
        "output",
    }:
        raise ValueError("overfit config schema mismatch")
    fixed = {
        "protocol": OVERFIT_PROTOCOL,
        "training_protocol": "functional-paired-randomness-v1",
        "cache_freeze_sha256": "94417eab65fd04a5827bdef9aadc9a5b8b66c266700b6cfc7ab56d39f949d3f0",
        "initialization_sha256": "92968fea30d723864383f68f9e51ef1f8b8adae927e11acf735c3c1cf9b93747",
        "phase10_artifact_sha256": PHASE10_ARTIFACT_SHA256,
        "phase10_attestation_sha256": PHASE10_ATTESTATION_SHA256,
        "example_ids": list(range(8)),
        "K": 4,
        "h0_seed_indices": [0],
        "trajectory_loss_weight": 0.0,
        "objective": "global-teacher-to-student-KL-per-valid-answer-token",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 1e-4,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
        },
        "training": {
            "minimum_updates": 50,
            "maximum_updates": 300,
            "evaluate_every_updates": 10,
            "data_order_seed": 9100,
        },
        "pass_criteria": {
            "maximum_final_over_initial_kl_ratio": 0.2,
            "maximum_final_kl_per_token": 0.5,
            "maximum_first_token_distribution_kl_ratio": 0.5,
            "maximum_final_first_token_distribution_kl": 0.5,
            "minimum_final_mean_first_target_probability": 0.5,
            "minimum_final_first_target_top1_count": 6,
            "require_first_target_probability_increase": True,
            "require_first_target_top1_count_increase": True,
        },
        "generation": {
            "maximum_new_tokens": 384,
            "full_prefix_recomputation": True,
            "use_cache": False,
            "greedy": True,
            "minimum_natural_stop_count": 6,
            "minimum_mean_first_32_teacher_token_match_fraction": 0.5,
            "minimum_teacher_answer_match_count": 6,
            "require_all_nonempty": True,
            "require_no_detected_repetitive_degeneration": True,
        },
        "output": str(OUTPUT_ROOT),
    }
    if config != fixed:
        raise ValueError("overfit config does not exactly match the frozen gate")


def validate_phase10_prerequisites() -> dict:
    """Require the immutable corrected Phase 10 artifact and attestation."""
    if sha256_file(PHASE10_ARTIFACT) != PHASE10_ARTIFACT_SHA256:
        raise ValueError("Phase 10 coda-v3 artifact checksum mismatch")
    if sha256_file(PHASE10_ATTESTATION) != PHASE10_ATTESTATION_SHA256:
        raise ValueError("Phase 10 coda-v3 attestation checksum mismatch")
    artifact = json.loads(PHASE10_ARTIFACT.read_text(encoding="utf-8"))
    attestation = json.loads(PHASE10_ATTESTATION.read_text(encoding="utf-8"))
    if artifact.get("protocol") != "functional-correctness-gate-coda-v3" or artifact.get("status") != "pass":
        raise ValueError("Phase 10 coda-v3 artifact is not an authoritative pass")
    if (
        attestation.get("protocol")
        != "functional-correctness-gate-coda-v3-runtime-attestation-v1"
        or attestation.get("status") != "pass"
        or attestation.get("artifact") != str(PHASE10_ARTIFACT)
        or attestation.get("artifact_sha256") != PHASE10_ARTIFACT_SHA256
        or attestation.get("is_symlink") is not False
        or attestation.get("lstat_type") != "regular-file"
        or attestation.get("mode") != "0444"
    ):
        raise ValueError("Phase 10 coda-v3 attestation is invalid")
    return attestation


def load_examples(device: torch.device) -> list[dict]:
    manifest = json.loads((CACHE_ROOT / "manifest.json").read_text())
    validate_cache_manifest(manifest)
    examples = []
    for example_id in range(8):
        path = CACHE_ROOT / "train" / f"{example_id:05d}.npz"
        validate_cache_item(path, manifest)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        attention = torch.from_numpy(arrays["attention_mask"])[None].to(device)
        start = int(arrays["answer_start"])
        end = int(arrays["valid_end"])
        examples.append(
            {
                "example_id": example_id,
                "input_ids": torch.from_numpy(arrays["input_ids"].astype(np.int64))[
                    None
                ].to(device),
                "attention_mask": attention,
                "h0": torch.from_numpy(arrays["h0_full"].astype(np.float32))[
                    None
                ].to(device),
                "x": torch.from_numpy(arrays["x_full"].astype(np.float32))[None].to(
                    device
                ),
                "h16_teacher": torch.from_numpy(
                    arrays["h16_teacher"].astype(np.float32)
                )[None].to(device),
                "answer_start": start,
                "valid_end": end,
                "loss_mask": causal_answer_loss_mask(attention, [start], [end]),
            }
        )
    return examples


def coda_logits(huginn, example: dict, normalized_state: torch.Tensor) -> torch.Tensor:
    """Apply the sole authoritative coda to an already-normalized state."""
    frequencies = huginn.freqs_cis[:, : example["input_ids"].shape[1]]
    return normalized_state_coda_logits(huginn, normalized_state, frequencies)


def transient_teacher_logits(huginn, example: dict) -> torch.Tensor:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return coda_logits(
            huginn, example, example["h16_teacher"].to(torch.bfloat16)
        ).detach()


def compute_metrics(a2, huginn, examples: list[dict]) -> dict:
    numerator = 0.0
    count = 0
    first_distribution_kl = 0.0
    first_target_probability = 0.0
    first_target_top1_count = 0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for example in examples:
            teacher = transient_teacher_logits(huginn, example)
            output, _, _ = a2(
                example["h0"],
                example["x"],
                token_mask=example["attention_mask"],
            )
            student = coda_logits(huginn, example, output[:, 15].to(torch.bfloat16))
            _, item_sum, item_count = functional_kl_loss(
                student, teacher, example["loss_mask"]
            )
            numerator += float(item_sum.item())
            count += int(item_count.item())
            position = example["answer_start"] - 1
            target = int(example["input_ids"][0, example["answer_start"]].item())
            teacher_first = teacher[:, position].float()
            student_first = student[:, position].float()
            first_distribution_kl += float(
                torch.nn.functional.kl_div(
                    torch.log_softmax(student_first, -1),
                    torch.softmax(teacher_first, -1),
                    reduction="sum",
                ).item()
            )
            probabilities = torch.softmax(student_first, -1)
            first_target_probability += float(probabilities[0, target].item())
            first_target_top1_count += int(student_first.argmax(-1).item() == target)
            del student, teacher, output
    return {
        "kl_per_token": numerator / count,
        "kl_numerator": numerator,
        "valid_token_count": count,
        "first_token_distribution_kl": first_distribution_kl / len(examples),
        "mean_first_target_probability": first_target_probability / len(examples),
        "first_target_top1_count": first_target_top1_count,
    }


def train(a2, huginn, examples: list[dict], config: dict) -> tuple[list[dict], dict]:
    optimizer_spec = OptimizerSpec()
    optimizer = optimizer_spec.build(a2.parameters())
    total_tokens = sum(int(example["loss_mask"].sum().item()) for example in examples)
    trace = []
    initial = compute_metrics(a2, huginn, examples)
    trace.append({"update": 0, **initial})
    print(json.dumps(trace[-1], sort_keys=True), flush=True)
    final = initial
    criteria = config["pass_criteria"]
    training = config["training"]
    for update in range(1, training["maximum_updates"] + 1):
        optimizer.zero_grad(set_to_none=True)
        order = paired_epoch_order(
            range(8),
            epoch=(update - 1) % 128,
            order_seed=training["data_order_seed"],
        )
        for index in order:
            example = examples[index]
            teacher = transient_teacher_logits(huginn, example)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output, _, _ = a2(
                    example["h0"],
                    example["x"],
                    token_mask=example["attention_mask"],
                )
                student = coda_logits(
                    huginn, example, output[:, 15].to(torch.bfloat16)
                )
                _, item_sum, _ = functional_kl_loss(
                    student, teacher, example["loss_mask"]
                )
                scaled_loss = item_sum / total_tokens
            scaled_loss.backward()
            del student, teacher, output, scaled_loss
        torch.nn.utils.clip_grad_norm_(a2.parameters(), config["optimizer"]["gradient_clip_norm"])
        optimizer.step()
        if update % training["evaluate_every_updates"] == 0:
            final = compute_metrics(a2, huginn, examples)
            trace.append({"update": update, **final})
            print(json.dumps(trace[-1], sort_keys=True), flush=True)
            checks = quantitative_overfit_checks(
                initial,
                final,
                maximum_kl_ratio=criteria["maximum_final_over_initial_kl_ratio"],
                maximum_final_kl=criteria["maximum_final_kl_per_token"],
                maximum_first_distribution_kl_ratio=criteria[
                    "maximum_first_token_distribution_kl_ratio"
                ],
                maximum_final_first_distribution_kl=criteria[
                    "maximum_final_first_token_distribution_kl"
                ],
                minimum_final_first_target_probability=criteria[
                    "minimum_final_mean_first_target_probability"
                ],
                minimum_final_first_target_top1_count=criteria[
                    "minimum_final_first_target_top1_count"
                ],
            )
            if update >= training["minimum_updates"] and gate_passed(checks):
                return trace, checks
    checks = quantitative_overfit_checks(
        initial,
        final,
        maximum_kl_ratio=criteria["maximum_final_over_initial_kl_ratio"],
        maximum_final_kl=criteria["maximum_final_kl_per_token"],
        maximum_first_distribution_kl_ratio=criteria[
            "maximum_first_token_distribution_kl_ratio"
        ],
        maximum_final_first_distribution_kl=criteria[
            "maximum_final_first_token_distribution_kl"
        ],
        minimum_final_first_target_probability=criteria[
            "minimum_final_mean_first_target_probability"
        ],
        minimum_final_first_target_top1_count=criteria[
            "minimum_final_first_target_top1_count"
        ],
    )
    return trace, checks


def fixed_width_teacher_prefix_match(
    generated: list[int], teacher_tokens: list[int], width: int = 32
) -> float:
    denominator = min(width, len(teacher_tokens))
    if denominator == 0:
        return 0.0
    matches = sum(
        index < len(generated) and generated[index] == teacher_tokens[index]
        for index in range(denominator)
    )
    return matches / denominator


def generate_same_examples(a2, huginn, tokenizer, examples: list[dict], max_new: int) -> list[dict]:
    """Generate only through the corrected canonical full-prefix A² route."""
    evaluator = FullPrefixAttention2Evaluator(
        huginn, a2, tokenizer, base_seed=3000, seed_index=0
    )
    generations = []
    a2.eval()
    for example in examples:
        prompt = example["input_ids"][:, : example["answer_start"]]
        result = evaluator.generate(
            prompt, example_id=example["example_id"], max_new_tokens=max_new
        )
        generated = list(result.token_ids)
        teacher_tokens = example["input_ids"][
            0, example["answer_start"] : example["valid_end"]
        ].tolist()
        prefix_match = fixed_width_teacher_prefix_match(generated, teacher_tokens)
        substantive_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        teacher_text = tokenizer.decode(teacher_tokens, skip_special_tokens=False)
        score = score_generation(
            result.text,
            teacher_text,
            hit_max_new_tokens=result.hit_max_new_tokens,
        )
        generations.append(
            {
                "example_id": example["example_id"],
                "generated_tokens": len(generated),
                "generated_token_ids": generated,
                "teacher_valid_answer_tokens": len(teacher_tokens),
                "ended_naturally": result.ended_naturally,
                "hit_max_new_tokens": result.hit_max_new_tokens,
                "full_prefix_lengths": list(result.prefix_lengths),
                "full_prefix_traces": [
                    {
                        "step": trace.step,
                        "prefix_length": trace.prefix_length,
                        "prefix_token_ids_sha256": trace.prefix_token_ids_sha256,
                    }
                    for trace in result.prefix_traces
                ],
                "first_32_teacher_token_match_fraction": prefix_match,
                "repetition_onset": find_repetition_onset(
                    generated, chunk_size=8, repetitions=3
                ),
                "generated_answer": score.predicted_answer,
                "teacher_answer": score.gold_answer,
                "teacher_answer_match": score.correct,
                "fallback_allowed": score.fallback_allowed,
                "nonempty_text": bool(substantive_text),
                "substantive_text": substantive_text,
                "text": result.text,
                "teacher_text": teacher_text,
            }
        )
    return generations


def save_model_atomic(path: Path, a2, metadata: dict) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite overfit model: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(
            {
                "protocol": OVERFIT_PROTOCOL,
                "state_dict": {k: v.detach().cpu() for k, v in a2.state_dict().items()},
                "metadata": metadata,
            },
            temporary,
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        reopened = torch.load(temporary, map_location="cpu", weights_only=True)
        if (
            set(reopened) != {"protocol", "state_dict", "metadata"}
            or reopened["protocol"] != OVERFIT_PROTOCOL
            or set(reopened["state_dict"]) != set(a2.state_dict())
            or reopened["metadata"] != metadata
        ):
            raise RuntimeError("overfit model failed post-write validation")
        del reopened
        os.chmod(temporary, 0o444)
        if stat.S_IMODE(temporary.stat().st_mode) != 0o444:
            raise RuntimeError("overfit model is not read-only before publication")
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_publish_attempt(attempt: Path, output_root: Path) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to replace final overfit output: {output_root}")
    for name in ("model.pt", "result.json"):
        path = attempt / name
        if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise RuntimeError(f"attempt output is not a read-only regular file: {path}")
    directory = os.open(attempt, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    os.chmod(attempt, 0o555)
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(
        -100,
        os.fsencode(attempt),
        -100,
        os.fsencode(output_root),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output_root))
    parent = os.open(output_root.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def validate_prelaunch_attestation(path: Path, commit: str) -> dict:
    attempts_root = OUTPUT_ROOT.parent / f"{OUTPUT_ROOT.name}_attempts"
    if (
        path.is_symlink()
        or not path.is_file()
        or path.parent.resolve() != attempts_root.resolve()
        or not path.name.startswith("prelaunch-")
        or stat.S_IMODE(path.stat().st_mode) != 0o444
    ):
        raise ValueError("invalid prelaunch attestation path, type, or mode")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("protocol") != PRELAUNCH_PROTOCOL
        or value.get("status") != "pass"
        or value.get("git_commit") != commit
        or value.get("final_output") != str(OUTPUT_ROOT)
        or value.get("final_output_absent") is not True
        or value.get("matching_training_processes") != []
        or value.get("cache_freeze_sha256")
        != sha256_file(CACHE_ROOT / "FROZEN.json")
        or value.get("initialization_sha256") != sha256_file(INITIALIZATION)
        or value.get("phase10_artifact_sha256") != PHASE10_ARTIFACT_SHA256
        or value.get("phase10_attestation_sha256") != PHASE10_ATTESTATION_SHA256
    ):
        raise ValueError("prelaunch attestation does not match this gate")
    age = time.time() - float(value["created_unix_seconds"])
    if age < 0 or age > 300:
        raise ValueError("prelaunch attestation is not recent")
    return value


def main(config_path: Path, output_root: Path, prelaunch_attestation: Path) -> dict:
    if config_path != CONFIG or output_root != OUTPUT_ROOT:
        raise ValueError("overfit gate requires exact production paths")
    commit = clean_git_commit()
    prelaunch = validate_prelaunch_attestation(prelaunch_attestation, commit)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)
    phase10_attestation = validate_phase10_prerequisites()
    if sha256_file(CACHE_ROOT / "FROZEN.json") != config["cache_freeze_sha256"]:
        raise ValueError("cache freeze checksum mismatch")
    if sha256_file(INITIALIZATION) != config["initialization_sha256"]:
        raise ValueError("shared initialization checksum mismatch")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to replace final overfit output: {output_root}")
    attempts_root = output_root.parent / f"{output_root.name}_attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    attempt_id = uuid.uuid4().hex
    # Keep the staging directory beside the final path: Runpod's durable FUSE
    # mount supports atomic no-replace rename within one parent, but rejected
    # the original cross-directory publication after a complete first run.
    attempt = output_root.parent / f".{output_root.name}.attempt-{attempt_id}"
    attempt.mkdir(exist_ok=False)
    print(f"preserved attempt directory: {attempt}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    examples = load_examples(device)
    huginn = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    freeze_module(huginn)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    a2 = load_shared_initialization(
        INITIALIZATION, rounds=config["K"], architecture=Attention2Architecture()
    ).to(device)
    a2.train()
    trace, checks = train(a2, huginn, examples, config)
    quantitative_pass = gate_passed(checks)
    final_metrics = {key: value for key, value in trace[-1].items() if key != "update"}
    torch.cuda.empty_cache()

    generations = generate_same_examples(
        a2, huginn, tokenizer, examples, config["generation"]["maximum_new_tokens"]
    )
    generation_config = config["generation"]
    natural_stop_count = sum(item["ended_naturally"] for item in generations)
    cap_hit_count = sum(item["hit_max_new_tokens"] for item in generations)
    mean_prefix_match = sum(
        item["first_32_teacher_token_match_fraction"] for item in generations
    ) / len(generations)
    teacher_answer_match_count = sum(
        item["teacher_answer_match"] for item in generations
    )
    generation_mechanical_checks = {
        "all_nonempty": (
            all(item["nonempty_text"] for item in generations)
            if generation_config["require_all_nonempty"]
            else True
        ),
        "no_detected_repetitive_degeneration": (
            all(item["repetition_onset"] is None for item in generations)
            if generation_config["require_no_detected_repetitive_degeneration"]
            else True
        ),
        "natural_stop_count_is_substantive": (
            natural_stop_count >= generation_config["minimum_natural_stop_count"]
        ),
        "teacher_prefix_agreement_is_substantive": (
            mean_prefix_match
            >= generation_config[
                "minimum_mean_first_32_teacher_token_match_fraction"
            ]
        ),
        "teacher_answer_agreement_is_substantive": (
            teacher_answer_match_count
            >= generation_config["minimum_teacher_answer_match_count"]
        ),
    }
    automated_pass = quantitative_pass and all(generation_mechanical_checks.values())
    result = {
        "protocol": OVERFIT_PROTOCOL,
        "status": "pending_scientific_review" if automated_pass else "fail",
        "explicit_scientific_generation_review_required": True,
        "git_commit": commit,
        "prelaunch_attestation": str(prelaunch_attestation),
        "prelaunch_created_unix_seconds": prelaunch["created_unix_seconds"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "cache_freeze_sha256": config["cache_freeze_sha256"],
        "initialization_sha256": config["initialization_sha256"],
        "phase10_artifact_sha256": config["phase10_artifact_sha256"],
        "phase10_attestation_sha256": config["phase10_attestation_sha256"],
        "phase10_git_commit": phase10_attestation["git_commit"],
        "authoritative_coda_semantics": "already-normalized state -> coda blocks -> final ln_f -> lm_head; no extra initial ln_f",
        "example_ids": config["example_ids"],
        "K": config["K"],
        "trajectory_loss_weight": config["trajectory_loss_weight"],
        "objective": config["objective"],
        "updates_completed": trace[-1]["update"],
        "initial_metrics": {key: value for key, value in trace[0].items() if key != "update"},
        "final_metrics": final_metrics,
        "quantitative_checks": checks,
        "generation_mechanical_checks": generation_mechanical_checks,
        "generation_summary": {
            "natural_stop_count": natural_stop_count,
            "cap_hit_count": cap_hit_count,
            "mean_first_32_teacher_token_match_fraction": mean_prefix_match,
            "teacher_answer_match_count": teacher_answer_match_count,
        },
        "generations": generations,
        "training_trace": trace,
        "huginn_parameters_with_gradient": int(any_parameter_gradient(huginn.parameters())),
        "full_vocabulary_logits_persisted": False,
    }
    result["attempt_id"] = attempt_id
    save_model_atomic(attempt / "model.pt", a2, result)
    result["model_sha256"] = sha256_file(attempt / "model.pt")
    write_json_exclusive_fsync(attempt / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not automated_pass:
        raise RuntimeError(f"eight-example functional overfit gate failed; preserved {attempt}")
    atomic_publish_attempt(attempt, output_root)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--prelaunch-attestation", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.config, arguments.output, arguments.prelaunch_attestation)
