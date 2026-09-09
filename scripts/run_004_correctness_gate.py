#!/usr/bin/env python3
"""Run the mandatory real-model correctness gate before functional training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.create_004_a2_initialization import write_json_exclusive_fsync
from src.data.functional_cache import validate_cache_item, validate_cache_manifest
from src.evaluation.correctness import (
    aligned_answer_logits_and_targets,
    answer_bounds,
)
from src.training.functional_objective import (
    any_parameter_gradient,
    causal_answer_loss_mask,
    freeze_module,
    functional_kl_loss,
    nonzero_gradient_parameter_count,
)
from src.training.functional_protocol import (
    Attention2Architecture,
    load_shared_initialization,
)


CACHE_ROOT = Path("/workspace/functional_cache_v2")
INITIALIZATION = Path("/workspace/functional_protocol/a2_init_seed_0.pt")
OUTPUT = Path("/workspace/functional_protocol/correctness_gate.json")
MODEL_ID = "tomg-group-umd/huginn-0125"
MODEL_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
PROTOCOL = "functional-correctness-gate-v1"
EXAMPLE_ID = 0


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
        raise RuntimeError("correctness gate requires a clean committed checkout")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def literal_alignment_control() -> dict:
    input_ids = torch.tensor([[9, 8, 7, 4, 5, 6]])
    logits = torch.full((1, 6, 10), -100.0)
    logits[0, 2, 4] = 100.0
    logits[0, 3, 5] = 100.0
    logits[0, 4, 6] = 100.0
    selected, targets = aligned_answer_logits_and_targets(
        logits, input_ids, answer_bounds(3, 6)
    )
    predictions = selected.argmax(-1)
    if not torch.equal(predictions, targets):
        raise RuntimeError("literal A/B/C causal alignment control failed")
    return {
        "answer_tokens": targets[0].tolist(),
        "predicting_logit_positions": [2, 3, 4],
        "predictions": predictions[0].tolist(),
        "passed": True,
    }


def coda_logits(model, input_ids, attention_mask, state):
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_states=state,
        num_steps=0,
        use_cache=False,
        return_dict=True,
    ).logits


def load_example(cache_root: Path, device: torch.device) -> dict:
    manifest = json.loads((cache_root / "manifest.json").read_text())
    validate_cache_manifest(manifest)
    path = cache_root / "train" / f"{EXAMPLE_ID:05d}.npz"
    metadata = validate_cache_item(path, manifest)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return {
        "path": path,
        "metadata": metadata,
        "input_ids": torch.from_numpy(arrays["input_ids"].astype(np.int64))[None].to(device),
        "attention_mask": torch.from_numpy(arrays["attention_mask"])[None].to(device),
        "h0": torch.from_numpy(arrays["h0_full"].astype(np.float32))[None].to(device),
        "x": torch.from_numpy(arrays["x_full"].astype(np.float32))[None].to(device),
        "h16_teacher": torch.from_numpy(
            arrays["h16_teacher"].astype(np.float32)
        )[None].to(device),
        "answer_start": int(arrays["answer_start"]),
        "valid_end": int(arrays["valid_end"]),
    }


def run_gate(cache_root: Path, initialization: Path) -> dict:
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("real-model correctness gate requires CUDA")
    device = torch.device("cuda")
    example = load_example(cache_root, device)
    input_ids = example["input_ids"]
    attention_mask = example["attention_mask"]
    loss_mask = causal_answer_loss_mask(
        attention_mask,
        [example["answer_start"]],
        [example["valid_end"]],
    )
    expected_count = example["valid_end"] - example["answer_start"]
    if int(loss_mask.sum().item()) != expected_count:
        raise RuntimeError("causal answer mask count is incorrect")
    if loss_mask[:, : example["answer_start"] - 1].any():
        raise RuntimeError("prompt logits entered the functional loss")
    if loss_mask[:, example["valid_end"] - 1 :].any():
        raise RuntimeError("post-valid_end logits entered the functional loss")

    huginn = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    freeze_module(huginn)
    with torch.no_grad():
        teacher_logits = coda_logits(
            huginn,
            input_ids,
            attention_mask,
            example["h16_teacher"].to(torch.bfloat16),
        ).detach()
        self_kl, _, self_count = functional_kl_loss(
            teacher_logits, teacher_logits, loss_mask
        )
    if abs(float(self_kl.item())) >= 1e-6:
        raise RuntimeError(f"teacher self-KL is not approximately zero: {self_kl.item()}")

    architecture = Attention2Architecture()
    baseline_model = load_shared_initialization(
        initialization, rounds=4, architecture=architecture
    ).to(device)
    baseline_model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        baseline_output, _, _ = baseline_model(
            example["h0"], example["x"], token_mask=attention_mask
        )
        baseline_logits = coda_logits(
            huginn,
            input_ids,
            attention_mask,
            baseline_output[:, 15].to(torch.bfloat16),
        )
        baseline_kl, _, baseline_count = functional_kl_loss(
            baseline_logits, teacher_logits, loss_mask
        )
    baseline_value = float(baseline_kl.item())
    if not np.isfinite(baseline_value) or baseline_value <= 0:
        raise RuntimeError("initial untrained A² KL/token must be finite and positive")
    del baseline_logits, baseline_output, baseline_model
    torch.cuda.empty_cache()

    a2 = load_shared_initialization(
        initialization, rounds=1, architecture=architecture
    ).to(device)
    a2.train()
    a2.zero_grad(set_to_none=True)
    huginn.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, _, _ = a2(example["h0"], example["x"], token_mask=attention_mask)
        z16 = output[:, 15]
        z16.retain_grad()
        student_logits = coda_logits(
            huginn, input_ids, attention_mask, z16.to(torch.bfloat16)
        )
        gradient_loss, _, gradient_count = functional_kl_loss(
            student_logits, teacher_logits, loss_mask
        )
    gradient_loss.backward()

    a2_nonzero = nonzero_gradient_parameter_count(a2.parameters())
    huginn_has_grad = any_parameter_gradient(huginn.parameters())
    if a2_nonzero == 0:
        raise RuntimeError("Attention² received no nonzero parameter gradients")
    if huginn_has_grad:
        raise RuntimeError("frozen Huginn/coda parameters received gradients")
    if z16.grad is None or not torch.isfinite(z16.grad).all() or not torch.count_nonzero(z16.grad):
        raise RuntimeError("loss gradient did not flow through frozen coda to z16")

    return {
        "protocol": PROTOCOL,
        "status": "pass",
        "example_id": EXAMPLE_ID,
        "cache_item": str(example["path"]),
        "sequence_length": int(input_ids.shape[1]),
        "answer_start": example["answer_start"],
        "valid_end": example["valid_end"],
        "valid_answer_token_count": expected_count,
        "loss_mask_count": int(loss_mask.sum().item()),
        "prompt_loss_positions": int(
            loss_mask[:, : example["answer_start"] - 1].sum().item()
        ),
        "post_valid_end_loss_positions": int(
            loss_mask[:, example["valid_end"] - 1 :].sum().item()
        ),
        "padding_loss_positions": int(loss_mask.masked_select(~attention_mask).sum().item()),
        "teacher_self_kl_per_token": float(self_kl.item()),
        "teacher_self_kl_token_count": int(self_count.item()),
        "untrained_a2_K4_kl_per_token": baseline_value,
        "untrained_a2_K4_token_count": int(baseline_count.item()),
        "gradient_gate_K": 1,
        "gradient_loss_per_token": float(gradient_loss.detach().item()),
        "gradient_token_count": int(gradient_count.item()),
        "a2_parameter_count": sum(parameter.numel() for parameter in a2.parameters()),
        "a2_parameters_with_nonzero_gradient": a2_nonzero,
        "huginn_parameter_count": sum(parameter.numel() for parameter in huginn.parameters()),
        "huginn_parameters_with_gradient": int(huginn_has_grad),
        "all_huginn_parameters_frozen": all(
            not parameter.requires_grad for parameter in huginn.parameters()
        ),
        "z16_gradient_nonzero_count": int(torch.count_nonzero(z16.grad).item()),
        "z16_gradient_l2": float(z16.grad.float().norm().item()),
        "teacher_state_requires_grad": bool(example["h16_teacher"].requires_grad),
        "teacher_logits_has_grad_fn": teacher_logits.grad_fn is not None,
    }


def main(cache_root: Path, initialization: Path, output: Path) -> dict:
    if (cache_root, initialization, output) != (CACHE_ROOT, INITIALIZATION, OUTPUT):
        raise ValueError("correctness gate requires exact production paths")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite correctness gate: {output}")
    commit = clean_git_commit()
    result = run_gate(cache_root, initialization)
    result.update(
        {
            "git_commit": commit,
            "initialization_sha256": sha256_file(initialization),
            "cache_freeze_sha256": sha256_file(cache_root / "FROZEN.json"),
            "literal_alignment_control": literal_alignment_control(),
            "padding_influence_control": "tests/test_attention2_padding.py",
        }
    )
    write_json_exclusive_fsync(output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--initialization", type=Path, default=INITIALIZATION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    main(arguments.cache_root, arguments.initialization, arguments.output)
