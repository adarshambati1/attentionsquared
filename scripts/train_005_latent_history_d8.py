#!/usr/bin/env python3
"""Train only latent-history attention on GSM8K provided solutions at Huginn D=8."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.latent_history_huginn import LatentHistoryHuginn
from src.training.latent_history import answer_cross_entropy, encode_supervised_example, generate_cached, materialize_h0_schedule

CONFIG = Path("configs/005_latent_history_attention_d8.json")


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_torch_save(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def encode(dataset, tokenizer, example_id: int, config: dict):
    item = encode_supervised_example(tokenizer, dataset[example_id]["question"], dataset[example_id]["answer"], config["system_instruction"])
    if len(item.input_ids) > config["maximum_sequence_tokens"]:
        raise ValueError(f"provided solution exceeds frozen 2048-token limit ID={example_id}")
    return item


def validation_loss(wrapper, huginn, tokenizer, dataset, ids, config) -> tuple[float, int]:
    wrapper.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for example_id in ids:
            item = encode(dataset, tokenizer, example_id, config)
            input_ids = item.input_ids[None].cuda()
            schedule = materialize_h0_schedule(huginn, device=input_ids.device, example_id=example_id, base_seed=config["h0_base_seed"])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = wrapper(input_ids, schedule[:, : input_ids.shape[1]], depth=8, mode="learned")
                loss, count = answer_cross_entropy(output.logits, input_ids, item.answer_start)
            total_loss += float(loss) * count
            total_tokens += count
            del input_ids, schedule, output, loss
    return total_loss / total_tokens, total_tokens


def checkpoint_payload(wrapper, optimizer, step, validation, config_path):
    return {
        "protocol": "latent-history-attention-d8-checkpoint-v1",
        "step": step,
        "validation_answer_token_cross_entropy": validation,
        "history_attention_state_dict": {k: v.detach().cpu() for k, v in wrapper.history_attention.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "git_commit": git_commit(),
        "config_sha256": config_sha256(config_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=CONFIG); args = parser.parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    output = Path(config["output_root"])
    if output.exists():
        if output.is_symlink() or {path.name for path in output.iterdir()} != {"correctness_gate.json"}:
            raise FileExistsError(f"output must contain only the immutable correctness gate: {output}")
    else:
        output.mkdir(parents=True)
    (output / "checkpoints").mkdir(exist_ok=False)
    curve = output / "training_curve.jsonl"
    if curve.exists() or (output / "best.pt").exists() or (output / "training_summary.json").exists():
        raise FileExistsError("refusing to replace an existing training artifact")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"], local_files_only=True)
    dataset = load_dataset(config["dataset_id"], config["dataset_config"], split="train", revision=config["dataset_revision"])
    huginn = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["model_revision"], torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True).eval().cuda()
    wrapper = LatentHistoryHuginn(huginn, config["projection_size"], config["attention_heads"]).cuda().train()
    optimizer = torch.optim.AdamW(wrapper.trainable_parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    train_ids = list(range(config["train_ids"][0], config["train_ids"][1] + 1))
    validation_ids = list(range(config["validation_ids"][0], config["validation_ids"][1] + 1))
    rng = random.Random(config["training_shuffle_seed"])
    order: list[int] = []
    cursor = 0
    best_validation = math.inf
    best_step = None
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, config["optimizer_steps"] + 1):
        wrapper.train()
        accumulated_loss = 0.0
        batch = []
        for _ in range(config["gradient_accumulation_examples"]):
            if cursor >= len(order):
                order = train_ids.copy(); rng.shuffle(order); cursor = 0
            example_id = order[cursor]; cursor += 1
            batch.append((example_id, encode(dataset, tokenizer, example_id, config)))
        accumulated_tokens = sum(len(item.input_ids) - item.answer_start for _, item in batch)
        for example_id, item in batch:
            input_ids = item.input_ids[None].cuda()
            schedule = materialize_h0_schedule(huginn, device=input_ids.device, example_id=example_id, base_seed=config["h0_base_seed"])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_value = wrapper(input_ids, schedule[:, : input_ids.shape[1]], depth=8, mode="learned")
                loss, target_count = answer_cross_entropy(output_value.logits, input_ids, item.answer_start)
            (loss * (target_count / accumulated_tokens)).backward()
            accumulated_loss += float(loss.detach()) * target_count
            del input_ids, schedule, output_value, loss
        grad_norm = float(torch.nn.utils.clip_grad_norm_(list(wrapper.trainable_parameters()), config["gradient_clip_norm"]))
        if not math.isfinite(grad_norm) or not math.isfinite(accumulated_loss):
            raise RuntimeError(f"nonfinite training state at step {step}")
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        record = {"type": "train", "step": step, "answer_token_cross_entropy": accumulated_loss / accumulated_tokens, "answer_tokens": accumulated_tokens, "gradient_norm": grad_norm, "elapsed_seconds": time.time() - started}
        with curve.open("a") as stream:
            stream.write(json.dumps(record) + "\n"); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)

        if step == config["startup_gate_step"]:
            sample_ids = validation_ids[: config["startup_validation_examples"]]
            startup_loss, startup_tokens = validation_loss(wrapper, huginn, tokenizer, dataset, sample_ids, config)
            prompt_item = encode(dataset, tokenizer, validation_ids[0], config)
            prompt_ids = prompt_item.input_ids[: prompt_item.answer_start][None].cuda()
            schedule = materialize_h0_schedule(huginn, device=prompt_ids.device, example_id=validation_ids[0], base_seed=config["h0_base_seed"])
            generation = generate_cached(wrapper, tokenizer, prompt_ids, schedule, depth=8, mode="learned", max_new_tokens=32)
            startup = {"type": "startup_gate", "step": step, "validation_loss": startup_loss, "validation_tokens": startup_tokens, "generation_tokens": generation.generated_tokens, "finite": math.isfinite(startup_loss), "checkpoint_write_test": True}
            test_path = output / "checkpoints" / "startup_step_00010.pt"
            atomic_torch_save(test_path, checkpoint_payload(wrapper, optimizer, step, startup_loss, config_path))
            with curve.open("a") as stream:
                stream.write(json.dumps(startup) + "\n"); stream.flush(); os.fsync(stream.fileno())
            print(json.dumps(startup), flush=True)
            if not startup["finite"] or generation.generated_tokens < 1:
                raise RuntimeError("10-step startup gate failed")
            del prompt_ids, schedule

        if step % config["validation_every_steps"] == 0:
            value, tokens = validation_loss(wrapper, huginn, tokenizer, dataset, validation_ids, config)
            validation_record = {"type": "validation", "step": step, "answer_token_cross_entropy": value, "answer_tokens": tokens, "elapsed_seconds": time.time() - started}
            with curve.open("a") as stream:
                stream.write(json.dumps(validation_record) + "\n"); stream.flush(); os.fsync(stream.fileno())
            print(json.dumps(validation_record), flush=True)
            checkpoint = checkpoint_payload(wrapper, optimizer, step, value, config_path)
            atomic_torch_save(output / "checkpoints" / f"step_{step:05d}.pt", checkpoint)
            if value < best_validation:
                best_validation = value; best_step = step
                atomic_torch_save(output / "best.pt", checkpoint)
    summary = {
        "protocol": "latent-history-attention-d8-training-v1",
        "status": "complete",
        "optimizer_steps": config["optimizer_steps"],
        "best_validation_answer_token_cross_entropy": best_validation,
        "best_step": best_step,
        "selection_criterion": config["selection_criterion"],
        "training_seconds": time.time() - started,
        "git_commit": git_commit(),
        "config_sha256": config_sha256(config_path),
        "test_set_used_for_selection": False,
        "huginn_parameters_trained": False,
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
