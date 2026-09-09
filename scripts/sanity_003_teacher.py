#!/usr/bin/env python3
"""Re-run the frozen Huginn D16 teacher control with shared Phase 1 mechanics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.evaluation.correctness import (
    CORRECTED_V2_PROTOCOL,
    SCORER_PROTOCOL,
    SEED_PROTOCOL,
    build_chat_prompt,
    prepare_corrected_v2_output,
    generation_config_kwargs,
    generation_status,
    score_generation,
    seed_for_example,
    stop_token_ids,
    synchronized_cuda_timer,
    tokenize_prompt,
    write_json_exclusive,
)


def main(cfg_path: Path, output: Path) -> None:
    prepare_corrected_v2_output(output)
    cfg = json.loads(cfg_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"]
    )
    dataset = load_dataset(
        cfg["dataset_id"],
        cfg["dataset_config"],
        split=cfg["dataset_split"],
        revision=cfg["dataset_revision"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        revision=cfg["model_revision"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval().cuda()
    correct = 0
    latencies: list[float] = []
    cap_hits = 0
    max_new_tokens = int(cfg.get("max_new_tokens", 1024))
    start_id, end_id = cfg["splits"]["test"]
    for number, example_id in enumerate(range(start_id, end_id + 1), 1):
        seed_for_example(cfg["seed"], example_id)
        prompt = build_chat_prompt(
            tokenizer, dataset[example_id]["question"], cfg["system_instruction"]
        )
        encoded = {
            key: value.cuda()
            for key, value in tokenize_prompt(tokenizer, prompt).items()
        }
        with synchronized_cuda_timer("cuda") as timing:
            with torch.inference_mode():
                generated_output = model.generate(
                    **encoded,
                    generation_config=GenerationConfig(
                        **generation_config_kwargs(tokenizer, max_new_tokens)
                    ),
                    num_steps=16,
                    tokenizer=tokenizer,
                )
        latencies.append(timing.seconds)
        generated = generated_output.sequences[0][encoded["input_ids"].shape[-1] :]
        status = generation_status(
            generated.tolist(),
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids(tokenizer),
        )
        cap_hits += int(status.hit_max_new_tokens)
        text = tokenizer.decode(generated, skip_special_tokens=False)
        score = score_generation(
            text,
            dataset[example_id]["answer"],
            hit_max_new_tokens=status.hit_max_new_tokens,
        )
        correct += int(score.correct)
        if number % 10 == 0:
            print(f"teacher generation {number}/{end_id - start_id + 1}", flush=True)
    examples = end_id - start_id + 1
    result = {
        "model": "huginn_d16_teacher",
        "protocol": CORRECTED_V2_PROTOCOL,
        "seed_protocol": SEED_PROTOCOL,
        "base_seed": cfg["seed"],
        "model_revision": cfg["model_revision"],
        "dataset_revision": cfg["dataset_revision"],
        "examples": examples,
        "gsm8k_accuracy": correct / examples,
        "cap_hits": cap_hits,
        "mean_generation_latency_seconds": sum(latencies) / len(latencies),
        "timing_protocol": "cuda-synchronized-wall-clock-v1",
        "scorer_protocol": SCORER_PROTOCOL,
        "scientific_validity": "teacher_control; no Attention2 execution",
    }
    write_json_exclusive(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/003_direct_jump.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/003_direct_jump/corrected-v2/teacher_control.json"),
    )
    arguments = parser.parse_args()
    main(arguments.config, arguments.output)
