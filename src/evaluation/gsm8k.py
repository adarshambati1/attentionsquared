"""Fixed-example GSM8K evaluation for the depth-scaling baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.correctness import (
    SCORER_PROTOCOL,
    SEED_PROTOCOL,
    ensure_new_output_path,
    extract_answer,
    score_generation,
    seed_for_example,
)
from src.models.huginn import HuginnAdapter


def evaluate(config: dict[str, Any], output_path: Path) -> None:
    from datasets import load_dataset

    dataset = load_dataset(
        config["dataset_id"],
        config["dataset_config"],
        split=config["dataset_split"],
        revision=config["dataset_revision"],
    )
    examples = [dataset[index] for index in config["example_ids"]]
    adapter = HuginnAdapter(config["model_id"], config["model_revision"], config["dtype"])
    ensure_new_output_path(output_path)

    with output_path.open("x", encoding="utf-8") as stream:
        for depth in config["depths"]:
            for example_id, example in zip(config["example_ids"], examples):
                derived_seed = seed_for_example(config["seed"], example_id)
                result = adapter.generate(
                    example["question"],
                    depth=depth,
                    system_instruction=config["system_instruction"],
                    max_new_tokens=config["max_new_tokens"],
                )
                score = score_generation(
                    result.text,
                    example["answer"],
                    hit_max_new_tokens=result.hit_max_new_tokens,
                )
                record = {
                    "example_id": example_id,
                    "depth": depth,
                    "model_answer": result.text,
                    "gold_answer": score.gold_answer,
                    "predicted_answer": score.predicted_answer,
                    "correct": score.correct,
                    "generation_latency_seconds": result.generation_latency_seconds,
                    "time_to_first_token_seconds": result.time_to_first_token_seconds,
                    "single_forward_latency_seconds": result.single_forward_latency_seconds,
                    "generated_tokens": result.generated_tokens,
                    "generated_token_ids": list(result.generated_token_ids),
                    "scorer_protocol": SCORER_PROTOCOL,
                    "seed_protocol": SEED_PROTOCOL,
                    "base_seed": config["seed"],
                    "derived_seed": derived_seed,
                    "prompt_tokens": result.prompt_tokens,
                    "tokens_per_second": result.generated_tokens / result.generation_latency_seconds if result.generation_latency_seconds else None,
                    "seconds_per_generated_token": result.generation_latency_seconds / result.generated_tokens if result.generated_tokens else None,
                    "mean_decode_step_latency_seconds": result.generation_latency_seconds / result.generated_tokens if result.generated_tokens else None,
                    "ended_naturally": result.ended_naturally,
                    "hit_max_new_tokens": result.hit_max_new_tokens,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(json.dumps(record, ensure_ascii=False))
