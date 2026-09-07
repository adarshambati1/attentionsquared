"""Fixed-example GSM8K evaluation for the depth-scaling baseline."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from src.models.huginn import HuginnAdapter


_NUMBER = re.compile(r"####\s*([-+]?[\d,]+(?:\.\d+)?)")


def extract_answer(text: str) -> str | None:
    matches = _NUMBER.findall(text.replace("\u202f", ""))
    if not matches:
        return None
    return matches[-1].replace(",", "").strip()


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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as stream:
        for depth in config["depths"]:
            for example_id, example in zip(config["example_ids"], examples):
                adapter._synchronize()
                start = time.perf_counter()
                result = adapter.generate(
                    example["question"],
                    depth=depth,
                    system_instruction=config["system_instruction"],
                    max_new_tokens=config["max_new_tokens"],
                )
                adapter._synchronize()
                latency = time.perf_counter() - start
                gold = extract_answer(example["answer"])
                predicted = extract_answer(result.text)
                record = {
                    "example_id": example_id,
                    "depth": depth,
                    "model_answer": result.text,
                    "gold_answer": gold,
                    "predicted_answer": predicted,
                    "correct": predicted is not None and predicted == gold,
                    "generation_latency_seconds": latency,
                    "generated_tokens": result.generated_tokens,
                    "prompt_tokens": result.prompt_tokens,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(json.dumps(record, ensure_ascii=False))
