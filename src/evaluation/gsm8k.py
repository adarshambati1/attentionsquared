"""Fixed-example GSM8K evaluation for the depth-scaling baseline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.models.huginn import HuginnAdapter


_NUMERIC = r"[-+]?\$?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?"
_MARKED_NUMBER = re.compile(rf"####\s*({_NUMERIC})")
_EXPLICIT_NUMBER = re.compile(
    rf"(?:answer(?: is|:)|final answer(?: is|:)|therefore(?:,|\s+))[^\d$+-]{{0,48}}({_NUMERIC})",
    re.IGNORECASE,
)
_FALLBACK_NUMBER = re.compile(_NUMERIC)


def _normalize_number(value: str) -> str:
    return value.replace(",", "").replace("$", "").strip()


def extract_answer(text: str, *, allow_fallback: bool = True) -> str | None:
    text = text.replace("\u202f", "")
    matches = _MARKED_NUMBER.findall(text)
    if matches:
        return _normalize_number(matches[-1])
    explicit = _EXPLICIT_NUMBER.findall(text)
    if explicit:
        return _normalize_number(explicit[-1])
    # Natural-language answers often have no marker, but fallback extraction
    # is unsafe for a response truncated at max_new_tokens. In that case only
    # marked/explicit answers are eligible for exact-match scoring.
    if not allow_fallback:
        return None
    fallback = _FALLBACK_NUMBER.findall(text)
    return _normalize_number(fallback[-1]) if fallback else None


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
                result = adapter.generate(
                    example["question"],
                    depth=depth,
                    system_instruction=config["system_instruction"],
                    max_new_tokens=config["max_new_tokens"],
                )
                gold = extract_answer(example["answer"])
                predicted = extract_answer(result.text, allow_fallback=not result.hit_max_new_tokens)
                record = {
                    "example_id": example_id,
                    "depth": depth,
                    "model_answer": result.text,
                    "gold_answer": gold,
                    "predicted_answer": predicted,
                    "correct": predicted is not None and predicted == gold,
                    "generation_latency_seconds": result.generation_latency_seconds,
                    "time_to_first_token_seconds": result.time_to_first_token_seconds,
                    "single_forward_latency_seconds": result.single_forward_latency_seconds,
                    "generated_tokens": result.generated_tokens,
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
