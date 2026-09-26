"""Authoritative MATH-500 context-cap amendment rule."""
from __future__ import annotations
from typing import Any

def effective_math500_cap(*, dataset: str, prompt_tokens: int, config: dict[str, Any], amendment: dict[str, Any]) -> int:
    if dataset != "math500":
        raise ValueError("context-cap amendment is MATH-500-only")
    if amendment.get("scope") != "math500-only" or amendment.get("requested_max_new_tokens") != config.get("max_new_tokens") or amendment.get("maximum_total_tokens") != config.get("maximum_schedule_tokens"):
        raise ValueError("context-cap amendment/config mismatch")
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be a positive integer")
    effective = min(config["max_new_tokens"], config["maximum_schedule_tokens"] - prompt_tokens)
    if effective <= 0:
        raise ValueError("complete prompt exhausts the frozen context")
    return effective
