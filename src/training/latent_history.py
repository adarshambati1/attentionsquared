"""Data, loss, and generation utilities for latent-history Huginn."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from src.evaluation.correctness import (
    build_chat_prompt,
    generation_status,
    stop_token_ids,
    tokenize_prompt,
)
from src.training.functional_protocol import fixed_huginn_h0_schedule


@dataclass(frozen=True)
class SupervisedExample:
    input_ids: torch.Tensor
    answer_start: int


def encode_supervised_example(tokenizer: Any, question: str, answer: str, system_instruction: str) -> SupervisedExample:
    prompt = build_chat_prompt(tokenizer, question, system_instruction)
    prompt_ids = tokenize_prompt(tokenizer, prompt)["input_ids"][0]
    answer_text = answer.strip() + "<|end_turn|>"
    answer_ids = tokenizer(answer_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    full = torch.cat((prompt_ids, answer_ids), dim=0)
    if not torch.equal(full[: len(prompt_ids)], prompt_ids):
        raise RuntimeError("answer formatting changed the evaluation prompt prefix")
    return SupervisedExample(full, len(prompt_ids))


def materialize_h0_schedule(huginn: Any, *, device: torch.device, example_id: int, base_seed: int = 3000) -> torch.Tensor:
    hidden = int(huginn.config.n_embd)
    template = torch.empty((1, 2048, hidden), device=device, dtype=torch.bfloat16)
    schedule, _ = fixed_huginn_h0_schedule(huginn, template, base_seed=base_seed, example_id=example_id, seed_index=0)
    return schedule


def answer_cross_entropy(logits: torch.Tensor, input_ids: torch.Tensor, answer_start: int) -> tuple[torch.Tensor, int]:
    if not 1 <= answer_start < input_ids.shape[1]:
        raise ValueError("answer_start must leave at least one answer target")
    selected_logits = logits[:, answer_start - 1 : input_ids.shape[1] - 1]
    targets = input_ids[:, answer_start:]
    loss_sum = F.cross_entropy(selected_logits.reshape(-1, selected_logits.shape[-1]), targets.reshape(-1), reduction="sum")
    return loss_sum / targets.numel(), int(targets.numel())


def full_prefix_next_logits(model: Any, prefix_ids: torch.Tensor, h0_schedule: torch.Tensor, *, depth: int, mode: str) -> torch.Tensor:
    output = model(prefix_ids, h0_schedule[:, : prefix_ids.shape[1]], depth=depth, mode=mode)
    return output.logits[:, -1]


@dataclass
class CachedGeneration:
    token_ids: list[int]
    text: str
    latency_seconds: float
    generated_tokens: int
    hit_max_new_tokens: bool
    ended_naturally: bool
    peak_memory_bytes: int


def generate_cached(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    h0_schedule: torch.Tensor,
    *,
    depth: int,
    mode: str,
    max_new_tokens: int,
) -> CachedGeneration:
    device = prompt_ids.device
    stops = {int(value) for value in stop_token_ids(tokenizer) if value is not None and int(value) >= 0}
    generated: list[int] = []
    cache = None
    current_input = prompt_ids
    position = None
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(max_new_tokens):
            if position is None:
                states = h0_schedule[:, : current_input.shape[1]]
                cache_position = None
            else:
                states = h0_schedule[:, position : position + 1]
                cache_position = torch.tensor([position], device=device, dtype=torch.long)
            output = model(current_input, states, depth=depth, mode=mode, past_key_values=cache, use_cache=True, cache_position=cache_position)
            if not bool(torch.isfinite(output.logits).all()):
                raise FloatingPointError("non-finite logits during cached generation")
            if hasattr(output, "latent_states") and not bool(torch.isfinite(output.latent_states).all()):
                raise FloatingPointError("non-finite latent state during cached generation")
            cache = output.past_key_values
            token = int(output.logits[0, -1].argmax().item())
            generated.append(token)
            if token in stops:
                break
            position = prompt_ids.shape[1] + len(generated) - 1
            current_input = torch.tensor([[token]], device=device, dtype=prompt_ids.dtype)
    torch.cuda.synchronize(device)
    latency = time.perf_counter() - start
    status = generation_status(generated, max_new_tokens=max_new_tokens, stop_token_ids=stop_token_ids(tokenizer))
    return CachedGeneration(
        token_ids=generated,
        text=tokenizer.decode(generated, skip_special_tokens=False),
        latency_seconds=latency,
        generated_tokens=len(generated),
        hit_max_new_tokens=status.hit_max_new_tokens,
        ended_naturally=status.ended_naturally,
        peak_memory_bytes=int(torch.cuda.max_memory_allocated(device)),
    )


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
