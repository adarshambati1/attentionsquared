"""Causal masked functional-distillation objective for Experiment 004."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from src.evaluation.correctness import token_kl_sum_and_count


def causal_answer_loss_mask(
    attention_mask: torch.Tensor,
    answer_start: torch.Tensor | Iterable[int],
    valid_end: torch.Tensor | Iterable[int],
) -> torch.Tensor:
    """Mask logits at t-1 that predict valid answer tokens at t."""
    if attention_mask.ndim != 2 or attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be bool with shape [B,T]")
    batch, tokens = attention_mask.shape
    starts = torch.as_tensor(answer_start, device=attention_mask.device, dtype=torch.long)
    ends = torch.as_tensor(valid_end, device=attention_mask.device, dtype=torch.long)
    if starts.shape != (batch,) or ends.shape != (batch,):
        raise ValueError("answer_start and valid_end must each have shape [B]")
    if not ((starts >= 1) & (starts < ends) & (ends <= tokens)).all():
        raise ValueError("expected 1 <= answer_start < valid_end <= padded length")

    positions = torch.arange(tokens, device=attention_mask.device)[None, :]
    answer_predictors = (positions >= starts[:, None] - 1) & (
        positions < ends[:, None] - 1
    )
    target_is_real = torch.zeros_like(attention_mask)
    target_is_real[:, :-1] = attention_mask[:, 1:]
    return answer_predictors & attention_mask & target_is_real


def functional_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return token-normalized teacher-to-student KL, numerator, and count."""
    numerator, count = token_kl_sum_and_count(
        student_logits, teacher_logits, loss_mask
    )
    return numerator / count, numerator, count


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    module.requires_grad_(False)
    return module


def nonzero_gradient_parameter_count(parameters: Iterable[nn.Parameter]) -> int:
    return sum(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool(torch.count_nonzero(parameter.grad).item())
        for parameter in parameters
    )


def any_parameter_gradient(parameters: Iterable[nn.Parameter]) -> bool:
    return any(parameter.grad is not None for parameter in parameters)
