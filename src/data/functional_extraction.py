"""Frozen-Huginn state extraction for compact functional cache v2."""

from __future__ import annotations

from types import MethodType
from typing import Any

import numpy as np


EXTRACTION_LOCATION = (
    "h0_full=initialize_state output before first recurrent noise/core step; "
    "x_full=post-prelude input_embeds passed to every recurrent core step; "
    "h16_teacher=iterate_forward layer-normalized output immediately before frozen coda"
)


def capture_huginn_functional_states(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    depth: int = 16,
) -> dict[str, np.ndarray]:
    """Capture h0, recurrent input x, and the exact coda input at fixed depth."""
    import torch

    if depth != 16:
        raise ValueError("functional cache v2 is frozen to Huginn D=16")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1,T]")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")

    captured: dict[str, Any] = {}
    recurrent_calls = 0
    original = model.core_block_forward

    def wrapped(this, x, input_embeds, *args, **kwargs):
        nonlocal recurrent_calls
        if recurrent_calls == 0:
            captured["h0_full"] = x.detach().to(torch.float32).cpu()[0]
            captured["x_full"] = input_embeds.detach().to(torch.float32).cpu()[0]
        recurrent_calls += 1
        return original(x, input_embeds, *args, **kwargs)

    model.core_block_forward = MethodType(wrapped, model)
    try:
        with torch.inference_mode():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_steps=depth,
                use_cache=False,
                output_details={
                    "return_logits": False,
                    "return_latents": True,
                    "return_head": False,
                    "return_stats": False,
                },
            )
    finally:
        model.core_block_forward = original

    if recurrent_calls != depth:
        raise RuntimeError(
            f"expected exactly {depth} recurrent calls, observed {recurrent_calls}"
        )
    if set(captured) != {"h0_full", "x_full"}:
        raise RuntimeError("failed to capture initial recurrent state and input")
    latent = result.latent_states
    if latent is None or latent.ndim != 3 or latent.shape[0] != 1:
        raise RuntimeError("Huginn did not return one batch of recurrent latent states")
    captured["h16_teacher"] = latent.detach().to(torch.float32).cpu()[0]

    expected_shape = (int(input_ids.shape[1]), int(captured["h0_full"].shape[1]))
    arrays: dict[str, np.ndarray] = {}
    for key, tensor in captured.items():
        value = tensor.numpy().astype(np.float16)
        if value.shape != expected_shape:
            raise RuntimeError(f"{key} has shape {value.shape}, expected {expected_shape}")
        if not np.isfinite(value).all():
            raise RuntimeError(f"{key} contains non-finite values")
        arrays[key] = value
    return arrays
