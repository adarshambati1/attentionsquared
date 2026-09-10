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
    input_states: Any | None = None,
) -> dict[str, np.ndarray]:
    """Capture h0, recurrent input x, and the exact coda input at fixed depth.

    ``input_states`` optionally injects a caller-materialized h0. This supports
    fixed-schedule replay without changing the legacy cache-v2 call path.
    """
    import torch

    if depth != 16:
        raise ValueError("functional cache v2 is frozen to Huginn D=16")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1,T]")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    if input_states is not None and input_states.shape[:2] != input_ids.shape:
        raise ValueError("explicit input_states must match input_ids [B,T]")

    captured: dict[str, Any] = {}
    recurrent_calls = 0
    original = model.core_block_forward

    def wrapped(this, x, input_embeds, *args, **kwargs):
        nonlocal recurrent_calls
        if recurrent_calls == 0:
            if input_states is not None and not torch.equal(x, input_states):
                raise RuntimeError("Huginn did not consume the explicit input_states")
            captured["h0_full"] = x.detach().to(torch.float32).cpu()[0]
            captured["x_full"] = input_embeds.detach().to(torch.float32).cpu()[0]
        recurrent_calls += 1
        return original(x, input_embeds, *args, **kwargs)

    model.core_block_forward = MethodType(wrapped, model)
    try:
        with torch.inference_mode():
            model_kwargs = {}
            if input_states is not None:
                model_kwargs["input_states"] = input_states
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
                **model_kwargs,
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
    if input_states is not None:
        injected = input_states.detach().to(torch.float32).cpu()[0].numpy().astype(np.float16)
        if not np.array_equal(arrays["h0_full"], injected):
            raise RuntimeError("captured h0 does not equal explicit input_states")
    return arrays
