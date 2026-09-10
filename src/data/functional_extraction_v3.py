"""Prefix-stable frozen-Huginn extraction for the Phase 9R cache-v3 smoke."""

from __future__ import annotations

from contextlib import nullcontext
from types import MethodType
from typing import Any

import numpy as np


NORMALIZED_H16_SEMANTICS = (
    "D16 recurrent output normalized exactly once by transformer.ln_f; "
    "input to coda blocks; coda then final ln_f then lm_head; no extra initial ln_f"
)
EXTRACTION_LOCATION_V3 = (
    "explicit scheduled input_states -> frozen Huginn D16; "
    "x=post-prelude recurrent input; h16=returned normalized pre-coda latent"
)


def capture_scheduled_huginn_states(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    input_states: Any,
    *,
    depth: int = 16,
) -> dict[str, np.ndarray]:
    """Run D16 with an explicit schedule slice and capture its corresponding x/h16."""
    import torch

    if depth != 16:
        raise ValueError("cache-v3 smoke is frozen to Huginn D=16")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("input_ids must have shape [1,T], T>=1")
    if attention_mask.shape != input_ids.shape or not bool(attention_mask.all()):
        raise ValueError("attention_mask must be all-true and match input_ids")
    if input_states.shape[:2] != input_ids.shape:
        raise ValueError("scheduled input_states must match [1,T,H]")
    if input_states.dtype != torch.bfloat16:
        raise ValueError("scheduled input_states must be BF16")
    parameters = getattr(model, "parameters", None)
    if parameters is not None and any(parameter.requires_grad for parameter in parameters()):
        raise ValueError("Huginn must be frozen before cache-v3 extraction")

    captured: dict[str, Any] = {}
    recurrent_calls = 0
    original = model.core_block_forward

    def wrapped(this, state, recurrent_input, *args, **kwargs):
        nonlocal recurrent_calls
        if recurrent_calls == 0:
            if not torch.equal(state, input_states):
                raise RuntimeError("Huginn did not consume the explicit schedule slice")
            captured["x"] = recurrent_input.detach().float().cpu()[0]
        recurrent_calls += 1
        return original(state, recurrent_input, *args, **kwargs)

    model.core_block_forward = MethodType(wrapped, model)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if input_states.device.type == "cuda"
        else nullcontext()
    )
    try:
        with torch.inference_mode(), autocast:
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_states=input_states,
                num_steps=depth,
                use_cache=False,
                return_dict=True,
                output_details={
                    "return_logits": False,
                    "return_latents": True,
                    "return_head": False,
                    "return_stats": False,
                },
            )
    finally:
        model.core_block_forward = original

    if recurrent_calls != depth or set(captured) != {"x"}:
        raise RuntimeError(f"expected exactly {depth} recurrent calls, got {recurrent_calls}")
    latent = result.latent_states
    if latent is None or latent.shape != input_states.shape:
        raise RuntimeError("Huginn did not return normalized pre-coda h16")

    tensors = {
        "h0": input_states.detach().float().cpu()[0],
        "x": captured["x"],
        "h16": latent.detach().float().cpu()[0],
    }
    arrays = {key: value.numpy().astype(np.float16) for key, value in tensors.items()}
    injected = input_states.detach().cpu()[0].float().numpy().astype(np.float16)
    if not np.array_equal(arrays["h0"], injected):
        raise RuntimeError("captured h0 does not equal the injected schedule slice")
    if any(value.shape != arrays["h0"].shape for value in arrays.values()):
        raise RuntimeError("h0, x, and h16 shapes differ")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("captured state contains non-finite values")
    return arrays
