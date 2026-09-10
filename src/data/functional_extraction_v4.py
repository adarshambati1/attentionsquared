"""Lossless-BF16 prefix-stable Huginn state extraction for cache-v4 smoke."""
from __future__ import annotations

from contextlib import nullcontext
from types import MethodType
from typing import Any

import numpy as np

from src.data.functional_extraction_v3 import (
    NORMALIZED_H16_SEMANTICS,
)

EXTRACTION_LOCATION_V4 = (
    "explicit scheduled BF16 input_states -> frozen Huginn D16; "
    "x=post-prelude recurrent input; h16=returned normalized pre-coda latent; "
    "all state tensors persisted losslessly as raw BF16 uint16 bits"
)


def bfloat16_tensor_to_uint16(value: Any) -> np.ndarray:
    import torch

    if value.dtype != torch.bfloat16:
        raise ValueError("state tensor must be BF16")
    return value.detach().cpu().contiguous().view(torch.uint16).numpy().copy()


def uint16_to_bfloat16_tensor(value: np.ndarray, *, device: Any = None):
    import torch

    if value.dtype != np.uint16 or not value.flags.c_contiguous:
        value = np.ascontiguousarray(value, dtype=np.uint16)
    tensor = torch.from_numpy(value.copy()).view(torch.bfloat16)
    return tensor.to(device) if device is not None else tensor


def capture_scheduled_huginn_states(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    input_states: Any,
    *,
    depth: int = 16,
) -> dict[str, np.ndarray]:
    import torch

    if depth != 16:
        raise ValueError("cache-v4 smoke is frozen to Huginn D=16")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("input_ids must have shape [1,T], T>=1")
    if attention_mask.shape != input_ids.shape or not bool(attention_mask.all()):
        raise ValueError("attention_mask must be all true and match input_ids")
    if input_states.shape[:2] != input_ids.shape or input_states.dtype != torch.bfloat16:
        raise ValueError("input_states must be BF16 with shape [1,T,H]")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Huginn must be frozen")

    captured: dict[str, Any] = {}
    calls = 0
    original = model.core_block_forward

    def wrapped(this, state, recurrent_input, *args, **kwargs):
        nonlocal calls
        if calls == 0:
            if not torch.equal(state, input_states):
                raise RuntimeError("Huginn did not consume the schedule slice")
            captured["x"] = recurrent_input.detach().clone()
        calls += 1
        return original(state, recurrent_input, *args, **kwargs)

    model.core_block_forward = MethodType(wrapped, model)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if input_states.device.type == "cuda" else nullcontext()
    try:
        with torch.inference_mode(), autocast:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_states=input_states,
                num_steps=16,
                use_cache=False,
                return_dict=True,
                output_details={"return_logits": False, "return_latents": True, "return_head": False, "return_stats": False},
            )
    finally:
        model.core_block_forward = original
    if calls != 16 or output.latent_states is None:
        raise RuntimeError(f"expected exactly 16 recurrent calls, got {calls}")
    tensors = {"h0": input_states[0], "x": captured["x"][0], "h16": output.latent_states[0]}
    if any(value.dtype != torch.bfloat16 for value in tensors.values()):
        raise RuntimeError("all captured states must remain BF16")
    arrays = {key: bfloat16_tensor_to_uint16(value) for key, value in tensors.items()}
    for key, value in arrays.items():
        restored = uint16_to_bfloat16_tensor(value)
        if not torch.equal(restored, tensors[key].detach().cpu()):
            raise RuntimeError(f"lossless BF16 round trip failed for {key}")
    return arrays
