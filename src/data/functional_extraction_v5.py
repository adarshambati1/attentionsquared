"""Lossless native-dtype Huginn state extraction for cache-v5 smoke."""
from __future__ import annotations

from contextlib import nullcontext
from types import MethodType
from typing import Any

import numpy as np

from src.data.functional_extraction_v3 import NORMALIZED_H16_SEMANTICS

EXTRACTION_LOCATION_V5 = (
    "explicit fixed-schedule BF16 input_states -> frozen canonical Huginn D16; "
    "h0 persisted as raw BF16 uint16 bits; x post-prelude recurrent input and "
    "normalized pre-coda h16 persisted as native FP32"
)


def bfloat16_tensor_to_uint16(value: Any) -> np.ndarray:
    import torch

    if value.dtype != torch.bfloat16:
        raise ValueError("h0 must be native BF16")
    return value.detach().cpu().contiguous().view(torch.uint16).numpy().copy()


def uint16_to_bfloat16_tensor(value: np.ndarray, *, device: Any = None):
    import torch

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
        raise ValueError("cache-v5 smoke requires Huginn D=16")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("input_ids must have shape [1,T], T>=1")
    if attention_mask.shape != input_ids.shape or not bool(attention_mask.all()):
        raise ValueError("attention_mask must be all true and match input_ids")
    if input_states.shape[:2] != input_ids.shape or input_states.dtype != torch.bfloat16:
        raise ValueError("input_states must be native BF16 [1,T,H]")
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
    x = captured["x"]
    h16 = output.latent_states
    if x.dtype != torch.float32 or h16.dtype != torch.float32:
        raise RuntimeError(f"canonical native dtypes changed: x={x.dtype}, h16={h16.dtype}")
    arrays = {
        "h0": bfloat16_tensor_to_uint16(input_states[0]),
        "x": x.detach().cpu().contiguous().numpy()[0].copy(),
        "h16": h16.detach().cpu().contiguous().numpy()[0].copy(),
    }
    if arrays["x"].dtype != np.float32 or arrays["h16"].dtype != np.float32:
        raise RuntimeError("FP32 state conversion changed dtype")
    if not torch.equal(uint16_to_bfloat16_tensor(arrays["h0"]), input_states.detach().cpu()[0]):
        raise RuntimeError("h0 BF16 bit round trip failed")
    if not np.array_equal(arrays["x"], x.detach().cpu().numpy()[0]):
        raise RuntimeError("x FP32 round trip failed")
    if not np.array_equal(arrays["h16"], h16.detach().cpu().numpy()[0]):
        raise RuntimeError("h16 FP32 round trip failed")
    return arrays
