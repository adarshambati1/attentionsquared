import json
from collections import Counter
from types import MethodType

import numpy as np
import torch
from transformers import AutoModelForCausalLM

import sys
sys.path.insert(0, "/workspace/attentionsquared")
from src.training.functional_protocol import fixed_huginn_h0_schedule

MODEL = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    revision=REVISION,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
).eval().cuda().requires_grad_(False)

result = {
    "load_argument": "torch_dtype=torch.bfloat16",
    "model_config_torch_dtype": str(getattr(model.config, "torch_dtype", None)),
    "parameter_dtype_counts": dict(Counter(str(p.dtype) for p in model.parameters())),
    "wte_dtype": str(model.transformer.wte.weight.dtype),
    "lm_head_dtype": str(model.lm_head.weight.dtype),
    "examples": [],
}

for example_id in (0, 2, 7):
    with np.load(f"results/004_functional/teacher_sequences/train/{example_id:05d}.npz", allow_pickle=False) as z:
        ids = torch.from_numpy(z["input_ids"].astype(np.int64))[None].cuda()
    schedule, _ = fixed_huginn_h0_schedule(
        model,
        torch.empty((1, 2048, 5280), device="cuda", dtype=torch.bfloat16),
        base_seed=3000,
        example_id=example_id,
        seed_index=0,
    )
    observed = {"example_id": example_id, "schedule_dtype": str(schedule.dtype)}
    original = model.core_block_forward
    call_counter = {"value": 0}

    def wrapped(this, state, recurrent_input, *args, **kwargs):
        call_counter["value"] += 1
        calls = call_counter["value"]
        if calls == 1:
            observed["first_core_input_state_dtype"] = str(state.dtype)
            observed["recurrent_input_x_dtype"] = str(recurrent_input.dtype)
        output = original(state, recurrent_input, *args, **kwargs)
        value = output[0] if isinstance(output, tuple) else output
        if calls in (1, 16):
            observed[f"core_output_{calls}_dtype"] = str(value.dtype)
        return output

    model.core_block_forward = MethodType(wrapped, model)
    try:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids, dtype=torch.bool),
                input_states=schedule[:, : ids.shape[1]],
                num_steps=16,
                use_cache=False,
                return_dict=True,
                output_details={
                    "return_logits": True,
                    "return_latents": True,
                    "return_head": True,
                    "return_stats": False,
                },
            )
    finally:
        model.core_block_forward = original
    observed.update({
        "core_calls": call_counter["value"],
        "returned_latent_h16_dtype": str(output.latent_states.dtype),
        "returned_hidden_post_coda_dtype": str(output.hidden_states.dtype),
        "returned_logits_dtype": str(output.logits.dtype),
        "h16_exact_after_bfloat16_roundtrip": bool(torch.equal(output.latent_states, output.latent_states.to(torch.bfloat16).float())),
        "h16_bfloat16_roundtrip_max_abs": float((output.latent_states - output.latent_states.to(torch.bfloat16).float()).abs().max()),
        "x_claim": "recurrent_input_x captured at first core call",
    })
    result["examples"].append(observed)

print(json.dumps(result, indent=2, sort_keys=True))
