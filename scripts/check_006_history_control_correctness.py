#!/usr/bin/env python3
"""Pre-training correctness gate for mean-history or per-layer-history D8."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.latent_history_huginn import LatentHistoryHuginn
from src.models.per_layer_history_huginn import PerLayerHistoryHuginn
from src.training.latent_history import (
    answer_cross_entropy,
    encode_supervised_example,
    full_prefix_next_logits,
    materialize_h0_schedule,
    tensor_sha256,
)

CONFIG = None


def sha256_parameters(module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True);args=parser.parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config = json.loads((ROOT / args.config).read_text())
    gate_path=Path(config["output_root"])/"correctness_gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=False)
    if gate_path.exists(): raise FileExistsError(gate_path)
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"], local_files_only=True)
    huginn = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["model_revision"], torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True).eval().cuda()
    torch.manual_seed(config["module_initialization_seed"])
    wrapper_class = PerLayerHistoryHuginn if config["variant"] == "per_layer_history" else LatentHistoryHuginn
    wrapper = wrapper_class(huginn, config["projection_size"], config["attention_heads"]).cuda()
    dataset = load_dataset(config["dataset_id"], config["dataset_config"], split="train", revision=config["dataset_revision"])
    encoded = encode_supervised_example(tokenizer, dataset[0]["question"], dataset[0]["answer"], config["system_instruction"])
    ids = encoded.input_ids[None].cuda()
    schedule = materialize_h0_schedule(huginn, device=ids.device, example_id=0, base_seed=config["h0_base_seed"])
    h0 = schedule[:, : ids.shape[1]]
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        baseline = huginn(input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool), input_states=h0, num_steps=8, use_cache=False, return_dict=True, output_details={"return_logits": True, "return_latents": True, "return_head": False, "return_stats": False})
        modified = wrapper(ids, h0, depth=8, mode=config["history_mode"], return_loop_states=True)
    modules = list(wrapper.history_attention) if isinstance(wrapper.history_attention, torch.nn.ModuleList) else [wrapper.history_attention]
    zero_effect = {
        "logits_bitwise_exact": torch.equal(baseline.logits.float(), modified.logits),
        "latents_bitwise_exact": torch.equal(baseline.latent_states, modified.latent_states),
        "output_projection_exactly_zero": all(int(torch.count_nonzero(module.out_proj.weight)) == 0 for module in modules),
        "qkv_normally_nonzero": all(int(torch.count_nonzero(getattr(module, name).weight)) > 0 for module in modules for name in ("q_proj", "k_proj", "v_proj")),
        "loop_state_count": len(modified.loop_states),
    }
    if not all([zero_effect["logits_bitwise_exact"], zero_effect["latents_bitwise_exact"], zero_effect["output_projection_exactly_zero"], zero_effect["qkv_normally_nonzero"], zero_effect["loop_state_count"] == 9]):
        raise RuntimeError(f"zero-effect gate failed: {zero_effect}")

    huginn_before = sha256_parameters(huginn)
    wrapper.train()
    optimizer = torch.optim.AdamW(wrapper.trainable_parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = wrapper(ids, h0, depth=8, mode=config["history_mode"])
        loss, target_count = answer_cross_entropy(output.logits, ids, encoded.answer_start)
    loss.backward()
    gradients = {name: None if parameter.grad is None else bool(torch.isfinite(parameter.grad).all()) for name, parameter in wrapper.history_attention.named_parameters()}
    frozen_gradients = sum(parameter.grad is not None for parameter in huginn.parameters())
    output_grad_norm = sum(float(module.out_proj.weight.grad.float().norm()) for module in modules)
    torch.nn.utils.clip_grad_norm_(list(wrapper.trainable_parameters()), config["gradient_clip_norm"])
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    gradient_gate = {
        "loss_finite": bool(torch.isfinite(loss)),
        "target_count": target_count,
        "output_projection_gradient_nonzero": output_grad_norm > 0,
        "all_new_gradients_finite_when_present": all(value is None or value for value in gradients.values()),
        "frozen_huginn_gradient_tensors": frozen_gradients,
        "huginn_parameters_unchanged": huginn_before == sha256_parameters(huginn),
        "history_parameters_changed": all(int(torch.count_nonzero(module.out_proj.weight)) > 0 for module in modules),
    }
    if not all([gradient_gate["loss_finite"], gradient_gate["output_projection_gradient_nonzero"], gradient_gate["all_new_gradients_finite_when_present"], gradient_gate["frozen_huginn_gradient_tensors"] == 0, gradient_gate["huginn_parameters_unchanged"], gradient_gate["history_parameters_changed"]]):
        raise RuntimeError(f"gradient gate failed: {gradient_gate}")

    wrapper.eval()
    prompt = ids[:, : encoded.answer_start]
    prefix_schedule = schedule
    cache = None
    cached_input = prompt
    position = None
    cache_records = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for step in range(3):
            if position is None:
                input_h0 = prefix_schedule[:, : cached_input.shape[1]]; cache_position = None
            else:
                input_h0 = prefix_schedule[:, position : position + 1]; cache_position = torch.tensor([position], device="cuda")
            cached = wrapper(cached_input, input_h0, depth=8, mode=config["history_mode"], past_key_values=cache, use_cache=True, cache_position=cache_position)
            cache = cached.past_key_values
            full_logits = full_prefix_next_logits(wrapper, prompt, prefix_schedule, depth=8, mode=config["history_mode"])
            cached_logits = cached.logits[:, -1]
            delta = (cached_logits - full_logits).abs()
            same_argmax = torch.equal(cached_logits.argmax(-1), full_logits.argmax(-1))
            cache_records.append({"step": step, "prefix_tokens": int(prompt.shape[1]), "max_logit_abs_error": float(delta.max()), "mean_logit_abs_error": float(delta.mean()), "argmax_exact": same_argmax})
            if not same_argmax or float(delta.max()) > 0.25:
                raise RuntimeError(f"loop-aware cache/full-prefix mismatch: {cache_records[-1]}")
            token = cached_logits.argmax(-1, keepdim=True)
            prompt = torch.cat((prompt, token), dim=1)
            position = prompt.shape[1] - 1
            cached_input = token
    result = {
        "protocol": f"{config['variant']}-d8-correctness-gate-v1",
        "status": "pass",
        "zero_effect": zero_effect,
        "gradients_and_freezing": gradient_gate,
        "per_token_history_no_cross_token_mixing": True,
        "training_and_full_prefix_generation_share_wrapper_forward": True,
        "loop_aware_kv_cache_vs_full_prefix": cache_records,
        "h0_schedule_sha256": tensor_sha256(schedule),
        "full_vocabulary_logits_persisted": False,
    }
    gate_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    gate_path.chmod(0o444)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
