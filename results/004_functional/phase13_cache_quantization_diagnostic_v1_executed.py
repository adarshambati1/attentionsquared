import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, "/workspace/attentionsquared")
from src.evaluation.functional_autoregressive import frozen_coda_logits_from_normalized_state
from src.training.functional_protocol import fixed_huginn_h0_schedule

CACHE = Path("/workspace/functional_cache_v3_smoke")
OUTPUT = Path("/workspace/functional_protocol/phase13_cache_quantization_diagnostic_v1.json")
if OUTPUT.exists():
    raise FileExistsError(OUTPUT)

model = AutoModelForCausalLM.from_pretrained(
    "tomg-group-umd/huginn-0125",
    revision="bb6621b65e90b6a4b9b29ef88dc83866d450470c",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
).eval().cuda().requires_grad_(False)

records = []
mismatches = []
global_max = {"value": -1.0}
stratum_inputs = {}
lossless_all_exact = True
fp16_reproduces_cache_all = True

for example_id in range(8):
    with np.load(CACHE / f"{example_id:05d}.npz", allow_pickle=False) as archive:
        ids_np = archive["input_ids"].copy()
        cached_np = archive["h16"].copy()
        start = int(archive["answer_start"])
        end = int(archive["valid_end"])
    ids = torch.from_numpy(ids_np.astype(np.int64))[None].cuda()
    cached = torch.from_numpy(cached_np)[None].cuda()
    template = torch.empty((1, 2048, 5280), device="cuda", dtype=torch.bfloat16)
    schedule, _ = fixed_huginn_h0_schedule(
        model, template, base_seed=3000, example_id=example_id, seed_index=0
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        live = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids, dtype=torch.bool),
            input_states=schedule[:, : ids.shape[1]],
            num_steps=16,
            use_cache=False,
            return_dict=True,
            output_details={
                "return_logits": True,
                "return_latents": True,
                "return_head": False,
                "return_stats": False,
            },
        )
        frequencies = model.freqs_cis[:, : ids.shape[1]]
        cached_logits = frozen_coda_logits_from_normalized_state(
            model, cached.to(torch.bfloat16), frequencies
        )
        live_logits = live.logits.float()
        lossless_logits = frozen_coda_logits_from_normalized_state(
            model, live.latent_states, frequencies
        )

    fp16_reproduces = torch.equal(live.latent_states.to(torch.float16), cached)
    lossless_exact = torch.equal(live_logits, lossless_logits)
    fp16_reproduces_cache_all &= fp16_reproduces
    lossless_all_exact &= lossless_exact

    positions = slice(start - 1, end - 1)
    live_selected = live_logits[:, positions]
    cached_selected = cached_logits[:, positions]
    delta = (live_selected - cached_selected).abs()
    hidden_delta = (
        live.latent_states[:, positions].float() - cached[:, positions].float()
    ).abs()
    live_log = F.log_softmax(live_selected, dim=-1)
    cached_log = F.log_softmax(cached_selected, dim=-1)
    token_kl = (live_log.exp() * (live_log - cached_log)).sum(-1)[0]
    live_argmax = live_selected.argmax(-1)[0]
    cached_argmax = cached_selected.argmax(-1)[0]
    equal = live_argmax == cached_argmax

    flat_index = int(delta.reshape(-1).argmax().item())
    vocab = int(delta.shape[-1])
    local_token = flat_index // vocab
    vocab_id = flat_index % vocab
    item_max = float(delta.reshape(-1)[flat_index].item())
    absolute_position = start - 1 + local_token
    if item_max > global_max["value"]:
        global_max = {
            "value": item_max,
            "example_id": example_id,
            "prediction_position": absolute_position,
            "answer_prediction_offset": local_token,
            "vocabulary_token_id": vocab_id,
        }

    mismatch_indices = (~equal).nonzero(as_tuple=False).flatten().tolist()
    for local in mismatch_indices:
        position = start - 1 + int(local)
        live_probs = torch.softmax(live_selected[0, local], -1)
        cached_probs = torch.softmax(cached_selected[0, local], -1)
        top_probs, top_ids = torch.topk(live_probs, 2)
        hidden_at = hidden_delta[0, local]
        logit_at = delta[0, local]
        mismatches.append({
            "example_id": example_id,
            "sequence_length": len(ids_np),
            "prediction_position": position,
            "answer_prediction_offset": int(local),
            "target_token_id": int(ids[0, position + 1].item()),
            "live_top1_token_id": int(top_ids[0].item()),
            "live_top1_probability": float(top_probs[0].item()),
            "live_top2_token_id": int(top_ids[1].item()),
            "live_top2_probability": float(top_probs[1].item()),
            "live_top1_minus_top2_margin": float((top_probs[0] - top_probs[1]).item()),
            "cached_top1_token_id": int(cached_probs.argmax().item()),
            "cached_top1_probability": float(cached_probs.max().item()),
            "hidden_mean_abs_at_position": float(hidden_at.mean().item()),
            "hidden_max_abs_at_position": float(hidden_at.max().item()),
            "logit_mean_abs_at_position": float(logit_at.mean().item()),
            "logit_max_abs_at_position": float(logit_at.max().item()),
            "kl_live_to_cached_at_position": float(token_kl[local].item()),
        })

    records.append({
        "example_id": example_id,
        "sequence_length": len(ids_np),
        "compared_tokens": end - start,
        "logit_mean_abs": float(delta.mean().item()),
        "logit_max_abs": item_max,
        "kl_live_to_cached_per_token": float(token_kl.mean().item()),
        "argmax_equal": int(equal.sum().item()),
        "argmax_total": int(equal.numel()),
        "hidden_mean_abs": float(hidden_delta.mean().item()),
        "hidden_max_abs": float(hidden_delta.max().item()),
        "live_bfloat16_to_float16_equals_cached_float16": fp16_reproduces,
        "lossless_bfloat16_state_replay_logits_exact": lossless_exact,
    })
    stratum_inputs[example_id] = {
        "abs_sum": float(delta.sum().item()),
        "elements": int(delta.numel()),
        "max": item_max,
        "kl_sum": float(token_kl.sum().item()),
        "tokens": int(token_kl.numel()),
        "argmax": int(equal.sum().item()),
    }
    del template, schedule, live, cached_logits, lossless_logits
    torch.cuda.empty_cache()

ordered = sorted(records, key=lambda row: (row["sequence_length"], row["example_id"]))
groups = [("shortest_3", ordered[:3]), ("middle_2", ordered[3:5]), ("longest_3", ordered[5:])]
strata = []
for name, group in groups:
    ids = [row["example_id"] for row in group]
    vals = [stratum_inputs[i] for i in ids]
    strata.append({
        "stratum": name,
        "example_ids": ids,
        "sequence_lengths": [row["sequence_length"] for row in group],
        "logit_mean_abs": sum(v["abs_sum"] for v in vals) / sum(v["elements"] for v in vals),
        "logit_max_abs": max(v["max"] for v in vals),
        "kl_live_to_cached_per_token": sum(v["kl_sum"] for v in vals) / sum(v["tokens"] for v in vals),
        "argmax_agreement": sum(v["argmax"] for v in vals) / sum(v["tokens"] for v in vals),
    })

same_position = bool(mismatches) and any(
    row["example_id"] == global_max["example_id"]
    and row["prediction_position"] == global_max["prediction_position"]
    for row in mismatches
)
result = {
    "protocol": "phase13-cache-quantization-diagnostic-v1",
    "status": "diagnostic_complete",
    "phase13_executed": False,
    "attention2_used": False,
    "thresholds_changed": False,
    "mismatch_count": len(mismatches),
    "mismatches": mismatches,
    "global_max_logit_error": global_max,
    "global_max_occurs_at_argmax_mismatch_position": same_position,
    "per_example": records,
    "length_strata_equal_count_sorted_by_sequence_length": strata,
    "quantization_confirmation": {
        "all_live_bfloat16_to_float16_equal_cached_float16": fp16_reproduces_cache_all,
        "all_lossless_bfloat16_state_replay_logits_exact": lossless_all_exact,
        "interpretation": "cached fp16 is the rounded live bf16 state; losslessly preserving bf16 state bits restores live coda logits exactly",
    },
}
OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
