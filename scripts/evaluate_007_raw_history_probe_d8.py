#!/usr/bin/env python3
"""Compare plain D8, parameter-matched current-only D8, and raw-mean-history D8."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.correctness import build_chat_prompt, score_generation, tokenize_prompt
from src.models.latent_history_huginn import LatentHistoryHuginn
from src.models.per_layer_history_huginn import PerLayerHistoryHuginn
from src.training.latent_history import generate_cached, materialize_h0_schedule

CONFIG = Path("configs/005_latent_history_attention_d8.json")


def mean(values):
    return sum(values) / len(values)


def percentile(values, p):
    ordered = sorted(values); index = (len(ordered) - 1) * p; low = int(index); high = min(low + 1, len(ordered) - 1); fraction = index - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def fixed_forward_benchmark(wrapper, ids, schedule, *, depth, mode, warmup, repetitions):
    for _ in range(warmup):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            wrapper(ids, schedule[:, : ids.shape[1]], depth=depth, mode=mode)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repetitions):
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            wrapper(ids, schedule[:, : ids.shape[1]], depth=depth, mode=mode)
        torch.cuda.synchronize(); samples.append(time.perf_counter() - start)
    return {"tokens": int(ids.shape[1]), "repetitions": repetitions, "mean_seconds": mean(samples), "median_seconds": statistics.median(samples), "p95_seconds": percentile(samples, 0.95), "peak_memory_bytes": int(torch.cuda.max_memory_allocated())}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=CONFIG); args = parser.parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config_path = ROOT / args.config; config = json.loads(config_path.read_text())
    output_root = Path("/workspace/raw_history_probe_d8")
    if output_root.exists(): raise FileExistsError(output_root)
    output_root.mkdir()
    output_path = output_root / "comparison.json"
    current_checkpoint_path = Path("/workspace/current_state_only_d8/best.pt")
    current_checkpoint = torch.load(current_checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"], local_files_only=True)
    huginn = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["model_revision"], torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True).eval().cuda()
    plain_wrapper = LatentHistoryHuginn(huginn, config["projection_size"], config["attention_heads"]).cuda().eval()
    current_wrapper = LatentHistoryHuginn(huginn, config["projection_size"], config["attention_heads"]).cuda().eval()
    raw_mean_wrapper = LatentHistoryHuginn(huginn, config["projection_size"], config["attention_heads"]).cuda().eval()
    current_wrapper.history_attention.load_state_dict(current_checkpoint["history_attention_state_dict"])
    test = load_dataset(config["dataset_id"], config["dataset_config"], split="test", revision=config["dataset_revision"])
    test_ids = list(range(config["test_ids"][0], config["test_ids"][1] + 1))
    models = [
        ("original_huginn_d8", plain_wrapper, 8, "disabled"),
        ("current_state_only_huginn_d8", current_wrapper, 8, "current_only"),
        ("raw_mean_history_huginn_d8", raw_mean_wrapper, 8, "raw_mean"),
    ]
    records = []
    summaries = {}
    for name, active_wrapper, depth, mode in models:
        for example_id in test_ids[: config["generation_warmup_examples"]]:
            prompt = build_chat_prompt(tokenizer, test[example_id]["question"], config["system_instruction"])
            ids = tokenize_prompt(tokenizer, prompt)["input_ids"].cuda()
            schedule = materialize_h0_schedule(huginn, device=ids.device, example_id=example_id, base_seed=config["h0_base_seed"])
            generate_cached(active_wrapper, tokenizer, ids, schedule, depth=depth, mode=mode, max_new_tokens=min(32, config["max_new_tokens"]))
        for example_id in test_ids:
            prompt = build_chat_prompt(tokenizer, test[example_id]["question"], config["system_instruction"])
            ids = tokenize_prompt(tokenizer, prompt)["input_ids"].cuda()
            schedule = materialize_h0_schedule(huginn, device=ids.device, example_id=example_id, base_seed=config["h0_base_seed"])
            generation = generate_cached(active_wrapper, tokenizer, ids, schedule, depth=depth, mode=mode, max_new_tokens=config["max_new_tokens"])
            score = score_generation(generation.text, test[example_id]["answer"], hit_max_new_tokens=generation.hit_max_new_tokens)
            record = {"model": name, "depth": depth, "example_id": example_id, "correct": score.correct, "predicted_answer": score.predicted_answer, "gold_answer": score.gold_answer, "generated_token_ids": generation.token_ids, "generated_text": generation.text, "generation_latency_seconds": generation.latency_seconds, "generated_tokens": generation.generated_tokens, "hit_max_new_tokens": generation.hit_max_new_tokens, "ended_naturally": generation.ended_naturally, "peak_memory_bytes": generation.peak_memory_bytes}
            records.append(record); print(json.dumps({k:v for k,v in record.items() if k not in ("generated_text","generated_token_ids")}), flush=True)
        selected = [record for record in records if record["model"] == name]
        summaries[name] = {"loops": depth, "accuracy": sum(r["correct"] for r in selected) / len(selected), "correct": sum(r["correct"] for r in selected), "total": len(selected), "mean_generation_latency_seconds": mean([r["generation_latency_seconds"] for r in selected]), "median_generation_latency_seconds": statistics.median([r["generation_latency_seconds"] for r in selected]), "mean_generated_tokens": mean([r["generated_tokens"] for r in selected]), "cap_hit_rate": mean([r["hit_max_new_tokens"] for r in selected]), "peak_memory_bytes": max(r["peak_memory_bytes"] for r in selected)}
    fixed_tokens = []
    for example_id in test_ids:
        prompt = build_chat_prompt(tokenizer, test[example_id]["question"], config["system_instruction"])
        fixed_tokens.extend(tokenize_prompt(tokenizer, prompt)["input_ids"][0].tolist())
        if len(fixed_tokens) >= config["fixed_forward_tokens"]: break
    fixed_ids = torch.tensor([fixed_tokens[: config["fixed_forward_tokens"]]], dtype=torch.long, device="cuda")
    fixed_schedule = materialize_h0_schedule(huginn, device=fixed_ids.device, example_id=0, base_seed=config["h0_base_seed"])
    for name, active_wrapper, depth, mode in models:
        summaries[name]["fixed_length_forward"] = fixed_forward_benchmark(active_wrapper, fixed_ids, fixed_schedule, depth=depth, mode=mode, warmup=config["fixed_forward_warmup"], repetitions=config["fixed_forward_repetitions"])
    prior_comparison_path = Path("/workspace/history_memory_comparison_d8/comparison.json")
    prior = json.loads(prior_comparison_path.read_text())
    references = {name: prior["models"][name] for name in ("mean_history_huginn_d8", "shared_history_attention_huginn_d8", "per_layer_history_huginn_d8")}
    references["original_huginn_d16"] = prior["d16_reference_from_frozen_prior_comparison"]
    result = {"protocol": "raw-history-probe-d8-comparison-v1", "current_checkpoint": str(current_checkpoint_path), "current_checkpoint_step": current_checkpoint["step"], "models": summaries, "frozen_references": references, "prior_comparison_sha256": hashlib.sha256(prior_comparison_path.read_bytes()).hexdigest(), "records": records, "decision_rule": "continue to heavy-ball/Anderson if raw mean beats plain/current-only or is within 0.02 absolute accuracy of the best existing history arm", "git_commit": subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip(), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
    temporary = output_path.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, output_path)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
