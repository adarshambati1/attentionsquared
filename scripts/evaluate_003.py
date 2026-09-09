#!/usr/bin/env python3
"""Corrected-v2 evaluation of Exp3 direct/residual jump controls.

Historical output remains preserved. This entry point writes a new result,
uses globally token-normalized KL, cap-safe scoring, paired per-example seeds,
and synchronized timing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.evaluation.correctness import (
    CORRECTED_V2_PROTOCOL,
    SCORER_PROTOCOL,
    SEED_PROTOCOL,
    TokenKLAggregator,
    build_chat_prompt,
    prepare_corrected_v2_output,
    per_example_seed,
    generation_config_kwargs,
    generation_status,
    score_generation,
    seed_for_example,
    sha256_file,
    stop_token_ids,
    synchronized_cuda_timer,
    tokenize_prompt,
    write_json_exclusive,
)
from scripts.archive.train_003_predictors import JumpMLP


def load_model(config):
    return AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval().cuda()


def patch_jump(model, predictor):
    original = model.iterate_forward

    def jumped(
        this,
        input_embeds,
        input_states,
        freqs_cis,
        block_idx,
        mask,
        past_key_values=None,
        num_steps=None,
        init_scale=1.0,
    ):
        h0 = (
            this.initialize_state(input_embeds, scale=init_scale)
            if input_states is None
            else input_states.clone()
        )
        predicted = predictor(h0.float(), input_embeds.float()).to(h0.dtype)
        return this.transformer.ln_f(predicted), 0, 0, h0.detach(), block_idx

    model.iterate_forward = MethodType(jumped, model)
    return original


def generate(model, tokenizer, prompt, depth, max_new_tokens):
    encoded = {
        key: value.cuda()
        for key, value in tokenize_prompt(tokenizer, prompt).items()
    }
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            generation_config=GenerationConfig(
                **generation_config_kwargs(tokenizer, max_new_tokens)
            ),
            num_steps=depth,
            tokenizer=tokenizer,
        )
    generated = output.sequences[0][encoded["input_ids"].shape[-1] :]
    status = generation_status(
        generated.tolist(),
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids(tokenizer),
    )
    return tokenizer.decode(generated, skip_special_tokens=False), status


def validate_cache_provenance(cache: Path, config: dict) -> None:
    metadata_path = cache / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"cache provenance metadata is required: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    for key in ("model_revision", "dataset_revision", "seed", "splits", "seed_protocol"):
        if metadata.get(key) != config.get(key) and not (key == "seed_protocol" and metadata.get(key) == SEED_PROTOCOL):
            raise ValueError(f"cache metadata mismatch for {key}")
    expected = {cache / split / f"{i:05d}.npz" for split,(start,end) in config["splits"].items() for i in range(start,end+1)}
    actual = set(cache.glob("train/*.npz")) | set(cache.glob("val/*.npz")) | set(cache.glob("test/*.npz"))
    if actual != expected:
        raise ValueError("cache file set does not match configuration")
    required={"h0","x","h16","input_ids","base_seed","derived_seed","seed_protocol","example_id","split","model_revision","dataset_revision"}
    legacy={"h0","x","h16","input_ids"}
    provenance_mode=None
    for path in sorted(actual):
        with np.load(path) as item:
            keys=set(item.files)
            if keys == legacy:
                if provenance_mode == "authoritative-v3":
                    raise ValueError("cache mixes legacy and authoritative item formats")
                provenance_mode="legacy-historical"
            elif keys == required:
                if provenance_mode == "legacy-historical":
                    raise ValueError("cache mixes legacy and authoritative item formats")
                provenance_mode="authoritative-v3"
                example_id=int(path.stem); split=path.parent.name
                if int(item["example_id"].item()) != example_id or str(item["split"].item()) != split or int(item["base_seed"].item()) != int(config["seed"]):
                    raise ValueError(f"{path} has invalid identity provenance")
                if int(item["derived_seed"].item()) != per_example_seed(config["seed"], example_id) or str(item["seed_protocol"].item()) != SEED_PROTOCOL:
                    raise ValueError(f"{path} has invalid seed provenance")
                if str(item["model_revision"].item()) != config["model_revision"] or str(item["dataset_revision"].item()) != config["dataset_revision"]:
                    raise ValueError(f"{path} has invalid source provenance")
            else:
                raise ValueError(f"{path} has invalid cache keys")
    return provenance_mode


def cache_seed_protocol(cache: Path, mode: str) -> str:
    metadata_path = cache / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"cache provenance metadata is required: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if mode == "legacy-historical":
        return "historical-cache-legacy-item-provenance-v1"
    if metadata.get("seed_protocol") != SEED_PROTOCOL:
        raise ValueError(f"cache seed protocol is not authoritative: {metadata.get('seed_protocol')!r}")
    return SEED_PROTOCOL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/003_direct_jump.json"))
    parser.add_argument("--cache", type=Path, default=Path("results/003_direct_jump/cache"))
    parser.add_argument("--models", type=Path, default=Path("results/003_direct_jump/models"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/003_direct_jump/corrected-v2/evaluation.json"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"]
    )
    dataset = load_dataset(
        config["dataset_id"],
        config["dataset_config"],
        split=config["dataset_split"],
        revision=config["dataset_revision"],
    )
    device = torch.device("cuda")
    results = []
    test_start, test_end = config["splits"]["test"]
    max_new_tokens = int(config.get("max_new_tokens", 1024))
    cache_mode = validate_cache_provenance(args.cache, config)
    cache_seed_policy = cache_seed_protocol(args.cache, cache_mode)
    for name in ("direct", "residual"):
        checkpoint = torch.load(args.models / f"{name}.pt", map_location="cpu")
        if checkpoint.get("cache_root") != str(args.cache.resolve()) and cache_mode != "legacy-historical":
            raise ValueError(f"{name} checkpoint is not bound to cache {args.cache}")
    prepare_corrected_v2_output(args.output)

    for name in ("direct", "residual"):
        checkpoint = torch.load(args.models / f"{name}.pt", map_location="cpu")
        if checkpoint.get("cache_root") != str(args.cache.resolve()) and cache_mode != "legacy-historical":
            raise ValueError(f"{name} checkpoint is not bound to cache {args.cache}")
        predictor = JumpMLP(checkpoint["hidden"], checkpoint["residual"])
        predictor.load_state_dict(checkpoint["state_dict"])
        predictor.to(device).eval()
        model = load_model(config)
        cosine = []
        relative_l2 = []
        kl = TokenKLAggregator()

        for cache_path in sorted((args.cache / "test").glob("*.npz")):
            with np.load(cache_path) as item:
                h0 = torch.from_numpy(item["h0"].astype(np.float32)).cuda()
                x = torch.from_numpy(item["x"].astype(np.float32)).cuda()
                target = torch.from_numpy(item["h16"].astype(np.float32)).cuda()
                input_ids = torch.from_numpy(item["input_ids"]).cuda().unsqueeze(0)
            with torch.inference_mode():
                predicted = predictor(h0, x)
                cosine.append(
                    torch.nn.functional.cosine_similarity(
                        predicted, target, dim=-1
                    ).mean().item()
                )
                relative_l2.append(
                    ((predicted - target).norm(dim=-1) / (target.norm(dim=-1) + 1e-8))
                    .mean()
                    .item()
                )
                student_logits = model(
                    input_ids=input_ids,
                    input_states=predicted.to(torch.bfloat16).unsqueeze(0),
                    num_steps=0,
                    use_cache=False,
                ).logits
                teacher_logits = model(
                    input_ids=input_ids,
                    input_states=target.to(torch.bfloat16).unsqueeze(0),
                    num_steps=0,
                    use_cache=False,
                ).logits
                kl.update_logits(student_logits, teacher_logits)

        original = patch_jump(model, predictor)
        correct = 0
        cap_hits = 0
        latencies = []
        for number, example_id in enumerate(range(test_start, test_end + 1), 1):
            seed_for_example(config["seed"], example_id)
            prompt = build_chat_prompt(
                tokenizer,
                dataset[example_id]["question"],
                config["system_instruction"],
            )
            with synchronized_cuda_timer(device) as timing:
                generated, status = generate(
                    model, tokenizer, prompt, depth=16, max_new_tokens=max_new_tokens
                )
            latencies.append(timing.seconds)
            cap_hits += int(status.hit_max_new_tokens)
            score = score_generation(
                generated,
                dataset[example_id]["answer"],
                hit_max_new_tokens=status.hit_max_new_tokens,
            )
            correct += int(score.correct)
            if number % 10 == 0:
                print(f"{name} generation {number}/{test_end - test_start + 1}", flush=True)
        model.iterate_forward = original
        total = test_end - test_start + 1
        result = {
            "model": name,
            "protocol": CORRECTED_V2_PROTOCOL,
            "scorer_protocol": SCORER_PROTOCOL,
            "seed_protocol": SEED_PROTOCOL,
            "generation_seed_protocol": SEED_PROTOCOL,
            "cache_seed_protocol": cache_seed_policy,
            "base_seed": config["seed"],
            "model_revision": config["model_revision"],
            "dataset_revision": config["dataset_revision"],
            "checkpoint_path": str(args.models / f"{name}.pt"),
            "checkpoint_sha256": sha256_file(args.models / f"{name}.pt"),
            "checkpoint_result": checkpoint.get("result"),
            "timing_protocol": "cuda-synchronized-wall-clock-v1",
            "kl_protocol": "global-numerator-over-valid-token-count-v1",
            "latent_cosine": float(np.mean(cosine)),
            "latent_relative_l2": float(np.mean(relative_l2)),
            "teacher_logit_kl_per_token": kl.mean,
            "teacher_logit_kl_valid_tokens": kl.count,
            "gsm8k_accuracy": correct / total,
            "cap_hits": cap_hits,
            "mean_generation_latency_seconds": float(np.mean(latencies)),
            "examples": total,
        }
        results.append(result)
        print(result, flush=True)
        del model, predictor
        torch.cuda.empty_cache()

    write_json_exclusive(args.output, results)


if __name__ == "__main__":
    main()
