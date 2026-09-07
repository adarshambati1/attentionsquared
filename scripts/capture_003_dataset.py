#!/usr/bin/env python3
"""Cache exact tokenwise Huginn (h0, input_embeds, h16) triples."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(config_path: Path, output: Path) -> None:
    cfg = json.loads(config_path.read_text())
    ds = load_dataset(cfg["dataset_id"], cfg["dataset_config"], split=cfg["dataset_split"], revision=cfg["dataset_revision"])
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=cfg["model_revision"], torch_dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
    output.mkdir(parents=True, exist_ok=True)
    for split, bounds in cfg["splits"].items():
        split_dir = output / split
        split_dir.mkdir(exist_ok=True)
        start, end = bounds
        for example_id in range(start, end + 1):
            out_path = split_dir / f"{example_id:05d}.npz"
            if out_path.exists():
                continue
            torch.manual_seed(cfg["seed"] + example_id)
            torch.cuda.manual_seed_all(cfg["seed"] + example_id)
            messages = [{"role": "system", "content": cfg["system_instruction"]}, {"role": "user", "content": ds[example_id]["question"]}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
            enc.pop("token_type_ids", None)
            if enc["input_ids"].shape[-1] > cfg["max_prompt_tokens"]:
                raise ValueError(f"Prompt {example_id} exceeds max_prompt_tokens")
            enc = {k: v.cuda() for k, v in enc.items()}
            captured = {}
            original = model.core_block_forward
            def wrapped(this, x, input_embeds, *args, **kwargs):
                if "h0" not in captured:
                    captured["h0"] = x.detach().float().cpu().numpy()[0]
                    captured["x"] = input_embeds.detach().float().cpu().numpy()[0]
                result = original(x, input_embeds, *args, **kwargs)
                captured["h16"] = result[0].detach().float().cpu().numpy()[0]
                return result
            model.core_block_forward = MethodType(wrapped, model)
            try:
                with torch.inference_mode():
                    model(**enc, num_steps=cfg["depth"], use_cache=False, output_details={"return_logits": False, "return_latents": False, "return_head": False, "return_stats": False})
            finally:
                model.core_block_forward = original
            np.savez(out_path, h0=captured["h0"].astype(np.float16), x=captured["x"].astype(np.float16), h16=captured["h16"].astype(np.float16), input_ids=enc["input_ids"].cpu().numpy()[0])
            print(f"{split} {example_id}: {captured['h16'].shape[0]} tokens", flush=True)
    (output / "metadata.json").write_text(json.dumps(cfg, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/003_direct_jump.json"))
    p.add_argument("--output", type=Path, default=Path("results/003_direct_jump/cache"))
    a = p.parse_args()
    main(a.config, a.output)
