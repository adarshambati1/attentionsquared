"""Capture Huginn recurrent-core trajectories without training or generation."""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np


class HuginnTrajectoryCapture:
    def __init__(self, model_id: str, revision: str, dtype: str = "bfloat16") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=dtype_map[dtype], trust_remote_code=True
        ).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def capture(self, question: str, system_instruction: str, depth: int, seed: int) -> dict[str, np.ndarray]:
        torch = self.torch
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded.pop("token_type_ids", None)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        final_token_states: list[np.ndarray] = []
        mean_token_states: list[np.ndarray] = []
        original = self.model.core_block_forward

        def pool(state: Any) -> None:
            final_token_states.append(state[:, -1, :].detach().float().cpu().numpy()[0])
            mean_token_states.append(state.mean(dim=1).detach().float().cpu().numpy()[0])

        def wrapped(this_model: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
            if not final_token_states:
                pool(x)  # h_0: initialized recurrent state
            result = original(x, *args, **kwargs)
            pool(result[0])  # h_{d+1}: state after one recurrent core application
            return result

        self.model.core_block_forward = MethodType(wrapped, self.model)
        try:
            with torch.inference_mode():
                self.model(
                    **encoded,
                    num_steps=depth,
                    use_cache=False,
                    output_details={
                        "return_logits": False,
                        "return_latents": False,
                        "return_head": False,
                        "return_stats": False,
                    },
                )
        finally:
            self.model.core_block_forward = original

        if len(final_token_states) != depth + 1:
            raise RuntimeError(f"Expected {depth + 1} states, captured {len(final_token_states)}")
        return {
            "final_token": np.stack(final_token_states),
            "mean_token": np.stack(mean_token_states),
            "prompt_tokens": np.asarray([encoded["input_ids"].shape[-1]], dtype=np.int32),
        }


def run(config: dict[str, Any], output: Path) -> None:
    from datasets import load_dataset

    dataset = load_dataset(
        config["dataset_id"], config["dataset_config"], split=config["dataset_split"], revision=config["dataset_revision"]
    )
    capture = HuginnTrajectoryCapture(config["model_id"], config["model_revision"], config["dtype"])
    trajectories: dict[str, np.ndarray] = {}
    prompt_tokens = []
    for example_id in config["example_ids"]:
        item = capture.capture(
            dataset[example_id]["question"], config["system_instruction"], config["depth"], config["seed"] + example_id
        )
        trajectories[f"example_{example_id}_final_token"] = item["final_token"]
        trajectories[f"example_{example_id}_mean_token"] = item["mean_token"]
        prompt_tokens.append(int(item["prompt_tokens"][0]))
        print(f"captured example {example_id}: {config['depth'] + 1} states")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **trajectories)
    output.with_suffix(".json").write_text(
        json.dumps({"config": config, "prompt_tokens": prompt_tokens, "state_layout": "[depth+1, hidden]"}, indent=2)
    )
