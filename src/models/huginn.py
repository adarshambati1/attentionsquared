"""Huginn-specific loading and generation adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationResult:
    text: str
    generated_tokens: int
    prompt_tokens: int


class HuginnAdapter:
    """Small stable interface around Huginn's remote-code Transformers model."""

    def __init__(self, model_id: str, revision: str, dtype: str = "bfloat16") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype_map[dtype],
            trust_remote_code=True,
        ).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def format_prompt(self, question: str, system_instruction: str) -> str:
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, question: str, depth: int, system_instruction: str, max_new_tokens: int) -> GenerationResult:
        from transformers import GenerationConfig

        prompt = self.format_prompt(question, system_instruction)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        self._synchronize()
        outputs = self.model.generate(
            **encoded,
            generation_config=GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            ),
            # Huginn's documented recurrent-depth control. Do not put this in GenerationConfig.
            num_steps=depth,
            tokenizer=self.tokenizer,
        )
        self._synchronize()
        sequence = outputs.sequences[0]
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        generated = sequence[prompt_tokens:]
        return GenerationResult(
            text=self.tokenizer.decode(generated, skip_special_tokens=False),
            generated_tokens=int(generated.shape[-1]),
            prompt_tokens=prompt_tokens,
        )

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
