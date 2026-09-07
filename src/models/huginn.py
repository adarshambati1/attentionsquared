"""Huginn-specific loading and generation adapter."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any


@dataclass
class GenerationResult:
    text: str
    generated_tokens: int
    prompt_tokens: int
    generation_latency_seconds: float
    time_to_first_token_seconds: float | None
    single_forward_latency_seconds: float
    ended_naturally: bool
    hit_max_new_tokens: bool


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
        from transformers import GenerationConfig, TextIteratorStreamer

        prompt = self.format_prompt(question, system_instruction)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        # Huginn is decoder-only and rejects tokenizer-generated segment IDs.
        encoded.pop("token_type_ids", None)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)
        result_holder: dict[str, Any] = {}

        def run_generation() -> None:
            try:
                result_holder["outputs"] = self.model.generate(
                    **encoded,
                    generation_config=GenerationConfig(
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        return_dict_in_generate=True,
                        # Huginn's custom DynamicCache is sparse by design; do not force
                        # Transformers 4.44.2 to convert it back to a legacy tuple.
                        return_legacy_cache=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    ),
                    streamer=streamer,
                    # Huginn's documented recurrent-depth control. Do not put this in GenerationConfig.
                    num_steps=depth,
                    tokenizer=self.tokenizer,
                )
            except BaseException as error:  # propagate errors from the generation thread
                result_holder["error"] = error

        self._synchronize()
        start = time.perf_counter()
        worker = threading.Thread(target=run_generation, daemon=True)
        worker.start()
        first_token_time: float | None = None
        for _ in streamer:
            if first_token_time is None:
                first_token_time = time.perf_counter()
        worker.join()
        self._synchronize()
        generation_latency = time.perf_counter() - start
        if "error" in result_holder:
            raise result_holder["error"]
        outputs = result_holder["outputs"]
        sequence = outputs.sequences[0]
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        generated = sequence[prompt_tokens:]
        generated_tokens = int(generated.shape[-1])
        eos_ids = {self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|end_text|>"), self.tokenizer.convert_tokens_to_ids("<|end_turn|>")}
        ended_naturally = bool(generated_tokens and int(generated[-1]) in eos_ids)
        self._synchronize()
        single_start = time.perf_counter()
        with self.torch.inference_mode():
            self.model(**encoded, num_steps=depth, use_cache=False, return_dict=True)
        self._synchronize()
        single_forward_latency = time.perf_counter() - single_start
        return GenerationResult(
            text=self.tokenizer.decode(generated, skip_special_tokens=False),
            generated_tokens=generated_tokens,
            prompt_tokens=prompt_tokens,
            generation_latency_seconds=generation_latency,
            time_to_first_token_seconds=(first_token_time - start) if first_token_time is not None else None,
            single_forward_latency_seconds=single_forward_latency,
            ended_naturally=ended_naturally,
            hit_max_new_tokens=generated_tokens >= max_new_tokens and not ended_naturally,
        )

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
