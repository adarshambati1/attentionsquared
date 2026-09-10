"""Slow, canonical full-prefix autoregressive evaluator for Attention²."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from src.evaluation.correctness import generation_status, stop_token_ids
from src.training.functional_protocol import paired_huginn_h0


FULL_PREFIX_PROTOCOL = "attention2-full-prefix-autoregressive-v1"


@dataclass(frozen=True)
class FullPrefixGeneration:
    token_ids: tuple[int, ...]
    text: str
    prefix_lengths: tuple[int, ...]
    ended_naturally: bool
    hit_max_new_tokens: bool


class FullPrefixAttention2Evaluator:
    """Recompute prelude, h0, Attention², and frozen coda for every token."""

    def __init__(
        self,
        huginn: Any,
        attention2: Any,
        tokenizer: Any,
        *,
        base_seed: int = 3000,
        seed_index: int = 0,
    ) -> None:
        self.huginn = huginn
        self.attention2 = attention2
        self.tokenizer = tokenizer
        self.base_seed = int(base_seed)
        self.seed_index = int(seed_index)

    def _prelude(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, Any, torch.Tensor]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("canonical evaluator currently requires input_ids [1,T]")
        frequencies = self.huginn.freqs_cis[:, : input_ids.shape[1]]
        x = self.huginn.transformer.wte(input_ids)
        if self.huginn.emb_scale != 1:
            x = x * self.huginn.emb_scale
        block_index = torch.tensor(-1, device="cpu", dtype=torch.long)
        for block in self.huginn.transformer.prelude:
            block_index += 1
            x = block(x, frequencies, block_index, None, None)
        return x, frequencies, block_index

    def _coda_last_logits(
        self, z16: torch.Tensor, frequencies: Any
    ) -> torch.Tensor:
        # Training and the historical patched evaluator pass predicted z16 as
        # input_states with num_steps=0, whose iterate_forward applies ln_f
        # before coda. Preserve that exact functional interface here.
        state = self.huginn.transformer.ln_f(z16)
        block_index = torch.tensor(0, device="cpu", dtype=torch.long)
        for block in self.huginn.transformer.coda:
            block_index -= 1
            state = block(state, frequencies, block_index, None, None)
        state = self.huginn.transformer.ln_f(state)
        return self.huginn.lm_head(state[:, -1:]).float()[:, 0]

    def next_token_logits(
        self, full_prefix_ids: torch.Tensor, *, example_id: int
    ) -> torch.Tensor:
        """Return next-token logits after recomputing the complete prefix."""
        if full_prefix_ids.shape[1] < 1:
            raise ValueError("full prefix must contain at least one token")
        with torch.inference_mode(), torch.autocast(
            device_type=full_prefix_ids.device.type,
            dtype=torch.bfloat16,
            enabled=full_prefix_ids.device.type == "cuda",
        ):
            x, frequencies, _ = self._prelude(full_prefix_ids)
            h0, _ = paired_huginn_h0(
                self.huginn,
                x,
                base_seed=self.base_seed,
                example_id=example_id,
                seed_index=self.seed_index,
                step=0,
            )
            token_mask = torch.ones(
                full_prefix_ids.shape, dtype=torch.bool, device=full_prefix_ids.device
            )
            z, _, _ = self.attention2(
                h0.float(), x.float(), token_mask=token_mask
            )
            return self._coda_last_logits(z[:, 15].to(h0.dtype), frequencies)

    def generate(
        self,
        prompt_ids: torch.Tensor,
        *,
        example_id: int,
        max_new_tokens: int,
    ) -> FullPrefixGeneration:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError("prompt_ids must have shape [1,T]")
        current = prompt_ids.clone()
        generated: list[int] = []
        lengths: list[int] = []
        stops = {
            int(token)
            for token in stop_token_ids(self.tokenizer)
            if token is not None and int(token) >= 0
        }
        for _ in range(max_new_tokens):
            lengths.append(int(current.shape[1]))
            logits = self.next_token_logits(current, example_id=example_id)
            next_token = int(logits[0].argmax().item())
            del logits
            generated.append(next_token)
            current = torch.cat(
                [
                    current,
                    torch.tensor(
                        [[next_token]], device=current.device, dtype=current.dtype
                    ),
                ],
                dim=1,
            )
            if next_token in stops:
                break
        status = generation_status(
            generated,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids(self.tokenizer),
        )
        return FullPrefixGeneration(
            token_ids=tuple(generated),
            text=self.tokenizer.decode(generated, skip_special_tokens=False),
            prefix_lengths=tuple(lengths),
            ended_naturally=status.ended_naturally,
            hit_max_new_tokens=status.hit_max_new_tokens,
        )


def strictly_growing_full_prefix_lengths(
    lengths: Sequence[int], prompt_length: int
) -> bool:
    return tuple(int(value) for value in lengths) == tuple(
        range(prompt_length, prompt_length + len(lengths))
    )
