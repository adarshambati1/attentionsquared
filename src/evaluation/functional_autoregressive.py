"""Slow canonical full-prefix evaluators for frozen Huginn and Attention²."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Sequence

import torch

from src.evaluation.correctness import generation_status, stop_token_ids
from src.training.functional_protocol import paired_huginn_h0


FULL_PREFIX_PROTOCOL = "attention2-full-prefix-autoregressive-v1"
HUGINN_D16_FULL_PREFIX_PROTOCOL = "huginn-d16-full-prefix-autoregressive-v1"


@dataclass(frozen=True)
class PrefixTrace:
    step: int
    prefix_length: int
    prefix_token_ids_sha256: str


@dataclass(frozen=True)
class FullPrefixGeneration:
    token_ids: tuple[int, ...]
    text: str
    prefix_lengths: tuple[int, ...]
    ended_naturally: bool
    hit_max_new_tokens: bool
    prefix_traces: tuple[PrefixTrace, ...] = ()


def _prefix_sha256(input_ids: torch.Tensor) -> str:
    values = [int(token) for token in input_ids[0].tolist()]
    payload = b"".join(value.to_bytes(8, "little", signed=True) for value in values)
    return hashlib.sha256(payload).hexdigest()


def normalized_state_coda_logits(
    huginn: Any,
    normalized_state: torch.Tensor,
    frequencies: Any,
    *,
    last_only: bool = False,
) -> torch.Tensor:
    """Apply the frozen coda to an already-normalized true Huginn latent."""
    state = normalized_state
    block_index = torch.tensor(0, device="cpu", dtype=torch.long)
    for block in huginn.transformer.coda:
        block_index -= 1
        state = block(state, frequencies, block_index, None, None)
    state = huginn.transformer.ln_f(state)
    selected = state[:, -1:] if last_only else state
    logits = huginn.lm_head(selected).float()
    return logits[:, 0] if last_only else logits


def canonical_direct_coda_logits(
    huginn: Any,
    state: torch.Tensor,
    frequencies: Any,
    *,
    last_only: bool = False,
) -> torch.Tensor:
    """Apply the exact no-step frozen coda path without a model KV cache.

    ``state`` is treated exactly like ``input_states`` passed to Huginn with
    ``num_steps=0``: final recurrent normalization is applied before coda.
    The returned logits are transient and are never stored by this module.
    """
    normalized = huginn.transformer.ln_f(state)
    return normalized_state_coda_logits(
        huginn, normalized, frequencies, last_only=last_only
    )


class _FullPrefixGreedyEvaluator:
    tokenizer: Any

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
        traces: list[PrefixTrace] = []
        stop_ids = stop_token_ids(self.tokenizer)
        stops = {
            int(token)
            for token in stop_ids
            if token is not None and int(token) >= 0
        }
        for step in range(max_new_tokens):
            lengths.append(int(current.shape[1]))
            traces.append(PrefixTrace(step, int(current.shape[1]), _prefix_sha256(current)))
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
            stop_token_ids=stop_ids,
        )
        return FullPrefixGeneration(
            token_ids=tuple(generated),
            text=self.tokenizer.decode(generated, skip_special_tokens=False),
            prefix_lengths=tuple(lengths),
            ended_naturally=status.ended_naturally,
            hit_max_new_tokens=status.hit_max_new_tokens,
            prefix_traces=tuple(traces),
        )


class FullPrefixAttention2Evaluator(_FullPrefixGreedyEvaluator):
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
        return canonical_direct_coda_logits(
            self.huginn, z16, frequencies, last_only=True
        )

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


class FullPrefixHuginnD16Evaluator(_FullPrefixGreedyEvaluator):
    """Recompute frozen Huginn D16 over the complete prefix at every token."""

    def __init__(
        self,
        huginn: Any,
        tokenizer: Any,
        *,
        base_seed: int = 0,
        seed_index: int = 0,
    ) -> None:
        self.huginn = huginn
        self.tokenizer = tokenizer
        self.base_seed = int(base_seed)
        self.seed_index = int(seed_index)

    def _prelude(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, Any]:
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
        return x, frequencies

    def materialize_h0(
        self, full_prefix_ids: torch.Tensor, *, example_id: int
    ) -> tuple[torch.Tensor, int]:
        """Materialize the one deterministic h0 for a corresponding full prefix."""
        if full_prefix_ids.ndim != 2 or full_prefix_ids.shape[0] != 1:
            raise ValueError("full_prefix_ids must have shape [1,T]")
        if full_prefix_ids.shape[1] < 1:
            raise ValueError("full prefix must contain at least one token")
        with torch.inference_mode(), torch.autocast(
            device_type=full_prefix_ids.device.type,
            dtype=torch.bfloat16,
            enabled=full_prefix_ids.device.type == "cuda",
        ):
            x, _ = self._prelude(full_prefix_ids)
            return paired_huginn_h0(
                self.huginn,
                x,
                base_seed=self.base_seed,
                example_id=example_id,
                seed_index=self.seed_index,
                step=0,
            )

    def full_prefix_logits_with_h0(
        self, full_prefix_ids: torch.Tensor, h0: torch.Tensor
    ) -> torch.Tensor:
        """Run D16 with the caller's materialized h0 object, without copying it."""
        if full_prefix_ids.ndim != 2 or full_prefix_ids.shape[0] != 1:
            raise ValueError("full_prefix_ids must have shape [1,T]")
        if full_prefix_ids.shape[1] < 1:
            raise ValueError("full prefix must contain at least one token")
        if h0.shape[:2] != full_prefix_ids.shape:
            raise ValueError("h0 batch/sequence shape must match the complete prefix")
        with torch.inference_mode(), torch.autocast(
            device_type=full_prefix_ids.device.type,
            dtype=torch.bfloat16,
            enabled=full_prefix_ids.device.type == "cuda",
        ):
            output = self.huginn(
                input_ids=full_prefix_ids,
                attention_mask=torch.ones_like(full_prefix_ids, dtype=torch.bool),
                input_states=h0,
                num_steps=16,
                use_cache=False,
                return_dict=True,
            )
            return output.logits.float()

    def next_token_logits_with_h0(
        self, full_prefix_ids: torch.Tensor, h0: torch.Tensor
    ) -> torch.Tensor:
        logits = self.full_prefix_logits_with_h0(full_prefix_ids, h0)
        last = logits[:, -1].clone()
        del logits
        return last

    def full_prefix_logits(
        self, full_prefix_ids: torch.Tensor, *, example_id: int
    ) -> torch.Tensor:
        """Return transient D16 logits after a fresh deterministic full-prefix pass."""
        h0, _ = self.materialize_h0(full_prefix_ids, example_id=example_id)
        return self.full_prefix_logits_with_h0(full_prefix_ids, h0)

    def next_token_logits(
        self, full_prefix_ids: torch.Tensor, *, example_id: int
    ) -> torch.Tensor:
        h0, _ = self.materialize_h0(full_prefix_ids, example_id=example_id)
        return self.next_token_logits_with_h0(full_prefix_ids, h0)


def strictly_growing_full_prefix_lengths(
    lengths: Sequence[int], prompt_length: int
) -> bool:
    return tuple(int(value) for value in lengths) == tuple(
        range(prompt_length, prompt_length + len(lengths))
    )
