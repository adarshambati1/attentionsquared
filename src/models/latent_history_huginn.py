"""Huginn with one shared per-token attention module over completed loop states."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

HistoryMode = Literal["learned", "current_only", "uniform", "disabled"]


class PerTokenHistoryAttention(nn.Module):
    """Attend across loop history independently at every token position."""

    def __init__(self, hidden_size: int, projection_size: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        if projection_size % num_heads:
            raise ValueError("projection_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.projection_size = int(projection_size)
        self.num_heads = int(num_heads)
        self.head_dim = projection_size // num_heads
        self.q_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.out_proj = nn.Linear(projection_size, hidden_size, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        current: torch.Tensor,
        completed_states: list[torch.Tensor],
        *,
        mode: HistoryMode = "learned",
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not completed_states or completed_states[-1] is not current:
            raise ValueError("history must end with the literal current-state object")
        if any(state.shape != current.shape for state in completed_states):
            raise ValueError("all completed states must match current [B,T,H]")
        if mode == "disabled":
            return torch.zeros_like(current)
        memory = completed_states[-1:] if mode == "current_only" else completed_states
        history = torch.stack(memory, dim=2)  # [B,T,L,H], never mixes token positions
        batch, tokens, loops, _ = history.shape
        values = self.v_proj(history).view(batch, tokens, loops, self.num_heads, self.head_dim)
        if mode == "uniform":
            mixed = values.mean(dim=2)
        else:
            query = self.q_proj(current).view(batch, tokens, self.num_heads, self.head_dim)
            keys = self.k_proj(history).view(batch, tokens, loops, self.num_heads, self.head_dim)
            scores = torch.einsum("bthd,btlhd->bthl", query, keys) * (self.head_dim ** -0.5)
            weights = scores.softmax(dim=-1)
            mixed = torch.einsum("bthl,btlhd->bthd", weights, values)
        output = self.out_proj(mixed.reshape(batch, tokens, self.projection_size))
        if token_mask is not None:
            if token_mask.shape != current.shape[:2]:
                raise ValueError("token_mask must have shape [B,T]")
            output = output * token_mask.unsqueeze(-1).to(output.dtype)
        return output


@dataclass
class LatentHistoryOutput:
    logits: torch.Tensor
    latent_states: torch.Tensor
    past_key_values: Any
    loop_states: tuple[torch.Tensor, ...]


class LatentHistoryHuginn(nn.Module):
    """A frozen Huginn execution path with trainable history attention inserted before each core loop."""

    def __init__(self, huginn: nn.Module, projection_size: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.huginn = huginn
        hidden_size = int(huginn.config.n_embd)
        self.history_attention = PerTokenHistoryAttention(hidden_size, projection_size, num_heads)
        for parameter in self.huginn.parameters():
            parameter.requires_grad_(False)
        self.huginn.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.huginn.eval()
        return self

    def new_dynamic_cache(self):
        cache_class = self.huginn.forward.__globals__.get("HuginnDynamicCache")
        if cache_class is None:
            raise RuntimeError("pinned Huginn module does not expose HuginnDynamicCache")
        return cache_class()

    def forward(
        self,
        input_ids: torch.Tensor,
        input_states: torch.Tensor,
        *,
        depth: int = 8,
        mode: HistoryMode = "learned",
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        return_loop_states: bool = False,
    ) -> LatentHistoryOutput:
        if depth <= 0:
            raise ValueError("depth must be positive")
        if input_ids.shape != input_states.shape[:2]:
            raise ValueError("input_ids and input_states must agree on [B,T]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if use_cache and past_key_values is None:
            past_key_values = self.new_dynamic_cache()
        if cache_position is None:
            frequencies = self.huginn.freqs_cis[:, : input_ids.shape[1]]
        else:
            frequencies = self.huginn.freqs_cis[:, cache_position]
        block_index = torch.tensor(-1, device="cpu", dtype=torch.long)
        x = self.huginn.transformer.wte(input_ids)
        if self.huginn.emb_scale != 1:
            x = x * self.huginn.emb_scale
        for block in self.huginn.transformer.prelude:
            block_index += 1
            x = block(x, frequencies, block_index, None, past_key_values)
        state = input_states
        completed = [state]
        exposed_states = [state] if return_loop_states else []
        for loop_index in range(depth):
            memory = self.history_attention(state, completed, mode=mode, token_mask=attention_mask)
            state, block_index = self.huginn.core_block_forward(
                state + memory, x, frequencies, None, past_key_values, block_index, loop_index
            )
            completed.append(state)
            if return_loop_states:
                exposed_states.append(state)
        normalized = self.huginn.transformer.ln_f(state)
        coda_state = normalized
        block_index = torch.tensor(0, device="cpu", dtype=torch.long)
        for block in self.huginn.transformer.coda:
            block_index -= 1
            coda_state = block(coda_state, frequencies, block_index, None, past_key_values)
        coda_state = self.huginn.transformer.ln_f(coda_state)
        logits = self.huginn.lm_head(coda_state).float()
        return LatentHistoryOutput(
            logits=logits,
            latent_states=normalized,
            past_key_values=past_key_values,
            loop_states=tuple(exposed_states),
        )

    def trainable_parameters(self):
        return self.history_attention.parameters()
