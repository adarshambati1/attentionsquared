"""Frozen Huginn with independent per-core-layer attention over prior-loop states."""
from __future__ import annotations

from typing import Any
import torch
from torch import nn

from src.models.latent_history_huginn import LatentHistoryOutput, PerTokenHistoryAttention


class PerLayerHistoryHuginn(nn.Module):
    def __init__(self, huginn: nn.Module, projection_size: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.huginn = huginn
        hidden = int(huginn.config.n_embd)
        self.history_attention = nn.ModuleList(
            PerTokenHistoryAttention(hidden, projection_size, num_heads)
            for _ in huginn.transformer.core_block
        )
        for parameter in huginn.parameters():
            parameter.requires_grad_(False)
        huginn.eval()

    def train(self, mode: bool = True):
        super().train(mode); self.huginn.eval(); return self

    def trainable_parameters(self):
        return self.history_attention.parameters()

    def new_dynamic_cache(self):
        cls = self.huginn.forward.__globals__.get("HuginnDynamicCache")
        if cls is None: raise RuntimeError("pinned Huginn does not expose HuginnDynamicCache")
        return cls()

    def forward(self, input_ids: torch.Tensor, input_states: torch.Tensor, *, depth: int = 8, mode: str = "learned", attention_mask: torch.Tensor | None = None, past_key_values: Any = None, use_cache: bool = False, cache_position: torch.Tensor | None = None, return_loop_states: bool = False) -> LatentHistoryOutput:
        if input_ids.shape != input_states.shape[:2]: raise ValueError("input/state shape mismatch")
        if attention_mask is None: attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if use_cache and past_key_values is None: past_key_values = self.new_dynamic_cache()
        frequencies = self.huginn.freqs_cis[:, :input_ids.shape[1]] if cache_position is None else self.huginn.freqs_cis[:, cache_position]
        block_index = torch.tensor(-1, device="cpu", dtype=torch.long)
        recurrent_input = self.huginn.transformer.wte(input_ids)
        if self.huginn.emb_scale != 1: recurrent_input = recurrent_input * self.huginn.emb_scale
        for block in self.huginn.transformer.prelude:
            block_index += 1; recurrent_input = block(recurrent_input, frequencies, block_index, None, past_key_values)
        state = input_states
        layer_histories: list[list[torch.Tensor]] = [[] for _ in self.huginn.transformer.core_block]
        exposed = [state] if return_loop_states else []
        for loop_index in range(depth):
            state = self.huginn._maybe_inject_noise(state, loop_index)
            state = self.huginn.transformer.adapter(torch.cat([state, recurrent_input.to(state.device)], dim=-1))
            for layer_index, block in enumerate(self.huginn.transformer.core_block):
                layer_histories[layer_index].append(state)
                memory = self.history_attention[layer_index](state, layer_histories[layer_index], mode=mode, token_mask=attention_mask)
                block_index += 1
                state = block(state + memory, frequencies, block_index, None, past_key_values)
            if return_loop_states: exposed.append(state)
        normalized = self.huginn.transformer.ln_f(state)
        coda = normalized; block_index = torch.tensor(0, device="cpu", dtype=torch.long)
        for block in self.huginn.transformer.coda:
            block_index -= 1; coda = block(coda, frequencies, block_index, None, past_key_values)
        logits = self.huginn.lm_head(self.huginn.transformer.ln_f(coda)).float()
        return LatentHistoryOutput(logits, normalized, past_key_values, tuple(exposed))
