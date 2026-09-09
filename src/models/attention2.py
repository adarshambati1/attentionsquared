"""Minimal shared-round Attention² operator for Experiment 004."""

from __future__ import annotations

import torch
from torch import nn


class Attention2Lite(nn.Module):
    def __init__(
        self,
        hidden: int,
        depth: int = 16,
        heads: int = 16,
        rounds: int = 4,
        mlp_ratio: int = 1,
        depth_scale: float = 1.0,
        film: bool = False,
    ) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden size must be divisible by attention heads")
        self.depth = depth
        self.rounds = rounds
        self.hidden = hidden
        self.depth_scale = depth_scale
        self.film = film
        self.init_proj = nn.Linear(2 * hidden, hidden)
        self.depth_embedding = nn.Parameter(torch.zeros(depth, hidden))
        nn.init.normal_(self.depth_embedding, std=0.02)
        if film:
            self.film_gamma = nn.Parameter(torch.ones(depth, hidden))
            self.film_beta = nn.Parameter(torch.zeros(depth, hidden))
        self.token_norm = nn.LayerNorm(hidden)
        self.token_attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.depth_norm = nn.LayerNorm(hidden)
        self.depth_attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        middle = hidden * mlp_ratio
        self.mlp_norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, middle),
            nn.GELU(),
            nn.Linear(middle, hidden),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.init_proj.weight)
        nn.init.zeros_(self.init_proj.bias)

    def initialize(self, h0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if h0.shape != x.shape or h0.ndim != 3 or h0.shape[-1] != self.hidden:
            raise ValueError("h0 and x must have matching shape [B,T,H]")
        base = self.init_proj(torch.cat([h0, x], dim=-1))
        if self.film:
            return (
                base[:, None, :, :] * self.film_gamma[None, :, None, :]
                + self.film_beta[None, :, None, :]
            )
        return (
            base[:, None, :, :]
            + self.depth_scale * self.depth_embedding[None, :, None, :]
        )

    def _validated_token_mask(
        self, token_mask: torch.Tensor | None, *, batch: int, tokens: int, device
    ) -> torch.Tensor:
        if token_mask is None:
            return torch.ones((batch, tokens), dtype=torch.bool, device=device)
        if token_mask.shape != (batch, tokens) or token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be bool with shape [B,T]")
        token_mask = token_mask.to(device=device)
        if not token_mask.any(dim=1).all():
            raise ValueError("every sequence must contain at least one real token")
        return token_mask

    def operator(
        self,
        z: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if z.ndim != 4:
            raise ValueError("z must have shape [B,D,T,H]")
        batch, depth, tokens, hidden = z.shape
        if depth != self.depth or hidden != self.hidden:
            raise ValueError("z depth/hidden dimensions do not match the model")
        valid_tokens = self._validated_token_mask(
            token_mask, batch=batch, tokens=tokens, device=z.device
        )
        valid_state = valid_tokens[:, None, :, None]
        z = z.masked_fill(~valid_state, 0)

        token_query = self.token_norm(z).reshape(batch * depth, tokens, hidden)
        causal_mask = torch.triu(
            torch.ones(tokens, tokens, device=z.device, dtype=torch.bool), diagonal=1
        )
        key_padding_mask = (
            ~valid_tokens[:, None, :]
            .expand(batch, depth, tokens)
            .reshape(batch * depth, tokens)
        )
        token_output, _ = self.token_attn(
            token_query,
            token_query,
            token_query,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        z = (z + token_output.reshape(batch, depth, tokens, hidden)).masked_fill(
            ~valid_state, 0
        )

        depth_query = (
            self.depth_norm(z).transpose(1, 2).reshape(batch * tokens, depth, hidden)
        )
        depth_output, depth_weights = self.depth_attn(
            depth_query,
            depth_query,
            depth_query,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        z = (
            z + depth_output.reshape(batch, tokens, depth, hidden).transpose(1, 2)
        ).masked_fill(~valid_state, 0)
        z = (z + self.mlp(self.mlp_norm(z))).masked_fill(~valid_state, 0)

        attention = {"depth": depth_weights} if return_attention else {}
        return z, attention

    def forward(
        self,
        h0: torch.Tensor,
        x: torch.Tensor,
        return_attention: bool = False,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[dict[str, torch.Tensor]]]:
        z = self.initialize(h0, x)
        all_states = []
        attentions = []
        for _ in range(self.rounds):
            z, attention = self.operator(
                z,
                token_mask=token_mask,
                return_attention=return_attention,
            )
            all_states.append(z)
            attentions.append(attention)
        return z, all_states, attentions
