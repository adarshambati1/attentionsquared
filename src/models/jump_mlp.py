"""Direct-jump baseline model shared by active evaluation code."""

from __future__ import annotations

import torch
from torch import nn


class JumpMLP(nn.Module):
    """Tokenwise MLP predicting a target latent state from ``(h0, x)``."""

    def __init__(self, hidden: int, residual: bool) -> None:
        super().__init__()
        self.residual = residual
        self.in_proj = nn.Linear(2 * hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, h0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([h0, x], dim=-1)
        delta = self.out_proj(torch.nn.functional.gelu(self.in_proj(inputs)))
        return h0 + delta if self.residual else delta
