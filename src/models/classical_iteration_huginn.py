"""Parameter-free classical recurrent-state updates around frozen Huginn."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal
import torch
from torch import nn

ClassicalMode = Literal["plain", "halpern", "heavy_ball", "anderson"]

@dataclass
class ClassicalOutput:
    logits: torch.Tensor
    latent_states: torch.Tensor
    past_key_values: Any
    loop_states: tuple[torch.Tensor, ...]
    anderson_fallbacks: int

class ClassicalIterationHuginn(nn.Module):
    def __init__(self, huginn: nn.Module) -> None:
        super().__init__(); self.huginn = huginn
        for parameter in huginn.parameters(): parameter.requires_grad_(False)
        huginn.eval()

    def train(self, mode: bool = True):
        super().train(False); self.huginn.eval(); return self

    def new_dynamic_cache(self):
        cls = self.huginn.forward.__globals__.get("HuginnDynamicCache")
        if cls is None: raise RuntimeError("pinned Huginn does not expose HuginnDynamicCache")
        return cls()

    @staticmethod
    def _anderson_mix(fs: list[torch.Tensor], residuals: list[torch.Tensor], *, window: int, ridge: float) -> tuple[torch.Tensor, int]:
        recent_f = fs[-window:]; recent_r = residuals[-window:]
        count = len(recent_f)
        if count == 1: return recent_f[-1], 0
        f = torch.stack([value.float() for value in recent_f], dim=2)
        r = torch.stack([value.float() for value in recent_r], dim=2)
        gram = torch.einsum("btlh,btmh->btlm", r, r) / r.shape[-1]
        eye = torch.eye(count, device=gram.device, dtype=gram.dtype)
        gram = gram + ridge * eye
        shape = (*gram.shape[:2], count + 1, count + 1)
        system = torch.zeros(shape, device=gram.device, dtype=gram.dtype)
        system[..., :count, :count] = gram
        system[..., :count, count] = 1
        system[..., count, :count] = 1
        rhs = torch.zeros((*gram.shape[:2], count + 1, 1), device=gram.device, dtype=gram.dtype)
        rhs[..., count, 0] = 1
        try:
            solution = torch.linalg.solve(system, rhs)[..., :count, 0]
            finite = torch.isfinite(solution).all(dim=-1)
        except RuntimeError:
            solution = torch.zeros((*gram.shape[:2], count), device=gram.device, dtype=gram.dtype)
            finite = torch.zeros(gram.shape[:2], device=gram.device, dtype=torch.bool)
        fallback = torch.zeros_like(solution); fallback[..., -1] = 1
        coefficients = torch.where(finite.unsqueeze(-1), solution, fallback)
        mixed = torch.einsum("btl,btlh->bth", coefficients, f)
        return mixed.to(recent_f[-1].dtype), int((~finite).sum().item())

    def forward(self, input_ids: torch.Tensor, input_states: torch.Tensor, *, depth: int = 8, mode: ClassicalMode = "plain", momentum: float = 0.1, anderson_window: int = 3, anderson_ridge: float = 1e-4, attention_mask: torch.Tensor | None = None, past_key_values: Any = None, use_cache: bool = False, cache_position: torch.Tensor | None = None, return_loop_states: bool = False) -> ClassicalOutput:
        if depth != 8: raise ValueError("classical comparison is frozen at D8")
        if input_ids.shape != input_states.shape[:2]: raise ValueError("input/state shape mismatch")
        if mode == "heavy_ball" and momentum not in (0.05, 0.1, 0.2): raise ValueError("momentum outside preregistered sweep")
        if mode == "anderson" and anderson_window not in (2, 3, 4): raise ValueError("window outside preregistered sweep")
        if anderson_ridge != 1e-4: raise ValueError("Anderson ridge is frozen at 1e-4")
        if use_cache and past_key_values is None: past_key_values = self.new_dynamic_cache()
        frequencies = self.huginn.freqs_cis[:, :input_ids.shape[1]] if cache_position is None else self.huginn.freqs_cis[:, cache_position]
        block_index = torch.tensor(-1, device="cpu", dtype=torch.long)
        recurrent_input = self.huginn.transformer.wte(input_ids)
        if self.huginn.emb_scale != 1: recurrent_input = recurrent_input * self.huginn.emb_scale
        for block in self.huginn.transformer.prelude:
            block_index += 1; recurrent_input = block(recurrent_input, frequencies, block_index, None, past_key_values)
        h0 = input_states; state = h0; previous = None; states = [state]; fs=[]; residuals=[]; fallbacks=0
        for loop_index in range(depth):
            ordinary, block_index = self.huginn.core_block_forward(state, recurrent_input, frequencies, None, past_key_values, block_index, loop_index)
            if mode == "plain": next_state = ordinary
            elif mode == "halpern":
                alpha = 1.0 / (loop_index + 2.0); next_state = alpha * h0 + (1.0 - alpha) * ordinary
            elif mode == "heavy_ball":
                next_state = ordinary if previous is None else ordinary + momentum * (state - previous)
            else:
                fs.append(ordinary); residuals.append(ordinary - state)
                next_state, count = self._anderson_mix(fs, residuals, window=anderson_window, ridge=anderson_ridge); fallbacks += count
            if not bool(torch.isfinite(next_state).all()): raise FloatingPointError(f"nonfinite {mode} state at loop {loop_index}")
            previous, state = state, next_state; states.append(state)
        normalized = self.huginn.transformer.ln_f(state)
        coda = normalized; block_index = torch.tensor(0, device="cpu", dtype=torch.long)
        for block in self.huginn.transformer.coda:
            block_index -= 1; coda = block(coda, frequencies, block_index, None, past_key_values)
        logits = self.huginn.lm_head(self.huginn.transformer.ln_f(coda)).float()
        return ClassicalOutput(logits, normalized, past_key_values, tuple(states) if return_loop_states else (), fallbacks)
