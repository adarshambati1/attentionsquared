from types import SimpleNamespace

import torch
from torch import nn

from scripts.run_004_correctness_gate import (
    OUTPUT,
    PROTOCOL,
    live_d16_decomposed_coda_gold_control,
)

from src.evaluation.correctness import (
    aligned_answer_logits_and_targets,
    answer_bounds,
    token_normalized_kl,
)
from src.models.attention2 import Attention2Lite
from src.training.functional_objective import (
    any_parameter_gradient,
    causal_answer_loss_mask,
    freeze_module,
    functional_kl_loss,
    nonzero_gradient_parameter_count,
)


def test_literal_a_b_c_causal_alignment():
    # Sequence is prompt | A B C, where token IDs A/B/C are 4/5/6.
    input_ids = torch.tensor([[9, 8, 7, 4, 5, 6]])
    logits = torch.full((1, 6, 10), -100.0)
    logits[0, 2, 4] = 100.0  # logit before A predicts A
    logits[0, 3, 5] = 100.0  # logit before B predicts B
    logits[0, 4, 6] = 100.0  # logit before C predicts C
    selected_logits, targets = aligned_answer_logits_and_targets(
        logits, input_ids, answer_bounds(3, 6)
    )
    assert torch.equal(selected_logits.argmax(-1), targets)
    assert targets.tolist() == [[4, 5, 6]]


def test_loss_mask_excludes_prompt_post_valid_end_and_padding():
    attention = torch.tensor(
        [
            [True, True, True, True, True, True, False, False],
            [True, True, True, True, False, False, False, False],
        ]
    )
    mask = causal_answer_loss_mask(
        attention,
        answer_start=torch.tensor([3, 2]),
        valid_end=torch.tensor([6, 5]),
    )
    assert mask[0].tolist() == [False, False, True, True, True, False, False, False]
    # The final nominal answer target is padding, so its preceding logit is excluded.
    assert mask[1].tolist() == [False, True, True, False, False, False, False, False]
    assert not mask[:, :1].any()
    assert not mask[:, 6:].any()


def test_logits_outside_loss_mask_have_exactly_zero_influence():
    torch.manual_seed(20)
    teacher = torch.randn(2, 7, 11)
    student = torch.randn(2, 7, 11)
    attention = torch.tensor(
        [[True] * 5 + [False] * 2, [True] * 7], dtype=torch.bool
    )
    mask = causal_answer_loss_mask(attention, [2, 3], [5, 6])
    changed = student.clone()
    changed[~mask] += torch.randn_like(changed[~mask]) * 1000
    original_loss, _, original_count = functional_kl_loss(student, teacher, mask)
    changed_loss, _, changed_count = functional_kl_loss(changed, teacher, mask)
    torch.testing.assert_close(original_loss, changed_loss, rtol=0, atol=0)
    assert original_count.item() == changed_count.item() == 6


def test_gradient_flow_reaches_a2_and_z16_but_not_frozen_teacher_coda():
    torch.manual_seed(30)
    a2 = Attention2Lite(hidden=8, depth=2, heads=2, rounds=1)
    coda = freeze_module(nn.Linear(8, 13))
    h0 = torch.randn(1, 5, 8)
    x = torch.randn(1, 5, 8)
    teacher_h16 = torch.randn(1, 5, 8)
    attention = torch.ones((1, 5), dtype=torch.bool)
    loss_mask = causal_answer_loss_mask(attention, [2], [5])

    with torch.no_grad():
        teacher_logits = coda(teacher_h16)
    output, _, _ = a2(h0, x, token_mask=attention)
    z16 = output[:, -1]
    z16.retain_grad()
    student_logits = coda(z16)
    loss, _, count = functional_kl_loss(student_logits, teacher_logits, loss_mask)
    loss.backward()

    assert count.item() == 3
    assert teacher_h16.grad is None
    assert teacher_logits.grad_fn is None
    assert not any_parameter_gradient(coda.parameters())
    assert nonzero_gradient_parameter_count(a2.parameters()) > 0
    assert z16.grad is not None
    assert torch.isfinite(z16.grad).all()
    assert torch.count_nonzero(z16.grad) > 0


class _TimesTwo(nn.Module):
    def forward(self, state):
        return state * 2


class _AddThree(nn.Module):
    def forward(self, state, frequencies, block_index, mask, cache):
        return state + 3


class _GoldControlHuginn(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = SimpleNamespace(ln_f=_TimesTwo(), coda=[_AddThree()])
        self.lm_head = nn.Identity()
        self.freqs_cis = torch.zeros(1, 32, 1)

    def core_block_forward(self, state, input_embeds, *args, **kwargs):
        return state + input_embeds, torch.tensor(0)

    def forward(self, *, input_ids, input_states, num_steps, **kwargs):
        state = input_states
        recurrent_input = torch.ones_like(state)
        for step in range(num_steps):
            state, _ = self.core_block_forward(
                state, recurrent_input, None, None, None, torch.tensor(0), step
            )
        latent = self.transformer.ln_f(state)
        coda_state = self.transformer.coda[0](
            latent, self.freqs_cis[:, : input_ids.shape[1]], torch.tensor(-1), None, None
        )
        logits = self.lm_head(self.transformer.ln_f(coda_state)).float()
        return SimpleNamespace(latent_states=latent, logits=logits)


def test_production_gold_control_captures_exactly_d16_and_normalizes_once():
    model = _GoldControlHuginn()
    original = model.core_block_forward
    result = live_d16_decomposed_coda_gold_control(
        model,
        torch.tensor([[1, 2, 3]]),
        torch.ones((1, 3), dtype=torch.bool),
        torch.zeros((1, 3, 2)),
    )
    assert model.core_block_forward == original
    assert result["recurrent_calls"] == 16
    assert result["initial_pre_coda_ln_f_applications"] == 1
    assert result["latent_exact"]
    assert result["logits_exact"]
    assert result["latent_max_abs"] == 0
    assert result["logit_max_abs"] == 0


def test_corrected_gate_uses_new_v3_artifact_without_overwriting_v1():
    assert OUTPUT.name == "correctness_gate_coda_v3.json"
    assert PROTOCOL == "functional-correctness-gate-coda-v3"


def test_teacher_self_kl_is_calculated_and_untrained_baseline_is_nonzero():
    torch.manual_seed(40)
    teacher = torch.randn(2, 4, 17)
    untrained_student = torch.randn(2, 4, 17)
    mask = torch.tensor(
        [[False, True, True, False], [False, False, True, True]]
    )
    self_kl = token_normalized_kl(teacher, teacher, mask)
    baseline = token_normalized_kl(untrained_student, teacher, mask)
    assert abs(self_kl.item()) < 1e-6
    assert torch.isfinite(baseline)
    assert baseline.item() > 0
