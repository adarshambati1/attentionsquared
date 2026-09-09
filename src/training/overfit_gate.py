"""Predeclared decision rules for the eight-example functional overfit gate."""

from __future__ import annotations

from typing import Mapping


OVERFIT_PROTOCOL = "functional-eight-example-overfit-v1"
QUANTITATIVE_CHECK_KEYS = frozenset(
    {
        "dramatic_global_kl_decrease",
        "absolute_global_kl_is_small",
        "first_distribution_approaches_teacher",
        "absolute_first_distribution_kl_is_small",
        "first_target_probability_increases",
        "first_target_probability_is_substantive",
        "first_target_top1_count_increases",
        "first_target_top1_count_is_substantive",
    }
)


def quantitative_overfit_checks(
    initial: Mapping[str, float | int],
    final: Mapping[str, float | int],
    *,
    maximum_kl_ratio: float,
    maximum_final_kl: float,
    maximum_first_distribution_kl_ratio: float,
    maximum_final_first_distribution_kl: float,
    minimum_final_first_target_probability: float,
    minimum_final_first_target_top1_count: int,
) -> dict[str, bool]:
    initial_kl = float(initial["kl_per_token"])
    final_kl = float(final["kl_per_token"])
    initial_first_kl = float(initial["first_token_distribution_kl"])
    final_first_kl = float(final["first_token_distribution_kl"])
    if initial_kl <= 0 or initial_first_kl <= 0:
        raise ValueError("initial KL controls must be positive")
    return {
        "dramatic_global_kl_decrease": final_kl <= maximum_kl_ratio * initial_kl,
        "absolute_global_kl_is_small": final_kl <= maximum_final_kl,
        "first_distribution_approaches_teacher": (
            final_first_kl <= maximum_first_distribution_kl_ratio * initial_first_kl
        ),
        "absolute_first_distribution_kl_is_small": (
            final_first_kl <= maximum_final_first_distribution_kl
        ),
        "first_target_probability_increases": (
            float(final["mean_first_target_probability"])
            > float(initial["mean_first_target_probability"])
        ),
        "first_target_probability_is_substantive": (
            float(final["mean_first_target_probability"])
            >= minimum_final_first_target_probability
        ),
        "first_target_top1_count_increases": (
            int(final["first_target_top1_count"])
            > int(initial["first_target_top1_count"])
        ),
        "first_target_top1_count_is_substantive": (
            int(final["first_target_top1_count"])
            >= minimum_final_first_target_top1_count
        ),
    }


def gate_passed(checks: Mapping[str, bool]) -> bool:
    if set(checks) != QUANTITATIVE_CHECK_KEYS:
        raise ValueError("overfit checks do not match the frozen gate")
    return all(checks.values())
