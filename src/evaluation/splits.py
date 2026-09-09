"""Frozen dataset split validation for corrected Experiment 004 work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SPLIT_PROTOCOL = "experiment-004-functional-splits-v1"
GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_DATASET_CONFIG = "main"
GSM8K_DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
SPLIT_GENERATION_POLICY = (
    "Take the first 2500 examples of the pinned GSM8K train split in dataset order; "
    "IDs 0-2249 are gradient-update examples and IDs 2250-2499 are permanent "
    "checkpoint-selection validation examples. Reserve IDs 0-249 of the pinned "
    "GSM8K test split for scientific evaluation only. No shuffling."
)


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a frozen split manifest."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed if train, validation, or evaluation roles can leak."""
    if manifest.get("protocol") != SPLIT_PROTOCOL:
        raise ValueError("unexpected split protocol")

    dataset = manifest.get("dataset", {})
    expected_dataset = {
        "id": GSM8K_DATASET_ID,
        "config": GSM8K_DATASET_CONFIG,
        "revision": GSM8K_DATASET_REVISION,
    }
    if dataset != expected_dataset:
        raise ValueError("dataset identity does not match the frozen GSM8K source")

    if manifest.get("split_generation_policy") != SPLIT_GENERATION_POLICY:
        raise ValueError("split generation policy does not match the frozen policy")

    training = manifest.get("training_side", {})
    evaluation = manifest.get("scientific_evaluation", {})
    if training.get("dataset_split") != "train":
        raise ValueError("training-side examples must come from GSM8K train")
    if evaluation.get("dataset_split") != "test":
        raise ValueError("scientific evaluation examples must come from GSM8K test")

    gradient_ids = _validated_ids(training, "gradient_update_ids", expected_count=2250)
    validation_ids = _validated_ids(
        training, "checkpoint_selection_ids", expected_count=250
    )
    evaluation_ids = _validated_ids(evaluation, "example_ids", expected_count=250)

    if gradient_ids != list(range(2250)):
        raise ValueError("gradient-update IDs must be exactly 0 through 2249")
    if validation_ids != list(range(2250, 2500)):
        raise ValueError("checkpoint-selection IDs must be exactly 2250 through 2499")
    if evaluation_ids != list(range(250)):
        raise ValueError("scientific evaluation IDs must be exactly 0 through 249")

    if training.get("checkpoint_selection_role") != "validation_only":
        raise ValueError("training-side validation role is not frozen")
    if evaluation.get("role") != "scientific_evaluation_only":
        raise ValueError("held-out test role is not frozen")
    if evaluation.get("allow_checkpoint_selection") is not False:
        raise ValueError("held-out test examples must not select checkpoints")


def validate_split_config(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Require a functional-v2 config to match the frozen split manifest."""
    validate_split_manifest(manifest)
    dataset = manifest["dataset"]
    if (
        config.get("dataset_id") != dataset["id"]
        or config.get("dataset_config") != dataset["config"]
        or config.get("dataset_revision") != dataset["revision"]
    ):
        raise ValueError("config dataset identity does not match split manifest")
    if config.get("training_dataset_split") != "train":
        raise ValueError("config training source must be GSM8K train")
    if config.get("training_splits") != {
        "train": [0, 2249],
        "validation": [2250, 2499],
    }:
        raise ValueError("config training ranges do not match split manifest")
    if config.get("scientific_evaluation") != {
        "dataset_split": "test",
        "example_range": [0, 249],
        "allow_checkpoint_selection": False,
    }:
        raise ValueError("config scientific-evaluation role does not match manifest")


def _validated_ids(
    section: dict[str, Any], key: str, *, expected_count: int
) -> list[int]:
    values = section.get(key)
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise ValueError(f"{key} must be an explicit integer list")
    if len(values) != expected_count or len(set(values)) != expected_count:
        raise ValueError(f"{key} must contain {expected_count} unique IDs")
    if values != sorted(values):
        raise ValueError(f"{key} must be sorted")
    return values
