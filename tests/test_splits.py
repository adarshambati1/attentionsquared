import copy
import json
from pathlib import Path

import pytest

from src.evaluation.splits import (
    load_split_manifest,
    validate_split_config,
    validate_split_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "004_functional_v2_splits.json"
CONFIG_PATH = ROOT / "configs" / "004_functional_v2.json"


def test_phase3_split_counts_roles_and_exact_ids_are_frozen():
    manifest = load_split_manifest(MANIFEST_PATH)
    training = manifest["training_side"]
    evaluation = manifest["scientific_evaluation"]

    assert training["dataset_split"] == "train"
    assert training["gradient_update_ids"] == list(range(2250))
    assert training["checkpoint_selection_ids"] == list(range(2250, 2500))
    assert training["checkpoint_selection_role"] == "validation_only"

    assert evaluation["dataset_split"] == "test"
    assert evaluation["example_ids"] == list(range(250))
    assert evaluation["role"] == "scientific_evaluation_only"
    assert evaluation["allow_checkpoint_selection"] is False


def test_phase3_config_matches_authoritative_manifest():
    manifest = load_split_manifest(MANIFEST_PATH)
    config = json.loads(CONFIG_PATH.read_text())
    validate_split_config(config, manifest)


def test_phase3_rejects_cross_role_swap_even_when_counts_and_union_match():
    manifest = load_split_manifest(MANIFEST_PATH)
    tampered = copy.deepcopy(manifest)
    tampered["training_side"]["gradient_update_ids"][-1] = 2250
    tampered["training_side"]["checkpoint_selection_ids"][0] = 2249
    with pytest.raises(ValueError, match="gradient-update IDs"):
        validate_split_manifest(tampered)


@pytest.mark.parametrize("replacement", [None, "different policy"])
def test_phase3_rejects_missing_or_changed_generation_policy(replacement):
    manifest = load_split_manifest(MANIFEST_PATH)
    tampered = copy.deepcopy(manifest)
    if replacement is None:
        del tampered["split_generation_policy"]
    else:
        tampered["split_generation_policy"] = replacement
    with pytest.raises(ValueError, match="generation policy"):
        validate_split_manifest(tampered)


def test_phase3_rejects_train_validation_overlap():
    manifest = load_split_manifest(MANIFEST_PATH)
    tampered = copy.deepcopy(manifest)
    tampered["training_side"]["checkpoint_selection_ids"][0] = 2249
    with pytest.raises(ValueError):
        validate_split_manifest(tampered)


def test_phase3_rejects_test_use_for_checkpoint_selection():
    manifest = load_split_manifest(MANIFEST_PATH)
    tampered = copy.deepcopy(manifest)
    tampered["scientific_evaluation"]["allow_checkpoint_selection"] = True
    with pytest.raises(ValueError, match="must not select checkpoints"):
        validate_split_manifest(tampered)


def test_phase3_rejects_wrong_dataset_split_or_revision():
    manifest = load_split_manifest(MANIFEST_PATH)
    wrong_split = copy.deepcopy(manifest)
    wrong_split["scientific_evaluation"]["dataset_split"] = "train"
    with pytest.raises(ValueError, match="GSM8K test"):
        validate_split_manifest(wrong_split)

    wrong_revision = copy.deepcopy(manifest)
    wrong_revision["dataset"]["revision"] = "unpinned"
    with pytest.raises(ValueError, match="frozen GSM8K source"):
        validate_split_manifest(wrong_revision)
