import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from src.models.per_layer_history_huginn import PerLayerHistoryHuginn

ROOT = Path(__file__).resolve().parents[1]


class FakeHuginn(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_embd=12)
        self.transformer = SimpleNamespace(core_block=nn.ModuleList([nn.Identity() for _ in range(4)]))


def test_per_layer_variant_has_one_zero_effect_module_per_core_layer():
    model = FakeHuginn()
    wrapper = PerLayerHistoryHuginn(model, projection_size=8, num_heads=2)
    assert len(wrapper.history_attention) == 4
    assert all(torch.count_nonzero(module.out_proj.weight) == 0 for module in wrapper.history_attention)
    assert all(torch.count_nonzero(module.q_proj.weight) > 0 for module in wrapper.history_attention)


def test_control_configs_freeze_same_budget_and_distinct_memory_modes():
    mean = json.loads((ROOT / "configs/006_mean_history_d8.json").read_text())
    layer = json.loads((ROOT / "configs/006_per_layer_history_d8.json").read_text())
    locked = ["model_revision", "dataset_revision", "train_ids", "validation_ids", "test_ids", "depth", "projection_size", "attention_heads", "h0_base_seed", "learning_rate", "weight_decay", "gradient_accumulation_examples", "optimizer_steps", "validation_every_steps", "training_shuffle_seed", "module_initialization_seed"]
    assert all(mean[key] == layer[key] for key in locked)
    assert mean["variant"] == "mean_history" and mean["history_mode"] == "uniform"
    assert layer["variant"] == "per_layer_history" and layer["history_mode"] == "learned"


def test_control_training_keeps_answer_ce_and_validation_selection():
    source = (ROOT / "scripts/train_006_history_control_d8.py").read_text()
    assert "answer_cross_entropy" in source
    assert 'depth=8, mode=config["history_mode"]' in source
    assert "lowest full validation answer-token cross-entropy" not in source  # comes from frozen config
    assert 'config["validation_every_steps"]' in source
    assert 'split="train"' in source
    assert 'config["test_ids"]' not in source
    evaluation = (ROOT / "scripts/evaluate_006_history_memory_d8.py").read_text()
    assert 'Path("/workspace/history_memory_comparison_d8")' in evaluation
    assert 'Path(config["output_root"])' not in evaluation
