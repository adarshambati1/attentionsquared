import copy
import json
from pathlib import Path

import pytest
import torch

from scripts import create_004_a2_initialization as initialization_script
from scripts.create_004_a2_initialization import validate_config
from src.evaluation.correctness import per_example_seed
from src.training.functional_protocol import (
    A2_INITIALIZATION_SEED,
    FINAL_H0_SEED_INDICES,
    SMOKE_H0_SEED_INDICES,
    Attention2Architecture,
    OptimizerSpec,
    create_shared_initialization,
    h0_seed_indices,
    load_shared_initialization,
    paired_epoch_order,
    paired_huginn_h0,
    state_dict_sha256,
    validate_shared_initialization,
)


class FakeHuginn:
    def initialize_state(self, input_embeds, scale=1.0):
        return torch.randn_like(input_embeds) * scale


def test_h0_seed_counts_are_frozen_for_smoke_and_final():
    assert h0_seed_indices("smoke") == SMOKE_H0_SEED_INDICES == (0,)
    assert h0_seed_indices("final") == FINAL_H0_SEED_INDICES == (0, 1, 2)
    with pytest.raises(ValueError):
        h0_seed_indices("other")


def test_huginn_h0_is_repeatable_paired_distinct_and_restores_rng():
    model = FakeHuginn()
    embeds = torch.zeros(1, 3, 4)
    torch.manual_seed(77)
    expected_after = torch.rand(4)
    torch.manual_seed(77)

    first, first_seed = paired_huginn_h0(
        model, embeds, base_seed=3000, example_id=12, seed_index=0
    )
    repeated, repeated_seed = paired_huginn_h0(
        model, embeds, base_seed=3000, example_id=12, seed_index=0
    )
    other, other_seed = paired_huginn_h0(
        model, embeds, base_seed=3000, example_id=12, seed_index=1
    )

    assert first_seed == repeated_seed == per_example_seed(3000, 12, seed_index=0)
    assert other_seed == per_example_seed(3000, 12, seed_index=1)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, other)
    assert torch.equal(torch.rand(4), expected_after)
    with pytest.raises(ValueError):
        paired_huginn_h0(
            model, embeds, base_seed=3000, example_id=12, seed_index=3
        )


def test_data_order_is_identical_across_k_and_changes_by_epoch():
    ids = range(100)
    for epoch in range(3):
        by_k = {k: paired_epoch_order(ids, epoch=epoch) for k in (1, 2, 4)}
        assert by_k[1] == by_k[2] == by_k[4]
        assert sorted(by_k[1]) == list(ids)
    assert paired_epoch_order(ids, epoch=0) != paired_epoch_order(ids, epoch=1)
    with pytest.raises(ValueError):
        paired_epoch_order([1, 1], epoch=0)


def test_one_shared_initialization_loads_byte_identical_weights_for_all_k(tmp_path):
    architecture = Attention2Architecture(
        hidden=8, depth=3, heads=2, mlp_ratio=1, depth_scale=3.0
    )
    path = tmp_path / "a2_init_seed_0.pt"
    torch.manual_seed(88)
    expected_after = torch.rand(3)
    torch.manual_seed(88)

    metadata = create_shared_initialization(path, architecture=architecture)
    assert metadata["initialization_seed"] == A2_INITIALIZATION_SEED
    assert validate_shared_initialization(path, architecture=architecture) == metadata
    assert torch.equal(torch.rand(3), expected_after)

    models = {
        k: load_shared_initialization(path, rounds=k, architecture=architecture)
        for k in (1, 2, 4)
    }
    hashes = {k: state_dict_sha256(model.state_dict()) for k, model in models.items()}
    assert set(hashes.values()) == {metadata["state_dict_sha256"]}
    reference = models[1].state_dict()
    for model in models.values():
        for key, tensor in model.state_dict().items():
            assert torch.equal(tensor, reference[key])
    assert [models[k].rounds for k in (1, 2, 4)] == [1, 2, 4]
    with pytest.raises(FileExistsError):
        create_shared_initialization(path, architecture=architecture)
    with pytest.raises(ValueError):
        load_shared_initialization(path, rounds=8, architecture=architecture)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    first_key = sorted(checkpoint["state_dict"])[0]
    checkpoint["state_dict"][first_key].view(-1)[0] += 1
    tampered = tmp_path / "tampered.pt"
    torch.save(checkpoint, tampered)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_shared_initialization(tampered, rounds=1, architecture=architecture)


def test_optimizer_setup_is_identical_across_k(tmp_path):
    architecture = Attention2Architecture(hidden=8, depth=2, heads=2)
    path = tmp_path / "a2_init_seed_0.pt"
    create_shared_initialization(path, architecture=architecture)
    spec = OptimizerSpec()
    signatures = []
    for k in (1, 2, 4):
        model = load_shared_initialization(path, rounds=k, architecture=architecture)
        optimizer = spec.build(model.parameters())
        group = optimizer.param_groups[0]
        signatures.append(
            (group["lr"], group["betas"], group["eps"], group["weight_decay"])
        )
    assert signatures[0] == signatures[1] == signatures[2]


def test_production_training_config_freezes_all_paired_conditions():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/004_functional_v2_training.json").read_text())
    output = Path(config["attention2"]["initialization_artifact"])
    assert validate_config(config, output) == Attention2Architecture()

    mutations = [
        ("cache_freeze_sha256", "wrong"),
        ("scientific_test_dataset_split", "train"),
        ("gradient_update_ids", [1, 2249]),
    ]
    for key, value in mutations:
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError, match="exactly match"):
            validate_config(changed, output)
    for section, key, value in [
        ("h0", "final_seed_indices", [0]),
        ("attention2", "paired_K", [1, 4]),
        ("data_order", "seed", 1),
        ("optimizer", "learning_rate", 2e-4),
        ("preprocessing", "padding_mask", "none"),
        ("loss", "normalization", "batchmean"),
    ]:
        changed = copy.deepcopy(config)
        changed[section][key] = value
        with pytest.raises(ValueError, match="exactly match"):
            validate_config(changed, output)


def test_metadata_record_publication_is_atomic_and_no_replace(tmp_path, monkeypatch):
    blocked_output = tmp_path / "blocked.pt"
    blocked_record = blocked_output.with_suffix(".json")
    blocked_record.write_text("preserved")
    with pytest.raises(FileExistsError):
        initialization_script._preflight_new_paths(blocked_output, blocked_record)
    assert not blocked_output.exists()

    record = tmp_path / "a2_init_seed_0.json"
    initialization_script.write_json_exclusive_fsync(record, {"valid": True})
    assert json.loads(record.read_text()) == {"valid": True}
    with pytest.raises(FileExistsError):
        initialization_script.write_json_exclusive_fsync(record, {"valid": False})
    assert json.loads(record.read_text()) == {"valid": True}

    failed = tmp_path / "failed.json"
    monkeypatch.setattr(
        initialization_script.os,
        "link",
        lambda *_: (_ for _ in ()).throw(OSError("injected")),
    )
    with pytest.raises(OSError, match="injected"):
        initialization_script.write_json_exclusive_fsync(failed, {"valid": True})
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed.json.*.tmp"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpu_scoped_initialization_restores_all_cuda_rng_states(tmp_path):
    torch.manual_seed(222)
    before_cpu = torch.random.get_rng_state().clone()
    before_cuda = [state.clone() for state in torch.cuda.get_rng_state_all()]
    architecture = Attention2Architecture(hidden=8, depth=2, heads=2)

    create_shared_initialization(
        tmp_path / "a2_init_seed_0.pt", architecture=architecture
    )
    paired_huginn_h0(
        FakeHuginn(),
        torch.zeros(1, 2, 8),
        base_seed=3000,
        example_id=1,
        seed_index=0,
    )

    assert torch.equal(torch.random.get_rng_state(), before_cpu)
    after_cuda = torch.cuda.get_rng_state_all()
    assert len(after_cuda) == len(before_cuda)
    assert all(
        torch.equal(after, before) for after, before in zip(after_cuda, before_cuda)
    )
