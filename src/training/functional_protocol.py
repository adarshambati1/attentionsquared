"""Paired-randomness and initialization protocol for functional Experiment 004."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import random
import stat
import tempfile
from typing import Any, Iterable, Iterator, Mapping

import torch

from src.evaluation.correctness import SEED_PROTOCOL, per_example_seed
from src.models.attention2 import Attention2Lite


PAIRED_RANDOMNESS_PROTOCOL = "functional-paired-randomness-v1"
A2_INITIALIZATION_PROTOCOL = "attention2-shared-initialization-v1"
SMOKE_H0_SEED_INDICES = (0,)
FINAL_H0_SEED_INDICES = (0, 1, 2)
A2_INITIALIZATION_SEED = 0
PAIRED_DATA_ORDER_SEED = 9100
SUPPORTED_K = (1, 2, 4)


@dataclass(frozen=True)
class Attention2Architecture:
    hidden: int = 5280
    depth: int = 16
    heads: int = 16
    mlp_ratio: int = 1
    depth_scale: float = 16.0
    film: bool = False

    def build(self, rounds: int) -> Attention2Lite:
        if rounds <= 0:
            raise ValueError("rounds/K must be positive")
        return Attention2Lite(
            hidden=self.hidden,
            depth=self.depth,
            heads=self.heads,
            rounds=rounds,
            mlp_ratio=self.mlp_ratio,
            depth_scale=self.depth_scale,
            film=self.film,
        )


@dataclass(frozen=True)
class OptimizerSpec:
    name: str = "AdamW"
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 0.0

    def build(self, parameters: Iterable[torch.nn.Parameter]) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.epsilon,
            weight_decay=self.weight_decay,
        )


def h0_seed_indices(mode: str) -> tuple[int, ...]:
    if mode == "smoke":
        return SMOKE_H0_SEED_INDICES
    if mode == "final":
        return FINAL_H0_SEED_INDICES
    raise ValueError("mode must be 'smoke' or 'final'")


@contextmanager
def _fork_torch_rng(device: torch.device) -> Iterator[None]:
    del device  # torch.manual_seed affects CPU and every CUDA generator.
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        yield


def paired_huginn_h0(
    model: Any,
    input_embeds: torch.Tensor,
    *,
    base_seed: int,
    example_id: int,
    seed_index: int,
    step: int = 0,
    init_scale: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """Create deterministic Huginn h0 while restoring process-global RNG state."""
    if seed_index not in FINAL_H0_SEED_INDICES:
        raise ValueError("seed_index must be one of the three frozen paired indices")
    seed = per_example_seed(
        base_seed, example_id, step=step, seed_index=seed_index
    )
    device = input_embeds.device
    with _fork_torch_rng(device):
        torch.manual_seed(seed)
        state = model.initialize_state(input_embeds, scale=init_scale)
    return state, seed


def paired_epoch_order(
    example_ids: Iterable[int], *, epoch: int, order_seed: int = PAIRED_DATA_ORDER_SEED
) -> tuple[int, ...]:
    """Return a K-independent deterministic permutation for one epoch."""
    ids = tuple(int(example_id) for example_id in example_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("example IDs must be unique")
    seed = per_example_seed(order_seed, 0, step=epoch)
    order = list(ids)
    random.Random(seed).shuffle(order)
    return tuple(order)


def initialization_metadata(architecture: Attention2Architecture) -> dict[str, Any]:
    return {
        "protocol": A2_INITIALIZATION_PROTOCOL,
        "seed_protocol": SEED_PROTOCOL,
        "initialization_seed": A2_INITIALIZATION_SEED,
        "architecture": asdict(architecture),
        "K_independent": True,
        "supported_K": list(SUPPORTED_K),
    }


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash ordered keys, dtype/shape metadata, and canonical CPU tensor bytes."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def create_shared_initialization(
    path: str | Path,
    *,
    architecture: Attention2Architecture = Attention2Architecture(),
) -> dict[str, Any]:
    """Create and atomically publish the one immutable K-independent state dict."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite shared initialization: {output}")

    with _fork_torch_rng(torch.device("cpu")):
        torch.manual_seed(A2_INITIALIZATION_SEED)
        model = architecture.build(rounds=1)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    metadata = initialization_metadata(architecture)
    metadata["state_dict_sha256"] = state_dict_sha256(state_dict)
    checkpoint = {"metadata": metadata, "state_dict": state_dict}

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(checkpoint, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        validate_shared_initialization(temporary, architecture=architecture)
        os.chmod(temporary, 0o444)
        if stat.S_IMODE(temporary.stat().st_mode) != 0o444:
            raise RuntimeError("shared initialization is not read-only before publication")
        os.link(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


def load_shared_initialization(
    path: str | Path,
    *,
    rounds: int,
    architecture: Attention2Architecture = Attention2Architecture(),
    map_location: str | torch.device = "cpu",
) -> Attention2Lite:
    """Build any K condition and load the exact shared parameters strictly."""
    if rounds not in SUPPORTED_K:
        raise ValueError(f"rounds must be one of {SUPPORTED_K}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    _validate_checkpoint(checkpoint, architecture)
    with _fork_torch_rng(torch.device("cpu")):
        model = architecture.build(rounds=rounds)
    model = model.to(map_location)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def validate_shared_initialization(
    path: str | Path,
    *,
    architecture: Attention2Architecture = Attention2Architecture(),
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _validate_checkpoint(checkpoint, architecture)
    observed_hash = state_dict_sha256(checkpoint["state_dict"])
    expected_hash = checkpoint["metadata"]["state_dict_sha256"]
    if observed_hash != expected_hash:
        raise ValueError("shared initialization state-dict checksum mismatch")
    return dict(checkpoint["metadata"])


def _validate_checkpoint(
    checkpoint: Any, architecture: Attention2Architecture
) -> None:
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"metadata", "state_dict"}:
        raise ValueError("invalid shared-initialization checkpoint schema")
    expected = initialization_metadata(architecture)
    metadata = checkpoint["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != {*expected, "state_dict_sha256"}:
        raise ValueError("invalid shared-initialization metadata schema")
    for key, value in expected.items():
        if metadata[key] != value:
            raise ValueError(f"shared-initialization metadata mismatch: {key}")
    if not isinstance(metadata["state_dict_sha256"], str) or len(metadata["state_dict_sha256"]) != 64:
        raise ValueError("invalid shared-initialization state-dict checksum")
    with _fork_torch_rng(torch.device("cpu")):
        reference = architecture.build(rounds=1).state_dict()
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, dict) or set(state_dict) != set(reference):
        raise ValueError("shared-initialization state-dict keys mismatch")
    for key, expected_tensor in reference.items():
        observed = state_dict[key]
        if (
            not isinstance(observed, torch.Tensor)
            or observed.shape != expected_tensor.shape
            or observed.dtype != expected_tensor.dtype
            or not torch.isfinite(observed).all()
        ):
            raise ValueError(f"invalid shared-initialization tensor: {key}")
    if state_dict_sha256(state_dict) != metadata["state_dict_sha256"]:
        raise ValueError("shared initialization state-dict checksum mismatch")
