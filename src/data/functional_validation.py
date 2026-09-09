"""Checksum and immutable-freeze helpers for functional cache v2."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


CHECKSUM_PROTOCOL = "sha256-bytes-relative-path-v1"


def arrays_byte_identical(observed: Any, expected: Any) -> bool:
    """Require identical dtype, shape, and C-order bytes (including signed zero)."""
    import numpy as np

    return (
        observed.dtype == expected.dtype
        and observed.shape == expected.shape
        and np.ascontiguousarray(observed).tobytes()
        == np.ascontiguousarray(expected).tobytes()
    )


def require_identical_regular_files(first: Path, second: Path) -> None:
    """Reject symlinks and require exact bytes for two regular files."""
    if first.is_symlink() or second.is_symlink():
        raise ValueError("frozen files must not be symlinks")
    if not first.is_file() or not second.is_file():
        raise ValueError("frozen files must be regular files")
    if first.read_bytes() != second.read_bytes():
        raise ValueError("frozen files are not byte-identical")


def validate_runpod_volume_attestation(
    attestation: dict[str, Any],
    *,
    pod_id: str,
    volume_id: str,
    data_center_id: str,
    mount_path: str,
    minimum_size_gb: int,
) -> None:
    """Require a control-plane capture binding Pod, volume, DC, and mount."""
    if attestation.get("protocol") != "runpod-cache-v2-runtime-attestation-v1":
        raise ValueError("unexpected runtime-attestation protocol")
    pod = attestation.get("pod", {})
    volume = attestation.get("network_volume", {})
    expected_pod = {
        "id": pod_id,
        "networkVolumeId": volume_id,
        "volumeMountPath": mount_path,
        "desiredStatus": "RUNNING",
        "runtimeStatus": "running",
    }
    for key, expected in expected_pod.items():
        if pod.get(key) != expected:
            raise ValueError(f"runtime attestation has wrong Pod {key}")
    if volume.get("id") != volume_id or volume.get("dataCenterId") != data_center_id:
        raise ValueError("runtime attestation has wrong network volume identity/DC")
    if type(volume.get("size")) is not int or volume["size"] < minimum_size_gb:
        raise ValueError("runtime attestation network volume is too small")


def validate_mount_output(
    cache_root: Path,
    mount_output: str,
    *,
    expected_mount: Path,
    expected_data_center_id: str,
    expected_volume_id: str,
) -> None:
    """Reject symlinked roots and require the exact expected RunPod MFS mount."""
    if cache_root.is_symlink() or expected_mount.is_symlink():
        raise ValueError("cache root and durable mount must not be symlinks")
    if cache_root.resolve() != expected_mount.resolve() / cache_root.name:
        raise ValueError("cache root is not directly beneath the expected mount")
    fields = mount_output.split()
    if len(fields) != 3:
        raise ValueError("unexpected findmnt output")
    source, filesystem_type, target = fields
    expected_source = (
        f"mfs#{expected_data_center_id.lower()}.runpod.net:9421"
        f"[/networkvolumes/{expected_volume_id}]"
    )
    if source != expected_source:
        raise ValueError("cache mount source has wrong RunPod volume or data center")
    if filesystem_type != "fuse" or Path(target) != expected_mount:
        raise ValueError("cache mount type or target is not the dedicated mount")


def _safe_inventory_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative:
        raise ValueError(f"inventory path is not normalized and relative: {relative}")
    path = root / candidate
    if path.is_symlink():
        raise ValueError(f"inventory payload must not be a symlink: {relative}")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"inventory path escapes root: {relative}") from error
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_inventory_bytes(root: Path, paths: Iterable[Path]) -> bytes:
    """Return deterministic sha256/size/path records for files below root."""
    rows = []
    for path in sorted(paths, key=lambda item: str(item.relative_to(root))):
        relative = path.relative_to(root).as_posix()
        if "\n" in relative or "\t" in relative:
            raise ValueError("inventory paths cannot contain tabs or newlines")
        regular = _safe_inventory_path(root, relative)
        rows.append(
            f"{file_sha256(regular)}\t{regular.stat().st_size}\t{relative}\n"
        )
    return "".join(rows).encode()


def verify_checksum_inventory(root: Path, payload: bytes) -> int:
    """Re-read every inventoried file and verify size and SHA-256."""
    count = 0
    seen: set[str] = set()
    for line in payload.decode().splitlines():
        digest, size_text, relative = line.split("\t", 2)
        if relative in seen:
            raise ValueError(f"duplicate inventory path: {relative}")
        seen.add(relative)
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"invalid inventory digest: {relative}")
        path = _safe_inventory_path(root, relative)
        if not path.is_file() or path.stat().st_size != int(size_text):
            raise ValueError(f"inventory size/path mismatch: {relative}")
        if file_sha256(path) != digest:
            raise ValueError(f"inventory checksum mismatch: {relative}")
        count += 1
    return count


def write_bytes_immutable(path: str | Path, payload: bytes) -> str:
    """Atomically publish bytes without replacing a differing existing file."""
    destination = Path(path)
    digest = hashlib.sha256(payload).hexdigest()
    if destination.exists():
        if destination.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace frozen metadata: {destination}")
        return digest

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != payload:
                raise FileExistsError(
                    f"refusing to replace frozen metadata: {destination}"
                ) from None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return digest
