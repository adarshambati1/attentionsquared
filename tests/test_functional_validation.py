import copy
import hashlib

import numpy as np
import pytest

import scripts.validate_004_functional_cache_v2 as validator
from src.data.functional_validation import (
    arrays_byte_identical,
    checksum_inventory_bytes,
    require_identical_regular_files,
    validate_mount_output,
    validate_runpod_volume_attestation,
    verify_checksum_inventory,
    write_bytes_immutable,
)


def test_inventory_is_sorted_and_verifies_all_files(tmp_path):
    second = tmp_path / "b.bin"
    first = tmp_path / "a.bin"
    second.write_bytes(b"second")
    first.write_bytes(b"first")

    payload = checksum_inventory_bytes(tmp_path, [second, first])
    lines = payload.decode().splitlines()
    assert lines[0].endswith("\ta.bin")
    assert lines[1].endswith("\tb.bin")
    assert verify_checksum_inventory(tmp_path, payload) == 2


def test_inventory_detects_byte_change_even_when_size_is_same(tmp_path):
    path = tmp_path / "item.npz"
    path.write_bytes(b"abc")
    payload = checksum_inventory_bytes(tmp_path, [path])
    path.write_bytes(b"abd")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksum_inventory(tmp_path, payload)


def test_inventory_rejects_traversal_absolute_and_symlink_paths(tmp_path):
    outside = tmp_path.parent / "outside-cache-item"
    outside.write_bytes(b"outside")
    digest = hashlib.sha256(b"outside").hexdigest()
    for relative in ("../outside-cache-item", str(outside)):
        payload = f"{digest}\t7\t{relative}\n".encode()
        with pytest.raises(ValueError, match="normalized and relative"):
            verify_checksum_inventory(tmp_path, payload)

    target = tmp_path / "target"
    target.write_bytes(b"target")
    symlink = tmp_path / "link"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        checksum_inventory_bytes(tmp_path, [symlink])


def test_inventory_rejects_duplicate_paths(tmp_path):
    path = tmp_path / "item.npz"
    path.write_bytes(b"abc")
    line = checksum_inventory_bytes(tmp_path, [path])
    with pytest.raises(ValueError, match="duplicate inventory path"):
        verify_checksum_inventory(tmp_path, line + line)


def test_state_comparison_is_raw_byte_identity_including_signed_zero():
    positive = np.array([0.0], dtype=np.float16)
    negative = np.array([-0.0], dtype=np.float16)
    assert np.array_equal(positive, negative)
    assert not arrays_byte_identical(positive, negative)
    assert arrays_byte_identical(positive, positive.copy())
    assert not arrays_byte_identical(positive, positive.astype(np.float32))


def test_manifest_identity_rejects_difference_and_symlink(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    require_identical_regular_files(first, second)
    second.write_bytes(b"diff")
    with pytest.raises(ValueError, match="not byte-identical"):
        require_identical_regular_files(first, second)
    second.unlink()
    second.symlink_to(first)
    with pytest.raises(ValueError, match="symlinks"):
        require_identical_regular_files(first, second)


def test_runtime_attestation_and_exact_mount_are_enforced(tmp_path):
    attestation = {
        "protocol": "runpod-cache-v2-runtime-attestation-v1",
        "pod": {
            "id": "pod",
            "networkVolumeId": "volume",
            "volumeMountPath": str(tmp_path),
            "desiredStatus": "RUNNING",
            "runtimeStatus": "running",
        },
        "network_volume": {
            "id": "volume",
            "dataCenterId": "US-GA-2",
            "size": 100,
        },
    }
    validate_runpod_volume_attestation(
        attestation,
        pod_id="pod",
        volume_id="volume",
        data_center_id="US-GA-2",
        mount_path=str(tmp_path),
        minimum_size_gb=100,
    )
    changed = copy.deepcopy(attestation)
    changed["pod"]["networkVolumeId"] = "other"
    with pytest.raises(ValueError, match="networkVolumeId"):
        validate_runpod_volume_attestation(
            changed,
            pod_id="pod",
            volume_id="volume",
            data_center_id="US-GA-2",
            mount_path=str(tmp_path),
            minimum_size_gb=100,
        )

    cache = tmp_path / "functional_cache_v2"
    cache.mkdir()
    mount = f"mfs#us-ga-2.runpod.net:9421 fuse {tmp_path}"
    validate_mount_output(
        cache,
        mount,
        expected_mount=tmp_path,
        expected_data_center_id="US-GA-2",
    )
    with pytest.raises(ValueError, match="dedicated mount"):
        validate_mount_output(
            cache,
            mount.replace("fuse", "ext4"),
            expected_mount=tmp_path,
            expected_data_center_id="US-GA-2",
        )


def test_exact_cache_item_file_set_and_roles(monkeypatch, tmp_path):
    root = tmp_path / "functional_cache_v2"
    (root / "train").mkdir(parents=True)
    (root / "validation").mkdir()
    (root / "manifest.json").touch()
    for example_id in range(2500):
        split = "train" if example_id < 2250 else "validation"
        (root / split / f"{example_id:05d}.npz").touch()
    monkeypatch.setattr(validator, "CACHE_ROOT", root)
    paths = validator._validate_file_set()
    assert len(paths) == 2500
    assert paths[2249].parent.name == "train"
    assert paths[2250].parent.name == "validation"
    (root / "validation/02250.npz").rename(root / "train/02250.npz")
    with pytest.raises(ValueError, match="file set mismatch"):
        validator._validate_file_set()


def test_immutable_metadata_reuses_identical_and_rejects_difference(tmp_path):
    path = tmp_path / "FROZEN.json"
    payload = b'{"status":"frozen"}\n'
    digest = write_bytes_immutable(path, payload)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert write_bytes_immutable(path, payload) == digest
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_bytes_immutable(path, b"different")
    assert path.read_bytes() == payload
