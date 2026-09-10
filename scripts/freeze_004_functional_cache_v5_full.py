#!/usr/bin/env python3
"""Copy, checksum, and freeze the independently validated Phase 14 cache."""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path

SOURCE = Path("/workspace/functional_cache_v5_full")
REPLICA = Path("/workspace/functional_cache_v5_full_replica")
FREEZE = Path("/workspace/functional_cache_v5_full_FROZEN.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> tuple[str, dict[str, str]]:
    hashes = {path.name: sha256_file(path) for path in sorted(root.iterdir()) if path.is_file()}
    digest = hashlib.sha256()
    for name, value in hashes.items():
        digest.update(name.encode() + b"\0" + value.encode() + b"\n")
    return digest.hexdigest(), hashes


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_no_replace(source: Path, target: Path) -> None:
    function = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if function is None:
        raise RuntimeError("Linux renameat2 is required")
    if function(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        observed = ctypes.get_errno()
        raise OSError(observed, os.strerror(observed), str(target))
    fsync_directory(target.parent)


def write_exclusive(path: Path, payload: dict) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    fsync_directory(path.parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--replica", type=Path, default=REPLICA)
    parser.add_argument("--freeze-record", type=Path, default=FREEZE)
    args = parser.parse_args()
    validation_bytes = args.validation.read_bytes()
    validation = json.loads(validation_bytes)
    if validation.get("status") != "pass" or validation.get("cache_root") != str(args.source):
        raise ValueError("independent validation is not a passing record for this source")
    if args.replica.exists() or args.replica.is_symlink() or args.freeze_record.exists():
        raise FileExistsError("refusing to replace replica or freeze record")
    source_tree, source_hashes = tree_sha256(args.source)
    if source_tree != validation["cache_tree_sha256"]:
        raise ValueError("source cache changed after independent validation")
    staging = Path(tempfile.mkdtemp(prefix=f".{args.replica.name}.copying-", dir=args.replica.parent))
    try:
        for name in sorted(source_hashes):
            source = args.source / name
            target = staging / name
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=16 * 1024 * 1024)
                outgoing.flush(); os.fsync(outgoing.fileno())
            os.chmod(target, 0o444)
        fsync_directory(staging)
        replica_tree, replica_hashes = tree_sha256(staging)
        if replica_tree != source_tree or replica_hashes != source_hashes:
            raise ValueError("durable replica checksum differs from source")
        os.chmod(staging, 0o555)
        fsync_directory(staging.parent)
        publish_no_replace(staging, args.replica)
        freeze = {
            "protocol": "functional-cache-v5-full-freeze-v1",
            "status": "FULL NATIVE CACHE FROZEN",
            "source": str(args.source),
            "replica": str(args.replica),
            "item_count": 2500,
            "source_tree_sha256": source_tree,
            "replica_tree_sha256": replica_tree,
            "tree_hashes_equal": True,
            "validation_path": str(args.validation),
            "validation_sha256": hashlib.sha256(validation_bytes).hexdigest(),
            "builder_git_commit": validation["builder_git_commit"],
            "full_vocabulary_logits_persisted": False,
        }
        write_exclusive(args.freeze_record, freeze)
        print(json.dumps(freeze, indent=2, sort_keys=True))
    except Exception:
        if staging.exists():
            for path in staging.iterdir():
                path.chmod(0o644); path.unlink()
            staging.chmod(0o755); staging.rmdir()
        raise


if __name__ == "__main__":
    main()
