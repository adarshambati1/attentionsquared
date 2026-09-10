#!/usr/bin/env python3
"""Record execution-host evidence before corrected Phase 11 coda-v3 training."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.create_004_a2_initialization import write_json_exclusive_fsync
from scripts.run_004_functional_overfit_gate_v3 import (
    BLOCKED_REASON,
    CACHE_ROOT,
    INITIALIZATION,
    OUTPUT_ROOT,
    PHASE10_ARTIFACT_SHA256,
    PHASE10_ATTESTATION_SHA256,
    PRELAUNCH_PROTOCOL,
    clean_git_commit,
    sha256_file,
    validate_phase10_prerequisites,
)


PROTOCOL = PRELAUNCH_PROTOCOL
ATTEMPTS_ROOT = Path("/workspace/functional_overfit_v3_attempts")


def matching_training_processes(proc_root: Path = Path("/proc")) -> list[dict[str, object]]:
    """Find actual Python runner argv entries without matching parent shell text."""
    matches = []
    for process_dir in proc_root.glob("[0-9]*"):
        try:
            raw = (process_dir / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
        if not argv or not Path(argv[0]).name.startswith("python"):
            continue
        if not any(
            Path(argument).name == "run_004_functional_overfit_gate_v3.py"
            for argument in argv[1:]
        ):
            continue
        matches.append(
            {
                "pid": int(process_dir.name),
                "argv": argv,
            }
        )
    return sorted(matches, key=lambda item: int(item["pid"]))


def main() -> Path:
    raise RuntimeError(BLOCKED_REASON)

    # Unreachable preserved implementation pending prefix-stable cache-v3.
    commit = clean_git_commit()
    validate_phase10_prerequisites()
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(f"final Phase 11 output already exists: {OUTPUT_ROOT}")
    processes = matching_training_processes()
    if processes:
        raise RuntimeError(f"Phase 11 training process already exists: {processes}")
    ATTEMPTS_ROOT.mkdir(parents=True, exist_ok=True)
    output = ATTEMPTS_ROOT / f"prelaunch-{uuid.uuid4().hex}.json"
    value = {
        "protocol": PROTOCOL,
        "status": "pass",
        "created_unix_seconds": time.time(),
        "git_commit": commit,
        "host": os.uname().nodename,
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        "final_output": str(OUTPUT_ROOT),
        "final_output_absent": True,
        "matching_training_processes": processes,
        "cache_freeze_sha256": sha256_file(CACHE_ROOT / "FROZEN.json"),
        "initialization_sha256": sha256_file(INITIALIZATION),
        "phase10_artifact_sha256": PHASE10_ARTIFACT_SHA256,
        "phase10_attestation_sha256": PHASE10_ATTESTATION_SHA256,
    }
    write_json_exclusive_fsync(output, value)
    print(str(output), flush=True)
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)
    return output


if __name__ == "__main__":
    main()
