#!/usr/bin/env python3
"""Record execution-host evidence immediately before Phase 11 training."""

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
from scripts.run_004_functional_overfit_gate import (
    CACHE_ROOT,
    INITIALIZATION,
    OUTPUT_ROOT,
    clean_git_commit,
    sha256_file,
)


PROTOCOL = "functional-eight-example-overfit-prelaunch-v1"
ATTEMPTS_ROOT = Path("/workspace/functional_overfit_v2_attempts")


def matching_training_processes() -> list[dict[str, object]]:
    output = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, text=True, capture_output=True
    ).stdout
    matches = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, arguments = stripped.split(maxsplit=1)
        if "scripts/run_004_functional_overfit_gate.py" in arguments:
            matches.append({"pid": int(pid_text), "arguments": arguments})
    return matches


def main() -> Path:
    commit = clean_git_commit()
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
    }
    write_json_exclusive_fsync(output, value)
    print(str(output), flush=True)
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)
    return output


if __name__ == "__main__":
    main()
