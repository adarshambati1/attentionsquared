#!/usr/bin/env python3
"""Run the fixed-subset Huginn depth-scaling experiment."""

import argparse
import json
import sys
from pathlib import Path

# Allow direct execution as `python scripts/run_001_depth_scaling.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.gsm8k import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/001_depth_scaling.json"))
    parser.add_argument("--output", type=Path, default=Path("results/001_depth_scaling/raw.jsonl"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = json.load(stream)
    evaluate(config, args.output)
