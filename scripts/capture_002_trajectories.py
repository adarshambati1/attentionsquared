#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.huginn_trajectory import run

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/002_trajectory_analysis.json"))
    parser.add_argument("--output", type=Path, default=Path("results/002_trajectory_analysis/raw_trajectories.npz"))
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)
