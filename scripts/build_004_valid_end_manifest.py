#!/usr/bin/env python3
"""Build and freeze Experiment 004's reviewed valid-end sidecar."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.valid_end_manifest import (
    build_valid_end_records,
    load_and_validate_manifest,
    write_immutable_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("results/004_functional/teacher_sequences"),
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("results/004_functional/capped_continuation_review.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/004_functional/valid_end_manifest.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_valid_end_records(args.source_root, args.review)
    digest = write_immutable_manifest(args.output, records)
    validated = load_and_validate_manifest(args.output)
    capped = sum(row["source_truncated_at_cap"] for row in validated)
    shortened = sum(
        row["reviewed_valid_end"] < row["source_sequence_length"]
        for row in validated
    )
    print(f"validated {len(validated)} records ({capped} capped, {shortened} shortened)")
    print(f"sha256 {digest}  {args.output}")


if __name__ == "__main__":
    main()
