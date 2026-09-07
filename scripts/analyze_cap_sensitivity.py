#!/usr/bin/env python3
"""Analyze the complete N=20 cap-sensitivity matrix without model execution."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def load(path: Path, cap: int) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            row["cap"] = cap
            rows.append(row)
    return rows


def classify_capped(row: dict) -> str:
    text = row["model_answer"]
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    words = re.findall(r"[a-z0-9]+", normalized)
    ngrams = Counter(tuple(words[i : i + 6]) for i in range(max(0, len(words) - 5)))
    if any(count >= 3 for count in ngrams.values()):
        return "degenerate/repetitive"
    if "<|end_text|>" in text or "<|end_turn|>" in text:
        return "stop-string/evaluator issue"
    # An explicit final-answer phrase near the truncation boundary indicates
    # the evaluator likely stopped before emitting the stop token.
    tail = normalized[-240:]
    if re.search(r"(?:therefore|final answer|the answer is|answer:)", tail):
        return "stop-string/evaluator issue"
    if len(words) >= 80:
        return "coherent but unfinished"
    return "other"


def main(args: argparse.Namespace) -> None:
    rows = []
    for cap, paths in [(256, args.raw256), (512, args.raw512), (1024, args.raw1024)]:
        for path in paths:
            rows.extend(load(path, cap))
    categories = ["degenerate/repetitive", "coherent but unfinished", "stop-string/evaluator issue", "other"]
    output = []
    for (cap, depth), group in sorted(_groups(rows), key=lambda x: (x[0][0], x[0][1])):
        capped = [r for r in group if r["hit_max_new_tokens"]]
        counts = Counter(classify_capped(r) for r in capped)
        output.append({
            "cap": cap,
            "depth": depth,
            "accuracy": sum(r["correct"] for r in group) / len(group),
            "cap_hit_percent": 100 * len(capped) / len(group),
            "degenerate_repetitive": counts[categories[0]],
            "coherent_but_unfinished": counts[categories[1]],
            "stop_string_evaluator_issue": counts[categories[2]],
            "other_capped": counts[categories[3]],
            "natural_end_percent": 100 * sum(r["ended_naturally"] for r in group) / len(group),
            "mean_ttft_seconds": sum(r["time_to_first_token_seconds"] for r in group) / len(group),
            "mean_single_forward_seconds": sum(r["single_forward_latency_seconds"] for r in group) / len(group),
            "examples": len(group),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    for row in output:
        print(row)


def _groups(rows: list[dict]):
    groups = {}
    for row in rows:
        groups.setdefault((row["cap"], row["depth"]), []).append(row)
    return groups.items()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw256", type=Path, nargs="+", required=True)
    parser.add_argument("--raw512", type=Path, nargs="+", required=True)
    parser.add_argument("--raw1024", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/001_depth_scaling/cap_sensitivity.csv"))
    main(parser.parse_args())
