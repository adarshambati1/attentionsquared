#!/usr/bin/env python3
"""Create summary CSV and accuracy/latency/frontier plots."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main(raw: Path, out_dir: Path) -> None:
    records = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    grouped = defaultdict(list)
    for record in records:
        grouped[record["depth"]].append(record)
    rows = []
    for depth, items in sorted(grouped.items()):
        rows.append({
            "depth": depth,
            "accuracy": sum(item["correct"] for item in items) / len(items),
            "mean_latency_seconds": statistics.mean(item["generation_latency_seconds"] for item in items),
            "mean_generated_tokens": statistics.mean(item["generated_tokens"] for item in items),
            "examples": len(items),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Summary written; matplotlib unavailable, skipping plots")
        return
    depths = [row["depth"] for row in rows]
    accuracy = [row["accuracy"] for row in rows]
    latency = [row["mean_latency_seconds"] for row in rows]
    tokens = [row["mean_generated_tokens"] for row in rows]
    for x, y, ylabel, filename in [
        (depths, accuracy, "GSM8K accuracy", "accuracy_vs_depth.png"),
        (depths, latency, "Mean generation latency (s)", "latency_vs_depth.png"),
    ]:
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel("Huginn recurrent depth")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=160)
        plt.close()
    plt.figure()
    plt.plot(latency, accuracy, marker="o")
    for row, x, y in zip(rows, latency, accuracy):
        plt.annotate(str(row["depth"]), (x, y))
    plt.xlabel("Mean generation latency (s)")
    plt.ylabel("GSM8K accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_latency.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/001_depth_scaling"))
    args = parser.parse_args()
    main(args.raw, args.out_dir)
