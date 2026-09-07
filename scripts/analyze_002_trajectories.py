#!/usr/bin/env python3
"""Compute recurrent trajectory geometry and plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def analyze(path: Path, output_dir: Path) -> None:
    data = np.load(path)
    keys = sorted(k for k in data.files if k.endswith("_final_token"))
    trajectories = np.stack([data[k] for k in keys])  # [examples, depth+1, hidden]
    updates = trajectories[:, 1:] - trajectories[:, :-1]
    norms = np.linalg.norm(trajectories[:, :-1], axis=-1)
    adjacent = cosine(trajectories[:, :-1], trajectories[:, 1:])
    step_size = np.linalg.norm(updates, axis=-1) / (norms + 1e-12)
    to_final = cosine(trajectories[:, 1:], trajectories[:, -1:, :])
    direction = cosine(updates[:, :-1], updates[:, 1:])

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "metrics.npz",
        adjacent_cosine=adjacent,
        normalized_step_size=step_size,
        cosine_to_final=to_final,
        update_direction_cosine=direction,
    )
    rows = []
    for depth in range(64):
        rows.append({
            "depth": depth,
            "adjacent_cosine_mean": float(adjacent[:, depth].mean()),
            "adjacent_cosine_std": float(adjacent[:, depth].std()),
            "normalized_step_mean": float(step_size[:, depth].mean()),
            "cosine_to_final_mean": float(to_final[:, depth].mean()),
            "update_direction_cosine_mean": float(direction[:, depth].mean()) if depth < 63 else None,
        })
    with (output_dir / "trajectory_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    updates_flat = updates.reshape(-1, updates.shape[-1])
    updates_flat -= updates_flat.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(updates_flat, full_matrices=False, compute_uv=False)
    explained = singular_values**2 / np.sum(singular_values**2)
    np.savetxt(output_dir / "update_pca_explained_variance.csv", explained, delimiter=",")
    (output_dir / "trajectory_metadata.json").write_text(json.dumps({"examples": len(keys), "depth": 64}, indent=2))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x = np.arange(64)
    for values, ylabel, filename in [
        (adjacent.mean(axis=0), "cos(h_d, h_{d+1})", "adjacent_cosine.png"),
        (step_size.mean(axis=0), "||h_{d+1}-h_d|| / ||h_d||", "normalized_step_size.png"),
        (to_final.mean(axis=0), "cos(h_d, h_64)", "cosine_to_final.png"),
        (direction.mean(axis=0), "cos(Δ_d, Δ_{d+1})", "update_direction_cosine.png"),
    ]:
        plt.figure()
        plt.plot(x, values)
        plt.xlabel("Recurrent depth d")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()
    plt.figure()
    plt.semilogy(np.arange(1, len(explained) + 1), np.cumsum(explained))
    plt.xlabel("Number of update principal components")
    plt.ylabel("Cumulative explained variance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "update_pca_cumulative.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/002_trajectory_analysis"))
    args = parser.parse_args()
    analyze(args.raw, args.output_dir)
