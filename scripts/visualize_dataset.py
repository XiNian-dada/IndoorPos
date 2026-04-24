#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize and diagnose WiFi + motion dataset quality.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def source_window_keys(source_index: np.ndarray) -> List[Tuple[int, ...]]:
    return [tuple(int(x) for x in row.tolist()) for row in source_index]


def coord_window_keys(coords: np.ndarray, decimals: int) -> List[Tuple[float, ...]]:
    flattened = np.round(coords.reshape(coords.shape[0], -1), decimals=decimals)
    return [tuple(float(x) for x in row.tolist()) for row in flattened]


def group_labels(groups: np.ndarray) -> List[str]:
    return [f"{int(building)}_{int(floor)}" for building, floor in groups.tolist()]


def build_metrics(
    train: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
    coord_decimals: int,
) -> Dict[str, object]:
    train_source_keys = source_window_keys(train["source_index"])
    val_source_keys = source_window_keys(val["source_index"])
    train_source_set = set(train_source_keys)
    val_source_set = set(val_source_keys)

    train_coord_keys = coord_window_keys(train["coords"], coord_decimals)
    val_coord_keys = coord_window_keys(val["coords"], coord_decimals)

    metrics: Dict[str, object] = {
        "num_train_sequences": int(train["X"].shape[0]),
        "num_val_sequences": int(val["X"].shape[0]),
        "num_total_sequences": int(train["X"].shape[0] + val["X"].shape[0]),
        "num_unique_source_windows_train": int(len(train_source_set)),
        "num_unique_source_windows_val": int(len(val_source_set)),
        "num_shared_source_windows_train_val": int(len(train_source_set.intersection(val_source_set))),
        "num_unique_coord_windows_train": int(len(set(train_coord_keys))),
        "num_unique_coord_windows_val": int(len(set(val_coord_keys))),
        "num_unique_coord_windows_total": int(len(set(train_coord_keys + val_coord_keys))),
        "interp_ratio_train": float(train["is_interpolated"].mean()),
        "interp_ratio_val": float(val["is_interpolated"].mean()),
        "speed_mean_train": float(train["speed"].mean()),
        "speed_mean_val": float(val["speed"].mean()),
        "speed_p95_train": float(np.percentile(train["speed"], 95)),
        "speed_p95_val": float(np.percentile(val["speed"], 95)),
    }

    if "trajectory_id" in train and "trajectory_id" in val:
        train_trajectory = set(int(x) for x in train["trajectory_id"].tolist())
        val_trajectory = set(int(x) for x in val["trajectory_id"].tolist())
        metrics["num_shared_trajectory_ids_train_val"] = int(
            len(train_trajectory.intersection(val_trajectory))
        )

    return metrics


def plot_group_distribution(
    ax: plt.Axes,
    train_groups: Iterable[str],
    val_groups: Iterable[str],
) -> None:
    train_counter = Counter(train_groups)
    val_counter = Counter(val_groups)
    labels = sorted(set(train_counter.keys()).union(val_counter.keys()))

    x = np.arange(len(labels))
    width = 0.4

    ax.bar(
        x - width / 2,
        [train_counter[label] for label in labels],
        width=width,
        label="Train",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        [val_counter[label] for label in labels],
        width=width,
        label="Val",
        alpha=0.85,
    )
    ax.set_title("Window Count by Building/Floor")
    ax.set_xlabel("Building_Floor")
    ax.set_ylabel("Windows")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()


def plot_sample_trajectories(
    ax: plt.Axes,
    coords: np.ndarray,
    color: str,
    max_windows: int,
    random_seed: int,
    label: str,
) -> None:
    if coords.shape[0] == 0:
        return

    rng = np.random.default_rng(random_seed)
    take = min(max_windows, coords.shape[0])
    indices = rng.choice(coords.shape[0], size=take, replace=False)

    for idx in indices.tolist():
        window = coords[idx]
        ax.plot(window[:, 0], window[:, 1], color=color, alpha=0.15, linewidth=1.0)

    ax.plot([], [], color=color, linewidth=2.0, alpha=0.8, label=label)


def create_figure(
    train: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
    output_png: str,
    max_trajectory_windows: int,
    random_seed: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_scatter, ax_speed, ax_group, ax_traj = axes.flatten()

    train_last = train["y_last"]
    val_last = val["y_last"]
    ax_scatter.scatter(train_last[:, 0], train_last[:, 1], s=5, alpha=0.15, label="Train")
    ax_scatter.scatter(val_last[:, 0], val_last[:, 1], s=5, alpha=0.15, label="Val")
    ax_scatter.set_title("Final Position Distribution")
    ax_scatter.set_xlabel("Longitude")
    ax_scatter.set_ylabel("Latitude")
    ax_scatter.legend()

    train_speed = train["speed"].reshape(-1)
    val_speed = val["speed"].reshape(-1)
    speed_upper = max(
        1e-6,
        float(np.percentile(np.concatenate([train_speed, val_speed]), 99)),
    )
    bins = np.linspace(0.0, speed_upper, 40)
    ax_speed.hist(train_speed, bins=bins, alpha=0.6, label="Train", density=True)
    ax_speed.hist(val_speed, bins=bins, alpha=0.6, label="Val", density=True)
    ax_speed.set_title("Speed Density (up to P99)")
    ax_speed.set_xlabel("Speed")
    ax_speed.set_ylabel("Density")
    ax_speed.legend()

    plot_group_distribution(
        ax_group,
        train_groups=group_labels(train["group"]),
        val_groups=group_labels(val["group"]),
    )

    plot_sample_trajectories(
        ax=ax_traj,
        coords=train["coords"],
        color="#1f77b4",
        max_windows=max_trajectory_windows,
        random_seed=random_seed,
        label="Train trajectories",
    )
    plot_sample_trajectories(
        ax=ax_traj,
        coords=val["coords"],
        color="#ff7f0e",
        max_windows=max_trajectory_windows,
        random_seed=random_seed + 1,
        label="Val trajectories",
    )
    ax_traj.set_title("Sampled Coordinate Windows")
    ax_traj.set_xlabel("Longitude")
    ax_traj.set_ylabel("Latitude")
    ax_traj.legend()

    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize and diagnose dataset quality.")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tag", type=str, default="dataset")
    parser.add_argument("--coord-decimals", type=int, default=4)
    parser.add_argument("--max-trajectory-windows", type=int, default=400)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    train = dict(np.load(os.path.join(args.dataset_dir, "train_sequences.npz")))
    val = dict(np.load(os.path.join(args.dataset_dir, "val_sequences.npz")))

    metrics = build_metrics(train, val, coord_decimals=args.coord_decimals)
    output_png = os.path.join(args.output_dir, f"{args.tag}_overview.png")
    output_json = os.path.join(args.output_dir, f"{args.tag}_metrics.json")

    create_figure(
        train=train,
        val=val,
        output_png=output_png,
        max_trajectory_windows=args.max_trajectory_windows,
        random_seed=args.random_seed,
    )

    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print("Saved figure:", output_png)
    print("Saved metrics:", output_json)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
