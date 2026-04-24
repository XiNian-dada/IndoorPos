#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from TrainTinyESP32Model import (
    TinySequenceDataset,
    build_tiny_model,
    concat_splits,
    load_npz,
    select_device,
)


def parse_group_to_class(raw_mapping: Dict[str, int]) -> Dict[Tuple[int, int], int]:
    parsed: Dict[Tuple[int, int], int] = {}
    for key, value in raw_mapping.items():
        b, f = key.split("_")
        parsed[(int(b), int(f))] = int(value)
    return parsed


def load_checkpoint_stats(payload: Dict[str, object]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in payload["stats"].items():
        dtype = np.int64 if key == "anchor_group_ids" else np.float32
        out[key] = np.asarray(value, dtype=dtype)
    return out


def build_test_loader(
    test_arrays: Dict[str, np.ndarray],
    checkpoint: Dict[str, object],
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, Dict[str, np.ndarray]]:
    stats = load_checkpoint_stats(checkpoint)
    group_to_class = parse_group_to_class(checkpoint["group_to_class"])
    args = checkpoint.get("args", {})
    rssi_feature_mode = "zscore"
    if isinstance(args, dict):
        rssi_feature_mode = str(args.get("rssi_feature_mode", "zscore"))

    dataset = TinySequenceDataset(
        arrays=test_arrays,
        feature_mean=stats["feature_mean"],
        feature_std=stats["feature_std"],
        motion_mean=stats["motion_mean"],
        motion_std=stats["motion_std"],
        coord_mean=stats["coord_mean"],
        coord_std=stats["coord_std"],
        group_to_class=group_to_class,
        anchor_centers_raw=stats["anchor_centers_raw"],
        anchor_centers_norm=stats["anchor_centers_norm"],
        anchor_group_ids=stats["anchor_group_ids"].astype(np.int64),
        rssi_feature_mode=rssi_feature_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return loader, stats


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    error = np.linalg.norm(pred - target, axis=1)
    return {
        "mean_error_m": float(error.mean()),
        "median_error_m": float(np.median(error)),
        "p75_error_m": float(np.quantile(error, 0.75)),
        "p90_error_m": float(np.quantile(error, 0.90)),
        "p95_error_m": float(np.quantile(error, 0.95)),
        "rmse_m": float(math.sqrt(np.mean(error ** 2))),
        "max_error_m": float(error.max()),
    }


def densest_cluster_centroid(points: np.ndarray, radius_m: float) -> np.ndarray:
    # points: [N, 2]
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    counts = (dist <= radius_m).sum(axis=1)
    max_count = counts.max()
    cand = np.where(counts == max_count)[0]
    if len(cand) == 1:
        center_idx = int(cand[0])
    else:
        mean_dist = dist[cand].mean(axis=1)
        center_idx = int(cand[np.argmin(mean_dist)])

    cluster_idx = np.where(dist[center_idx] <= radius_m)[0]
    cluster = points[cluster_idx]
    return cluster.mean(axis=0).astype(np.float32)


def consensus_from_samples(samples: np.ndarray, radius_m: float) -> np.ndarray:
    # samples: [B, N, 2]
    out = np.zeros((samples.shape[0], 2), dtype=np.float32)
    for i in range(samples.shape[0]):
        out[i] = densest_cluster_centroid(samples[i], radius_m=radius_m)
    return out


def evaluate_baseline_and_consensus(
    model: nn.Module,
    loader: DataLoader,
    coord_mean: np.ndarray,
    coord_std: np.ndarray,
    device: torch.device,
    n_runs: int,
    radius_m: float,
    noise_std: float,
    rssi_feature_dim: int,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    coord_mean_t = torch.from_numpy(coord_mean.astype(np.float32)).to(device)
    coord_std_t = torch.from_numpy(coord_std.astype(np.float32)).to(device)

    baseline_preds: List[np.ndarray] = []
    consensus_preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            target = batch["y_last_raw"].cpu().numpy()
            targets.append(target)

            base_out = model(inputs, None)
            base_pred = (base_out["pred_coord"] * coord_std_t + coord_mean_t).cpu().numpy()
            baseline_preds.append(base_pred)

            run_preds: List[np.ndarray] = [base_pred]
            for _ in range(max(1, n_runs) - 1):
                noisy_inputs = inputs.clone()
                if noise_std > 0:
                    noise = torch.randn_like(noisy_inputs[:, :, :rssi_feature_dim]) * noise_std
                    noisy_inputs[:, :, :rssi_feature_dim] = noisy_inputs[:, :, :rssi_feature_dim] + noise
                out = model(noisy_inputs, None)
                pred = (out["pred_coord"] * coord_std_t + coord_mean_t).cpu().numpy()
                run_preds.append(pred)

            run_stack = np.stack(run_preds, axis=1)  # [B, N, 2]
            consensus_pred = consensus_from_samples(run_stack, radius_m=radius_m)
            consensus_preds.append(consensus_pred)

    y_true = np.concatenate(targets, axis=0)
    y_base = np.concatenate(baseline_preds, axis=0)
    y_cons = np.concatenate(consensus_preds, axis=0)

    return {
        "baseline": regression_metrics(y_base, y_true),
        "consensus": regression_metrics(y_cons, y_true),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate tiny model with N-run dense-cluster consensus inference.",
    )
    parser.add_argument("--checkpoint", type=str, default="runs/tiny_esp32_best_now/best_tiny_esp32_model.pt")
    parser.add_argument("--test-dir", type=str, default="test_dataset")
    parser.add_argument("--output-dir", type=str, default="runs/tiny_consensus_eval")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-runs", type=int, default=7)
    parser.add_argument("--radius-m", type=float, default=4.0)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu") if args.cpu_only else select_device(cpu_only=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    test_dir = Path(args.test_dir)
    test_arrays = concat_splits(
        [
            load_npz(test_dir / "train_sequences.npz"),
            load_npz(test_dir / "val_sequences.npz"),
        ]
    )
    loader, stats = build_test_loader(
        test_arrays=test_arrays,
        checkpoint=checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    cfg = checkpoint["best_candidate"]
    checkpoint_args = checkpoint.get("args", {})
    model_arch = str(checkpoint.get("model_arch", "dscnn"))
    if model_arch == "dscnn" and isinstance(checkpoint_args, dict):
        model_arch = str(checkpoint_args.get("model_arch", model_arch))

    model = build_tiny_model(
        model_arch=model_arch,
        input_dim=int(checkpoint["input_dim"]),
        num_classes=len(checkpoint["class_names"]),
        num_anchors=int(checkpoint["num_anchors"]),
        anchor_centers_norm=stats["anchor_centers_norm"],
        anchor_group_ids=stats["anchor_group_ids"].astype(np.int64),
        step_hidden=int(cfg["step_hidden"]),
        temporal_hidden=int(cfg["temporal_hidden"]),
        head_hidden=int(cfg["head_hidden"]),
        dropout=float(checkpoint_args.get("dropout", 0.08)) if isinstance(checkpoint_args, dict) else 0.08,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    rssi_feature_dim = int(stats["feature_mean"].shape[0])
    result = evaluate_baseline_and_consensus(
        model=model,
        loader=loader,
        coord_mean=stats["coord_mean"],
        coord_std=stats["coord_std"],
        device=device,
        n_runs=args.n_runs,
        radius_m=args.radius_m,
        noise_std=args.noise_std,
        rssi_feature_dim=rssi_feature_dim,
    )

    payload = {
        "device": str(device),
        "checkpoint": args.checkpoint,
        "model_arch": model_arch,
        "n_runs": args.n_runs,
        "radius_m": args.radius_m,
        "noise_std": args.noise_std,
        "rssi_feature_dim": rssi_feature_dim,
        "baseline": result["baseline"],
        "consensus": result["consensus"],
        "delta_mean_m": float(result["consensus"]["mean_error_m"] - result["baseline"]["mean_error_m"]),
        "delta_p90_m": float(result["consensus"]["p90_error_m"] - result["baseline"]["p90_error_m"]),
        "delta_rmse_m": float(result["consensus"]["rmse_m"] - result["baseline"]["rmse_m"]),
    }
    (output_dir / "consensus_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved: {output_dir / 'consensus_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
