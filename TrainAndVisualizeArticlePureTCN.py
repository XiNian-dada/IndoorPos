#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ArticlePureTCNModel import (
    ArticlePureTCNConfig,
    ArticlePureTCNModel,
    apply_kalman,
    apply_speed_cap,
    count_parameters,
    rollout_positions,
)


@dataclass
class GridSpec:
    min_x: float
    min_y: float
    cell_size: float
    nx: int
    ny: int

    @property
    def num_classes(self) -> int:
        return int(self.nx * self.ny)


@dataclass
class PostProcessConfig:
    enabled: bool
    speed_cap_scale: float
    speed_cap_min: float
    speed_cap_global: float
    kalman_process_var: float
    kalman_measurement_var: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str, cpu_only: bool) -> torch.device:
    if cpu_only:
        return torch.device("cpu")

    d = device_arg.lower().strip()
    if d != "auto":
        return torch.device(d)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def concat_splits(splits: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not splits:
        raise ValueError("empty splits")
    keys = splits[0].keys()
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        out[k] = np.concatenate([s[k] for s in splits], axis=0)
    return out


def limit_arrays(arrays: Dict[str, np.ndarray], max_samples: int) -> Dict[str, np.ndarray]:
    if max_samples <= 0:
        return arrays
    first_key = next(iter(arrays.keys()))
    n = arrays[first_key].shape[0]
    if max_samples >= n:
        return arrays
    return {k: v[:max_samples] for k, v in arrays.items()}


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    err = np.linalg.norm(pred - target, axis=1)
    return {
        "mean_error_m": float(err.mean()),
        "median_error_m": float(np.median(err)),
        "p90_error_m": float(np.quantile(err, 0.90)),
        "p95_error_m": float(np.quantile(err, 0.95)),
        "rmse_m": float(math.sqrt(np.mean(err ** 2))),
        "max_error_m": float(err.max()),
    }


def build_grid_spec(coords: np.ndarray, cell_size: float, margin: float) -> GridSpec:
    if cell_size <= 0:
        raise ValueError("grid cell_size must be > 0")
    min_xy = coords.min(axis=0) - float(margin)
    max_xy = coords.max(axis=0) + float(margin)
    span = np.maximum(max_xy - min_xy, 1e-6)
    nx = max(1, int(math.ceil(float(span[0]) / cell_size)))
    ny = max(1, int(math.ceil(float(span[1]) / cell_size)))
    return GridSpec(
        min_x=float(min_xy[0]),
        min_y=float(min_xy[1]),
        cell_size=float(cell_size),
        nx=int(nx),
        ny=int(ny),
    )


def encode_grid(coords: np.ndarray, spec: GridSpec) -> np.ndarray:
    x_idx = np.floor((coords[:, 0] - spec.min_x) / spec.cell_size).astype(np.int64)
    y_idx = np.floor((coords[:, 1] - spec.min_y) / spec.cell_size).astype(np.int64)
    x_idx = np.clip(x_idx, 0, spec.nx - 1)
    y_idx = np.clip(y_idx, 0, spec.ny - 1)
    return (y_idx * spec.nx + x_idx).astype(np.int64)


def _read_seq_last(arrays: Dict[str, np.ndarray], key: str, default: Optional[np.ndarray] = None) -> np.ndarray:
    if key not in arrays:
        if default is None:
            raise KeyError(key)
        return default
    val = arrays[key]
    if val.ndim == 3 and val.shape[-1] == 1:
        return val[:, -1, 0].astype(np.float32)
    if val.ndim == 2:
        return val[:, -1].astype(np.float32)
    if val.ndim == 1:
        return val.astype(np.float32)
    raise ValueError(f"Unexpected shape for {key}: {val.shape}")


class SequenceDataset(Dataset):
    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        motion_mean: np.ndarray,
        motion_std: np.ndarray,
        delta_mean: np.ndarray,
        delta_std: np.ndarray,
        grid_spec: GridSpec,
    ) -> None:
        self.rssi = arrays["X"].astype(np.float32)

        self.coord_seq = arrays["y"].astype(np.float32)
        if self.coord_seq.ndim != 3 or self.coord_seq.shape[1] < 2 or self.coord_seq.shape[2] != 2:
            raise ValueError(f"Unexpected coord sequence shape: {self.coord_seq.shape}")

        motion = arrays["motion_features"].astype(np.float32)
        self.motion = ((motion - motion_mean[None, None, :]) / motion_std[None, None, :]).astype(np.float32)

        self.coord_prev = self.coord_seq[:, -2, :].astype(np.float32)
        self.coord_raw = arrays["y_last"].astype(np.float32)
        self.delta_raw = (self.coord_raw - self.coord_prev).astype(np.float32)
        self.delta_norm = ((self.delta_raw - delta_mean[None, :]) / delta_std[None, :]).astype(np.float32)

        self.grid_id = encode_grid(self.coord_raw, grid_spec)

        n = int(self.rssi.shape[0])
        if "trajectory_id" in arrays:
            self.trajectory_id = arrays["trajectory_id"].astype(np.int64).reshape(-1)
        else:
            self.trajectory_id = np.arange(n, dtype=np.int64)

        if "elapsed_time" in arrays:
            self.elapsed_last = _read_seq_last(arrays, "elapsed_time")
        elif "time_index" in arrays:
            self.elapsed_last = _read_seq_last(arrays, "time_index")
        else:
            self.elapsed_last = np.arange(n, dtype=np.float32)

        if "speed" in arrays:
            self.speed_last = _read_seq_last(arrays, "speed")
        else:
            self.speed_last = np.linalg.norm(self.delta_raw, axis=1).astype(np.float32)

    def __len__(self) -> int:
        return int(self.rssi.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "rssi": torch.from_numpy(self.rssi[idx]),
            "motion": torch.from_numpy(self.motion[idx]),
            "coord_seq": torch.from_numpy(self.coord_seq[idx]),
            "coord_raw": torch.from_numpy(self.coord_raw[idx]),
            "coord_prev": torch.from_numpy(self.coord_prev[idx]),
            "delta_norm": torch.from_numpy(self.delta_norm[idx]),
            "grid_id": torch.tensor(int(self.grid_id[idx]), dtype=torch.long),
            "trajectory_id": torch.tensor(int(self.trajectory_id[idx]), dtype=torch.long),
            "elapsed_last": torch.tensor(float(self.elapsed_last[idx]), dtype=torch.float32),
            "speed_last": torch.tensor(float(self.speed_last[idx]), dtype=torch.float32),
        }


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _trajectory_effective_stats(traj: np.ndarray) -> Tuple[bool, int, int]:
    if traj.size == 0:
        return False, 0, 0
    _, counts = np.unique(traj.astype(np.int64), return_counts=True)
    ge2 = int((counts >= 2).sum())
    return ge2 > 0, int(counts.size), ge2


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    delta_mean: np.ndarray,
    delta_std: np.ndarray,
    grid_loss_fn: nn.Module,
    cls_loss_weight: float,
    post_cfg: PostProcessConfig,
    return_predictions: bool = False,
) -> Dict[str, object]:
    model.eval()
    delta_mean_t = torch.from_numpy(delta_mean.astype(np.float32)).to(device)
    delta_std_t = torch.from_numpy(delta_std.astype(np.float32)).to(device)

    total_loss = 0.0
    total_delta_loss = 0.0
    total_grid_loss = 0.0
    n_batches = 0

    pred_coords_teacher: List[np.ndarray] = []
    pred_deltas_raw: List[np.ndarray] = []
    target_coords: List[np.ndarray] = []
    coord_prev: List[np.ndarray] = []
    pred_grid: List[np.ndarray] = []
    true_grid: List[np.ndarray] = []
    traj_id: List[np.ndarray] = []
    elapsed: List[np.ndarray] = []
    speed: List[np.ndarray] = []
    coord_seq: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            out = model(batch["rssi"], batch["motion"])

            delta_loss = nn.functional.smooth_l1_loss(out["delta_norm"], batch["delta_norm"])
            grid_loss = grid_loss_fn(out["grid_logits"], batch["grid_id"])
            loss = delta_loss + cls_loss_weight * grid_loss

            total_loss += float(loss.item())
            total_delta_loss += float(delta_loss.item())
            total_grid_loss += float(grid_loss.item())
            n_batches += 1

            delta_raw = out["delta_norm"] * delta_std_t + delta_mean_t
            coord_pred_teacher = batch["coord_prev"] + delta_raw

            pred_coords_teacher.append(coord_pred_teacher.detach().cpu().numpy())
            pred_deltas_raw.append(delta_raw.detach().cpu().numpy())
            target_coords.append(batch["coord_raw"].detach().cpu().numpy())
            coord_prev.append(batch["coord_prev"].detach().cpu().numpy())
            pred_grid.append(out["grid_logits"].argmax(dim=1).detach().cpu().numpy())
            true_grid.append(batch["grid_id"].detach().cpu().numpy())
            traj_id.append(batch["trajectory_id"].detach().cpu().numpy())
            elapsed.append(batch["elapsed_last"].detach().cpu().numpy())
            speed.append(batch["speed_last"].detach().cpu().numpy())
            coord_seq.append(batch["coord_seq"].detach().cpu().numpy())

    coord_pred_teacher = np.concatenate(pred_coords_teacher, axis=0)
    delta_pred_raw = np.concatenate(pred_deltas_raw, axis=0)
    coord_prev_np = np.concatenate(coord_prev, axis=0)
    coord_true = np.concatenate(target_coords, axis=0)
    g_pred = np.concatenate(pred_grid, axis=0)
    g_true = np.concatenate(true_grid, axis=0)
    traj_np = np.concatenate(traj_id, axis=0)
    elapsed_np = np.concatenate(elapsed, axis=0).astype(np.float32)
    speed_np = np.concatenate(speed, axis=0).astype(np.float32)
    coord_seq_np = np.concatenate(coord_seq, axis=0)

    coord_pred_raw = rollout_positions(
        delta_xy=delta_pred_raw,
        coord_prev=coord_prev_np,
        trajectory_id=traj_np,
        elapsed=elapsed_np,
    )

    reg_teacher = regression_metrics(coord_pred_teacher, coord_true)
    reg_raw = regression_metrics(coord_pred_raw, coord_true)

    traj_effective, num_traj, num_traj_ge2 = _trajectory_effective_stats(traj_np)
    if post_cfg.enabled and traj_effective:
        speed_capped = apply_speed_cap(
            coords=coord_pred_raw,
            trajectory_id=traj_np,
            elapsed=elapsed_np,
            speed=speed_np,
            speed_cap_scale=post_cfg.speed_cap_scale,
            speed_cap_min=post_cfg.speed_cap_min,
            speed_cap_global=post_cfg.speed_cap_global,
        )
        coord_post = apply_kalman(
            coords=speed_capped,
            trajectory_id=traj_np,
            elapsed=elapsed_np,
            process_var=post_cfg.kalman_process_var,
            measurement_var=post_cfg.kalman_measurement_var,
        )
    else:
        coord_post = coord_pred_raw

    reg_post = regression_metrics(coord_post, coord_true)

    result: Dict[str, object] = {
        "loss_total": total_loss / max(1, n_batches),
        "loss_delta": total_delta_loss / max(1, n_batches),
        "loss_grid": total_grid_loss / max(1, n_batches),
        "grid_classification_accuracy": float((g_pred == g_true).mean()),
        "regression_teacher": reg_teacher,
        "regression_raw": reg_raw,
        "regression_post": reg_post,
        "postprocess": {
            "enabled": bool(post_cfg.enabled),
            "effective": bool(post_cfg.enabled and traj_effective),
            "num_trajectories": int(num_traj),
            "num_trajectories_ge2": int(num_traj_ge2),
        },
    }

    if return_predictions:
        result["predictions"] = {
            "coord_true": coord_true,
            "coord_pred_teacher": coord_pred_teacher,
            "coord_pred_raw": coord_pred_raw,
            "coord_pred_post": coord_post,
            "coord_prev": coord_prev_np,
            "coord_seq": coord_seq_np,
            "trajectory_id": traj_np,
            "elapsed_last": elapsed_np,
            "speed_last": speed_np,
            "grid_true": g_true,
            "grid_pred": g_pred,
        }

    return result


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grid_loss_fn: nn.Module,
    cls_loss_weight: float,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_delta_loss = 0.0
    total_grid_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        out = model(batch["rssi"], batch["motion"])
        delta_loss = nn.functional.smooth_l1_loss(out["delta_norm"], batch["delta_norm"])
        grid_loss = grid_loss_fn(out["grid_logits"], batch["grid_id"])
        loss = delta_loss + cls_loss_weight * grid_loss

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        total_delta_loss += float(delta_loss.item())
        total_grid_loss += float(grid_loss.item())
        n_batches += 1

    return {
        "loss_total": total_loss / max(1, n_batches),
        "loss_delta": total_delta_loss / max(1, n_batches),
        "loss_grid": total_grid_loss / max(1, n_batches),
    }


def save_predictions_csv(pred: Dict[str, np.ndarray], out_csv: Path) -> None:
    true_xy = pred["coord_true"]
    raw_xy = pred["coord_pred_raw"]
    post_xy = pred["coord_pred_post"]
    traj = pred["trajectory_id"]
    elapsed = pred["elapsed_last"]

    raw_err = np.linalg.norm(raw_xy - true_xy, axis=1)
    post_err = np.linalg.norm(post_xy - true_xy, axis=1)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "index",
                "trajectory_id",
                "elapsed_last",
                "true_x",
                "true_y",
                "pred_raw_x",
                "pred_raw_y",
                "pred_post_x",
                "pred_post_y",
                "error_raw",
                "error_post",
            ]
        )
        for i in range(true_xy.shape[0]):
            w.writerow(
                [
                    i,
                    int(traj[i]),
                    float(elapsed[i]),
                    float(true_xy[i, 0]),
                    float(true_xy[i, 1]),
                    float(raw_xy[i, 0]),
                    float(raw_xy[i, 1]),
                    float(post_xy[i, 0]),
                    float(post_xy[i, 1]),
                    float(raw_err[i]),
                    float(post_err[i]),
                ]
            )


def _setup_matplotlib(output_dir: Path) -> None:
    mplcfg = output_dir / ".mplconfig"
    mplcfg.mkdir(parents=True, exist_ok=True)
    # Force a writable cache dir + headless backend to avoid GUI backend crashes.
    os.environ["MPLCONFIGDIR"] = str(mplcfg)
    os.environ["MPLBACKEND"] = "Agg"


def plot_global_scatter(pred: Dict[str, np.ndarray], out_png: Path) -> None:
    _setup_matplotlib(out_png.parent)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    true_xy = pred["coord_true"]
    raw_xy = pred["coord_pred_raw"]
    post_xy = pred["coord_pred_post"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    for ax, pxy, title in [
        (axes[0], raw_xy, "Raw Prediction"),
        (axes[1], post_xy, "Post-processed Prediction"),
    ]:
        ax.scatter(true_xy[:, 0], true_xy[:, 1], s=10, alpha=0.60, c="#111827", label="True")
        ax.scatter(pxy[:, 0], pxy[:, 1], s=10, alpha=0.55, c="#dc2626", label="Pred")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Test Set: True vs Predicted Coordinates", fontsize=14)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_window_trajectories(pred: Dict[str, np.ndarray], out_png: Path, max_samples: int, seed: int) -> None:
    _setup_matplotlib(out_png.parent)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    coord_seq = pred["coord_seq"]      # [N, T, 2]
    true_xy = pred["coord_true"]       # [N, 2]
    raw_xy = pred["coord_pred_raw"]    # [N, 2]
    post_xy = pred["coord_pred_post"]  # [N, 2]

    n = coord_seq.shape[0]
    m = min(max_samples, n)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(np.arange(n), size=m, replace=False))

    cols = 4
    rows = int(math.ceil(m / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes.flat[i]
        if i >= m:
            ax.axis("off")
            continue

        idx = int(chosen[i])
        path = coord_seq[idx]

        ax.plot(path[:, 0], path[:, 1], c="#111827", lw=1.8, marker="o", ms=3, label="True history")
        ax.scatter(true_xy[idx, 0], true_xy[idx, 1], c="#111827", s=34, marker="x", label="True last")

        prev = path[-2]
        ax.plot([prev[0], raw_xy[idx, 0]], [prev[1], raw_xy[idx, 1]], c="#dc2626", lw=1.5, ls="--")
        ax.scatter(raw_xy[idx, 0], raw_xy[idx, 1], c="#dc2626", s=30, label="Pred raw")

        ax.plot([prev[0], post_xy[idx, 0]], [prev[1], post_xy[idx, 1]], c="#16a34a", lw=1.4, ls=":")
        ax.scatter(post_xy[idx, 0], post_xy[idx, 1], c="#16a34a", s=26, label="Pred post")

        e_raw = float(np.linalg.norm(raw_xy[idx] - true_xy[idx]))
        e_post = float(np.linalg.norm(post_xy[idx] - true_xy[idx]))
        ax.set_title(f"sample {idx} | e_raw={e_raw:.2f} | e_post={e_post:.2f}", fontsize=9)
        ax.grid(alpha=0.22)
        ax.set_aspect("equal", adjustable="box")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle("Window-level trajectory check: 5-step true path + predicted final point", fontsize=14)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone trainer for ArticlePureTCNModel with trajectory visualizations on test set.",
    )
    p.add_argument("--train-dir", type=str, default="training_dataset")
    p.add_argument("--test-dir", type=str, default="test_dataset")
    p.add_argument("--output-dir", type=str, default="runs/article_pure_tcn_standalone")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--cls-loss-weight", type=float, default=0.20)
    p.add_argument("--label-smoothing", type=float, default=0.03)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--device", type=str, default="auto", help="auto|cuda|mps|cpu")
    p.add_argument("--cpu-only", action="store_true")

    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)

    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--token-hidden", type=int, default=64)
    p.add_argument("--temporal-hidden", type=int, default=128)
    p.add_argument("--head-hidden", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.08)

    p.add_argument("--grid-cell-size", type=float, default=20.0)
    p.add_argument("--grid-margin", type=float, default=1.0)

    p.add_argument("--disable-postprocess", action="store_true")
    p.add_argument("--speed-cap-scale", type=float, default=1.25)
    p.add_argument("--speed-cap-min", type=float, default=0.5)
    p.add_argument("--speed-cap-quantile", type=float, default=0.99)
    p.add_argument("--speed-cap-multiplier", type=float, default=1.2)
    p.add_argument("--kalman-process-var", type=float, default=4.0)
    p.add_argument("--kalman-measurement-var", type=float, default=9.0)

    p.add_argument("--selection-metric", choices=["raw", "post"], default="post")
    p.add_argument("--plot-samples", type=int, default=24)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "best_model.pt"

    device = select_device(args.device, args.cpu_only)

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    train_arrays = load_npz(train_dir / "train_sequences.npz")
    val_arrays = load_npz(train_dir / "val_sequences.npz")
    test_arrays = concat_splits(
        [
            load_npz(test_dir / "train_sequences.npz"),
            load_npz(test_dir / "val_sequences.npz"),
        ]
    )

    train_arrays = limit_arrays(train_arrays, args.max_train_samples)
    val_arrays = limit_arrays(val_arrays, args.max_val_samples)
    test_arrays = limit_arrays(test_arrays, args.max_test_samples)

    motion_train = train_arrays["motion_features"].astype(np.float32)
    motion_mean = motion_train.mean(axis=(0, 1)).astype(np.float32)
    motion_std = motion_train.std(axis=(0, 1)).astype(np.float32)
    motion_std = np.where(motion_std < 1e-6, 1.0, motion_std).astype(np.float32)

    train_y = train_arrays["y"].astype(np.float32)
    train_prev = train_y[:, -2, :]
    train_last = train_arrays["y_last"].astype(np.float32)
    train_delta = (train_last - train_prev).astype(np.float32)

    delta_mean = train_delta.mean(axis=0).astype(np.float32)
    delta_std = train_delta.std(axis=0).astype(np.float32)
    delta_std = np.where(delta_std < 1e-6, 1.0, delta_std).astype(np.float32)

    grid_spec = build_grid_spec(train_last, cell_size=float(args.grid_cell_size), margin=float(args.grid_margin))

    step_mag = np.linalg.norm(train_delta, axis=1)
    speed_cap_global = float(
        np.quantile(step_mag, min(max(float(args.speed_cap_quantile), 0.5), 0.9999)) * float(args.speed_cap_multiplier)
    )
    speed_cap_global = max(speed_cap_global, 1e-3)

    post_cfg = PostProcessConfig(
        enabled=not args.disable_postprocess,
        speed_cap_scale=float(args.speed_cap_scale),
        speed_cap_min=float(args.speed_cap_min),
        speed_cap_global=float(speed_cap_global),
        kalman_process_var=float(args.kalman_process_var),
        kalman_measurement_var=float(args.kalman_measurement_var),
    )

    train_ds = SequenceDataset(train_arrays, motion_mean, motion_std, delta_mean, delta_std, grid_spec)
    val_ds = SequenceDataset(val_arrays, motion_mean, motion_std, delta_mean, delta_std, grid_spec)
    test_ds = SequenceDataset(test_arrays, motion_mean, motion_std, delta_mean, delta_std, grid_spec)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_cfg = ArticlePureTCNConfig(
        num_aps=int(train_arrays["X"].shape[-1]),
        motion_dim=int(train_arrays["motion_features"].shape[-1]),
        num_grid_classes=int(grid_spec.num_classes),
        top_k=int(args.top_k),
        token_hidden=int(args.token_hidden),
        temporal_hidden=int(args.temporal_hidden),
        head_hidden=int(args.head_hidden),
        dropout=float(args.dropout),
    )
    model = ArticlePureTCNModel(model_cfg).to(device)

    print(f"Using device: {device}", flush=True)
    print(f"Train/Val/Test samples: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}", flush=True)
    print(
        f"Input: seq_len={train_arrays['X'].shape[1]}, num_aps={model_cfg.num_aps}, "
        f"motion_dim={model_cfg.motion_dim}, grid_classes={model_cfg.num_grid_classes}",
        flush=True,
    )
    print(f"Param count: {count_parameters(model)}", flush=True)

    grid_loss_fn = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    no_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grid_loss_fn=grid_loss_fn,
            cls_loss_weight=float(args.cls_loss_weight),
            grad_clip=float(args.grad_clip),
        )

        val_out = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            delta_mean=delta_mean,
            delta_std=delta_std,
            grid_loss_fn=grid_loss_fn,
            cls_loss_weight=float(args.cls_loss_weight),
            post_cfg=post_cfg,
            return_predictions=False,
        )

        key = "regression_post" if args.selection_metric == "post" else "regression_raw"
        val_metric = float(val_out[key]["mean_error_m"])
        scheduler.step(val_metric)

        history_row = {
            "epoch": float(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_loss["loss_total"]),
            "val_mean_raw": float(val_out["regression_raw"]["mean_error_m"]),
            "val_mean_post": float(val_out["regression_post"]["mean_error_m"]),
            "val_rmse_raw": float(val_out["regression_raw"]["rmse_m"]),
            "val_rmse_post": float(val_out["regression_post"]["rmse_m"]),
            "val_cls_acc": float(val_out["grid_classification_accuracy"]),
        }
        history.append(history_row)

        print(
            f"[epoch {epoch:03d}] train_loss={train_loss['loss_total']:.5f} | "
            f"val_raw={val_out['regression_raw']['mean_error_m']:.3f}m | "
            f"val_post={val_out['regression_post']['mean_error_m']:.3f}m | "
            f"val_cls={val_out['grid_classification_accuracy']:.3f}",
            flush=True,
        )

        if val_metric + 1e-6 < best_val:
            best_val = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stop at epoch {epoch} (patience={args.patience})", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training failed: no best_state captured.")

    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": model_cfg.__dict__,
            "delta_mean": delta_mean.tolist(),
            "delta_std": delta_std.tolist(),
            "motion_mean": motion_mean.tolist(),
            "motion_std": motion_std.tolist(),
            "grid_spec": grid_spec.__dict__,
            "best_epoch": int(best_epoch),
            "best_val": float(best_val),
        },
        ckpt_path,
    )

    model.load_state_dict(best_state)
    model.to(device)

    test_out = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        delta_mean=delta_mean,
        delta_std=delta_std,
        grid_loss_fn=grid_loss_fn,
        cls_loss_weight=float(args.cls_loss_weight),
        post_cfg=post_cfg,
        return_predictions=True,
    )

    pred = test_out["predictions"]
    assert isinstance(pred, dict)

    save_predictions_csv(pred, output_dir / "test_predictions.csv")
    plot_global_scatter(pred, output_dir / "test_scatter_true_vs_pred.png")
    plot_window_trajectories(pred, output_dir / "test_window_trajectories.png", max_samples=args.plot_samples, seed=args.seed)

    summary = {
        "device": str(device),
        "model_name": "article_pure_tcn_standalone",
        "best_epoch": int(best_epoch),
        "best_val_selection_metric": float(best_val),
        "selection_metric": args.selection_metric,
        "model_config": model_cfg.__dict__,
        "postprocess": post_cfg.__dict__,
        "param_count": int(count_parameters(model)),
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "test_samples": int(len(test_ds)),
        "val_history": history,
        "test_metrics": {
            "regression_teacher": test_out["regression_teacher"],
            "regression_raw": test_out["regression_raw"],
            "regression_post": test_out["regression_post"],
            "grid_classification_accuracy": test_out["grid_classification_accuracy"],
            "postprocess": test_out["postprocess"],
        },
        "artifacts": {
            "checkpoint": str(ckpt_path),
            "predictions_csv": str(output_dir / "test_predictions.csv"),
            "scatter_png": str(output_dir / "test_scatter_true_vs_pred.png"),
            "window_trajectories_png": str(output_dir / "test_window_trajectories.png"),
        },
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Done ===", flush=True)
    print(f"Best epoch : {best_epoch}", flush=True)
    print(f"Test mean(raw)  : {test_out['regression_raw']['mean_error_m']:.3f} m", flush=True)
    print(f"Test mean(post) : {test_out['regression_post']['mean_error_m']:.3f} m", flush=True)
    print(f"Test RMSE(raw)  : {test_out['regression_raw']['rmse_m']:.3f} m", flush=True)
    print(f"Test cls acc    : {test_out['grid_classification_accuracy']:.3f}", flush=True)
    print(f"Output dir      : {output_dir}", flush=True)


if __name__ == "__main__":
    main()
