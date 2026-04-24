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
class ModelConfig:
    num_aps: int
    seq_len: int
    frame_hidden: int = 256
    rnn_hidden: int = 192
    rnn_layers: int = 2
    head_hidden: int = 256
    dropout: float = 0.12
    num_grid_classes: int = 256


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
    z = np.load(path)
    return {k: z[k] for k in z.files}


def concat_splits(splits: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not splits:
        raise ValueError("Empty splits")
    keys = splits[0].keys()
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        out[k] = np.concatenate([s[k] for s in splits], axis=0)
    return out


def limit_arrays(arrays: Dict[str, np.ndarray], max_samples: int) -> Dict[str, np.ndarray]:
    if max_samples <= 0:
        return arrays
    n = arrays[next(iter(arrays.keys()))].shape[0]
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


class RSSISeqDataset(Dataset):
    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        coord_mean: np.ndarray,
        coord_std: np.ndarray,
        grid_spec: GridSpec,
    ) -> None:
        self.x = arrays["X"].astype(np.float32)
        self.coord_seq = arrays["y"].astype(np.float32)
        self.coord_raw = arrays["y_last"].astype(np.float32)
        self.coord_norm = ((self.coord_raw - coord_mean[None, :]) / coord_std[None, :]).astype(np.float32)
        self.grid_id = encode_grid(self.coord_raw, grid_spec)

        self.group = arrays.get("group", np.zeros((self.x.shape[0], 2), dtype=np.int32)).astype(np.int32)
        self.trajectory_id = arrays.get("trajectory_id", np.arange(self.x.shape[0], dtype=np.int32)).astype(np.int64)

        if "elapsed_time" in arrays:
            e = arrays["elapsed_time"]
            if e.ndim == 3:
                self.elapsed_last = e[:, -1, 0].astype(np.float32)
            elif e.ndim == 2:
                self.elapsed_last = e[:, -1].astype(np.float32)
            else:
                self.elapsed_last = e.astype(np.float32).reshape(-1)
        elif "time_index" in arrays:
            t = arrays["time_index"]
            if t.ndim == 3:
                self.elapsed_last = t[:, -1, 0].astype(np.float32)
            elif t.ndim == 2:
                self.elapsed_last = t[:, -1].astype(np.float32)
            else:
                self.elapsed_last = t.astype(np.float32).reshape(-1)
        else:
            self.elapsed_last = np.arange(self.x.shape[0], dtype=np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[idx]),
            "coord_raw": torch.from_numpy(self.coord_raw[idx]),
            "coord_norm": torch.from_numpy(self.coord_norm[idx]),
            "coord_seq": torch.from_numpy(self.coord_seq[idx]),
            "grid_id": torch.tensor(int(self.grid_id[idx]), dtype=torch.long),
            "trajectory_id": torch.tensor(int(self.trajectory_id[idx]), dtype=torch.long),
            "elapsed_last": torch.tensor(float(self.elapsed_last[idx]), dtype=torch.float32),
            "group": torch.from_numpy(self.group[idx]),
        }


class AbsoluteRSSIEncoder(nn.Module):
    """Pure RSSI absolute localization model (cold-start friendly)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.frame_proj = nn.Sequential(
            nn.Linear(cfg.num_aps, cfg.frame_hidden),
            nn.LayerNorm(cfg.frame_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.frame_hidden, cfg.frame_hidden),
            nn.ReLU(),
        )

        self.rnn = nn.GRU(
            input_size=cfg.frame_hidden,
            hidden_size=cfg.rnn_hidden,
            num_layers=cfg.rnn_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0,
            bidirectional=True,
        )

        rnn_out = cfg.rnn_hidden * 2
        self.temporal_proj = nn.Sequential(
            nn.Linear(rnn_out * 3, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        self.coord_head = nn.Linear(cfg.head_hidden, 2)
        self.grid_head = nn.Linear(cfg.head_hidden, int(cfg.num_grid_classes))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, T, N_AP]
        z = self.frame_proj(x)
        h, _ = self.rnn(z)

        last = h[:, -1, :]
        mean = h.mean(dim=1)
        maxv = h.max(dim=1).values
        fuse = torch.cat([last, mean, maxv], dim=-1)

        feat = self.temporal_proj(fuse)
        return {
            "coord_norm": self.coord_head(feat),
            "grid_logits": self.grid_head(feat),
        }


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    coord_mean: np.ndarray,
    coord_std: np.ndarray,
    grid_loss_fn: nn.Module,
    cls_loss_weight: float,
    use_amp: bool,
    return_preds: bool = False,
) -> Dict[str, object]:
    model.eval()
    cm = torch.from_numpy(coord_mean.astype(np.float32)).to(device)
    cs = torch.from_numpy(coord_std.astype(np.float32)).to(device)

    losses = []
    coord_losses = []
    grid_losses = []

    pred_raw: List[np.ndarray] = []
    true_raw: List[np.ndarray] = []
    pred_grid: List[np.ndarray] = []
    true_grid: List[np.ndarray] = []
    traj_id: List[np.ndarray] = []
    elapsed: List[np.ndarray] = []
    groups: List[np.ndarray] = []
    coord_seq: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            amp_enabled = use_amp and device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                out = model(batch["x"])
                loss_coord = nn.functional.smooth_l1_loss(out["coord_norm"], batch["coord_norm"])
                loss_grid = grid_loss_fn(out["grid_logits"], batch["grid_id"])
                loss = loss_coord + cls_loss_weight * loss_grid

            losses.append(float(loss.item()))
            coord_losses.append(float(loss_coord.item()))
            grid_losses.append(float(loss_grid.item()))

            p = out["coord_norm"] * cs + cm
            pred_raw.append(p.detach().cpu().numpy())
            true_raw.append(batch["coord_raw"].detach().cpu().numpy())
            pred_grid.append(out["grid_logits"].argmax(dim=1).detach().cpu().numpy())
            true_grid.append(batch["grid_id"].detach().cpu().numpy())
            traj_id.append(batch["trajectory_id"].detach().cpu().numpy())
            elapsed.append(batch["elapsed_last"].detach().cpu().numpy())
            groups.append(batch["group"].detach().cpu().numpy())
            coord_seq.append(batch["coord_seq"].detach().cpu().numpy())

    pred_xy = np.concatenate(pred_raw, axis=0)
    true_xy = np.concatenate(true_raw, axis=0)

    out: Dict[str, object] = {
        "loss_total": float(np.mean(losses) if losses else 0.0),
        "loss_coord": float(np.mean(coord_losses) if coord_losses else 0.0),
        "loss_grid": float(np.mean(grid_losses) if grid_losses else 0.0),
        "regression": regression_metrics(pred_xy, true_xy),
        "grid_classification_accuracy": float(
            (np.concatenate(pred_grid, axis=0) == np.concatenate(true_grid, axis=0)).mean()
        ),
    }

    if return_preds:
        out["predictions"] = {
            "pred_xy": pred_xy,
            "true_xy": true_xy,
            "pred_grid": np.concatenate(pred_grid, axis=0),
            "true_grid": np.concatenate(true_grid, axis=0),
            "trajectory_id": np.concatenate(traj_id, axis=0),
            "elapsed_last": np.concatenate(elapsed, axis=0),
            "group": np.concatenate(groups, axis=0),
            "coord_seq": np.concatenate(coord_seq, axis=0),
        }

    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    grid_loss_fn: nn.Module,
    cls_loss_weight: float,
    grad_clip: float,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()

    losses = []
    coord_losses = []
    grid_losses = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        amp_enabled = use_amp and device.type == "cuda"
        if amp_enabled:
            assert scaler is not None
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                out = model(batch["x"])
                loss_coord = nn.functional.smooth_l1_loss(out["coord_norm"], batch["coord_norm"])
                loss_grid = grid_loss_fn(out["grid_logits"], batch["grid_id"])
                loss = loss_coord + cls_loss_weight * loss_grid

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(batch["x"])
            loss_coord = nn.functional.smooth_l1_loss(out["coord_norm"], batch["coord_norm"])
            loss_grid = grid_loss_fn(out["grid_logits"], batch["grid_id"])
            loss = loss_coord + cls_loss_weight * loss_grid
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        losses.append(float(loss.item()))
        coord_losses.append(float(loss_coord.item()))
        grid_losses.append(float(loss_grid.item()))

    return {
        "loss_total": float(np.mean(losses) if losses else 0.0),
        "loss_coord": float(np.mean(coord_losses) if coord_losses else 0.0),
        "loss_grid": float(np.mean(grid_losses) if grid_losses else 0.0),
    }


def wknn_predict(
    train_x_last: np.ndarray,
    train_y: np.ndarray,
    query_x_last: np.ndarray,
    k: int,
    weighted: bool,
    chunk_size: int = 256,
) -> np.ndarray:
    # train_x_last: [N, D], query_x_last: [M, D]
    n = train_x_last.shape[0]
    k = max(1, min(int(k), n))
    out = np.zeros((query_x_last.shape[0], 2), dtype=np.float32)

    train_norm = np.sum(train_x_last ** 2, axis=1, keepdims=True).T  # [1, N]

    for start in range(0, query_x_last.shape[0], chunk_size):
        end = min(start + chunk_size, query_x_last.shape[0])
        q = query_x_last[start:end]  # [B, D]

        # squared euclidean via expansion
        q_norm = np.sum(q ** 2, axis=1, keepdims=True)  # [B,1]
        dist2 = q_norm + train_norm - 2.0 * (q @ train_x_last.T)
        dist2 = np.maximum(dist2, 0.0)

        idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]  # [B, k]
        picked_d2 = np.take_along_axis(dist2, idx, axis=1)
        picked_y = train_y[idx]  # [B, k, 2]

        if weighted:
            w = 1.0 / (np.sqrt(picked_d2) + 1e-6)
            w = w / np.sum(w, axis=1, keepdims=True)
            pred = np.sum(picked_y * w[:, :, None], axis=1)
        else:
            pred = picked_y.mean(axis=1)

        out[start:end] = pred.astype(np.float32)

    return out


def tune_wknn(
    train_arrays: Dict[str, np.ndarray],
    val_arrays: Dict[str, np.ndarray],
    k_candidates: List[int],
) -> Dict[str, object]:
    x_train = train_arrays["X"].astype(np.float32)[:, -1, :]
    y_train = train_arrays["y_last"].astype(np.float32)

    x_val = val_arrays["X"].astype(np.float32)[:, -1, :]
    y_val = val_arrays["y_last"].astype(np.float32)

    best = {"k": None, "weighted": None, "mean_error_m": float("inf")}
    candidates = []

    for weighted in [True, False]:
        for k in k_candidates:
            pred = wknn_predict(x_train, y_train, x_val, k=k, weighted=weighted)
            m = regression_metrics(pred, y_val)
            row = {
                "name": f"wknn_k{k}_{'weighted' if weighted else 'mean'}",
                "k": int(k),
                "weighted": bool(weighted),
                **m,
            }
            candidates.append(row)
            if m["mean_error_m"] < best["mean_error_m"]:
                best = {
                    "k": int(k),
                    "weighted": bool(weighted),
                    "mean_error_m": float(m["mean_error_m"]),
                }

    return {"best": best, "candidates": candidates}


def _setup_matplotlib(output_dir: Path) -> None:
    mplcfg = output_dir / ".mplconfig"
    mplcfg.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mplcfg)
    os.environ["MPLBACKEND"] = "Agg"


def save_predictions_csv(pred: Dict[str, np.ndarray], out_csv: Path) -> None:
    p = pred["pred_xy"]
    t = pred["true_xy"]
    g = pred["group"]
    traj = pred["trajectory_id"]
    el = pred["elapsed_last"]
    err = np.linalg.norm(p - t, axis=1)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "index",
                "trajectory_id",
                "elapsed_last",
                "group_building",
                "group_floor",
                "true_x",
                "true_y",
                "pred_x",
                "pred_y",
                "error_m",
            ]
        )
        for i in range(p.shape[0]):
            w.writerow(
                [
                    i,
                    int(traj[i]),
                    float(el[i]),
                    int(g[i, 0]),
                    int(g[i, 1]),
                    float(t[i, 0]),
                    float(t[i, 1]),
                    float(p[i, 0]),
                    float(p[i, 1]),
                    float(err[i]),
                ]
            )


def plot_scatter(pred: Dict[str, np.ndarray], out_png: Path) -> None:
    _setup_matplotlib(out_png.parent)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    p = pred["pred_xy"]
    t = pred["true_xy"]
    err = np.linalg.norm(p - t, axis=1)

    fig, ax = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
    ax.scatter(t[:, 0], t[:, 1], s=10, c="#111827", alpha=0.45, label="True")
    sc = ax.scatter(p[:, 0], p[:, 1], s=10, c=err, cmap="turbo", alpha=0.72, label="Pred")
    ax.set_title("Absolute Localization: True vs Pred (color=error)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Error (m)")
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_error_hist(pred: Dict[str, np.ndarray], out_png: Path) -> None:
    _setup_matplotlib(out_png.parent)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    p = pred["pred_xy"]
    t = pred["true_xy"]
    err = np.linalg.norm(p - t, axis=1)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.hist(err, bins=40, color="#1d4ed8", alpha=0.8)
    ax.axvline(float(np.mean(err)), color="#dc2626", ls="--", lw=2, label=f"mean={np.mean(err):.2f}")
    ax.axvline(float(np.quantile(err, 0.90)), color="#16a34a", ls=":", lw=2, label=f"p90={np.quantile(err,0.90):.2f}")
    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("Error (m)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_window_paths(pred: Dict[str, np.ndarray], out_png: Path, max_samples: int, seed: int) -> None:
    _setup_matplotlib(out_png.parent)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    seq = pred["coord_seq"]
    p = pred["pred_xy"]
    t = pred["true_xy"]

    n = seq.shape[0]
    m = min(max_samples, n)
    if m <= 0:
        return

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(np.arange(n), size=m, replace=False))

    cols = 4
    rows = int(math.ceil(m / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.5 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes.flat[i]
        if i >= m:
            ax.axis("off")
            continue
        idx = int(chosen[i])

        path = seq[idx]
        ax.plot(path[:, 0], path[:, 1], c="#111827", lw=1.8, marker="o", ms=3, label="True history")
        ax.scatter(t[idx, 0], t[idx, 1], c="#111827", marker="x", s=34, label="True last")
        ax.scatter(p[idx, 0], p[idx, 1], c="#dc2626", s=30, label="Pred")

        e = float(np.linalg.norm(p[idx] - t[idx]))
        ax.set_title(f"sample {idx} | err={e:.2f}m", fontsize=9)
        ax.grid(alpha=0.22)
        ax.set_aspect("equal", adjustable="box")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Window-level check: 5-step true path + predicted last point", fontsize=14)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Pure RSSI absolute indoor localization. "
            "No coord_prev, no motion proxy features, supports cold-start prediction."
        )
    )
    p.add_argument("--train-dir", type=str, default="training_dataset")
    p.add_argument("--test-dir", type=str, default="test_dataset")
    p.add_argument("--output-dir", type=str, default="runs/absolute_rssi_only")

    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=8e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=16)

    p.add_argument("--cls-loss-weight", type=float, default=0.12)
    p.add_argument("--label-smoothing", type=float, default=0.02)

    p.add_argument("--frame-hidden", type=int, default=256)
    p.add_argument("--rnn-hidden", type=int, default=192)
    p.add_argument("--rnn-layers", type=int, default=2)
    p.add_argument("--head-hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.12)

    p.add_argument("--grid-cell-size", type=float, default=16.0)
    p.add_argument("--grid-margin", type=float, default=1.0)

    p.add_argument("--k-candidates", type=str, default="1,3,5,7,11,15")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="auto|cuda|mps|cpu")
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--no-amp", action="store_true")

    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)

    p.add_argument("--plot-samples", type=int, default=24)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device, args.cpu_only)
    use_amp = (not args.no_amp) and (device.type == "cuda")

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

    train_y = train_arrays["y_last"].astype(np.float32)
    coord_mean = train_y.mean(axis=0).astype(np.float32)
    coord_std = train_y.std(axis=0).astype(np.float32)
    coord_std = np.where(coord_std < 1e-6, 1.0, coord_std).astype(np.float32)

    grid_spec = build_grid_spec(train_y, cell_size=float(args.grid_cell_size), margin=float(args.grid_margin))

    train_ds = RSSISeqDataset(train_arrays, coord_mean, coord_std, grid_spec)
    val_ds = RSSISeqDataset(val_arrays, coord_mean, coord_std, grid_spec)
    test_ds = RSSISeqDataset(test_arrays, coord_mean, coord_std, grid_spec)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_cfg = ModelConfig(
        num_aps=int(train_arrays["X"].shape[-1]),
        seq_len=int(train_arrays["X"].shape[1]),
        frame_hidden=int(args.frame_hidden),
        rnn_hidden=int(args.rnn_hidden),
        rnn_layers=int(args.rnn_layers),
        head_hidden=int(args.head_hidden),
        dropout=float(args.dropout),
        num_grid_classes=int(grid_spec.num_classes),
    )
    model = AbsoluteRSSIEncoder(model_cfg).to(device)

    print(f"Using device: {device}", flush=True)
    print(f"AMP: {use_amp}", flush=True)
    print(f"Train/Val/Test samples: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}", flush=True)
    print(
        f"Input: seq_len={model_cfg.seq_len}, num_aps={model_cfg.num_aps}, grid_classes={model_cfg.num_grid_classes}",
        flush=True,
    )
    print(f"Param count: {count_parameters(model)}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    grid_loss_fn = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    no_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler if use_amp else None,
            device=device,
            grid_loss_fn=grid_loss_fn,
            cls_loss_weight=float(args.cls_loss_weight),
            grad_clip=float(args.grad_clip),
            use_amp=use_amp,
        )

        va = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            coord_mean=coord_mean,
            coord_std=coord_std,
            grid_loss_fn=grid_loss_fn,
            cls_loss_weight=float(args.cls_loss_weight),
            use_amp=use_amp,
            return_preds=False,
        )

        val_mean = float(va["regression"]["mean_error_m"])
        scheduler.step(val_mean)

        row = {
            "epoch": float(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(tr["loss_total"]),
            "val_mean_m": val_mean,
            "val_rmse_m": float(va["regression"]["rmse_m"]),
            "val_cls_acc": float(va["grid_classification_accuracy"]),
        }
        history.append(row)

        print(
            f"[epoch {epoch:03d}] "
            f"train={tr['loss_total']:.5f} | "
            f"val_mean={va['regression']['mean_error_m']:.3f}m | "
            f"val_rmse={va['regression']['rmse_m']:.3f}m | "
            f"val_cls={va['grid_classification_accuracy']:.3f}",
            flush=True,
        )

        if val_mean + 1e-6 < best_val:
            best_val = val_mean
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stop at epoch {epoch} (patience={args.patience})", flush=True)
                break

    if best_state is None:
        raise RuntimeError("No best checkpoint captured.")

    ckpt_path = output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": model_cfg.__dict__,
            "coord_mean": coord_mean.tolist(),
            "coord_std": coord_std.tolist(),
            "grid_spec": grid_spec.__dict__,
            "best_epoch": int(best_epoch),
            "best_val_mean_m": float(best_val),
        },
        ckpt_path,
    )

    model.load_state_dict(best_state)
    model.to(device)

    test_out = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        coord_mean=coord_mean,
        coord_std=coord_std,
        grid_loss_fn=grid_loss_fn,
        cls_loss_weight=float(args.cls_loss_weight),
        use_amp=use_amp,
        return_preds=True,
    )

    pred = test_out["predictions"]
    assert isinstance(pred, dict)

    # WKNN baseline (cold-start, RSSI-only)
    ks = [int(x.strip()) for x in args.k_candidates.split(",") if x.strip()]
    wknn_tuned = tune_wknn(train_arrays, val_arrays, ks)
    wk_best = wknn_tuned["best"]

    x_train_last = train_arrays["X"].astype(np.float32)[:, -1, :]
    y_train = train_arrays["y_last"].astype(np.float32)
    x_test_last = test_arrays["X"].astype(np.float32)[:, -1, :]
    y_test = test_arrays["y_last"].astype(np.float32)

    wk_pred = wknn_predict(
        x_train_last,
        y_train,
        x_test_last,
        k=int(wk_best["k"]),
        weighted=bool(wk_best["weighted"]),
    )
    wk_metrics = regression_metrics(wk_pred, y_test)

    save_predictions_csv(pred, output_dir / "test_predictions.csv")
    plot_scatter(pred, output_dir / "test_scatter_true_vs_pred.png")
    plot_error_hist(pred, output_dir / "test_error_hist.png")
    plot_window_paths(pred, output_dir / "test_window_paths.png", max_samples=args.plot_samples, seed=args.seed)

    metrics = {
        "device": str(device),
        "use_amp": bool(use_amp),
        "model_name": "absolute_rssi_only",
        "best_epoch": int(best_epoch),
        "best_val_mean_m": float(best_val),
        "param_count": int(count_parameters(model)),
        "model_config": model_cfg.__dict__,
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "test_samples": int(len(test_ds)),
        "val_history": history,
        "test_metrics": {
            "regression": test_out["regression"],
            "grid_classification_accuracy": test_out["grid_classification_accuracy"],
        },
        "wknn_val_tuning": wknn_tuned,
        "wknn_test_metrics": wk_metrics,
        "artifacts": {
            "checkpoint": str(ckpt_path),
            "predictions_csv": str(output_dir / "test_predictions.csv"),
            "scatter_png": str(output_dir / "test_scatter_true_vs_pred.png"),
            "error_hist_png": str(output_dir / "test_error_hist.png"),
            "window_paths_png": str(output_dir / "test_window_paths.png"),
        },
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== Done ===", flush=True)
    print(f"Best epoch        : {best_epoch}", flush=True)
    print(f"Test mean (model) : {test_out['regression']['mean_error_m']:.3f} m", flush=True)
    print(f"Test rmse (model) : {test_out['regression']['rmse_m']:.3f} m", flush=True)
    print(f"Test cls acc      : {test_out['grid_classification_accuracy']:.3f}", flush=True)
    print(
        f"Test mean (WKNN)  : {wk_metrics['mean_error_m']:.3f} m "
        f"(k={wk_best['k']}, weighted={wk_best['weighted']})",
        flush=True,
    )
    print(f"Output dir        : {output_dir}", flush=True)


if __name__ == "__main__":
    main()
