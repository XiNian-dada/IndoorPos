#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from TrainTinyESP32Model import concat_splits, load_npz, select_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def limit_arrays(arrays: Dict[str, np.ndarray], max_samples: int) -> Dict[str, np.ndarray]:
    if max_samples <= 0:
        return arrays
    first_key = next(iter(arrays.keys()))
    n = arrays[first_key].shape[0]
    if max_samples >= n:
        return arrays
    return {key: value[:max_samples] for key, value in arrays.items()}


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    err = np.linalg.norm(pred - target, axis=1)
    return {
        "mean_error_m": float(err.mean()),
        "median_error_m": float(np.median(err)),
        "p75_error_m": float(np.quantile(err, 0.75)),
        "p90_error_m": float(np.quantile(err, 0.90)),
        "p95_error_m": float(np.quantile(err, 0.95)),
        "max_error_m": float(err.max()),
        "rmse_m": float(math.sqrt(np.mean(err ** 2))),
    }


@dataclass
class CandidateConfig:
    arch: str
    top_k: int
    token_hidden: int
    temporal_hidden: int
    head_hidden: int
    dropout: float

    @property
    def name(self) -> str:
        return (
            f"{self.arch}_k{self.top_k}_th{self.token_hidden}_"
            f"tc{self.temporal_hidden}_hh{self.head_hidden}_do{self.dropout:.2f}"
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


def parse_candidates(raw: str) -> List[CandidateConfig]:
    out: List[CandidateConfig] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 6:
            raise ValueError(
                "Invalid candidate token "
                f"'{token}', expected arch:top_k:token_hidden:temporal_hidden:head_hidden:dropout"
            )
        arch = parts[0].strip().lower()
        if arch not in {"set_tcn", "cnn_tcn", "pure_tcn"}:
            raise ValueError("Unsupported arch, expected set_tcn|cnn_tcn|pure_tcn.")
        out.append(
            CandidateConfig(
                arch=arch,
                top_k=int(parts[1]),
                token_hidden=int(parts[2]),
                temporal_hidden=int(parts[3]),
                head_hidden=int(parts[4]),
                dropout=float(parts[5]),
            )
        )
    if not out:
        raise ValueError("No candidates parsed.")
    return out


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

        motion = arrays["motion_features"].astype(np.float32)
        self.motion = ((motion - motion_mean[None, None, :]) / motion_std[None, None, :]).astype(np.float32)

        if "y" in arrays:
            coord_seq = arrays["y"].astype(np.float32)
        elif "coords" in arrays:
            coord_seq = arrays["coords"].astype(np.float32)
        else:
            raise KeyError("Need y or coords in dataset")
        if coord_seq.ndim != 3 or coord_seq.shape[1] < 2 or coord_seq.shape[2] != 2:
            raise ValueError(f"Unexpected coord sequence shape: {coord_seq.shape}")

        self.coord_prev = coord_seq[:, -2, :].astype(np.float32)
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
            elapsed = arrays["elapsed_time"]
            if elapsed.ndim == 3:
                self.elapsed_last = elapsed[:, -1, 0].astype(np.float32)
            elif elapsed.ndim == 2:
                self.elapsed_last = elapsed[:, -1].astype(np.float32)
            else:
                self.elapsed_last = elapsed.astype(np.float32).reshape(-1)
        elif "time_index" in arrays:
            ti = arrays["time_index"]
            if ti.ndim == 3:
                self.elapsed_last = ti[:, -1, 0].astype(np.float32)
            elif ti.ndim == 2:
                self.elapsed_last = ti[:, -1].astype(np.float32)
            else:
                self.elapsed_last = ti.astype(np.float32).reshape(-1)
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
            "coord_raw": torch.from_numpy(self.coord_raw[idx]),
            "coord_prev": torch.from_numpy(self.coord_prev[idx]),
            "delta_norm": torch.from_numpy(self.delta_norm[idx]),
            "grid_id": torch.tensor(int(self.grid_id[idx]), dtype=torch.long),
            "trajectory_id": torch.tensor(int(self.trajectory_id[idx]), dtype=torch.long),
            "elapsed_last": torch.tensor(float(self.elapsed_last[idx]), dtype=torch.float32),
            "speed_last": torch.tensor(float(self.speed_last[idx]), dtype=torch.float32),
        }


class TopKTokenExtractor(nn.Module):
    def __init__(self, num_aps: int, top_k: int, ap_emb_dim: int = 8) -> None:
        super().__init__()
        self.num_aps = int(num_aps)
        self.top_k = max(1, int(top_k))
        self.ap_emb = nn.Embedding(self.num_aps, ap_emb_dim)
        rank_template = torch.linspace(0.0, 1.0, steps=self.top_k, dtype=torch.float32).view(1, 1, self.top_k, 1)
        self.register_buffer("rank_template", rank_template)

    @property
    def token_dim(self) -> int:
        return int(self.ap_emb.embedding_dim + 4)

    def forward(self, rssi_seq: torch.Tensor) -> torch.Tensor:
        # rssi_seq: [B, T, N_AP]
        bsz, tlen, n_ap = rssi_seq.shape
        k = min(self.top_k, int(n_ap))
        vals, idx = torch.topk(rssi_seq, k=k, dim=-1, largest=True, sorted=True)

        prev = torch.cat([rssi_seq[:, :1, :], rssi_seq[:, :-1, :]], dim=1)
        prev_vals = torch.gather(prev, dim=-1, index=idx)
        delta = vals - prev_vals
        is_new = ((prev_vals <= 1e-6) & (vals > 1e-6)).float()

        rank = self.rank_template[:, :, :k, :].expand(bsz, tlen, k, 1)
        ap_feat = self.ap_emb(idx)

        token = torch.cat(
            [
                ap_feat,
                vals.unsqueeze(-1),
                delta.unsqueeze(-1),
                rank,
                is_new.unsqueeze(-1),
            ],
            dim=-1,
        )
        return token


class FrameSetEncoder(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    @property
    def out_dim(self) -> int:
        return int(self.mlp[0].out_features * 2)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        h = self.mlp(token)
        h_mean = h.mean(dim=2)
        h_max = h.max(dim=2).values
        return torch.cat([h_mean, h_max], dim=-1)


class FrameCNNEncoder(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(token_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    @property
    def out_dim(self) -> int:
        return int(self.conv2.out_channels * 2)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        bsz, tlen, k, feat = token.shape
        x = token.permute(0, 1, 3, 2).reshape(bsz * tlen, feat, k)
        x = self.act(self.conv1(x))
        x = self.dropout(x)
        x = self.act(self.conv2(x))
        x_mean = x.mean(dim=2)
        x_max = x.max(dim=2).values
        z = torch.cat([x_mean, x_max], dim=1)
        return z.reshape(bsz, tlen, -1)


class FrameFlatEncoder(nn.Module):
    def __init__(self, token_dim: int, top_k: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.net = nn.Sequential(
            nn.Linear(token_dim * self.top_k, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    @property
    def out_dim(self) -> int:
        return int(self.net[0].out_features)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        bsz, tlen, k, feat = token.shape
        if k < self.top_k:
            pad = torch.zeros(
                (bsz, tlen, self.top_k - k, feat),
                dtype=token.dtype,
                device=token.device,
            )
            token = torch.cat([token, pad], dim=2)
        x = token.reshape(bsz, tlen, self.top_k * feat)
        return self.net(x)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm(out + residual)
        out = self.act(out)
        out = self.dropout(out)
        return out


class TemporalDeltaHead(nn.Module):
    def __init__(self, num_aps: int, motion_dim: int, num_grid_classes: int, cfg: CandidateConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.extractor = TopKTokenExtractor(num_aps=num_aps, top_k=cfg.top_k, ap_emb_dim=8)
        token_dim = self.extractor.token_dim

        if cfg.arch == "set_tcn":
            self.frame_encoder: nn.Module = FrameSetEncoder(token_dim=token_dim, hidden_dim=cfg.token_hidden, dropout=cfg.dropout)
        elif cfg.arch == "cnn_tcn":
            self.frame_encoder = FrameCNNEncoder(token_dim=token_dim, hidden_dim=cfg.token_hidden, dropout=cfg.dropout)
        elif cfg.arch == "pure_tcn":
            self.frame_encoder = FrameFlatEncoder(token_dim=token_dim, top_k=cfg.top_k, hidden_dim=cfg.token_hidden, dropout=cfg.dropout)
        else:
            raise ValueError(f"Unsupported arch: {cfg.arch}")

        tcn_in = self.frame_encoder.out_dim + int(motion_dim)
        self.input_proj = nn.Linear(tcn_in, cfg.temporal_hidden)
        self.tcn1 = TCNBlock(cfg.temporal_hidden, dilation=1, dropout=cfg.dropout)
        self.tcn2 = TCNBlock(cfg.temporal_hidden, dilation=2, dropout=cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)
        self.trunk = nn.Sequential(
            nn.Linear(cfg.temporal_hidden * 3, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.delta_head = nn.Linear(cfg.head_hidden, 2)
        self.grid_head = nn.Linear(cfg.head_hidden, int(num_grid_classes))

    def forward(self, rssi: torch.Tensor, motion: torch.Tensor) -> Dict[str, torch.Tensor]:
        token = self.extractor(rssi)
        z = self.frame_encoder(token)
        x = torch.cat([z, motion], dim=-1)
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)

        last_feat = x[:, -1, :]
        mean_feat = x.mean(dim=1)
        max_feat = x.max(dim=1).values
        fuse = torch.cat([last_feat, mean_feat, max_feat], dim=1)
        h = self.trunk(fuse)
        return {
            "delta_norm": self.delta_head(h),
            "grid_logits": self.grid_head(h),
        }


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def wrap_loader(loader: DataLoader, enabled: bool, desc: str):
    if enabled and tqdm is not None:
        return tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    return loader


def _trajectory_effective_stats(traj: np.ndarray) -> Tuple[bool, int, int]:
    if traj.size == 0:
        return False, 0, 0
    uniq, counts = np.unique(traj.astype(np.int64), return_counts=True)
    ge2 = int((counts >= 2).sum())
    return ge2 > 0, int(uniq.size), ge2


def apply_speed_cap(
    coords: np.ndarray,
    trajectory_id: np.ndarray,
    elapsed: np.ndarray,
    speed: np.ndarray,
    speed_cap_scale: float,
    speed_cap_min: float,
    speed_cap_global: float,
) -> np.ndarray:
    out = coords.copy()
    n = int(out.shape[0])
    order = np.lexsort((np.arange(n), elapsed, trajectory_id))

    prev_idx_by_tid: Dict[int, int] = {}
    prev_time_by_tid: Dict[int, float] = {}
    for idx in order:
        tid = int(trajectory_id[idx])
        t = float(elapsed[idx])

        if tid not in prev_idx_by_tid:
            prev_idx_by_tid[tid] = int(idx)
            prev_time_by_tid[tid] = t
            continue

        prev_idx = prev_idx_by_tid[tid]
        prev_t = prev_time_by_tid[tid]
        dt = t - prev_t
        if not np.isfinite(dt) or dt <= 1e-6:
            dt = 1.0

        max_step = max(
            float(speed_cap_min),
            float(speed[idx]) * dt * float(speed_cap_scale) + float(speed_cap_min),
        )
        max_step = min(max_step, float(speed_cap_global) * dt)

        step = out[idx] - out[prev_idx]
        dist = float(np.linalg.norm(step))
        if dist > max_step and dist > 1e-8:
            out[idx] = out[prev_idx] + step * (max_step / dist)

        prev_idx_by_tid[tid] = int(idx)
        prev_time_by_tid[tid] = t

    return out


def rollout_positions(
    delta_xy: np.ndarray,
    coord_prev: np.ndarray,
    trajectory_id: np.ndarray,
    elapsed: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(delta_xy, dtype=np.float32)
    n = int(delta_xy.shape[0])
    order = np.lexsort((np.arange(n), elapsed, trajectory_id))

    active_tid: Optional[int] = None
    prev_est = np.zeros((2,), dtype=np.float32)
    for idx in order:
        tid = int(trajectory_id[idx])
        if active_tid is None or tid != active_tid:
            active_tid = tid
            prev_est = coord_prev[idx].astype(np.float32)
        cur = prev_est + delta_xy[idx].astype(np.float32)
        out[idx] = cur
        prev_est = cur
    return out


def apply_kalman(
    coords: np.ndarray,
    trajectory_id: np.ndarray,
    elapsed: np.ndarray,
    process_var: float,
    measurement_var: float,
) -> np.ndarray:
    out = coords.copy()
    n = int(out.shape[0])
    order = np.lexsort((np.arange(n), elapsed, trajectory_id))

    x = np.zeros((4,), dtype=np.float64)
    p = np.eye(4, dtype=np.float64)
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    r = np.eye(2, dtype=np.float64) * float(measurement_var)

    current_tid: Optional[int] = None
    prev_t = 0.0

    for idx in order:
        tid = int(trajectory_id[idx])
        t = float(elapsed[idx])
        z = out[idx].astype(np.float64)

        if current_tid is None or tid != current_tid:
            current_tid = tid
            prev_t = t
            x[:] = [z[0], z[1], 0.0, 0.0]
            p = np.eye(4, dtype=np.float64) * float(measurement_var * 2.0)
            out[idx] = x[:2].astype(np.float32)
            continue

        dt = t - prev_t
        if not np.isfinite(dt) or dt <= 1e-6:
            dt = 1.0
        prev_t = t

        f = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        q = np.diag(
            [
                process_var * dt * dt,
                process_var * dt * dt,
                process_var,
                process_var,
            ]
        )

        x = f @ x
        p = f @ p @ f.T + q

        y = z - (h @ x)
        s = h @ p @ h.T + r
        k = p @ h.T @ np.linalg.inv(s)
        x = x + k @ y
        p = (np.eye(4, dtype=np.float64) - k @ h) @ p

        out[idx] = x[:2].astype(np.float32)

    return out


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    delta_mean: np.ndarray,
    delta_std: np.ndarray,
    grid_loss_fn: nn.Module,
    cls_loss_weight: float,
    show_progress: bool,
    desc: str,
    post_cfg: PostProcessConfig,
) -> Dict[str, object]:
    model.eval()
    delta_mean_t = torch.from_numpy(delta_mean.astype(np.float32)).to(device)
    delta_std_t = torch.from_numpy(delta_std.astype(np.float32)).to(device)

    total_loss = 0.0
    total_delta_loss = 0.0
    total_grid_loss = 0.0
    num_batches = 0

    pred_coords_raw: List[np.ndarray] = []
    pred_delta_raw: List[np.ndarray] = []
    target_coords: List[np.ndarray] = []
    coord_prev_raw: List[np.ndarray] = []
    pred_grid: List[np.ndarray] = []
    true_grid: List[np.ndarray] = []
    traj_id: List[np.ndarray] = []
    elapsed: List[np.ndarray] = []
    speed: List[np.ndarray] = []

    with torch.no_grad():
        iterator = wrap_loader(loader, show_progress, desc)
        for batch in iterator:
            batch = batch_to_device(batch, device)
            out = model(batch["rssi"], batch["motion"])
            delta_loss = nn.functional.smooth_l1_loss(out["delta_norm"], batch["delta_norm"])
            grid_loss = grid_loss_fn(out["grid_logits"], batch["grid_id"])
            loss = delta_loss + cls_loss_weight * grid_loss

            total_loss += float(loss.item())
            total_delta_loss += float(delta_loss.item())
            total_grid_loss += float(grid_loss.item())
            num_batches += 1

            delta_raw = out["delta_norm"] * delta_std_t + delta_mean_t
            coord_pred = batch["coord_prev"] + delta_raw

            pred_coords_raw.append(coord_pred.detach().cpu().numpy())
            pred_delta_raw.append(delta_raw.detach().cpu().numpy())
            target_coords.append(batch["coord_raw"].detach().cpu().numpy())
            coord_prev_raw.append(batch["coord_prev"].detach().cpu().numpy())
            pred_grid.append(out["grid_logits"].argmax(dim=1).detach().cpu().numpy())
            true_grid.append(batch["grid_id"].detach().cpu().numpy())
            traj_id.append(batch["trajectory_id"].detach().cpu().numpy())
            elapsed.append(batch["elapsed_last"].detach().cpu().numpy())
            speed.append(batch["speed_last"].detach().cpu().numpy())

    coord_pred_teacher = np.concatenate(pred_coords_raw, axis=0)
    delta_pred_raw = np.concatenate(pred_delta_raw, axis=0)
    coord_prev = np.concatenate(coord_prev_raw, axis=0)
    coord_true = np.concatenate(target_coords, axis=0)
    g_pred = np.concatenate(pred_grid, axis=0)
    g_true = np.concatenate(true_grid, axis=0)
    traj = np.concatenate(traj_id, axis=0)
    t_last = np.concatenate(elapsed, axis=0).astype(np.float32)
    s_last = np.concatenate(speed, axis=0).astype(np.float32)

    coord_pred_raw = rollout_positions(
        delta_xy=delta_pred_raw,
        coord_prev=coord_prev,
        trajectory_id=traj,
        elapsed=t_last,
    )
    reg_teacher = regression_metrics(coord_pred_teacher, coord_true)
    reg_raw = regression_metrics(coord_pred_raw, coord_true)
    traj_effective, num_traj, num_traj_ge2 = _trajectory_effective_stats(traj)

    if post_cfg.enabled and traj_effective:
        speed_capped = apply_speed_cap(
            coords=coord_pred_raw,
            trajectory_id=traj,
            elapsed=t_last,
            speed=s_last,
            speed_cap_scale=post_cfg.speed_cap_scale,
            speed_cap_min=post_cfg.speed_cap_min,
            speed_cap_global=post_cfg.speed_cap_global,
        )
        coord_post = apply_kalman(
            coords=speed_capped,
            trajectory_id=traj,
            elapsed=t_last,
            process_var=post_cfg.kalman_process_var,
            measurement_var=post_cfg.kalman_measurement_var,
        )
    else:
        coord_post = coord_pred_raw

    reg_post = regression_metrics(coord_post, coord_true)

    return {
        "loss_total": total_loss / max(1, num_batches),
        "loss_delta": total_delta_loss / max(1, num_batches),
        "loss_grid": total_grid_loss / max(1, num_batches),
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grid_loss_fn: nn.Module,
    cls_loss_weight: float,
    grad_clip: float,
    show_progress: bool,
    desc: str,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_delta_loss = 0.0
    total_grid_loss = 0.0
    num_batches = 0

    iterator = wrap_loader(loader, show_progress, desc)
    for batch in iterator:
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
        num_batches += 1
        if show_progress and tqdm is not None:
            iterator.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss_total": total_loss / max(1, num_batches),
        "loss_delta": total_delta_loss / max(1, num_batches),
        "loss_grid": total_grid_loss / max(1, num_batches),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Article-style RSSI model: Top-K AP set/CNN encoder + TCN, "
            "predict delta position + grid classification, with speed-cap + Kalman postprocess."
        ),
    )
    parser.add_argument("--train-dir", type=str, default="training_dataset")
    parser.add_argument("--test-dir", type=str, default="test_dataset")
    parser.add_argument("--output-dir", type=str, default="runs/article_trajectory_model")
    parser.add_argument(
        "--candidates",
        type=str,
        default=(
            "set_tcn:12:24:48:64:0.10,"
            "cnn_tcn:12:24:48:64:0.10,"
            "pure_tcn:12:24:48:64:0.10"
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--cls-loss-weight", type=float, default=0.20)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)

    parser.add_argument("--grid-cell-size", type=float, default=20.0)
    parser.add_argument("--grid-margin", type=float, default=1.0)
    parser.add_argument("--selection-metric", choices=["raw", "post"], default="post")

    parser.add_argument("--disable-postprocess", action="store_true")
    parser.add_argument("--speed-cap-scale", type=float, default=1.25)
    parser.add_argument("--speed-cap-min", type=float, default=0.5)
    parser.add_argument("--speed-cap-quantile", type=float, default=0.99)
    parser.add_argument("--speed-cap-multiplier", type=float, default=1.2)
    parser.add_argument("--kalman-process-var", type=float, default=4.0)
    parser.add_argument("--kalman-measurement-var", type=float, default=9.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(cpu_only=args.cpu_only)
    show_progress = not args.no_progress

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

    grid_spec = build_grid_spec(
        coords=train_last,
        cell_size=float(args.grid_cell_size),
        margin=float(args.grid_margin),
    )

    step_mag = np.linalg.norm(train_delta, axis=1)
    speed_cap_global = float(
        np.quantile(step_mag, min(max(float(args.speed_cap_quantile), 0.5), 0.9999))
        * float(args.speed_cap_multiplier)
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

    train_ds = SequenceDataset(
        arrays=train_arrays,
        motion_mean=motion_mean,
        motion_std=motion_std,
        delta_mean=delta_mean,
        delta_std=delta_std,
        grid_spec=grid_spec,
    )
    val_ds = SequenceDataset(
        arrays=val_arrays,
        motion_mean=motion_mean,
        motion_std=motion_std,
        delta_mean=delta_mean,
        delta_std=delta_std,
        grid_spec=grid_spec,
    )
    test_ds = SequenceDataset(
        arrays=test_arrays,
        motion_mean=motion_mean,
        motion_std=motion_std,
        delta_mean=delta_mean,
        delta_std=delta_std,
        grid_spec=grid_spec,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    num_aps = int(train_arrays["X"].shape[-1])
    motion_dim = int(train_arrays["motion_features"].shape[-1])
    seq_len = int(train_arrays["X"].shape[1])
    num_grid_classes = int(grid_spec.num_classes)

    print(f"Using device: {device}", flush=True)
    print(f"Train/Val/Test samples: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}", flush=True)
    print(
        f"Input: seq_len={seq_len}, num_aps={num_aps}, motion_dim={motion_dim}, grid_classes={num_grid_classes}",
        flush=True,
    )
    print(
        "Postprocess: "
        f"enabled={post_cfg.enabled}, speed_cap_global={post_cfg.speed_cap_global:.3f}, "
        f"kalman_q={post_cfg.kalman_process_var:.3f}, kalman_r={post_cfg.kalman_measurement_var:.3f}",
        flush=True,
    )

    if show_progress and tqdm is None:
        print("tqdm not installed, fallback to epoch-level logs only.", flush=True)

    grid_loss_fn = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    candidates = parse_candidates(args.candidates)

    best_val = float("inf")
    best_result: Optional[Dict[str, object]] = None
    scheme_results: List[Dict[str, object]] = []
    candidate_histories: Dict[str, List[Dict[str, float]]] = {}

    for idx, cfg in enumerate(candidates, start=1):
        print(f"\n=== Candidate {idx}/{len(candidates)}: {cfg.name} ===", flush=True)
        model = TemporalDeltaHead(
            num_aps=num_aps,
            motion_dim=motion_dim,
            num_grid_classes=num_grid_classes,
            cfg=cfg,
        ).to(device)
        param_count = count_parameters(model)
        print(f"Param count: {param_count}", flush=True)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
        )

        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_candidate_val = float("inf")
        history: List[Dict[str, float]] = []
        no_improve = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                grid_loss_fn=grid_loss_fn,
                cls_loss_weight=args.cls_loss_weight,
                grad_clip=args.grad_clip,
                show_progress=show_progress,
                desc=f"{cfg.name} Train {epoch}/{args.epochs}",
            )
            val_out = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                delta_mean=delta_mean,
                delta_std=delta_std,
                grid_loss_fn=grid_loss_fn,
                cls_loss_weight=args.cls_loss_weight,
                show_progress=show_progress,
                desc=f"{cfg.name} Val {epoch}/{args.epochs}",
                post_cfg=post_cfg,
            )

            val_reg_raw = val_out["regression_raw"]
            val_reg_post = val_out["regression_post"]
            val_mean = float(
                val_reg_post["mean_error_m"] if args.selection_metric == "post" else val_reg_raw["mean_error_m"]
            )
            scheduler.step(val_mean)
            lr_now = float(optimizer.param_groups[0]["lr"])

            history.append(
                {
                    "epoch": float(epoch),
                    "lr": lr_now,
                    "train_loss": float(train_loss["loss_total"]),
                    "val_mean_raw_m": float(val_reg_raw["mean_error_m"]),
                    "val_mean_post_m": float(val_reg_post["mean_error_m"]),
                    "val_rmse_post_m": float(val_reg_post["rmse_m"]),
                    "val_grid_acc": float(val_out["grid_classification_accuracy"]),
                }
            )
            print(
                f"{cfg.name} Epoch {epoch:03d} | "
                f"train={train_loss['loss_total']:.4f} | "
                f"val_raw={float(val_reg_raw['mean_error_m']):.3f} m | "
                f"val_post={float(val_reg_post['mean_error_m']):.3f} m | "
                f"val_grid={float(val_out['grid_classification_accuracy']):.3f} | "
                f"lr={lr_now:.2e}",
                flush=True,
            )

            if val_mean < best_candidate_val - args.min_delta:
                best_candidate_val = val_mean
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"{cfg.name} early stop at epoch {epoch}", flush=True)
                    break

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)

        val_final = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            delta_mean=delta_mean,
            delta_std=delta_std,
            grid_loss_fn=grid_loss_fn,
            cls_loss_weight=args.cls_loss_weight,
            show_progress=False,
            desc="Val-final",
            post_cfg=post_cfg,
        )
        test_final = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            delta_mean=delta_mean,
            delta_std=delta_std,
            grid_loss_fn=grid_loss_fn,
            cls_loss_weight=args.cls_loss_weight,
            show_progress=False,
            desc="Test-final",
            post_cfg=post_cfg,
        )

        ckpt_path = ckpt_dir / f"{cfg.name}.pt"
        torch.save(
            {
                "model_state_dict": best_state,
                "candidate": {
                    "arch": cfg.arch,
                    "top_k": cfg.top_k,
                    "token_hidden": cfg.token_hidden,
                    "temporal_hidden": cfg.temporal_hidden,
                    "head_hidden": cfg.head_hidden,
                    "dropout": cfg.dropout,
                },
                "num_aps": num_aps,
                "motion_dim": motion_dim,
                "num_grid_classes": num_grid_classes,
                "grid_spec": {
                    "min_x": grid_spec.min_x,
                    "min_y": grid_spec.min_y,
                    "cell_size": grid_spec.cell_size,
                    "nx": grid_spec.nx,
                    "ny": grid_spec.ny,
                },
                "delta_mean": delta_mean.tolist(),
                "delta_std": delta_std.tolist(),
                "motion_mean": motion_mean.tolist(),
                "motion_std": motion_std.tolist(),
                "postprocess": {
                    "enabled": post_cfg.enabled,
                    "speed_cap_scale": post_cfg.speed_cap_scale,
                    "speed_cap_min": post_cfg.speed_cap_min,
                    "speed_cap_global": post_cfg.speed_cap_global,
                    "kalman_process_var": post_cfg.kalman_process_var,
                    "kalman_measurement_var": post_cfg.kalman_measurement_var,
                },
                "args": vars(args),
            },
            ckpt_path,
        )

        result = {
            "model_name": f"article_{cfg.name}",
            "arch": cfg.arch,
            "checkpoint_path": str(ckpt_path),
            "param_count": int(param_count),
            "val_metrics_raw": val_final["regression_raw"],
            "val_metrics_post": val_final["regression_post"],
            "val_grid_accuracy": float(val_final["grid_classification_accuracy"]),
            "test_metrics_raw": test_final["regression_raw"],
            "test_metrics_post": test_final["regression_post"],
            "test_grid_accuracy": float(test_final["grid_classification_accuracy"]),
            "postprocess": test_final["postprocess"],
            "selection_metric": args.selection_metric,
        }
        scheme_results.append(result)
        candidate_histories[cfg.name] = history

        print(
            f"{cfg.name} summary: "
            f"val_post={float(result['val_metrics_post']['mean_error_m']):.3f} m, "
            f"test_post={float(result['test_metrics_post']['mean_error_m']):.3f} m",
            flush=True,
        )

        selected_val = float(
            result["val_metrics_post"]["mean_error_m"] if args.selection_metric == "post" else result["val_metrics_raw"]["mean_error_m"]
        )
        if selected_val < best_val:
            best_val = selected_val
            best_result = result

    if best_result is None:
        raise RuntimeError("No candidate finished.")

    metrics_payload = {
        "device": str(device),
        "model_type": "article_trajectory_model",
        "best_candidate": best_result,
        "scheme_results": scheme_results,
        "history_by_candidate": candidate_histories,
        "num_train_samples": int(len(train_ds)),
        "num_val_samples": int(len(val_ds)),
        "num_test_samples": int(len(test_ds)),
        "input_info": {
            "seq_len": seq_len,
            "num_aps": num_aps,
            "motion_dim": motion_dim,
            "num_grid_classes": num_grid_classes,
            "grid_spec": {
                "min_x": grid_spec.min_x,
                "min_y": grid_spec.min_y,
                "cell_size": grid_spec.cell_size,
                "nx": grid_spec.nx,
                "ny": grid_spec.ny,
            },
        },
        "postprocess": {
            "enabled": post_cfg.enabled,
            "speed_cap_scale": post_cfg.speed_cap_scale,
            "speed_cap_min": post_cfg.speed_cap_min,
            "speed_cap_global": post_cfg.speed_cap_global,
            "kalman_process_var": post_cfg.kalman_process_var,
            "kalman_measurement_var": post_cfg.kalman_measurement_var,
        },
        "delta_stats": {
            "mean": delta_mean.tolist(),
            "std": delta_std.tolist(),
        },
        "selection_metric": args.selection_metric,
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "training_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_test = best_result["test_metrics_post"]
    print("\nFinal Article-Style Results", flush=True)
    print(f"  Best Model  : {best_result['model_name']}", flush=True)
    print(f"  Best Arch   : {best_result['arch']}", flush=True)
    print(f"  Mean Error  : {float(best_test['mean_error_m']):.3f} m", flush=True)
    print(f"  Median Error: {float(best_test['median_error_m']):.3f} m", flush=True)
    print(f"  P90 Error   : {float(best_test['p90_error_m']):.3f} m", flush=True)
    print(f"  P95 Error   : {float(best_test['p95_error_m']):.3f} m", flush=True)
    print(f"  RMSE        : {float(best_test['rmse_m']):.3f} m", flush=True)
    print(f"  Grid Acc    : {float(best_result['test_grid_accuracy']):.3f}", flush=True)
    print(f"  Metrics     : {output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
