#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone implementation of the best article model:
article_pure_tcn_k20_th64_tc128_hh192_do0.08_post

This file intentionally focuses on model architecture + inference utilities
for easier study and reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ArticlePureTCNConfig:
    # Input dimensions
    num_aps: int = 128
    motion_dim: int = 7
    num_grid_classes: int = 204

    # Architecture (matches best candidate)
    top_k: int = 20
    ap_emb_dim: int = 8
    token_hidden: int = 64
    temporal_hidden: int = 128
    head_hidden: int = 192
    dropout: float = 0.08


class TopKTokenExtractor(nn.Module):
    """Build per-frame Top-K AP tokens.

    Input:
      rssi_seq: [B, T, N_AP]

    Output:
      token: [B, T, K, D_token]

    Token features:
      [AP_embedding, RSSI(t), delta_RSSI, rank_norm, is_new]
    """

    def __init__(self, num_aps: int, top_k: int, ap_emb_dim: int = 8) -> None:
        super().__init__()
        self.num_aps = int(num_aps)
        self.top_k = max(1, int(top_k))
        self.ap_emb = nn.Embedding(self.num_aps, int(ap_emb_dim))
        rank_template = torch.linspace(0.0, 1.0, steps=self.top_k, dtype=torch.float32).view(1, 1, self.top_k, 1)
        self.register_buffer("rank_template", rank_template)

    @property
    def token_dim(self) -> int:
        return int(self.ap_emb.embedding_dim + 4)

    def forward(self, rssi_seq: torch.Tensor) -> torch.Tensor:
        # rssi_seq: [B, T, N_AP]
        bsz, tlen, n_ap = rssi_seq.shape
        k = min(self.top_k, int(n_ap))

        # Top-K by strongest RSSI in current frame
        vals, idx = torch.topk(rssi_seq, k=k, dim=-1, largest=True, sorted=True)

        # Previous-frame RSSI for the same AP ids
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


class FrameFlatEncoder(nn.Module):
    """pure_tcn frame encoder.

    Flatten K tokens per frame and apply a tiny MLP.

    Input:
      token: [B, T, K, token_dim]

    Output:
      z: [B, T, token_hidden]
    """

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

        # If the real AP count is smaller than configured top_k, pad with zeros.
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
    """Residual TCN block with kernel_size=3 and configurable dilation."""

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


class ArticlePureTCNModel(nn.Module):
    """Best-performing article model core.

    Architecture:
      Top-K token extractor
      -> FrameFlatEncoder (pure_tcn)
      -> concat motion + input projection
      -> TCN(d=1) + TCN(d=2)
      -> temporal pooling (last/mean/max)
      -> trunk
      -> dual heads:
           1) delta position regression (dx, dy)
           2) grid classification logits
    """

    def __init__(self, cfg: ArticlePureTCNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.extractor = TopKTokenExtractor(
            num_aps=cfg.num_aps,
            top_k=cfg.top_k,
            ap_emb_dim=cfg.ap_emb_dim,
        )
        token_dim = self.extractor.token_dim

        self.frame_encoder = FrameFlatEncoder(
            token_dim=token_dim,
            top_k=cfg.top_k,
            hidden_dim=cfg.token_hidden,
            dropout=cfg.dropout,
        )

        tcn_in = self.frame_encoder.out_dim + int(cfg.motion_dim)
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
        self.grid_head = nn.Linear(cfg.head_hidden, int(cfg.num_grid_classes))

    def forward(self, rssi: torch.Tensor, motion: torch.Tensor) -> Dict[str, torch.Tensor]:
        # rssi: [B, T, N_AP]
        # motion: [B, T, motion_dim]
        token = self.extractor(rssi)
        z = self.frame_encoder(token)

        x = torch.cat([z, motion], dim=-1)
        x = self.input_proj(x)

        # Conv1d expects [B, C, T]
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


def denorm_delta(delta_norm: torch.Tensor, delta_mean: torch.Tensor, delta_std: torch.Tensor) -> torch.Tensor:
    """Convert normalized delta output back to metric-space delta (dx, dy)."""
    return delta_norm * delta_std + delta_mean


def rollout_positions(
    delta_xy: np.ndarray,
    coord_prev: np.ndarray,
    trajectory_id: np.ndarray,
    elapsed: np.ndarray,
) -> np.ndarray:
    """Rollout absolute coordinates from predicted deltas."""
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


def apply_speed_cap(
    coords: np.ndarray,
    trajectory_id: np.ndarray,
    elapsed: np.ndarray,
    speed: np.ndarray,
    speed_cap_scale: float = 2.0,
    speed_cap_min: float = 0.4,
    speed_cap_global: float = 3.5,
) -> np.ndarray:
    """Optional post-processing: constrain per-step displacement by speed prior."""
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


def apply_kalman(
    coords: np.ndarray,
    trajectory_id: np.ndarray,
    elapsed: np.ndarray,
    process_var: float = 0.06,
    measurement_var: float = 0.35,
) -> np.ndarray:
    """Optional post-processing: constant-velocity Kalman smoothing."""
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
        q = np.diag([process_var * dt * dt, process_var * dt * dt, process_var, process_var])

        x = f @ x
        p = f @ p @ f.T + q

        y = z - (h @ x)
        s = h @ p @ h.T + r
        k = p @ h.T @ np.linalg.inv(s)

        x = x + k @ y
        p = (np.eye(4, dtype=np.float64) - k @ h) @ p
        out[idx] = x[:2].astype(np.float32)

    return out


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _demo() -> None:
    cfg = ArticlePureTCNConfig(
        num_aps=128,
        motion_dim=7,
        num_grid_classes=204,
        top_k=20,
        token_hidden=64,
        temporal_hidden=128,
        head_hidden=192,
        dropout=0.08,
    )
    model = ArticlePureTCNModel(cfg)

    bsz, tlen = 4, 10
    rssi = torch.randn(bsz, tlen, cfg.num_aps)
    motion = torch.randn(bsz, tlen, cfg.motion_dim)

    out = model(rssi, motion)
    print("Model:", model.__class__.__name__)
    print("Params:", count_parameters(model))
    print("delta_norm shape:", tuple(out["delta_norm"].shape))
    print("grid_logits shape:", tuple(out["grid_logits"].shape))


if __name__ == "__main__":
    _demo()
