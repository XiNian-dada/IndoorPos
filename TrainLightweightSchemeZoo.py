#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def build_group_mapping(*group_arrays: np.ndarray) -> Dict[Tuple[int, int], int]:
    unique = sorted(
        {
            (int(b), int(f))
            for groups in group_arrays
            for b, f in groups
        }
    )
    return {g: i for i, g in enumerate(unique)}


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    error = np.linalg.norm(pred - target, axis=1)
    return {
        "mean_error_m": float(error.mean()),
        "median_error_m": float(np.median(error)),
        "p75_error_m": float(np.quantile(error, 0.75)),
        "p90_error_m": float(np.quantile(error, 0.90)),
        "p95_error_m": float(np.quantile(error, 0.95)),
        "max_error_m": float(error.max()),
        "rmse_m": float(math.sqrt(np.mean(error ** 2))),
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
            raise ValueError(f"Unsupported arch '{arch}', expected set_tcn|cnn_tcn|pure_tcn.")
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


class SequenceDataset(Dataset):
    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        motion_mean: np.ndarray,
        motion_std: np.ndarray,
        coord_mean: np.ndarray,
        coord_std: np.ndarray,
        group_to_class: Dict[Tuple[int, int], int],
    ) -> None:
        self.rssi = arrays["X"].astype(np.float32)
        motion = arrays["motion_features"].astype(np.float32)
        self.motion = ((motion - motion_mean[None, None, :]) / motion_std[None, None, :]).astype(np.float32)
        self.coord_raw = arrays["y_last"].astype(np.float32)
        self.coord_norm = ((self.coord_raw - coord_mean[None, :]) / coord_std[None, :]).astype(np.float32)
        groups = arrays["group"].astype(np.int32)
        self.group_id = np.asarray(
            [group_to_class[(int(b), int(f))] for b, f in groups],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return int(self.rssi.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "rssi": torch.from_numpy(self.rssi[idx]),
            "motion": torch.from_numpy(self.motion[idx]),
            "coord_raw": torch.from_numpy(self.coord_raw[idx]),
            "coord_norm": torch.from_numpy(self.coord_norm[idx]),
            "group_id": torch.tensor(self.group_id[idx], dtype=torch.long),
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
        # token: [B, T, K, F]
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
        # token: [B, T, K, F]
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
        # token: [B, T, K, F]
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
        pad = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation)
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


class TemporalHead(nn.Module):
    def __init__(
        self,
        num_aps: int,
        motion_dim: int,
        num_classes: int,
        cfg: CandidateConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.extractor = TopKTokenExtractor(num_aps=num_aps, top_k=cfg.top_k, ap_emb_dim=8)
        token_dim = self.extractor.token_dim

        if cfg.arch == "set_tcn":
            self.frame_encoder: nn.Module = FrameSetEncoder(token_dim=token_dim, hidden_dim=cfg.token_hidden, dropout=cfg.dropout)
            frame_out_dim = self.frame_encoder.out_dim
        elif cfg.arch == "cnn_tcn":
            self.frame_encoder = FrameCNNEncoder(token_dim=token_dim, hidden_dim=cfg.token_hidden, dropout=cfg.dropout)
            frame_out_dim = self.frame_encoder.out_dim
        elif cfg.arch == "pure_tcn":
            self.frame_encoder = FrameFlatEncoder(
                token_dim=token_dim,
                top_k=cfg.top_k,
                hidden_dim=cfg.token_hidden,
                dropout=cfg.dropout,
            )
            frame_out_dim = self.frame_encoder.out_dim
        else:
            raise ValueError(f"Unsupported arch: {cfg.arch}")

        tcn_in = frame_out_dim + int(motion_dim)
        self.input_proj = nn.Linear(tcn_in, cfg.temporal_hidden)
        self.tcn1 = TCNBlock(cfg.temporal_hidden, dilation=1, dropout=cfg.dropout)
        self.tcn2 = TCNBlock(cfg.temporal_hidden, dilation=2, dropout=cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)
        self.trunk = nn.Sequential(
            nn.Linear(cfg.temporal_hidden * 3, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.coord_head = nn.Linear(cfg.head_hidden, 2)
        self.cls_head = nn.Linear(cfg.head_hidden, num_classes)

    def forward(self, rssi: torch.Tensor, motion: torch.Tensor) -> Dict[str, torch.Tensor]:
        token = self.extractor(rssi)
        z = self.frame_encoder(token)
        x = torch.cat([z, motion], dim=-1)
        x = self.input_proj(x)
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.dropout(x)

        last_feat = x[:, -1, :]
        mean_feat = x.mean(dim=1)
        max_feat = x.max(dim=1).values
        fuse = torch.cat([last_feat, mean_feat, max_feat], dim=1)
        h = self.trunk(fuse)
        coord = self.coord_head(h)
        logits = self.cls_head(h)
        return {
            "coord_norm": coord,
            "logits": logits,
        }


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def wrap_loader(loader: DataLoader, enabled: bool, desc: str):
    if enabled and tqdm is not None:
        return tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    return loader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    coord_mean: np.ndarray,
    coord_std: np.ndarray,
    cls_loss_fn: nn.Module,
    cls_loss_weight: float,
    show_progress: bool,
    desc: str,
) -> Dict[str, object]:
    model.eval()
    coord_mean_t = torch.from_numpy(coord_mean.astype(np.float32)).to(device)
    coord_std_t = torch.from_numpy(coord_std.astype(np.float32)).to(device)

    total_loss = 0.0
    total_coord_loss = 0.0
    total_cls_loss = 0.0
    num_batches = 0

    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    pred_group: List[np.ndarray] = []
    true_group: List[np.ndarray] = []

    with torch.no_grad():
        iterator = wrap_loader(loader, show_progress, desc)
        for batch in iterator:
            batch = batch_to_device(batch, device)
            out = model(batch["rssi"], batch["motion"])
            coord_loss = nn.functional.smooth_l1_loss(out["coord_norm"], batch["coord_norm"])
            cls_loss = cls_loss_fn(out["logits"], batch["group_id"])
            loss = coord_loss + cls_loss_weight * cls_loss

            total_loss += float(loss.item())
            total_coord_loss += float(coord_loss.item())
            total_cls_loss += float(cls_loss.item())
            num_batches += 1

            pred_coord = out["coord_norm"] * coord_std_t + coord_mean_t
            pred_cls = out["logits"].argmax(dim=1)

            preds.append(pred_coord.detach().cpu().numpy())
            targets.append(batch["coord_raw"].detach().cpu().numpy())
            pred_group.append(pred_cls.detach().cpu().numpy())
            true_group.append(batch["group_id"].detach().cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    g_pred = np.concatenate(pred_group, axis=0)
    g_true = np.concatenate(true_group, axis=0)

    reg = regression_metrics(y_pred, y_true)
    return {
        "loss_total": total_loss / max(1, num_batches),
        "loss_coord": total_coord_loss / max(1, num_batches),
        "loss_cls": total_cls_loss / max(1, num_batches),
        "classification_accuracy": float((g_pred == g_true).mean()),
        "regression": reg,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cls_loss_fn: nn.Module,
    cls_loss_weight: float,
    grad_clip: float,
    show_progress: bool,
    desc: str,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_coord_loss = 0.0
    total_cls_loss = 0.0
    num_batches = 0

    iterator = wrap_loader(loader, show_progress, desc)
    for batch in iterator:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["rssi"], batch["motion"])
        coord_loss = nn.functional.smooth_l1_loss(out["coord_norm"], batch["coord_norm"])
        cls_loss = cls_loss_fn(out["logits"], batch["group_id"])
        loss = coord_loss + cls_loss_weight * cls_loss
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        total_coord_loss += float(coord_loss.item())
        total_cls_loss += float(cls_loss.item())
        num_batches += 1
        if show_progress and tqdm is not None:
            iterator.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss_total": total_loss / max(1, num_batches),
        "loss_coord": total_coord_loss / max(1, num_batches),
        "loss_cls": total_cls_loss / max(1, num_batches),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train lightweight RSSI sequence schemes: set_tcn / cnn_tcn / pure_tcn.",
    )
    parser.add_argument("--train-dir", type=str, default="training_dataset")
    parser.add_argument("--test-dir", type=str, default="test_dataset")
    parser.add_argument("--output-dir", type=str, default="runs/lightweight_scheme_zoo")
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
    parser.add_argument("--lr", type=float, default=9e-4)
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

    coord_mean = train_arrays["y_last"].mean(axis=0).astype(np.float32)
    coord_std = train_arrays["y_last"].std(axis=0).astype(np.float32)
    coord_std = np.where(coord_std < 1e-6, 1.0, coord_std).astype(np.float32)

    group_to_class = build_group_mapping(train_arrays["group"], val_arrays["group"], test_arrays["group"])
    num_classes = len(group_to_class)
    num_aps = int(train_arrays["X"].shape[-1])
    motion_dim = int(train_arrays["motion_features"].shape[-1])
    seq_len = int(train_arrays["X"].shape[1])

    train_ds = SequenceDataset(
        arrays=train_arrays,
        motion_mean=motion_mean,
        motion_std=motion_std,
        coord_mean=coord_mean,
        coord_std=coord_std,
        group_to_class=group_to_class,
    )
    val_ds = SequenceDataset(
        arrays=val_arrays,
        motion_mean=motion_mean,
        motion_std=motion_std,
        coord_mean=coord_mean,
        coord_std=coord_std,
        group_to_class=group_to_class,
    )
    test_ds = SequenceDataset(
        arrays=test_arrays,
        motion_mean=motion_mean,
        motion_std=motion_std,
        coord_mean=coord_mean,
        coord_std=coord_std,
        group_to_class=group_to_class,
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

    print(f"Using device: {device}", flush=True)
    print(
        f"Train/Val/Test samples: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}",
        flush=True,
    )
    print(
        f"Input: seq_len={seq_len}, num_aps={num_aps}, motion_dim={motion_dim}, classes={num_classes}",
        flush=True,
    )
    if show_progress and tqdm is None:
        print("tqdm not installed, fallback to epoch-level logs only.", flush=True)

    cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    candidates = parse_candidates(args.candidates)

    best_val = float("inf")
    best_result: Optional[Dict[str, object]] = None
    scheme_results: List[Dict[str, object]] = []
    candidate_histories: Dict[str, List[Dict[str, float]]] = {}

    for idx, cfg in enumerate(candidates, start=1):
        print(f"\n=== Candidate {idx}/{len(candidates)}: {cfg.name} ===", flush=True)
        model = TemporalHead(
            num_aps=num_aps,
            motion_dim=motion_dim,
            num_classes=num_classes,
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
                cls_loss_fn=cls_loss_fn,
                cls_loss_weight=args.cls_loss_weight,
                grad_clip=args.grad_clip,
                show_progress=show_progress,
                desc=f"{cfg.name} Train {epoch}/{args.epochs}",
            )
            val_out = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                coord_mean=coord_mean,
                coord_std=coord_std,
                cls_loss_fn=cls_loss_fn,
                cls_loss_weight=args.cls_loss_weight,
                show_progress=show_progress,
                desc=f"{cfg.name} Val {epoch}/{args.epochs}",
            )
            val_reg = val_out["regression"]
            val_mean = float(val_reg["mean_error_m"])
            scheduler.step(val_mean)
            lr_now = float(optimizer.param_groups[0]["lr"])

            history.append(
                {
                    "epoch": float(epoch),
                    "lr": lr_now,
                    "train_loss": float(train_loss["loss_total"]),
                    "val_mean_error_m": val_mean,
                    "val_rmse_m": float(val_reg["rmse_m"]),
                    "val_cls_acc": float(val_out["classification_accuracy"]),
                }
            )
            print(
                f"{cfg.name} Epoch {epoch:03d} | "
                f"train={train_loss['loss_total']:.4f} | "
                f"val_mean={val_mean:.3f} m | "
                f"val_rmse={float(val_reg['rmse_m']):.3f} m | "
                f"val_cls={float(val_out['classification_accuracy']):.3f} | "
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
            coord_mean=coord_mean,
            coord_std=coord_std,
            cls_loss_fn=cls_loss_fn,
            cls_loss_weight=args.cls_loss_weight,
            show_progress=False,
            desc="Val-final",
        )
        test_final = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            coord_mean=coord_mean,
            coord_std=coord_std,
            cls_loss_fn=cls_loss_fn,
            cls_loss_weight=args.cls_loss_weight,
            show_progress=False,
            desc="Test-final",
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
                "num_classes": num_classes,
                "coord_mean": coord_mean.tolist(),
                "coord_std": coord_std.tolist(),
                "motion_mean": motion_mean.tolist(),
                "motion_std": motion_std.tolist(),
                "group_to_class": {f"{k[0]}_{k[1]}": v for k, v in group_to_class.items()},
                "args": vars(args),
            },
            ckpt_path,
        )

        val_reg = val_final["regression"]
        test_reg = test_final["regression"]
        result = {
            "model_name": f"lw_{cfg.name}",
            "arch": cfg.arch,
            "checkpoint_path": str(ckpt_path),
            "param_count": int(param_count),
            "val_metrics": val_reg,
            "val_classification_accuracy": float(val_final["classification_accuracy"]),
            "test_metrics": test_reg,
            "test_classification_accuracy": float(test_final["classification_accuracy"]),
        }
        scheme_results.append(result)
        candidate_histories[cfg.name] = history

        print(
            f"{cfg.name} summary: "
            f"val_mean={float(val_reg['mean_error_m']):.3f} m, "
            f"test_mean={float(test_reg['mean_error_m']):.3f} m",
            flush=True,
        )

        if float(val_reg["mean_error_m"]) < best_val:
            best_val = float(val_reg["mean_error_m"])
            best_result = result

    if best_result is None:
        raise RuntimeError("No candidate finished.")

    metrics_payload = {
        "device": str(device),
        "model_type": "lightweight_scheme_zoo",
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
            "num_classes": num_classes,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "training_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_test = best_result["test_metrics"]
    print("\nFinal Lightweight Scheme Zoo Results", flush=True)
    print(f"  Best Model  : {best_result['model_name']}", flush=True)
    print(f"  Best Arch   : {best_result['arch']}", flush=True)
    print(f"  Mean Error  : {float(best_test['mean_error_m']):.3f} m", flush=True)
    print(f"  Median Error: {float(best_test['median_error_m']):.3f} m", flush=True)
    print(f"  P90 Error   : {float(best_test['p90_error_m']):.3f} m", flush=True)
    print(f"  P95 Error   : {float(best_test['p95_error_m']):.3f} m", flush=True)
    print(f"  RMSE        : {float(best_test['rmse_m']):.3f} m", flush=True)
    print(f"  Cls Acc     : {float(best_result['test_classification_accuracy']):.3f}", flush=True)
    print(f"  Metrics     : {output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
