#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from TrainTinyESP32Model import (
    build_group_mapping,
    build_motion_feature_array,
    build_rssi_robust_features,
    concat_splits,
    limit_arrays,
    load_metadata,
    load_npz,
    select_device,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


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
    step_hidden: int
    rnn_hidden: int
    rnn_layers: int
    head_hidden: int
    dropout: float

    @property
    def name(self) -> str:
        return (
            f"sh{self.step_hidden}_rh{self.rnn_hidden}_"
            f"rl{self.rnn_layers}_hh{self.head_hidden}_do{self.dropout:.2f}"
        )


def parse_candidates(raw: str) -> List[CandidateConfig]:
    candidates: List[CandidateConfig] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 5:
            raise ValueError(
                "Invalid candidate token "
                f"'{token}', expected step_hidden:rnn_hidden:rnn_layers:head_hidden:dropout"
            )
        candidates.append(
            CandidateConfig(
                step_hidden=int(parts[0]),
                rnn_hidden=int(parts[1]),
                rnn_layers=int(parts[2]),
                head_hidden=int(parts[3]),
                dropout=float(parts[4]),
            )
        )
    if not candidates:
        raise ValueError("No candidates parsed.")
    return candidates


def parse_int_list(raw: str) -> List[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Empty integer list.")
    return values


def parse_float_list(raw: str) -> List[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Empty float list.")
    return values


class SequenceDataset(Dataset):
    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        motion_mean: np.ndarray,
        motion_std: np.ndarray,
        coord_mean: np.ndarray,
        coord_std: np.ndarray,
        group_to_class: Dict[Tuple[int, int], int],
        rssi_feature_mode: str,
    ) -> None:
        raw_rssi = arrays["X"].astype(np.float32)
        rssi_z = zscore(raw_rssi, feature_mean[None, None, :], feature_std[None, None, :])
        rssi_centered, rssi_mask, rssi_count_ratio = build_rssi_robust_features(raw_rssi)

        if rssi_feature_mode == "zscore":
            rssi_features = rssi_z
        elif rssi_feature_mode == "robust":
            rssi_features = np.concatenate([rssi_centered, rssi_mask, rssi_count_ratio], axis=2).astype(np.float32)
        elif rssi_feature_mode == "hybrid":
            rssi_features = np.concatenate([rssi_z, rssi_centered, rssi_mask, rssi_count_ratio], axis=2).astype(np.float32)
        else:
            raise ValueError(
                f"Unknown rssi_feature_mode='{rssi_feature_mode}', expected zscore|robust|hybrid."
            )

        motion = zscore(
            build_motion_feature_array(arrays),
            motion_mean[None, None, :],
            motion_std[None, None, :],
        )
        self.inputs = np.concatenate([rssi_features, motion], axis=2).astype(np.float32)

        self.y_raw = arrays["y_last"].astype(np.float32)
        self.y_norm = zscore(self.y_raw, coord_mean[None, :], coord_std[None, :])
        groups = arrays["group"].astype(np.int32)
        self.group_ids = np.asarray(
            [group_to_class[(int(b), int(f))] for b, f in groups],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "inputs": torch.from_numpy(self.inputs[idx]),
            "coord_norm": torch.from_numpy(self.y_norm[idx]),
            "coord_raw": torch.from_numpy(self.y_raw[idx]),
            "group_id": torch.tensor(self.group_ids[idx], dtype=torch.long),
        }


class HighAccuracySequenceNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        step_hidden: int,
        rnn_hidden: int,
        rnn_layers: int,
        head_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, step_hidden)
        self.temporal_rnn = nn.GRU(
            input_size=step_hidden,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        temporal_dim = rnn_hidden * 2
        self.temporal_norm = nn.LayerNorm(temporal_dim)
        self.attn_score = nn.Linear(temporal_dim, 1)
        self.dropout = nn.Dropout(dropout)

        fusion_dim = temporal_dim * 3
        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
        )
        self.embed_head = nn.Linear(head_hidden, head_hidden)
        self.cls_head = nn.Linear(head_hidden, num_classes)
        self.coord_head = nn.Linear(head_hidden, 2)

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.nn.functional.gelu(self.input_proj(inputs))
        x = self.dropout(x)
        x, _ = self.temporal_rnn(x)
        x = self.temporal_norm(x)

        attn_logits = self.attn_score(x).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_feat = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)

        mean_feat = x.mean(dim=1)
        max_feat = x.max(dim=1).values
        fused = torch.cat([attn_feat, mean_feat, max_feat], dim=1)
        hidden = self.trunk(fused)

        embedding = torch.nn.functional.normalize(self.embed_head(hidden), dim=1)
        logits = self.cls_head(hidden)
        coord_norm = self.coord_head(hidden)
        return {
            "coord_norm": coord_norm,
            "logits": logits,
            "embedding": embedding,
        }


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def clone_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    base = unwrap_model(model)
    return {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}


def load_model_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    base = unwrap_model(model)
    base.load_state_dict(state)


def count_parameters(model: nn.Module) -> int:
    base = unwrap_model(model)
    return int(sum(p.numel() for p in base.parameters()))


def wrap_loader(loader: DataLoader, enabled: bool, desc: str):
    if enabled and tqdm is not None:
        return tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    return loader


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def eval_model(
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

    all_pred_coord: List[np.ndarray] = []
    all_true_coord: List[np.ndarray] = []
    all_pred_group: List[np.ndarray] = []
    all_true_group: List[np.ndarray] = []
    all_embed: List[np.ndarray] = []

    with torch.no_grad():
        iterator = wrap_loader(loader, show_progress, desc)
        for batch in iterator:
            batch = batch_to_device(batch, device)
            out = model(batch["inputs"])

            coord_loss = torch.nn.functional.smooth_l1_loss(out["coord_norm"], batch["coord_norm"])
            cls_loss = cls_loss_fn(out["logits"], batch["group_id"])
            loss = coord_loss + cls_loss_weight * cls_loss

            total_loss += float(loss.item())
            total_coord_loss += float(coord_loss.item())
            total_cls_loss += float(cls_loss.item())
            num_batches += 1

            pred_coord = out["coord_norm"] * coord_std_t + coord_mean_t
            pred_group = out["logits"].argmax(dim=1)

            all_pred_coord.append(pred_coord.detach().cpu().numpy())
            all_true_coord.append(batch["coord_raw"].detach().cpu().numpy())
            all_pred_group.append(pred_group.detach().cpu().numpy())
            all_true_group.append(batch["group_id"].detach().cpu().numpy())
            all_embed.append(out["embedding"].detach().cpu().numpy())

    pred_coord = np.concatenate(all_pred_coord, axis=0)
    true_coord = np.concatenate(all_true_coord, axis=0)
    pred_group = np.concatenate(all_pred_group, axis=0)
    true_group = np.concatenate(all_true_group, axis=0)
    embed = np.concatenate(all_embed, axis=0)

    metrics = regression_metrics(pred_coord, true_coord)
    return {
        "loss_total": total_loss / max(1, num_batches),
        "loss_coord": total_coord_loss / max(1, num_batches),
        "loss_cls": total_cls_loss / max(1, num_batches),
        "classification_accuracy": float((pred_group == true_group).mean()),
        "pred_coord": pred_coord,
        "true_coord": true_coord,
        "pred_group": pred_group,
        "true_group": true_group,
        "embedding": embed,
        "regression": metrics,
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
        out = model(batch["inputs"])

        coord_loss = torch.nn.functional.smooth_l1_loss(out["coord_norm"], batch["coord_norm"])
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


def weighted_knn_predict(
    nn_index: NearestNeighbors,
    bank_coords: np.ndarray,
    query_embed: np.ndarray,
    k: int,
) -> np.ndarray:
    dist, idx = nn_index.kneighbors(query_embed, n_neighbors=k, return_distance=True)
    weights = 1.0 / (dist + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)
    coords = bank_coords[idx]
    return (coords * weights[:, :, None]).sum(axis=1).astype(np.float32)


def pick_best_knn_refine(
    val_embed: np.ndarray,
    val_pred_coord: np.ndarray,
    val_true_coord: np.ndarray,
    bank_embed: np.ndarray,
    bank_coords: np.ndarray,
    k_values: Sequence[int],
    alpha_values: Sequence[float],
) -> Dict[str, object]:
    max_k = int(max(k_values))
    nn_index = NearestNeighbors(n_neighbors=max_k, metric="euclidean")
    nn_index.fit(bank_embed)

    best = None
    for k in k_values:
        knn_coord = weighted_knn_predict(
            nn_index=nn_index,
            bank_coords=bank_coords,
            query_embed=val_embed,
            k=int(k),
        )
        for alpha in alpha_values:
            fused = alpha * val_pred_coord + (1.0 - alpha) * knn_coord
            reg = regression_metrics(fused, val_true_coord)
            row = (float(reg["mean_error_m"]), float(reg["p90_error_m"]), int(k), float(alpha), reg)
            if best is None or row < best:
                best = row

    if best is None:
        raise RuntimeError("Failed to pick kNN refinement parameters.")
    return {
        "k": int(best[2]),
        "alpha": float(best[3]),
        "val_metrics": best[4],
        "nn_index": nn_index,
    }


def build_loaders_and_stats(
    train_arrays: Dict[str, np.ndarray],
    val_arrays: Dict[str, np.ndarray],
    test_arrays: Dict[str, np.ndarray],
    batch_size: int,
    num_workers: int,
    rssi_feature_mode: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, np.ndarray], Dict[Tuple[int, int], int], List[str]]:
    feature_mean = train_arrays["X"].mean(axis=(0, 1)).astype(np.float32)
    feature_std = train_arrays["X"].std(axis=(0, 1)).astype(np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)

    train_motion = build_motion_feature_array(train_arrays).astype(np.float32)
    motion_mean = train_motion.mean(axis=(0, 1)).astype(np.float32)
    motion_std = train_motion.std(axis=(0, 1)).astype(np.float32)
    motion_std = np.where(motion_std < 1e-6, 1.0, motion_std).astype(np.float32)

    coord_mean = train_arrays["y_last"].mean(axis=0).astype(np.float32)
    coord_std = train_arrays["y_last"].std(axis=0).astype(np.float32)
    coord_std = np.where(coord_std < 1e-6, 1.0, coord_std).astype(np.float32)

    group_to_class, class_names = build_group_mapping(
        train_arrays["group"],
        val_arrays["group"],
        test_arrays["group"],
    )
    stats = {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "motion_mean": motion_mean,
        "motion_std": motion_std,
        "coord_mean": coord_mean,
        "coord_std": coord_std,
    }

    train_ds = SequenceDataset(
        arrays=train_arrays,
        feature_mean=feature_mean,
        feature_std=feature_std,
        motion_mean=motion_mean,
        motion_std=motion_std,
        coord_mean=coord_mean,
        coord_std=coord_std,
        group_to_class=group_to_class,
        rssi_feature_mode=rssi_feature_mode,
    )
    val_ds = SequenceDataset(
        arrays=val_arrays,
        feature_mean=feature_mean,
        feature_std=feature_std,
        motion_mean=motion_mean,
        motion_std=motion_std,
        coord_mean=coord_mean,
        coord_std=coord_std,
        group_to_class=group_to_class,
        rssi_feature_mode=rssi_feature_mode,
    )
    test_ds = SequenceDataset(
        arrays=test_arrays,
        feature_mean=feature_mean,
        feature_std=feature_std,
        motion_mean=motion_mean,
        motion_std=motion_std,
        coord_mean=coord_mean,
        coord_std=coord_std,
        group_to_class=group_to_class,
        rssi_feature_mode=rssi_feature_mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader, stats, group_to_class, class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GPU-accelerated high-accuracy sequence model with kNN refinement.",
    )
    parser.add_argument("--train-dir", type=str, default="training_dataset")
    parser.add_argument("--test-dir", type=str, default="test_dataset")
    parser.add_argument("--output-dir", type=str, default="runs/high_accuracy_torch")
    parser.add_argument(
        "--candidates",
        type=str,
        default=(
            "192:192:2:256:0.15,"
            "256:256:2:320:0.18,"
            "320:320:2:384:0.20"
        ),
    )
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--cls-loss-weight", type=float, default=0.16)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument(
        "--rssi-feature-mode",
        type=str,
        default="hybrid",
        choices=["zscore", "robust", "hybrid"],
    )
    parser.add_argument("--knn-ks", type=str, default="5,9,13,17,21")
    parser.add_argument("--knn-alphas", type=str, default="0.2,0.35,0.5,0.65,0.8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    show_progress = not args.no_progress
    device = select_device(cpu_only=args.cpu_only)

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists():
        raise FileNotFoundError(f"Train dir not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")

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

    train_loader, val_loader, test_loader, stats, group_to_class, class_names = build_loaders_and_stats(
        train_arrays=train_arrays,
        val_arrays=val_arrays,
        test_arrays=test_arrays,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rssi_feature_mode=args.rssi_feature_mode,
    )
    input_dim = int(train_loader.dataset[0]["inputs"].shape[-1])  # type: ignore[index]
    candidates = parse_candidates(args.candidates)

    print(f"Using device: {device}", flush=True)
    if args.multi_gpu and device.type == "cuda":
        print(f"CUDA visible devices: {torch.cuda.device_count()}", flush=True)
    print(
        f"Train/Val/Test samples: {len(train_loader.dataset)}/"
        f"{len(val_loader.dataset)}/{len(test_loader.dataset)}",
        flush=True,
    )
    print(f"Input dim: {input_dim}, classes: {len(class_names)}", flush=True)
    if show_progress and tqdm is None:
        print("tqdm not installed, using epoch-level logs only.", flush=True)

    cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best_val_mean = float("inf")
    best_payload: Dict[str, object] = {}
    best_state: Dict[str, torch.Tensor] = {}
    candidate_records: List[Dict[str, object]] = []

    for idx, cfg in enumerate(candidates, start=1):
        print(f"\n=== Candidate {idx}/{len(candidates)}: {cfg.name} ===", flush=True)
        model = HighAccuracySequenceNet(
            input_dim=input_dim,
            num_classes=len(class_names),
            step_hidden=cfg.step_hidden,
            rnn_hidden=cfg.rnn_hidden,
            rnn_layers=cfg.rnn_layers,
            head_hidden=cfg.head_hidden,
            dropout=cfg.dropout,
        ).to(device)
        if args.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
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

        best_epoch_val = float("inf")
        best_epoch_state: Dict[str, torch.Tensor] = {}
        early_stop_count = 0
        history: List[Dict[str, float]] = []

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
            val_out = eval_model(
                model=model,
                loader=val_loader,
                device=device,
                coord_mean=stats["coord_mean"],
                coord_std=stats["coord_std"],
                cls_loss_fn=cls_loss_fn,
                cls_loss_weight=args.cls_loss_weight,
                show_progress=show_progress,
                desc=f"{cfg.name} Val {epoch}/{args.epochs}",
            )
            val_reg = val_out["regression"]
            val_mean = float(val_reg["mean_error_m"])
            scheduler.step(val_mean)

            cur_lr = float(optimizer.param_groups[0]["lr"])
            history.append(
                {
                    "epoch": float(epoch),
                    "lr": cur_lr,
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
                f"lr={cur_lr:.2e}",
                flush=True,
            )

            if val_mean < best_epoch_val - args.min_delta:
                best_epoch_val = val_mean
                best_epoch_state = clone_model_state(model)
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count >= args.patience:
                    print(f"{cfg.name} early stop at epoch {epoch}", flush=True)
                    break

        if not best_epoch_state:
            best_epoch_state = clone_model_state(model)
        load_model_state(model, best_epoch_state)

        val_final = eval_model(
            model=model,
            loader=val_loader,
            device=device,
            coord_mean=stats["coord_mean"],
            coord_std=stats["coord_std"],
            cls_loss_fn=cls_loss_fn,
            cls_loss_weight=args.cls_loss_weight,
            show_progress=False,
            desc="Val-final",
        )
        test_final = eval_model(
            model=model,
            loader=test_loader,
            device=device,
            coord_mean=stats["coord_mean"],
            coord_std=stats["coord_std"],
            cls_loss_fn=cls_loss_fn,
            cls_loss_weight=args.cls_loss_weight,
            show_progress=False,
            desc="Test-final",
        )
        val_reg = val_final["regression"]
        test_reg = test_final["regression"]

        candidate_records.append(
            {
                "candidate": cfg.name,
                "param_count": param_count,
                "val_mean_error_m": float(val_reg["mean_error_m"]),
                "val_rmse_m": float(val_reg["rmse_m"]),
                "val_cls_acc": float(val_final["classification_accuracy"]),
                "test_mean_error_m": float(test_reg["mean_error_m"]),
                "test_rmse_m": float(test_reg["rmse_m"]),
                "test_cls_acc": float(test_final["classification_accuracy"]),
            }
        )

        if float(val_reg["mean_error_m"]) < best_val_mean:
            best_val_mean = float(val_reg["mean_error_m"])
            best_state = best_epoch_state
            best_payload = {
                "cfg": cfg,
                "param_count": param_count,
                "history": history,
                "val_final": val_final,
                "test_final": test_final,
            }

    if not best_payload or not best_state:
        raise RuntimeError("Failed to train any candidate.")

    best_cfg: CandidateConfig = best_payload["cfg"]  # type: ignore[assignment]
    model = HighAccuracySequenceNet(
        input_dim=input_dim,
        num_classes=len(class_names),
        step_hidden=best_cfg.step_hidden,
        rnn_hidden=best_cfg.rnn_hidden,
        rnn_layers=best_cfg.rnn_layers,
        head_hidden=best_cfg.head_hidden,
        dropout=best_cfg.dropout,
    ).to(device)
    if args.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    load_model_state(model, best_state)

    val_out = eval_model(
        model=model,
        loader=val_loader,
        device=device,
        coord_mean=stats["coord_mean"],
        coord_std=stats["coord_std"],
        cls_loss_fn=cls_loss_fn,
        cls_loss_weight=args.cls_loss_weight,
        show_progress=False,
        desc="Val-best",
    )
    test_out = eval_model(
        model=model,
        loader=test_loader,
        device=device,
        coord_mean=stats["coord_mean"],
        coord_std=stats["coord_std"],
        cls_loss_fn=cls_loss_fn,
        cls_loss_weight=args.cls_loss_weight,
        show_progress=False,
        desc="Test-best",
    )

    full_train_arrays = concat_splits([train_arrays, val_arrays])
    full_train_ds = SequenceDataset(
        arrays=full_train_arrays,
        feature_mean=stats["feature_mean"],
        feature_std=stats["feature_std"],
        motion_mean=stats["motion_mean"],
        motion_std=stats["motion_std"],
        coord_mean=stats["coord_mean"],
        coord_std=stats["coord_std"],
        group_to_class=group_to_class,
        rssi_feature_mode=args.rssi_feature_mode,
    )
    full_train_loader = DataLoader(
        full_train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    full_train_out = eval_model(
        model=model,
        loader=full_train_loader,
        device=device,
        coord_mean=stats["coord_mean"],
        coord_std=stats["coord_std"],
        cls_loss_fn=cls_loss_fn,
        cls_loss_weight=args.cls_loss_weight,
        show_progress=False,
        desc="Bank",
    )

    k_values = parse_int_list(args.knn_ks)
    alpha_values = parse_float_list(args.knn_alphas)
    refine = pick_best_knn_refine(
        val_embed=val_out["embedding"],  # type: ignore[arg-type]
        val_pred_coord=val_out["pred_coord"],  # type: ignore[arg-type]
        val_true_coord=val_out["true_coord"],  # type: ignore[arg-type]
        bank_embed=full_train_out["embedding"],  # type: ignore[arg-type]
        bank_coords=full_train_out["true_coord"],  # type: ignore[arg-type]
        k_values=k_values,
        alpha_values=alpha_values,
    )
    best_k = int(refine["k"])
    best_alpha = float(refine["alpha"])
    nn_index: NearestNeighbors = refine["nn_index"]  # type: ignore[assignment]

    test_knn = weighted_knn_predict(
        nn_index=nn_index,
        bank_coords=full_train_out["true_coord"],  # type: ignore[arg-type]
        query_embed=test_out["embedding"],  # type: ignore[arg-type]
        k=best_k,
    )
    test_fused = best_alpha * test_out["pred_coord"] + (1.0 - best_alpha) * test_knn  # type: ignore[operator]
    test_refined_metrics = regression_metrics(
        test_fused.astype(np.float32),
        test_out["true_coord"],  # type: ignore[arg-type]
    )

    train_meta = load_metadata(train_dir / "metadata.json")
    test_meta = load_metadata(test_dir / "metadata.json")

    ckpt_payload = {
        "model_state_dict": best_state,
        "stats": {k: v.tolist() for k, v in stats.items()},
        "group_to_class": {f"{k[0]}_{k[1]}": v for k, v in group_to_class.items()},
        "class_names": class_names,
        "best_candidate": {
            "step_hidden": best_cfg.step_hidden,
            "rnn_hidden": best_cfg.rnn_hidden,
            "rnn_layers": best_cfg.rnn_layers,
            "head_hidden": best_cfg.head_hidden,
            "dropout": best_cfg.dropout,
        },
        "input_dim": input_dim,
        "num_classes": len(class_names),
        "args": vars(args),
        "multi_gpu": bool(args.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1),
        "knn_refine": {
            "k": best_k,
            "alpha": best_alpha,
        },
    }
    torch.save(ckpt_payload, output_dir / "best_high_accuracy_torch_model.pt")

    metrics_payload = {
        "device": str(device),
        "model_type": "high_accuracy_torch",
        "best_candidate": {
            "name": best_cfg.name,
            "step_hidden": best_cfg.step_hidden,
            "rnn_hidden": best_cfg.rnn_hidden,
            "rnn_layers": best_cfg.rnn_layers,
            "head_hidden": best_cfg.head_hidden,
            "dropout": best_cfg.dropout,
        },
        "val_best_metrics": val_out["regression"],
        "val_refined_metrics": refine["val_metrics"],
        "test_baseline_metrics": test_out["regression"],
        "test_metrics": test_refined_metrics,
        "test_classification_accuracy": float(test_out["classification_accuracy"]),
        "param_count": int(best_payload["param_count"]),
        "multi_gpu": bool(args.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1),
        "knn_refine": {
            "k": best_k,
            "alpha": best_alpha,
        },
        "candidate_summaries": candidate_records,
        "history_best_candidate": best_payload["history"],
        "num_train_samples": int(len(full_train_loader.dataset)),
        "num_test_samples": int(len(test_loader.dataset)),
        "train_csv": train_meta.get("config", {}).get("input_csv") if train_meta else None,
        "test_csv": test_meta.get("config", {}).get("input_csv") if test_meta else None,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "training_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nFinal High-Accuracy Torch Results", flush=True)
    print(f"  Best Candidate : {best_cfg.name}", flush=True)
    print(f"  Test Mean (raw): {float(test_out['regression']['mean_error_m']):.3f} m", flush=True)
    print(f"  Test Mean (ref): {float(test_refined_metrics['mean_error_m']):.3f} m", flush=True)
    print(f"  Test P90  (ref): {float(test_refined_metrics['p90_error_m']):.3f} m", flush=True)
    print(f"  Test RMSE (ref): {float(test_refined_metrics['rmse_m']):.3f} m", flush=True)
    print(f"  Test Cls Acc    : {float(test_out['classification_accuracy']):.3f}", flush=True)
    print(f"  kNN refine      : k={best_k}, alpha={best_alpha:.2f}", flush=True)
    print(f"  Checkpoint      : {output_dir / 'best_high_accuracy_torch_model.pt'}", flush=True)
    print(f"  Metrics         : {output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
