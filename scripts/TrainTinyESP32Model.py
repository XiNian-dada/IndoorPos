#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(cpu_only: bool) -> torch.device:
    if cpu_only:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_metadata(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def concat_splits(split_dicts: Iterable[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    split_dicts = list(split_dicts)
    keys = split_dicts[0].keys()
    return {
        key: np.concatenate([split[key] for split in split_dicts], axis=0)
        for key in keys
    }


def limit_arrays(arrays: Dict[str, np.ndarray], max_samples: int) -> Dict[str, np.ndarray]:
    if max_samples <= 0:
        return arrays
    first_key = next(iter(arrays.keys()))
    n = arrays[first_key].shape[0]
    if max_samples >= n:
        return arrays
    return {key: value[:max_samples] for key, value in arrays.items()}


def zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def build_motion_feature_array(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    if "motion_features" in arrays:
        return arrays["motion_features"].astype(np.float32)

    heading = arrays["heading"].astype(np.float32)
    heading_sin = np.sin(heading).astype(np.float32)
    heading_cos = np.cos(heading).astype(np.float32)
    motion_valid = arrays["motion_valid"].astype(np.float32)

    return np.concatenate(
        [
            arrays["velocity"].astype(np.float32),
            arrays["speed"].astype(np.float32),
            heading_sin,
            heading_cos,
            arrays["delta_t"].astype(np.float32),
            motion_valid,
        ],
        axis=2,
    ).astype(np.float32)


def build_rssi_robust_features(
    rssi_01: np.ndarray,
    eps: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # rssi_01 is expected in [0, 1], where 0 usually means no signal.
    visible_mask = (rssi_01 > 1e-6).astype(np.float32)
    visible_count = visible_mask.sum(axis=2, keepdims=True)
    safe_count = np.where(visible_count < 1.0, 1.0, visible_count)

    visible_mean = (rssi_01 * visible_mask).sum(axis=2, keepdims=True) / safe_count
    visible_var = (((rssi_01 - visible_mean) * visible_mask) ** 2).sum(axis=2, keepdims=True) / safe_count
    visible_std = np.sqrt(visible_var + eps).astype(np.float32)

    centered = ((rssi_01 - visible_mean) / visible_std) * visible_mask
    count_ratio = (safe_count / float(rssi_01.shape[2])).astype(np.float32)
    return centered.astype(np.float32), visible_mask.astype(np.float32), count_ratio


def build_group_mapping(*group_arrays: np.ndarray) -> Tuple[Dict[Tuple[int, int], int], List[str]]:
    unique_groups = sorted(
        {
            (int(building), int(floor))
            for groups in group_arrays
            for building, floor in groups
        }
    )
    group_to_class = {group: idx for idx, group in enumerate(unique_groups)}
    class_names = [f"B{building}_F{floor}" for building, floor in unique_groups]
    return group_to_class, class_names


def build_anchor_bank(
    train_groups: np.ndarray,
    train_y_last: np.ndarray,
    group_to_class: Dict[Tuple[int, int], int],
    coord_mean: np.ndarray,
    coord_std: np.ndarray,
    anchor_divisor: int,
    min_anchors_per_group: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor_centers_raw_list: List[np.ndarray] = []
    anchor_group_ids_list: List[int] = []

    for group_key, class_id in sorted(group_to_class.items(), key=lambda item: item[1]):
        mask = (
            (train_groups[:, 0] == group_key[0]) &
            (train_groups[:, 1] == group_key[1])
        )
        coords = train_y_last[mask].astype(np.float32)
        if len(coords) == 0:
            continue

        unique_coords = np.unique(np.round(coords, 3), axis=0)
        target_anchors = max(
            int(min_anchors_per_group),
            len(unique_coords) // max(1, int(anchor_divisor)),
        )
        target_anchors = min(max(1, target_anchors), len(unique_coords))

        if target_anchors >= len(unique_coords):
            centers = unique_coords.astype(np.float32)
        else:
            kmeans = KMeans(
                n_clusters=target_anchors,
                random_state=42,
                n_init=10,
            )
            kmeans.fit(coords)
            centers = kmeans.cluster_centers_.astype(np.float32)

        anchor_centers_raw_list.append(centers)
        anchor_group_ids_list.extend([int(class_id)] * len(centers))

    anchor_centers_raw = np.concatenate(anchor_centers_raw_list, axis=0).astype(np.float32)
    anchor_group_ids = np.asarray(anchor_group_ids_list, dtype=np.int64)
    anchor_centers_norm = zscore(
        anchor_centers_raw,
        coord_mean[None, :],
        coord_std[None, :],
    ).astype(np.float32)
    return anchor_centers_raw, anchor_centers_norm, anchor_group_ids


def assign_anchor_ids(
    coords_raw: np.ndarray,
    group_ids: np.ndarray,
    anchor_centers_raw: np.ndarray,
    anchor_group_ids: np.ndarray,
) -> np.ndarray:
    anchor_ids = np.full(len(coords_raw), -1, dtype=np.int64)
    unique_group_ids = sorted(set(int(x) for x in group_ids.tolist()))

    for group_id in unique_group_ids:
        sample_mask = group_ids == group_id
        if sample_mask.sum() == 0:
            continue

        group_anchor_indices = np.where(anchor_group_ids == group_id)[0]
        if len(group_anchor_indices) == 0:
            neighbors = NearestNeighbors(n_neighbors=1, metric="euclidean")
            neighbors.fit(anchor_centers_raw)
            _, nearest = neighbors.kneighbors(coords_raw[sample_mask])
            anchor_ids[sample_mask] = nearest[:, 0]
            continue

        neighbors = NearestNeighbors(n_neighbors=1, metric="euclidean")
        neighbors.fit(anchor_centers_raw[group_anchor_indices])
        _, local_idx = neighbors.kneighbors(coords_raw[sample_mask])
        anchor_ids[sample_mask] = group_anchor_indices[local_idx[:, 0]]

    if np.any(anchor_ids < 0):
        raise ValueError("Failed to assign anchor ids for some samples.")
    return anchor_ids


class TinySequenceDataset(Dataset):
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
        anchor_centers_raw: np.ndarray,
        anchor_centers_norm: np.ndarray,
        anchor_group_ids: np.ndarray,
        rssi_feature_mode: str,
    ) -> None:
        raw_rssi = arrays["X"].astype(np.float32)
        rssi_z = zscore(
            arrays["X"].astype(np.float32),
            feature_mean[None, None, :],
            feature_std[None, None, :],
        ).astype(np.float32)
        rssi_centered, rssi_mask, rssi_count_ratio = build_rssi_robust_features(raw_rssi)
        if rssi_feature_mode == "zscore":
            rssi = rssi_z
        elif rssi_feature_mode == "robust":
            rssi = np.concatenate(
                [rssi_centered, rssi_mask, rssi_count_ratio],
                axis=2,
            ).astype(np.float32)
        elif rssi_feature_mode == "hybrid":
            rssi = np.concatenate(
                [rssi_z, rssi_centered, rssi_mask, rssi_count_ratio],
                axis=2,
            ).astype(np.float32)
        else:
            raise ValueError(
                f"Unknown rssi_feature_mode='{rssi_feature_mode}', expected zscore|robust|hybrid."
            )

        motion = zscore(
            build_motion_feature_array(arrays),
            motion_mean[None, None, :],
            motion_std[None, None, :],
        ).astype(np.float32)
        self.inputs = np.concatenate([rssi, motion], axis=2).astype(np.float32)

        self.y_last_raw = arrays["y_last"].astype(np.float32)
        self.y_last_norm = zscore(
            self.y_last_raw,
            coord_mean[None, :],
            coord_std[None, :],
        ).astype(np.float32)

        self.groups = arrays["group"].astype(np.int32)
        self.group_ids = np.asarray(
            [group_to_class[(int(b), int(f))] for b, f in self.groups],
            dtype=np.int64,
        )
        self.anchor_ids = assign_anchor_ids(
            coords_raw=self.y_last_raw,
            group_ids=self.group_ids,
            anchor_centers_raw=anchor_centers_raw,
            anchor_group_ids=anchor_group_ids,
        )
        self.anchor_offset_norm = (
            self.y_last_norm - anchor_centers_norm[self.anchor_ids]
        ).astype(np.float32)

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "inputs": torch.from_numpy(self.inputs[idx]),
            "y_last_norm": torch.from_numpy(self.y_last_norm[idx]),
            "y_last_raw": torch.from_numpy(self.y_last_raw[idx]),
            "group_id": torch.tensor(self.group_ids[idx], dtype=torch.long),
            "anchor_id": torch.tensor(self.anchor_ids[idx], dtype=torch.long),
            "anchor_offset_norm": torch.from_numpy(self.anchor_offset_norm[idx]),
        }


class TinyBaseNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_anchors: int,
        anchor_centers_norm: np.ndarray,
        anchor_group_ids: np.ndarray,
        step_hidden: int,
        temporal_hidden: int,
        head_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        fusion_dim = temporal_hidden * 3
        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(head_hidden, num_classes)
        self.anchor_head = nn.Linear(head_hidden, num_anchors)
        self.residual_head = nn.Linear(head_hidden, 2)

        self.register_buffer(
            "anchor_centers_norm",
            torch.from_numpy(anchor_centers_norm.astype(np.float32)),
        )
        self.register_buffer(
            "anchor_group_ids",
            torch.from_numpy(anchor_group_ids.astype(np.int64)),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        inputs: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.encode(inputs)

        last_feat = x[:, -1, :]
        mean_feat = x.mean(dim=1)
        max_feat = x.max(dim=1).values
        fused = torch.cat([last_feat, mean_feat, max_feat], dim=1)
        hidden = self.fuse(fused)

        logits = self.cls_head(hidden)
        raw_anchor_logits = self.anchor_head(hidden)
        residual = self.residual_head(hidden)

        if group_ids is not None:
            valid_anchor_mask = self.anchor_group_ids.unsqueeze(0) == group_ids.unsqueeze(1)
            anchor_logits = raw_anchor_logits.masked_fill(~valid_anchor_mask, -1e9)
        else:
            group_prior = torch.softmax(logits, dim=1)
            anchor_group_prior = group_prior[:, self.anchor_group_ids].clamp_min(1e-8)
            anchor_logits = raw_anchor_logits + torch.log(anchor_group_prior)

        anchor_probs = torch.softmax(anchor_logits, dim=1)
        pred_anchor_coord = anchor_probs @ self.anchor_centers_norm
        pred_coord = pred_anchor_coord + residual
        return {
            "pred_coord": pred_coord,
            "pred_anchor_coord": pred_anchor_coord,
            "pred_residual": residual,
            "logits": logits,
            "anchor_logits": anchor_logits,
        }


class TinyESP32Net(TinyBaseNet):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_anchors: int,
        anchor_centers_norm: np.ndarray,
        anchor_group_ids: np.ndarray,
        step_hidden: int,
        temporal_hidden: int,
        head_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            num_anchors=num_anchors,
            anchor_centers_norm=anchor_centers_norm,
            anchor_group_ids=anchor_group_ids,
            step_hidden=step_hidden,
            temporal_hidden=temporal_hidden,
            head_hidden=head_hidden,
            dropout=dropout,
        )
        self.step_proj = nn.Linear(input_dim, step_hidden)
        self.temporal_dw = nn.Conv1d(
            step_hidden,
            step_hidden,
            kernel_size=3,
            padding=1,
            groups=step_hidden,
            bias=True,
        )
        self.temporal_pw = nn.Conv1d(step_hidden, temporal_hidden, kernel_size=1, bias=True)
        self.temporal_norm = nn.LayerNorm(temporal_hidden)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.act(self.step_proj(inputs))
        x = x.transpose(1, 2)
        x = self.act(self.temporal_dw(x))
        x = self.act(self.temporal_pw(x))
        x = x.transpose(1, 2)
        x = self.temporal_norm(x)
        x = self.dropout(x)
        return x


class TinyGRUNet(TinyBaseNet):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_anchors: int,
        anchor_centers_norm: np.ndarray,
        anchor_group_ids: np.ndarray,
        step_hidden: int,
        temporal_hidden: int,
        head_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            num_anchors=num_anchors,
            anchor_centers_norm=anchor_centers_norm,
            anchor_group_ids=anchor_group_ids,
            step_hidden=step_hidden,
            temporal_hidden=temporal_hidden,
            head_hidden=head_hidden,
            dropout=dropout,
        )
        self.step_proj = nn.Linear(input_dim, step_hidden)
        self.temporal_rnn = nn.GRU(
            input_size=step_hidden,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(temporal_hidden)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.act(self.step_proj(inputs))
        x = self.dropout(x)
        x, _ = self.temporal_rnn(x)
        x = self.temporal_norm(x)
        x = self.dropout(x)
        return x


class TinyTCNNet(TinyBaseNet):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_anchors: int,
        anchor_centers_norm: np.ndarray,
        anchor_group_ids: np.ndarray,
        step_hidden: int,
        temporal_hidden: int,
        head_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            num_anchors=num_anchors,
            anchor_centers_norm=anchor_centers_norm,
            anchor_group_ids=anchor_group_ids,
            step_hidden=step_hidden,
            temporal_hidden=temporal_hidden,
            head_hidden=head_hidden,
            dropout=dropout,
        )
        self.step_proj = nn.Linear(input_dim, step_hidden)
        self.dw_conv_1 = nn.Conv1d(
            step_hidden,
            step_hidden,
            kernel_size=3,
            padding=1,
            groups=step_hidden,
            bias=True,
        )
        self.pw_conv_1 = nn.Conv1d(step_hidden, temporal_hidden, kernel_size=1, bias=True)
        self.dw_conv_2 = nn.Conv1d(
            temporal_hidden,
            temporal_hidden,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=temporal_hidden,
            bias=True,
        )
        self.pw_conv_2 = nn.Conv1d(temporal_hidden, temporal_hidden, kernel_size=1, bias=True)
        self.temporal_norm = nn.LayerNorm(temporal_hidden)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.act(self.step_proj(inputs))
        x = x.transpose(1, 2)

        x = self.act(self.dw_conv_1(x))
        x = self.act(self.pw_conv_1(x))
        residual = x

        y = self.act(self.dw_conv_2(x))
        y = self.act(self.pw_conv_2(y))
        x = residual + y

        x = x.transpose(1, 2)
        x = self.temporal_norm(x)
        x = self.dropout(x)
        return x


def build_tiny_model(
    model_arch: str,
    input_dim: int,
    num_classes: int,
    num_anchors: int,
    anchor_centers_norm: np.ndarray,
    anchor_group_ids: np.ndarray,
    step_hidden: int,
    temporal_hidden: int,
    head_hidden: int,
    dropout: float,
) -> nn.Module:
    arch = str(model_arch).strip().lower()
    if arch == "dscnn":
        return TinyESP32Net(
            input_dim=input_dim,
            num_classes=num_classes,
            num_anchors=num_anchors,
            anchor_centers_norm=anchor_centers_norm,
            anchor_group_ids=anchor_group_ids,
            step_hidden=step_hidden,
            temporal_hidden=temporal_hidden,
            head_hidden=head_hidden,
            dropout=dropout,
        )
    if arch == "gru":
        return TinyGRUNet(
            input_dim=input_dim,
            num_classes=num_classes,
            num_anchors=num_anchors,
            anchor_centers_norm=anchor_centers_norm,
            anchor_group_ids=anchor_group_ids,
            step_hidden=step_hidden,
            temporal_hidden=temporal_hidden,
            head_hidden=head_hidden,
            dropout=dropout,
        )
    if arch == "tcn":
        return TinyTCNNet(
            input_dim=input_dim,
            num_classes=num_classes,
            num_anchors=num_anchors,
            anchor_centers_norm=anchor_centers_norm,
            anchor_group_ids=anchor_group_ids,
            step_hidden=step_hidden,
            temporal_hidden=temporal_hidden,
            head_hidden=head_hidden,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model_arch='{model_arch}', expected dscnn|gru|tcn.")


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_value: Optional[float] = None
        self.counter = 0

    def step(self, value: float) -> bool:
        if self.best_value is None or value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


@dataclass
class LossWeights:
    coord: float
    cls: float
    anchor: float
    residual: float


@dataclass
class CandidateConfig:
    step_hidden: int
    temporal_hidden: int
    head_hidden: int

    @property
    def name(self) -> str:
        return f"s{self.step_hidden}_t{self.temporal_hidden}_h{self.head_hidden}"


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_weights: LossWeights,
    cls_loss_fn: nn.Module,
    anchor_loss_fn: nn.Module,
) -> Dict[str, torch.Tensor]:
    coord_loss = nn.functional.smooth_l1_loss(outputs["pred_coord"], batch["y_last_norm"])
    cls_loss = cls_loss_fn(outputs["logits"], batch["group_id"])
    anchor_loss = anchor_loss_fn(outputs["anchor_logits"], batch["anchor_id"])
    residual_loss = nn.functional.smooth_l1_loss(
        outputs["pred_residual"],
        batch["anchor_offset_norm"],
    )
    total_loss = (
        loss_weights.coord * coord_loss +
        loss_weights.cls * cls_loss +
        loss_weights.anchor * anchor_loss +
        loss_weights.residual * residual_loss
    )
    return {
        "total": total_loss,
        "coord": coord_loss,
        "cls": cls_loss,
        "anchor": anchor_loss,
        "residual": residual_loss,
    }


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
    loss_weights: LossWeights,
    cls_loss_fn: nn.Module,
    anchor_loss_fn: nn.Module,
    show_progress: bool,
    desc: str,
) -> Dict[str, object]:
    model.eval()
    losses = {"total": 0.0, "coord": 0.0, "cls": 0.0, "anchor": 0.0, "residual": 0.0}
    num_batches = 0
    all_errors: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []
    all_group_ids: List[np.ndarray] = []
    all_pred_ids: List[np.ndarray] = []

    coord_mean_t = torch.from_numpy(coord_mean.astype(np.float32)).to(device)
    coord_std_t = torch.from_numpy(coord_std.astype(np.float32)).to(device)

    with torch.no_grad():
        iterator = wrap_loader(loader, show_progress, desc)
        for batch in iterator:
            batch = batch_to_device(batch, device)

            train_mode_outputs = model(batch["inputs"], batch["group_id"])
            infer_outputs = model(batch["inputs"], None)
            loss_dict = compute_loss(
                outputs=train_mode_outputs,
                batch=batch,
                loss_weights=loss_weights,
                cls_loss_fn=cls_loss_fn,
                anchor_loss_fn=anchor_loss_fn,
            )

            for key in losses:
                losses[key] += float(loss_dict[key].item())
            num_batches += 1

            pred_coord_raw = infer_outputs["pred_coord"] * coord_std_t + coord_mean_t
            error = torch.linalg.norm(pred_coord_raw - batch["y_last_raw"], dim=1)
            pred_ids = infer_outputs["logits"].argmax(dim=1)

            all_errors.append(error.detach().cpu().numpy())
            all_targets.append(batch["y_last_raw"].detach().cpu().numpy())
            all_predictions.append(pred_coord_raw.detach().cpu().numpy())
            all_group_ids.append(batch["group_id"].detach().cpu().numpy())
            all_pred_ids.append(pred_ids.detach().cpu().numpy())

            if show_progress and tqdm is not None:
                iterator.set_postfix(loss=f"{loss_dict['total'].item():.4f}")

    errors = np.concatenate(all_errors, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)
    group_ids = np.concatenate(all_group_ids, axis=0)
    pred_ids = np.concatenate(all_pred_ids, axis=0)

    avg_losses = {key: value / max(1, num_batches) for key, value in losses.items()}
    return {
        "losses": avg_losses,
        "mean_error_m": float(errors.mean()),
        "median_error_m": float(np.median(errors)),
        "p75_error_m": float(np.quantile(errors, 0.75)),
        "p90_error_m": float(np.quantile(errors, 0.90)),
        "p95_error_m": float(np.quantile(errors, 0.95)),
        "max_error_m": float(errors.max()),
        "rmse_m": float(math.sqrt(np.mean(errors ** 2))),
        "classification_accuracy": float((group_ids == pred_ids).mean()),
        "errors": errors,
        "targets": targets,
        "predictions": predictions,
        "group_ids": group_ids,
        "pred_ids": pred_ids,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: LossWeights,
    cls_loss_fn: nn.Module,
    anchor_loss_fn: nn.Module,
    grad_clip: float,
    show_progress: bool,
    desc: str,
) -> Dict[str, float]:
    model.train()
    losses = {"total": 0.0, "coord": 0.0, "cls": 0.0, "anchor": 0.0, "residual": 0.0}
    num_batches = 0

    iterator = wrap_loader(loader, show_progress, desc)
    for batch in iterator:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["inputs"], batch["group_id"])
        loss_dict = compute_loss(
            outputs=outputs,
            batch=batch,
            loss_weights=loss_weights,
            cls_loss_fn=cls_loss_fn,
            anchor_loss_fn=anchor_loss_fn,
        )
        loss_dict["total"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for key in losses:
            losses[key] += float(loss_dict[key].item())
        num_batches += 1

        if show_progress and tqdm is not None:
            iterator.set_postfix(
                total=f"{loss_dict['total'].item():.4f}",
                coord=f"{loss_dict['coord'].item():.4f}",
                cls=f"{loss_dict['cls'].item():.4f}",
                anchor=f"{loss_dict['anchor'].item():.4f}",
            )

    return {key: value / max(1, num_batches) for key, value in losses.items()}


def parse_candidate_configs(raw: str) -> List[CandidateConfig]:
    configs: List[CandidateConfig] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) == 2:
            step_hidden = int(parts[0])
            temporal_hidden = int(parts[1])
            head_hidden = int(parts[1])
        elif len(parts) == 3:
            step_hidden = int(parts[0])
            temporal_hidden = int(parts[1])
            head_hidden = int(parts[2])
        else:
            raise ValueError(
                f"Invalid candidate config '{token}', use 'step:temporal[:head]'."
            )
        configs.append(
            CandidateConfig(
                step_hidden=step_hidden,
                temporal_hidden=temporal_hidden,
                head_hidden=head_hidden,
            )
        )
    if not configs:
        raise ValueError("No valid candidate configs provided.")
    return configs


def to_serializable_stats(stats: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
    return {key: value.tolist() for key, value in stats.items()}


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def quantize_symmetric_int8(array: np.ndarray) -> Tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(array)))
    if max_abs < 1e-12:
        return np.zeros_like(array, dtype=np.int8), 1.0
    scale = max_abs / 127.0
    q = np.clip(np.round(array / scale), -127, 127).astype(np.int8)
    return q, float(scale)


def c_identifier(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", name)


def format_c_array(name: str, array: np.ndarray, c_type: str, values_per_line: int = 16) -> str:
    flat = array.reshape(-1)
    if c_type == "int8_t":
        values = [str(int(v)) for v in flat.tolist()]
    elif c_type == "int32_t":
        values = [str(int(v)) for v in flat.tolist()]
    else:
        values = [f"{float(v):.8g}f" for v in flat.tolist()]

    lines: List[str] = []
    for i in range(0, len(values), values_per_line):
        chunk = ", ".join(values[i : i + values_per_line])
        lines.append(f"  {chunk}")
    body = ",\n".join(lines)
    return f"static const {c_type} {name}[{len(values)}] = {{\n{body}\n}};\n"


def export_quantized_artifacts(
    model: nn.Module,
    stats: Dict[str, np.ndarray],
    selected_waps: Optional[Sequence[str]],
    class_names: Sequence[str],
    output_dir: Path,
    rssi_feature_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    quant_payload: Dict[str, np.ndarray] = {}

    header_parts: List[str] = []
    header_parts.append("#pragma once\n")
    header_parts.append("#include <stdint.h>\n")
    header_parts.append("\n")

    for key in (
        "feature_mean",
        "feature_std",
        "motion_mean",
        "motion_std",
        "coord_mean",
        "coord_std",
        "anchor_centers_norm",
        "anchor_centers_raw",
    ):
        arr = stats[key].astype(np.float32)
        quant_payload[key] = arr
        name = c_identifier(f"tiny_{key}")
        header_parts.append(format_c_array(name, arr, "float"))

    anchor_group_ids = stats["anchor_group_ids"].astype(np.int32)
    quant_payload["anchor_group_ids"] = anchor_group_ids
    header_parts.append(format_c_array("tiny_anchor_group_ids", anchor_group_ids, "int32_t"))

    for param_name, tensor in state_dict.items():
        array = tensor.detach().cpu().numpy()
        safe_name = c_identifier(f"tiny_{param_name}")

        if np.issubdtype(array.dtype, np.floating):
            q, scale = quantize_symmetric_int8(array.astype(np.float32))
            quant_payload[f"{param_name}__q"] = q
            quant_payload[f"{param_name}__scale"] = np.asarray([scale], dtype=np.float32)
            quant_payload[f"{param_name}__shape"] = np.asarray(array.shape, dtype=np.int32)

            header_parts.append(format_c_array(f"{safe_name}_q", q, "int8_t"))
            header_parts.append(
                f"static const float {safe_name}_scale = {scale:.8g}f;\n"
            )
            shape_array = np.asarray(array.shape, dtype=np.int32)
            header_parts.append(format_c_array(f"{safe_name}_shape", shape_array, "int32_t"))
        else:
            int_arr = array.astype(np.int32)
            quant_payload[f"{param_name}__int"] = int_arr
            quant_payload[f"{param_name}__shape"] = np.asarray(array.shape, dtype=np.int32)
            header_parts.append(format_c_array(f"{safe_name}_int", int_arr, "int32_t"))
            shape_array = np.asarray(array.shape, dtype=np.int32)
            header_parts.append(format_c_array(f"{safe_name}_shape", shape_array, "int32_t"))

    np.savez_compressed(output_dir / "esp32_tiny_model_int8.npz", **quant_payload)
    (output_dir / "esp32_tiny_model_int8.h").write_text(
        "".join(header_parts),
        encoding="utf-8",
    )

    manifest = {
        "selected_waps": list(selected_waps) if selected_waps is not None else None,
        "class_names": list(class_names),
        "stats_keys": list(stats.keys()),
        "rssi_feature_mode": rssi_feature_mode,
        "notes": (
            "Inputs must be transformed using the selected rssi_feature_mode "
            "and motion z-score stats before inference."
        ),
    }
    (output_dir / "deploy_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_dataloaders_and_stats(
    train_arrays: Dict[str, np.ndarray],
    val_arrays: Dict[str, np.ndarray],
    test_arrays: Dict[str, np.ndarray],
    batch_size: int,
    num_workers: int,
    anchor_divisor: int,
    min_anchors_per_group: int,
    rssi_feature_mode: str,
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    Dict[str, np.ndarray],
    Dict[Tuple[int, int], int],
    List[str],
]:
    feature_mean = train_arrays["X"].mean(axis=(0, 1)).astype(np.float32)
    feature_std = train_arrays["X"].std(axis=(0, 1)).astype(np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)

    train_motion = build_motion_feature_array(train_arrays)
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
    anchor_centers_raw, anchor_centers_norm, anchor_group_ids = build_anchor_bank(
        train_groups=train_arrays["group"],
        train_y_last=train_arrays["y_last"],
        group_to_class=group_to_class,
        coord_mean=coord_mean,
        coord_std=coord_std,
        anchor_divisor=anchor_divisor,
        min_anchors_per_group=min_anchors_per_group,
    )

    stats = {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "motion_mean": motion_mean,
        "motion_std": motion_std,
        "coord_mean": coord_mean,
        "coord_std": coord_std,
        "anchor_centers_raw": anchor_centers_raw,
        "anchor_centers_norm": anchor_centers_norm,
        "anchor_group_ids": anchor_group_ids,
    }

    train_dataset = TinySequenceDataset(
        arrays=train_arrays,
        feature_mean=feature_mean,
        feature_std=feature_std,
        motion_mean=motion_mean,
        motion_std=motion_std,
        coord_mean=coord_mean,
        coord_std=coord_std,
        group_to_class=group_to_class,
        anchor_centers_raw=anchor_centers_raw,
        anchor_centers_norm=anchor_centers_norm,
        anchor_group_ids=anchor_group_ids,
        rssi_feature_mode=rssi_feature_mode,
    )
    val_dataset = TinySequenceDataset(
        arrays=val_arrays,
        feature_mean=feature_mean,
        feature_std=feature_std,
        motion_mean=motion_mean,
        motion_std=motion_std,
        coord_mean=coord_mean,
        coord_std=coord_std,
        group_to_class=group_to_class,
        anchor_centers_raw=anchor_centers_raw,
        anchor_centers_norm=anchor_centers_norm,
        anchor_group_ids=anchor_group_ids,
        rssi_feature_mode=rssi_feature_mode,
    )
    test_dataset = TinySequenceDataset(
        arrays=test_arrays,
        feature_mean=feature_mean,
        feature_std=feature_std,
        motion_mean=motion_mean,
        motion_std=motion_std,
        coord_mean=coord_mean,
        coord_std=coord_std,
        group_to_class=group_to_class,
        anchor_centers_raw=anchor_centers_raw,
        anchor_centers_norm=anchor_centers_norm,
        anchor_group_ids=anchor_group_ids,
        rssi_feature_mode=rssi_feature_mode,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader, stats, group_to_class, class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny ESP32S3-friendly indoor positioning network on non-fixed datasets.",
    )
    parser.add_argument("--train-dir", type=str, default="training_dataset")
    parser.add_argument("--test-dir", type=str, default="test_dataset")
    parser.add_argument("--output-dir", type=str, default="runs/tiny_esp32")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--eval-only", action="store_true")

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--coord-loss-weight", type=float, default=1.0)
    parser.add_argument("--cls-loss-weight", type=float, default=0.18)
    parser.add_argument("--anchor-loss-weight", type=float, default=0.30)
    parser.add_argument("--residual-loss-weight", type=float, default=0.20)
    parser.add_argument("--anchor-divisor", type=int, default=4)
    parser.add_argument("--min-anchors-per-group", type=int, default=6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--min-delta", type=float, default=1e-4)

    parser.add_argument("--candidate-configs", type=str, default="16:24,24:32,32:48")
    parser.add_argument(
        "--model-arch",
        type=str,
        default="dscnn",
        choices=["dscnn", "gru", "tcn"],
    )
    parser.add_argument(
        "--rssi-feature-mode",
        type=str,
        default="hybrid",
        choices=["zscore", "robust", "hybrid"],
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
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

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Train dir '{train_dir}' not found. Build non-fixed dataset first with scripts/DatasetProc.py."
        )
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test dir '{test_dir}' not found. Build non-fixed dataset first with scripts/DatasetProc.py."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_meta = load_metadata(train_dir / "metadata.json")
    test_meta = load_metadata(test_dir / "metadata.json")

    test_arrays = concat_splits(
        [
            load_npz(test_dir / "train_sequences.npz"),
            load_npz(test_dir / "val_sequences.npz"),
        ]
    )
    train_arrays = load_npz(train_dir / "train_sequences.npz")
    val_arrays = load_npz(train_dir / "val_sequences.npz")

    train_arrays = limit_arrays(train_arrays, args.max_train_samples)
    val_arrays = limit_arrays(val_arrays, args.max_val_samples)
    test_arrays = limit_arrays(test_arrays, args.max_test_samples)

    train_loader, val_loader, test_loader, stats, group_to_class, class_names = build_dataloaders_and_stats(
        train_arrays=train_arrays,
        val_arrays=val_arrays,
        test_arrays=test_arrays,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        anchor_divisor=args.anchor_divisor,
        min_anchors_per_group=args.min_anchors_per_group,
        rssi_feature_mode=args.rssi_feature_mode,
    )

    input_dim = int(train_loader.dataset[0]["inputs"].shape[-1])  # type: ignore[index]
    motion_dim = int(build_motion_feature_array(train_arrays).shape[-1])
    feature_dim = int(train_arrays["X"].shape[-1])
    num_anchors = int(stats["anchor_centers_norm"].shape[0])

    print(f"Using device: {device}", flush=True)
    print(
        f"Train/Val/Test samples: {len(train_loader.dataset)}/"
        f"{len(val_loader.dataset)}/{len(test_loader.dataset)}",
        flush=True,
    )
    print(
        f"Input dims: RSSI={feature_dim}, motion={motion_dim}, total={input_dim}, "
        f"classes={len(class_names)}, anchors={num_anchors}",
        flush=True,
    )
    if show_progress and tqdm is None:
        print("tqdm not installed, falling back to epoch-level logs only.", flush=True)

    loss_weights = LossWeights(
        coord=args.coord_loss_weight,
        cls=args.cls_loss_weight,
        anchor=args.anchor_loss_weight,
        residual=args.residual_loss_weight,
    )
    cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    anchor_loss_fn = nn.CrossEntropyLoss()

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint.")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        cfg = checkpoint["best_candidate"]
        checkpoint_args = checkpoint.get("args", {})
        ckpt_model_arch = "dscnn"
        if isinstance(checkpoint_args, dict):
            ckpt_model_arch = str(checkpoint_args.get("model_arch", "dscnn"))
        model = build_tiny_model(
            model_arch=ckpt_model_arch,
            input_dim=int(checkpoint["input_dim"]),
            num_classes=len(checkpoint["class_names"]),
            num_anchors=int(checkpoint["num_anchors"]),
            anchor_centers_norm=np.asarray(checkpoint["stats"]["anchor_centers_norm"], dtype=np.float32),
            anchor_group_ids=np.asarray(checkpoint["stats"]["anchor_group_ids"], dtype=np.int64),
            step_hidden=int(cfg["step_hidden"]),
            temporal_hidden=int(cfg["temporal_hidden"]),
            head_hidden=int(cfg["head_hidden"]),
            dropout=float(checkpoint_args.get("dropout", args.dropout)) if isinstance(checkpoint_args, dict) else args.dropout,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            coord_mean=np.asarray(checkpoint["stats"]["coord_mean"], dtype=np.float32),
            coord_std=np.asarray(checkpoint["stats"]["coord_std"], dtype=np.float32),
            loss_weights=loss_weights,
            cls_loss_fn=cls_loss_fn,
            anchor_loss_fn=anchor_loss_fn,
            show_progress=show_progress,
            desc="Test",
        )
        payload = {
            "device": str(device),
            "checkpoint": args.checkpoint,
            "test_mean_error_m": test_metrics["mean_error_m"],
            "test_median_error_m": test_metrics["median_error_m"],
            "test_p75_error_m": test_metrics["p75_error_m"],
            "test_p90_error_m": test_metrics["p90_error_m"],
            "test_p95_error_m": test_metrics["p95_error_m"],
            "test_rmse_m": test_metrics["rmse_m"],
            "test_classification_accuracy": test_metrics["classification_accuracy"],
        }
        (output_dir / "metrics_eval_only.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return

    candidates = parse_candidate_configs(args.candidate_configs)
    candidate_summaries: List[Dict[str, object]] = []

    best_val_error = float("inf")
    best_checkpoint_payload: Optional[Dict[str, object]] = None
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    best_candidate_config: Optional[CandidateConfig] = None
    best_history: List[Dict[str, float]] = []

    for idx, candidate in enumerate(candidates, start=1):
        print(
            f"\n=== Candidate {idx}/{len(candidates)}: {candidate.name} ===",
            flush=True,
        )
        model = build_tiny_model(
            model_arch=args.model_arch,
            input_dim=input_dim,
            num_classes=len(class_names),
            num_anchors=num_anchors,
            anchor_centers_norm=stats["anchor_centers_norm"],
            anchor_group_ids=stats["anchor_group_ids"].astype(np.int64),
            step_hidden=candidate.step_hidden,
            temporal_hidden=candidate.temporal_hidden,
            head_hidden=candidate.head_hidden,
            dropout=args.dropout,
        ).to(device)

        param_count = count_parameters(model)
        print(
            f"Param count: {param_count} "
            f"(~{param_count * 4 / 1024:.1f} KiB float32, ~{param_count / 1024:.1f} KiB int8)",
            flush=True,
        )

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
        early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
        history: List[Dict[str, float]] = []
        candidate_best_val_error = float("inf")
        candidate_best_state: Optional[Dict[str, torch.Tensor]] = None

        for epoch in range(1, args.epochs + 1):
            train_losses = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                loss_weights=loss_weights,
                cls_loss_fn=cls_loss_fn,
                anchor_loss_fn=anchor_loss_fn,
                grad_clip=args.grad_clip,
                show_progress=show_progress,
                desc=f"{candidate.name} Train {epoch}/{args.epochs}",
            )
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                coord_mean=stats["coord_mean"],
                coord_std=stats["coord_std"],
                loss_weights=loss_weights,
                cls_loss_fn=cls_loss_fn,
                anchor_loss_fn=anchor_loss_fn,
                show_progress=show_progress,
                desc=f"{candidate.name} Val {epoch}/{args.epochs}",
            )
            scheduler.step(val_metrics["mean_error_m"])

            current_lr = optimizer.param_groups[0]["lr"]
            record = {
                "epoch": float(epoch),
                "lr": float(current_lr),
                "train_total_loss": float(train_losses["total"]),
                "val_total_loss": float(val_metrics["losses"]["total"]),
                "val_mean_error_m": float(val_metrics["mean_error_m"]),
                "val_rmse_m": float(val_metrics["rmse_m"]),
                "val_cls_acc": float(val_metrics["classification_accuracy"]),
            }
            history.append(record)
            print(
                f"{candidate.name} Epoch {epoch:03d} | "
                f"train_loss={train_losses['total']:.4f} | "
                f"val_loss={val_metrics['losses']['total']:.4f} | "
                f"val_mean_error={val_metrics['mean_error_m']:.3f} m | "
                f"val_rmse={val_metrics['rmse_m']:.3f} m | "
                f"val_cls_acc={val_metrics['classification_accuracy']:.3f} | "
                f"lr={current_lr:.2e}",
                flush=True,
            )

            if val_metrics["mean_error_m"] < candidate_best_val_error:
                candidate_best_val_error = float(val_metrics["mean_error_m"])
                candidate_best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

            if early_stopper.step(val_metrics["mean_error_m"]):
                print(f"{candidate.name} early stop at epoch {epoch}", flush=True)
                break

        if candidate_best_state is None:
            candidate_best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        model.load_state_dict(candidate_best_state)
        val_final = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            coord_mean=stats["coord_mean"],
            coord_std=stats["coord_std"],
            loss_weights=loss_weights,
            cls_loss_fn=cls_loss_fn,
            anchor_loss_fn=anchor_loss_fn,
            show_progress=False,
            desc="Val-final",
        )
        test_final = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            coord_mean=stats["coord_mean"],
            coord_std=stats["coord_std"],
            loss_weights=loss_weights,
            cls_loss_fn=cls_loss_fn,
            anchor_loss_fn=anchor_loss_fn,
            show_progress=False,
            desc="Test-final",
        )

        summary = {
            "candidate": candidate.name,
            "params": count_parameters(model),
            "val_mean_error_m": float(val_final["mean_error_m"]),
            "val_rmse_m": float(val_final["rmse_m"]),
            "val_cls_acc": float(val_final["classification_accuracy"]),
            "test_mean_error_m": float(test_final["mean_error_m"]),
            "test_rmse_m": float(test_final["rmse_m"]),
            "test_cls_acc": float(test_final["classification_accuracy"]),
        }
        candidate_summaries.append(summary)
        print(
            f"{candidate.name} summary: "
            f"val_mean={val_final['mean_error_m']:.3f} m, "
            f"test_mean={test_final['mean_error_m']:.3f} m",
            flush=True,
        )

        if val_final["mean_error_m"] < best_val_error:
            best_val_error = float(val_final["mean_error_m"])
            best_model_state = candidate_best_state
            best_candidate_config = candidate
            best_history = history
            best_checkpoint_payload = {
                "model_state_dict": candidate_best_state,
                "stats": to_serializable_stats(stats),
                "group_to_class": {f"{k[0]}_{k[1]}": v for k, v in group_to_class.items()},
                "class_names": class_names,
                "model_arch": args.model_arch,
                "input_dim": input_dim,
                "num_anchors": num_anchors,
                "best_candidate": {
                    "step_hidden": candidate.step_hidden,
                    "temporal_hidden": candidate.temporal_hidden,
                    "head_hidden": candidate.head_hidden,
                },
                "args": vars(args),
                "best_val_error_m": best_val_error,
                "selected_waps": (
                    train_meta.get("selected_waps")
                    if train_meta and isinstance(train_meta.get("selected_waps"), list)
                    else None
                ),
            }

    if best_checkpoint_payload is None or best_model_state is None or best_candidate_config is None:
        raise RuntimeError("Failed to train any candidate.")

    best_model = build_tiny_model(
        model_arch=args.model_arch,
        input_dim=input_dim,
        num_classes=len(class_names),
        num_anchors=num_anchors,
        anchor_centers_norm=stats["anchor_centers_norm"],
        anchor_group_ids=stats["anchor_group_ids"].astype(np.int64),
        step_hidden=best_candidate_config.step_hidden,
        temporal_hidden=best_candidate_config.temporal_hidden,
        head_hidden=best_candidate_config.head_hidden,
        dropout=args.dropout,
    ).to(device)
    best_model.load_state_dict(best_model_state)

    test_metrics = evaluate(
        model=best_model,
        loader=test_loader,
        device=device,
        coord_mean=stats["coord_mean"],
        coord_std=stats["coord_std"],
        loss_weights=loss_weights,
        cls_loss_fn=cls_loss_fn,
        anchor_loss_fn=anchor_loss_fn,
        show_progress=show_progress,
        desc="Best Test",
    )

    checkpoint_path = output_dir / "best_tiny_esp32_model.pt"
    torch.save(best_checkpoint_payload, checkpoint_path)

    metrics_payload = {
        "device": str(device),
        "model_arch": args.model_arch,
        "best_candidate": best_candidate_config.name,
        "best_candidate_detail": {
            "step_hidden": best_candidate_config.step_hidden,
            "temporal_hidden": best_candidate_config.temporal_hidden,
            "head_hidden": best_candidate_config.head_hidden,
        },
        "best_val_error_m": float(best_val_error),
        "test_mean_error_m": float(test_metrics["mean_error_m"]),
        "test_median_error_m": float(test_metrics["median_error_m"]),
        "test_p75_error_m": float(test_metrics["p75_error_m"]),
        "test_p90_error_m": float(test_metrics["p90_error_m"]),
        "test_p95_error_m": float(test_metrics["p95_error_m"]),
        "test_max_error_m": float(test_metrics["max_error_m"]),
        "test_rmse_m": float(test_metrics["rmse_m"]),
        "test_classification_accuracy": float(test_metrics["classification_accuracy"]),
        "num_train_samples": int(len(train_loader.dataset)),
        "num_val_samples": int(len(val_loader.dataset)),
        "num_test_samples": int(len(test_loader.dataset)),
        "num_anchors": num_anchors,
        "input_dims": {
            "rssi_dim": feature_dim,
            "motion_dim": motion_dim,
            "total": input_dim,
        },
        "rssi_feature_mode": args.rssi_feature_mode,
        "param_count": count_parameters(best_model),
        "candidate_summaries": candidate_summaries,
        "history_best_candidate": best_history,
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

    export_quantized_artifacts(
        model=best_model.cpu(),
        stats=stats,
        selected_waps=(
            train_meta.get("selected_waps")
            if train_meta and isinstance(train_meta.get("selected_waps"), list)
            else None
        ),
        class_names=class_names,
        output_dir=output_dir,
        rssi_feature_mode=args.rssi_feature_mode,
    )

    print("\nFinal Tiny ESP32 Model Results", flush=True)
    print(f"  Model Arch  : {args.model_arch}", flush=True)
    print(f"  Best Candidate: {best_candidate_config.name}", flush=True)
    print(f"  Mean Error  : {test_metrics['mean_error_m']:.3f} m", flush=True)
    print(f"  Median Error: {test_metrics['median_error_m']:.3f} m", flush=True)
    print(f"  P90 Error   : {test_metrics['p90_error_m']:.3f} m", flush=True)
    print(f"  P95 Error   : {test_metrics['p95_error_m']:.3f} m", flush=True)
    print(f"  RMSE        : {test_metrics['rmse_m']:.3f} m", flush=True)
    print(f"  Cls Acc     : {test_metrics['classification_accuracy']:.3f}", flush=True)
    print(f"  Model params: {count_parameters(best_model)}", flush=True)
    print(f"  Checkpoint  : {checkpoint_path}", flush=True)
    print(f"  Metrics     : {output_dir / 'metrics.json'}", flush=True)
    print(f"  ESP32 int8  : {output_dir / 'esp32_tiny_model_int8.h'}", flush=True)


if __name__ == "__main__":
    main()
