#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
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


def resolve_dataset_dir(preferred: str, fallback: str) -> Path:
    preferred_path = Path(preferred)
    if preferred_path.exists():
        return preferred_path

    fallback_path = Path(fallback)
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        f"Neither dataset directory exists: '{preferred_path}' or '{fallback_path}'."
    )


def concat_splits(split_dicts: Iterable[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    split_dicts = list(split_dicts)
    keys = split_dicts[0].keys()
    return {
        key: np.concatenate([split[key] for split in split_dicts], axis=0)
        for key in keys
    }


def validate_selected_waps(
    train_meta: Optional[Dict[str, object]],
    test_meta: Optional[Dict[str, object]],
) -> None:
    if not train_meta or not test_meta:
        return

    train_waps = train_meta.get("selected_waps")
    test_waps = test_meta.get("selected_waps")
    if not isinstance(train_waps, list) or not isinstance(test_waps, list):
        return

    if train_waps != test_waps:
        train_set = set(train_waps)
        test_set = set(test_waps)
        overlap = len(train_set & test_set)
        raise ValueError(
            "Train/test selected_waps are not identical. "
            f"train={len(train_waps)}, test={len(test_waps)}, overlap={overlap}. "
            "Rebuild the test dataset with the training metadata via "
            "'DatasetProc.py --selected-waps-json training_dataset_fixed/metadata.json'."
        )


def limit_arrays(arrays: Dict[str, np.ndarray], max_samples: int) -> Dict[str, np.ndarray]:
    if max_samples <= 0 or max_samples >= len(next(iter(arrays.values()))):
        return arrays
    return {key: value[:max_samples] for key, value in arrays.items()}


def pick_group_count(num_channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


def zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def unzscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std + mean


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


class FixedSequenceDataset(Dataset):
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
        group_coord_mean: np.ndarray,
        group_coord_std: np.ndarray,
    ) -> None:
        self.x = zscore(
            arrays["X"].astype(np.float32),
            feature_mean[None, None, :],
            feature_std[None, None, :],
        ).astype(np.float32)
        self.motion_x = zscore(
            build_motion_feature_array(arrays),
            motion_mean[None, None, :],
            motion_std[None, None, :],
        ).astype(np.float32)

        self.coords_raw = arrays["coords"].astype(np.float32)
        self.y_last_raw = arrays["y_last"].astype(np.float32)
        self.coords_norm = zscore(
            self.coords_raw,
            coord_mean[None, None, :],
            coord_std[None, None, :],
        ).astype(np.float32)
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
        self.group_coord_mean = group_coord_mean.astype(np.float32)
        self.group_coord_std = group_coord_std.astype(np.float32)
        self.y_last_local_norm = zscore(
            self.y_last_raw,
            self.group_coord_mean[self.group_ids],
            self.group_coord_std[self.group_ids],
        ).astype(np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[idx]),
            "motion_x": torch.from_numpy(self.motion_x[idx]),
            "coords_norm": torch.from_numpy(self.coords_norm[idx]),
            "y_last_norm": torch.from_numpy(self.y_last_norm[idx]),
            "y_last_local_norm": torch.from_numpy(self.y_last_local_norm[idx]),
            "y_last_raw": torch.from_numpy(self.y_last_raw[idx]),
            "group_id": torch.tensor(self.group_ids[idx], dtype=torch.long),
        }


class ResidualFeatureBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(pick_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(pick_group_count(out_channels), out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.act(x)
        return x


class FingerprintCNNEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels: Tuple[int, int, int] = (32, 64, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        c1, c2, c3 = hidden_channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(pick_group_count(c1), c1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualFeatureBlock(c1, c1, kernel_size=5, stride=1, dropout=dropout),
            ResidualFeatureBlock(c1, c2, kernel_size=5, stride=2, dropout=dropout),
            ResidualFeatureBlock(c2, c2, kernel_size=5, stride=1, dropout=dropout),
            ResidualFeatureBlock(c2, c3, kernel_size=3, stride=2, dropout=dropout),
            ResidualFeatureBlock(c3, c3, kernel_size=3, stride=1, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = c3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feature_dim = x.shape
        x = x.reshape(batch_size * seq_len, 1, feature_dim)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return x.reshape(batch_size, seq_len, self.output_dim)


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.act(x)
        return x


class MotionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            ResidualMLPBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResidualMLPBlock(hidden_dim, hidden_dim, dropout=dropout),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        return x


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(pick_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(pick_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.act(x)
        return x


class TemporalConvEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dilations: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False)
        self.blocks = nn.Sequential(
            *[
                TemporalResidualBlock(
                    hidden_dim,
                    hidden_dim,
                    dilation=dilation,
                    kernel_size=3,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.blocks(x)
        return x


class HybridCNNTCN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        motion_input_dim: int,
        dropout: float = 0.1,
        cnn_channels: Tuple[int, int, int] = (32, 64, 128),
        motion_hidden_dim: int = 64,
        tcn_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.fingerprint_encoder = FingerprintCNNEncoder(
            hidden_channels=cnn_channels,
            dropout=dropout,
        )
        self.motion_encoder = MotionEncoder(
            input_dim=motion_input_dim,
            hidden_dim=motion_hidden_dim,
            dropout=dropout,
        )
        self.temporal_encoder = TemporalConvEncoder(
            input_dim=self.fingerprint_encoder.output_dim + self.motion_encoder.output_dim,
            hidden_dim=tcn_hidden_dim,
            dilations=(1, 2, 4),
            dropout=dropout,
        )

        fusion_dim = tcn_hidden_dim * 2
        self.coord_backbone = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, tcn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.coord_experts = nn.ModuleList(
            [nn.Linear(tcn_hidden_dim, 2) for _ in range(num_classes)]
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, tcn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tcn_hidden_dim, num_classes),
        )
        self.traj_head = nn.Sequential(
            nn.Conv1d(tcn_hidden_dim, tcn_hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(tcn_hidden_dim, 2, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        motion_x: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        rssi_features = self.fingerprint_encoder(x)
        motion_features = self.motion_encoder(motion_x)
        fused_step_features = torch.cat([rssi_features, motion_features], dim=-1)

        temporal_features = self.temporal_encoder(fused_step_features)
        temporal_seq = temporal_features.transpose(1, 2)

        last_state = temporal_seq[:, -1]
        mean_state = temporal_seq.mean(dim=1)
        fused = torch.cat([last_state, mean_state], dim=-1)

        logits = self.cls_head(fused)
        coord_hidden = self.coord_backbone(fused)
        coord_candidates = torch.stack(
            [expert(coord_hidden) for expert in self.coord_experts],
            dim=1,
        )
        if group_ids is not None:
            pred_coord = coord_candidates[
                torch.arange(coord_candidates.size(0), device=coord_candidates.device),
                group_ids,
            ]
        else:
            probs = torch.softmax(logits, dim=1)
            pred_coord = torch.sum(coord_candidates * probs.unsqueeze(-1), dim=1)
        pred_traj = self.traj_head(temporal_features).transpose(1, 2).contiguous()

        return {
            "pred_coord": pred_coord,
            "pred_coord_candidates": coord_candidates,
            "pred_traj": pred_traj,
            "logits": logits,
        }


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_value: Optional[float] = None
        self.counter = 0

    def step(self, value: float) -> bool:
        if self.best_value is None or value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


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


def build_group_coord_stats(
    train_groups: np.ndarray,
    train_y_last: np.ndarray,
    group_to_class: Dict[Tuple[int, int], int],
) -> Tuple[np.ndarray, np.ndarray]:
    num_classes = len(group_to_class)
    group_coord_mean = np.zeros((num_classes, 2), dtype=np.float32)
    group_coord_std = np.ones((num_classes, 2), dtype=np.float32)

    for group_key, class_id in group_to_class.items():
        mask = (
            (train_groups[:, 0] == group_key[0]) &
            (train_groups[:, 1] == group_key[1])
        )
        coords = train_y_last[mask]
        if len(coords) == 0:
            continue
        group_coord_mean[class_id] = coords.mean(axis=0).astype(np.float32)
        std = coords.std(axis=0).astype(np.float32)
        group_coord_std[class_id] = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    return group_coord_mean, group_coord_std


def build_dataloaders(
    train_arrays: Dict[str, np.ndarray],
    val_arrays: Dict[str, np.ndarray],
    test_arrays: Dict[str, np.ndarray],
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, np.ndarray], Dict[Tuple[int, int], int], List[str]]:
    feature_mean = train_arrays["X"].mean(axis=(0, 1)).astype(np.float32)
    feature_std = train_arrays["X"].std(axis=(0, 1)).astype(np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)

    train_motion = build_motion_feature_array(train_arrays)
    motion_mean = train_motion.mean(axis=(0, 1)).astype(np.float32)
    motion_std = train_motion.std(axis=(0, 1)).astype(np.float32)
    motion_std = np.where(motion_std < 1e-6, 1.0, motion_std).astype(np.float32)

    coord_mean = train_arrays["coords"].reshape(-1, 2).mean(axis=0).astype(np.float32)
    coord_std = train_arrays["coords"].reshape(-1, 2).std(axis=0).astype(np.float32)
    coord_std = np.where(coord_std < 1e-6, 1.0, coord_std).astype(np.float32)

    group_to_class, class_names = build_group_mapping(
        train_arrays["group"],
        val_arrays["group"],
        test_arrays["group"],
    )
    group_coord_mean, group_coord_std = build_group_coord_stats(
        train_groups=train_arrays["group"],
        train_y_last=train_arrays["y_last"],
        group_to_class=group_to_class,
    )

    stats = {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "motion_mean": motion_mean,
        "motion_std": motion_std,
        "coord_mean": coord_mean,
        "coord_std": coord_std,
        "group_coord_mean": group_coord_mean,
        "group_coord_std": group_coord_std,
    }

    train_dataset = FixedSequenceDataset(
        train_arrays,
        feature_mean=feature_mean,
        feature_std=feature_std,
        motion_mean=motion_mean,
        motion_std=motion_std,
        coord_mean=coord_mean,
        coord_std=coord_std,
        group_to_class=group_to_class,
        group_coord_mean=group_coord_mean,
        group_coord_std=group_coord_std,
    )
    val_dataset = FixedSequenceDataset(
        val_arrays,
        feature_mean=feature_mean,
        feature_std=feature_std,
        motion_mean=motion_mean,
        motion_std=motion_std,
        coord_mean=coord_mean,
        coord_std=coord_std,
        group_to_class=group_to_class,
        group_coord_mean=group_coord_mean,
        group_coord_std=group_coord_std,
    )
    test_dataset = FixedSequenceDataset(
        test_arrays,
        feature_mean=feature_mean,
        feature_std=feature_std,
        motion_mean=motion_mean,
        motion_std=motion_std,
        coord_mean=coord_mean,
        coord_std=coord_std,
        group_to_class=group_to_class,
        group_coord_mean=group_coord_mean,
        group_coord_std=group_coord_std,
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


def build_test_loader_from_stats(
    test_arrays: Dict[str, np.ndarray],
    batch_size: int,
    num_workers: int,
    stats: Dict[str, np.ndarray],
    group_to_class: Dict[Tuple[int, int], int],
) -> DataLoader:
    test_dataset = FixedSequenceDataset(
        test_arrays,
        feature_mean=stats["feature_mean"],
        feature_std=stats["feature_std"],
        motion_mean=stats["motion_mean"],
        motion_std=stats["motion_std"],
        coord_mean=stats["coord_mean"],
        coord_std=stats["coord_std"],
        group_to_class=group_to_class,
        group_coord_mean=stats["group_coord_mean"],
        group_coord_std=stats["group_coord_std"],
    )
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )


def parse_group_to_class(raw_mapping: Dict[str, int]) -> Dict[Tuple[int, int], int]:
    parsed: Dict[Tuple[int, int], int] = {}
    for key, value in raw_mapping.items():
        building_str, floor_str = key.split("_")
        parsed[(int(building_str), int(floor_str))] = int(value)
    return parsed


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    coord_weight: float,
    traj_weight: float,
    cls_weight: float,
    cls_loss_fn: nn.Module,
) -> Dict[str, torch.Tensor]:
    coord_loss = nn.functional.smooth_l1_loss(outputs["pred_coord"], batch["y_last_local_norm"])
    traj_loss = nn.functional.smooth_l1_loss(outputs["pred_traj"], batch["coords_norm"])
    cls_loss = cls_loss_fn(outputs["logits"], batch["group_id"])
    total_loss = coord_weight * coord_loss + traj_weight * traj_loss + cls_weight * cls_loss

    return {
        "total": total_loss,
        "coord": coord_loss,
        "traj": traj_loss,
        "cls": cls_loss,
    }


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def wrap_loader(
    loader: DataLoader,
    enabled: bool,
    desc: str,
):
    if enabled and tqdm is not None:
        return tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    return loader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    coord_mean: np.ndarray,
    coord_std: np.ndarray,
    group_coord_mean: np.ndarray,
    group_coord_std: np.ndarray,
    coord_weight: float,
    traj_weight: float,
    cls_weight: float,
    cls_loss_fn: nn.Module,
    show_progress: bool = False,
    desc: str = "Eval",
) -> Dict[str, object]:
    model.eval()

    losses = {"total": 0.0, "coord": 0.0, "traj": 0.0, "cls": 0.0}
    num_batches = 0

    all_errors: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []
    all_group_ids: List[np.ndarray] = []
    all_pred_ids: List[np.ndarray] = []

    coord_mean_t = torch.from_numpy(coord_mean).to(device)
    coord_std_t = torch.from_numpy(coord_std).to(device)
    group_coord_mean_t = torch.from_numpy(group_coord_mean).to(device)
    group_coord_std_t = torch.from_numpy(group_coord_std).to(device)

    with torch.no_grad():
        iterator = wrap_loader(loader, enabled=show_progress, desc=desc)
        for batch in iterator:
            batch = batch_to_device(batch, device)
            outputs = model(batch["x"], batch["motion_x"], batch["group_id"])
            loss_dict = compute_loss(
                outputs,
                batch,
                coord_weight=coord_weight,
                traj_weight=traj_weight,
                cls_weight=cls_weight,
                cls_loss_fn=cls_loss_fn,
            )

            for key in losses:
                losses[key] += float(loss_dict[key].item())
            num_batches += 1

            pred_ids = outputs["logits"].argmax(dim=1)
            pred_coord_candidates_raw = (
                outputs["pred_coord_candidates"] *
                group_coord_std_t.unsqueeze(0) +
                group_coord_mean_t.unsqueeze(0)
            )
            pred_coord_raw = pred_coord_candidates_raw[
                torch.arange(pred_coord_candidates_raw.size(0), device=device),
                pred_ids,
            ]
            error = torch.linalg.norm(pred_coord_raw - batch["y_last_raw"], dim=1)

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
    metrics = {
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
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    coord_weight: float,
    traj_weight: float,
    cls_weight: float,
    cls_loss_fn: nn.Module,
    grad_clip: float,
    show_progress: bool = False,
    desc: str = "Train",
) -> Dict[str, float]:
    model.train()
    losses = {"total": 0.0, "coord": 0.0, "traj": 0.0, "cls": 0.0}
    num_batches = 0

    iterator = wrap_loader(loader, enabled=show_progress, desc=desc)
    for batch in iterator:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(batch["x"], batch["motion_x"], batch["group_id"])
        loss_dict = compute_loss(
            outputs,
            batch,
            coord_weight=coord_weight,
            traj_weight=traj_weight,
            cls_weight=cls_weight,
            cls_loss_fn=cls_loss_fn,
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
            )

    return {key: value / max(1, num_batches) for key, value in losses.items()}


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> np.ndarray:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Test Confusion Matrix for (BUILDINGID, FLOOR)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return matrix


def save_confusion_matrix_csv(matrix: np.ndarray, class_names: List[str], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred"] + class_names)
        for name, row in zip(class_names, matrix):
            writer.writerow([name] + row.tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a hybrid 1D CNN + TCN model on the fixed IndoorPos dataset.",
    )
    parser.add_argument("--train-dir", type=str, default="training_dataset_fixed")
    parser.add_argument("--test-dir", type=str, default="test_dataset_fixed")
    parser.add_argument("--output-dir", type=str, default="runs/fixed_hybrid_cnn_tcn")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--coord-loss-weight", type=float, default=1.0)
    parser.add_argument("--traj-loss-weight", type=float, default=0.35)
    parser.add_argument("--cls-loss-weight", type=float, default=0.20)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_dir = resolve_dataset_dir(args.train_dir, "training_dataset")
    test_dir = resolve_dataset_dir(args.test_dir, "test_dataset")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_meta = load_metadata(train_dir / "metadata.json")
    test_meta = load_metadata(test_dir / "metadata.json")
    validate_selected_waps(train_meta, test_meta)

    test_arrays = concat_splits(
        [
            load_npz(test_dir / "train_sequences.npz"),
            load_npz(test_dir / "val_sequences.npz"),
        ]
    )

    test_arrays = limit_arrays(test_arrays, args.max_test_samples)
    test_motion_dim = int(build_motion_feature_array(test_arrays).shape[-1])

    device = torch.device("cpu") if args.cpu_only else select_device(prefer_mps=True)
    show_progress = not args.no_progress

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint.")

        checkpoint = torch.load(args.checkpoint, map_location=device)
        stats = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in checkpoint["stats"].items()
        }
        group_to_class = parse_group_to_class(checkpoint["group_to_class"])
        class_names = checkpoint["class_names"]
        motion_input_dim = int(checkpoint.get("motion_input_dim", test_motion_dim))

        test_loader = build_test_loader_from_stats(
            test_arrays=test_arrays,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            stats=stats,
            group_to_class=group_to_class,
        )

        print(f"Using device: {device}", flush=True)
        print(f"Eval-only test samples: {len(test_loader.dataset)}", flush=True)
        if show_progress and tqdm is None:
            print("tqdm not installed, falling back to epoch-level logs only.", flush=True)

        model = HybridCNNTCN(
            num_classes=len(class_names),
            motion_input_dim=motion_input_dim,
            dropout=args.dropout,
            cnn_channels=(32, 64, 128),
            motion_hidden_dim=64,
            tcn_hidden_dim=128,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            coord_mean=stats["coord_mean"],
            coord_std=stats["coord_std"],
            group_coord_mean=stats["group_coord_mean"],
            group_coord_std=stats["group_coord_std"],
            coord_weight=args.coord_loss_weight,
            traj_weight=args.traj_loss_weight,
            cls_weight=args.cls_loss_weight,
            cls_loss_fn=cls_loss_fn,
            show_progress=show_progress,
            desc="Test",
        )

        confusion_path = output_dir / "test_confusion_matrix.png"
        confusion_csv_path = output_dir / "test_confusion_matrix.csv"
        confusion = save_confusion_matrix(
            y_true=test_metrics["group_ids"],
            y_pred=test_metrics["pred_ids"],
            class_names=class_names,
            output_path=confusion_path,
        )
        save_confusion_matrix_csv(confusion, class_names, confusion_csv_path)

        metrics_payload = {
            "device": str(device),
            "checkpoint": args.checkpoint,
            "test_mean_error_m": test_metrics["mean_error_m"],
            "test_median_error_m": test_metrics["median_error_m"],
            "test_p75_error_m": test_metrics["p75_error_m"],
            "test_p90_error_m": test_metrics["p90_error_m"],
            "test_p95_error_m": test_metrics["p95_error_m"],
            "test_max_error_m": test_metrics["max_error_m"],
            "test_rmse_m": test_metrics["rmse_m"],
            "test_classification_accuracy": test_metrics["classification_accuracy"],
            "num_test_samples": int(len(test_loader.dataset)),
            "class_names": class_names,
        }
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)

        print("\nTest Results", flush=True)
        print(f"  Mean Error : {test_metrics['mean_error_m']:.3f} m", flush=True)
        print(f"  Median Error: {test_metrics['median_error_m']:.3f} m", flush=True)
        print(f"  P90 Error  : {test_metrics['p90_error_m']:.3f} m", flush=True)
        print(f"  P95 Error  : {test_metrics['p95_error_m']:.3f} m", flush=True)
        print(f"  RMSE       : {test_metrics['rmse_m']:.3f} m", flush=True)
        print(f"  Cls Acc    : {test_metrics['classification_accuracy']:.3f}", flush=True)
        print(f"  Confusion Matrix: {confusion_path}", flush=True)
        print(f"  Metrics JSON    : {output_dir / 'metrics.json'}", flush=True)
        return

    train_arrays = load_npz(train_dir / "train_sequences.npz")
    val_arrays = load_npz(train_dir / "val_sequences.npz")

    train_arrays = limit_arrays(train_arrays, args.max_train_samples)
    val_arrays = limit_arrays(val_arrays, args.max_val_samples)
    motion_input_dim = int(build_motion_feature_array(train_arrays).shape[-1])

    train_loader, val_loader, test_loader, stats, group_to_class, class_names = build_dataloaders(
        train_arrays=train_arrays,
        val_arrays=val_arrays,
        test_arrays=test_arrays,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Using device: {device}", flush=True)
    print(
        f"Train/Val/Test samples: {len(train_loader.dataset)}/"
        f"{len(val_loader.dataset)}/{len(test_loader.dataset)}",
        flush=True,
    )
    print(
        f"Early stopping enabled: patience={args.patience}, min_delta={args.min_delta}, "
        f"max_epochs={args.epochs}",
        flush=True,
    )
    if show_progress and tqdm is None:
        print("tqdm not installed, falling back to epoch-level logs only.", flush=True)

    model = HybridCNNTCN(
        num_classes=len(class_names),
        motion_input_dim=motion_input_dim,
        dropout=args.dropout,
        cnn_channels=(32, 64, 128),
        motion_hidden_dim=64,
        tcn_hidden_dim=128,
    ).to(device)

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
    cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    best_model_path = output_dir / "best_model.pt"
    history: List[Dict[str, float]] = []
    best_val_error = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_losses = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            coord_weight=args.coord_loss_weight,
            traj_weight=args.traj_loss_weight,
            cls_weight=args.cls_loss_weight,
            cls_loss_fn=cls_loss_fn,
            grad_clip=args.grad_clip,
            show_progress=show_progress,
            desc=f"Train {epoch}/{args.epochs}",
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            coord_mean=stats["coord_mean"],
            coord_std=stats["coord_std"],
            group_coord_mean=stats["group_coord_mean"],
            group_coord_std=stats["group_coord_std"],
            coord_weight=args.coord_loss_weight,
            traj_weight=args.traj_loss_weight,
            cls_weight=args.cls_loss_weight,
            cls_loss_fn=cls_loss_fn,
            show_progress=show_progress,
            desc=f"Val {epoch}/{args.epochs}",
        )

        scheduler.step(val_metrics["mean_error_m"])

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_total_loss": train_losses["total"],
            "train_coord_loss": train_losses["coord"],
            "train_traj_loss": train_losses["traj"],
            "train_cls_loss": train_losses["cls"],
            "val_total_loss": val_metrics["losses"]["total"],
            "val_coord_loss": val_metrics["losses"]["coord"],
            "val_traj_loss": val_metrics["losses"]["traj"],
            "val_cls_loss": val_metrics["losses"]["cls"],
            "val_mean_error_m": val_metrics["mean_error_m"],
            "val_rmse_m": val_metrics["rmse_m"],
            "val_cls_acc": val_metrics["classification_accuracy"],
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_losses['total']:.4f} | "
            f"val_loss={val_metrics['losses']['total']:.4f} | "
            f"val_mean_error={val_metrics['mean_error_m']:.3f} m | "
            f"val_rmse={val_metrics['rmse_m']:.3f} m | "
            f"val_cls_acc={val_metrics['classification_accuracy']:.3f} | "
            f"lr={current_lr:.2e}",
            flush=True,
        )

        if val_metrics["mean_error_m"] < best_val_error:
            best_val_error = val_metrics["mean_error_m"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "stats": {key: value.tolist() for key, value in stats.items()},
                    "group_to_class": {f"{k[0]}_{k[1]}": v for k, v in group_to_class.items()},
                    "class_names": class_names,
                    "motion_input_dim": motion_input_dim,
                    "selected_waps": train_meta.get("selected_waps") if train_meta else None,
                    "args": vars(args),
                    "best_val_error_m": best_val_error,
                },
                best_model_path,
            )

        if early_stopper.step(val_metrics["mean_error_m"]):
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        coord_mean=stats["coord_mean"],
        coord_std=stats["coord_std"],
        group_coord_mean=stats["group_coord_mean"],
        group_coord_std=stats["group_coord_std"],
        coord_weight=args.coord_loss_weight,
        traj_weight=args.traj_loss_weight,
        cls_weight=args.cls_loss_weight,
        cls_loss_fn=cls_loss_fn,
        show_progress=show_progress,
        desc="Test",
    )

    confusion_path = output_dir / "test_confusion_matrix.png"
    confusion_csv_path = output_dir / "test_confusion_matrix.csv"
    confusion = save_confusion_matrix(
        y_true=test_metrics["group_ids"],
        y_pred=test_metrics["pred_ids"],
        class_names=class_names,
        output_path=confusion_path,
    )
    save_confusion_matrix_csv(confusion, class_names, confusion_csv_path)

    metrics_payload = {
        "device": str(device),
        "best_val_error_m": float(best_val_error),
        "test_mean_error_m": test_metrics["mean_error_m"],
        "test_median_error_m": test_metrics["median_error_m"],
        "test_p75_error_m": test_metrics["p75_error_m"],
        "test_p90_error_m": test_metrics["p90_error_m"],
        "test_p95_error_m": test_metrics["p95_error_m"],
        "test_max_error_m": test_metrics["max_error_m"],
        "test_rmse_m": test_metrics["rmse_m"],
        "test_classification_accuracy": test_metrics["classification_accuracy"],
        "num_train_samples": int(len(train_loader.dataset)),
        "num_val_samples": int(len(val_loader.dataset)),
        "num_test_samples": int(len(test_loader.dataset)),
        "class_names": class_names,
        "history": history,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)

    with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    print("\nTest Results", flush=True)
    print(f"  Mean Error : {test_metrics['mean_error_m']:.3f} m", flush=True)
    print(f"  Median Error: {test_metrics['median_error_m']:.3f} m", flush=True)
    print(f"  P90 Error  : {test_metrics['p90_error_m']:.3f} m", flush=True)
    print(f"  P95 Error  : {test_metrics['p95_error_m']:.3f} m", flush=True)
    print(f"  RMSE       : {test_metrics['rmse_m']:.3f} m", flush=True)
    print(f"  Cls Acc    : {test_metrics['classification_accuracy']:.3f}", flush=True)
    print(f"  Confusion Matrix: {confusion_path}", flush=True)
    print(f"  Metrics JSON    : {output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
