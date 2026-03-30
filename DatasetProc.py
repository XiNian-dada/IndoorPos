#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Construct a pseudo-temporal WiFi fingerprint dataset from the raw CSV files in
archive/.

Designed for UJIIndoorLoc-like data:
- WAP001 ... WAP520
- LONGITUDE, LATITUDE
- FLOOR, BUILDINGID

This builder keeps the full coordinate trajectory for every sequence window
instead of only the final coordinate, and it also derives motion annotations
from the coordinate sequence.

By default, each time step is a real sampled fingerprint point from the CSV.
No synthetic interpolation frames are inserted unless interpolation_steps > 0.

Each output .npz contains:
    X:                [num_samples, seq_len, num_rssi_features]
    y / coords:       [num_samples, seq_len, 2]
    y_last:           [num_samples, 2]
    trajectory_id:    [num_samples]
    source_window_id: [num_samples]
    motion_features:  [num_samples, seq_len, num_motion_features]
    displacement:     [num_samples, seq_len, 2]
    velocity:         [num_samples, seq_len, 2]
    step_distance:    [num_samples, seq_len, 1]
    speed:            [num_samples, seq_len, 1]
    heading:          [num_samples, seq_len, 1]      # radians
    direction_vector: [num_samples, seq_len, 2]
    delta_t:          [num_samples, seq_len, 1]
    elapsed_time:     [num_samples, seq_len, 1]
    time_index:       [num_samples, seq_len, 1]
    motion_valid:     [num_samples, seq_len, 1]
    is_interpolated:  [num_samples, seq_len]
    source_index:     [num_samples, seq_len]
    group:            [num_samples, 2]               # (building_id, floor)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


# =========================
# Configuration
# =========================


@dataclass
class Config:
    input_csv: str = "archive/TrainingData.csv"
    output_dir: str = "training_dataset"
    selected_waps_json: str = ""

    random_seed: int = 42

    # sequence construction
    seq_len: int = 5
    trajectories_per_group: int = 2000
    interpolation_steps: int = 0

    # graph construction
    k_neighbors: int = 8
    max_neighbor_distance: float = 4.0
    min_transition_distance: float = 0.5
    min_group_size: int = 30

    # motion prior
    enforce_direction_consistency: bool = True
    direction_similarity_threshold: float = -0.2

    # feature filtering
    min_ap_presence_ratio: float = 0.02
    top_k_aps: int = 128

    # data augmentation
    add_rssi_noise: bool = True
    rssi_noise_std_dbm: float = 2.0
    ap_dropout_prob: float = 0.01
    apply_augmentation_to_real_frames_only: bool = False

    # normalization
    rssi_min: float = -100.0
    rssi_max: float = 0.0
    normalize_rssi_to_01: bool = True

    # synthetic motion definition
    synthetic_step_seconds: float = 1.0

    # split
    val_ratio: float = 0.2
    deduplicate_source_windows: bool = True
    split_by_trajectory: bool = True
    coord_key_round_decimals: int = 4


# =========================
# Utilities
# =========================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_rssi(x: np.ndarray, rssi_min: float, rssi_max: float) -> np.ndarray:
    x = np.clip(x, rssi_min, rssi_max)
    return (x - rssi_min) / (rssi_max - rssi_min)


def denormalize_rssi(x: np.ndarray, rssi_min: float, rssi_max: float) -> np.ndarray:
    return x * (rssi_max - rssi_min) + rssi_min


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < eps or norm_b < eps:
        return 1.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def linear_interpolate(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * a + alpha * b


def add_bool_argument(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool,
    help_text: str,
) -> None:
    option = name.replace("_", "-")
    parser.add_argument(f"--{option}", dest=name, action="store_true", help=help_text)
    parser.add_argument(
        f"--no-{option}",
        dest=name,
        action="store_false",
        help=f"Disable: {help_text}",
    )
    parser.set_defaults(**{name: default})


# =========================
# Core pipeline
# =========================


class PseudoTemporalBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.wap_cols: List[str] = []
        self.coord_cols = ["LONGITUDE", "LATITUDE"]
        self.group_cols = ["BUILDINGID", "FLOOR"]
        self.motion_feature_names = [
            "velocity_x",
            "velocity_y",
            "speed",
            "heading_sin",
            "heading_cos",
            "delta_t",
            "motion_valid",
        ]

    def load_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.input_csv)

        self.wap_cols = [col for col in df.columns if col.startswith("WAP")]
        required = set(self.coord_cols + self.group_cols)
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        # UJIIndoorLoc convention: 100 means no signal
        df[self.wap_cols] = df[self.wap_cols].replace(100, -100)

        df = df.dropna(subset=self.coord_cols + self.group_cols).reset_index(drop=True)
        return df

    def load_reference_waps(self) -> List[str] | None:
        if not self.cfg.selected_waps_json:
            return None

        with open(self.cfg.selected_waps_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            if "selected_waps" not in payload:
                raise ValueError(
                    "selected_waps_json points to a JSON object without 'selected_waps'."
                )
            selected_waps = payload["selected_waps"]
        elif isinstance(payload, list):
            selected_waps = payload
        else:
            raise ValueError("selected_waps_json must be a metadata JSON or a JSON list.")

        if not isinstance(selected_waps, list) or not all(isinstance(item, str) for item in selected_waps):
            raise ValueError("selected_waps_json must contain a string list of WAP column names.")

        missing = [wap for wap in selected_waps if wap not in self.wap_cols]
        if missing:
            raise ValueError(
                f"Reference selected_waps contains columns missing from current CSV: {missing[:20]}"
            )

        return selected_waps

    def select_useful_aps(self, df: pd.DataFrame) -> List[str]:
        reference_waps = self.load_reference_waps()
        if reference_waps is not None:
            return reference_waps

        visibility = (df[self.wap_cols].to_numpy() > -100).mean(axis=0)

        selected_pairs = [
            (col, float(visibility[idx]))
            for idx, col in enumerate(self.wap_cols)
            if visibility[idx] >= self.cfg.min_ap_presence_ratio
        ]

        if self.cfg.top_k_aps > 0 and len(selected_pairs) > self.cfg.top_k_aps:
            selected_pairs.sort(key=lambda item: item[1], reverse=True)
            selected_pairs = selected_pairs[: self.cfg.top_k_aps]

        selected = [col for col, _ in selected_pairs]
        if not selected:
            raise ValueError("No AP columns selected. Relax min_ap_presence_ratio or top_k_aps.")

        return selected

    def preprocess_features(
        self,
        df: pd.DataFrame,
        selected_waps: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rssi = df[selected_waps].to_numpy(dtype=np.float32)
        coords = df[self.coord_cols].to_numpy(dtype=np.float32)
        groups = df[self.group_cols].to_numpy(dtype=np.int32)

        if self.cfg.normalize_rssi_to_01:
            rssi = normalize_rssi(rssi, self.cfg.rssi_min, self.cfg.rssi_max).astype(np.float32)

        return rssi, coords, groups

    def build_group_indices(self, groups: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
        grouped: Dict[Tuple[int, int], List[int]] = {}
        for idx, (building_id, floor) in enumerate(groups):
            key = (int(building_id), int(floor))
            grouped.setdefault(key, []).append(idx)

        output: Dict[Tuple[int, int], np.ndarray] = {}
        for key, indices in grouped.items():
            if len(indices) >= self.cfg.min_group_size:
                output[key] = np.asarray(indices, dtype=np.int32)

        return output

    def build_knn_graph(self, coords: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if len(coords) < 2:
            raise ValueError("Need at least 2 points to build graph.")

        neighbors = NearestNeighbors(radius=self.cfg.max_neighbor_distance, metric="euclidean")
        neighbors.fit(coords)
        distances_list, indices_list = neighbors.radius_neighbors(coords, sort_results=True)

        neighbor_indices: List[np.ndarray] = []
        neighbor_distances: List[np.ndarray] = []
        for self_idx, (indices, distances) in enumerate(zip(indices_list, distances_list)):
            keep_mask = (indices != self_idx) & (distances >= self.cfg.min_transition_distance)
            indices = indices[keep_mask]
            distances = distances[keep_mask]

            if self.cfg.k_neighbors > 0 and len(indices) > self.cfg.k_neighbors:
                indices = indices[: self.cfg.k_neighbors]
                distances = distances[: self.cfg.k_neighbors]

            neighbor_indices.append(indices.astype(np.int32, copy=False))
            neighbor_distances.append(distances.astype(np.float32, copy=False))

        return neighbor_indices, neighbor_distances

    def choose_next_anchor(
        self,
        current_local_idx: int,
        previous_local_idx: int | None,
        coords_local: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_distances: np.ndarray,
    ) -> int | None:
        candidates: List[int] = []
        candidate_weights: List[float] = []

        current_coord = coords_local[current_local_idx]

        prev_direction = None
        if previous_local_idx is not None:
            prev_coord = coords_local[previous_local_idx]
            prev_direction = current_coord - prev_coord

        for neigh_local_idx, distance in zip(
            neighbor_indices[current_local_idx],
            neighbor_distances[current_local_idx],
        ):
            if distance > self.cfg.max_neighbor_distance:
                continue
            if distance < self.cfg.min_transition_distance:
                continue
            if neigh_local_idx == current_local_idx:
                continue
            if previous_local_idx is not None and neigh_local_idx == previous_local_idx:
                continue

            direction_ok = True
            if self.cfg.enforce_direction_consistency and prev_direction is not None:
                cand_direction = coords_local[neigh_local_idx] - current_coord
                similarity = cosine_similarity(prev_direction, cand_direction)
                if similarity < self.cfg.direction_similarity_threshold:
                    direction_ok = False

            if direction_ok:
                candidates.append(int(neigh_local_idx))
                candidate_weights.append(1.0 / (float(distance) + 1e-6))

        if not candidates:
            for neigh_local_idx, distance in zip(
                neighbor_indices[current_local_idx],
                neighbor_distances[current_local_idx],
            ):
                if distance > self.cfg.max_neighbor_distance:
                    continue
                if distance < self.cfg.min_transition_distance:
                    continue
                if neigh_local_idx == current_local_idx:
                    continue
                if previous_local_idx is not None and neigh_local_idx == previous_local_idx:
                    continue

                candidates.append(int(neigh_local_idx))
                candidate_weights.append(1.0 / (float(distance) + 1e-6))

        if not candidates:
            return None

        probs = np.asarray(candidate_weights, dtype=np.float64)
        probs /= probs.sum()
        return int(np.random.choice(candidates, p=probs))

    def generate_anchor_trajectory(
        self,
        coords_local: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_distances: np.ndarray,
        num_anchor_points: int,
    ) -> List[int] | None:
        if len(coords_local) < 2:
            return None

        start_idx = np.random.randint(0, len(coords_local))
        trajectory = [int(start_idx)]

        while len(trajectory) < num_anchor_points:
            current_idx = trajectory[-1]
            previous_idx = trajectory[-2] if len(trajectory) >= 2 else None

            next_idx = self.choose_next_anchor(
                current_local_idx=current_idx,
                previous_local_idx=previous_idx,
                coords_local=coords_local,
                neighbor_indices=neighbor_indices,
                neighbor_distances=neighbor_distances,
            )
            if next_idx is None:
                return None

            trajectory.append(next_idx)

        return trajectory

    def augment_rssi_frame(self, frame: np.ndarray) -> np.ndarray:
        x = frame.copy()

        if self.cfg.normalize_rssi_to_01:
            dbm = denormalize_rssi(x, self.cfg.rssi_min, self.cfg.rssi_max)

            if self.cfg.add_rssi_noise:
                dbm = dbm + np.random.normal(
                    loc=0.0,
                    scale=self.cfg.rssi_noise_std_dbm,
                    size=dbm.shape,
                ).astype(np.float32)

            if self.cfg.ap_dropout_prob > 0:
                visible_mask = dbm > self.cfg.rssi_min + 1e-6
                dropout_mask = (np.random.rand(*dbm.shape) < self.cfg.ap_dropout_prob) & visible_mask
                dbm[dropout_mask] = self.cfg.rssi_min

            dbm = np.clip(dbm, self.cfg.rssi_min, self.cfg.rssi_max)
            return normalize_rssi(dbm, self.cfg.rssi_min, self.cfg.rssi_max).astype(np.float32)

        if self.cfg.add_rssi_noise:
            x = x + np.random.normal(
                loc=0.0,
                scale=self.cfg.rssi_noise_std_dbm,
                size=x.shape,
            ).astype(np.float32)

        if self.cfg.ap_dropout_prob > 0:
            visible_mask = x > self.cfg.rssi_min + 1e-6
            dropout_mask = (np.random.rand(*x.shape) < self.cfg.ap_dropout_prob) & visible_mask
            x[dropout_mask] = self.cfg.rssi_min

        return np.clip(x, self.cfg.rssi_min, self.cfg.rssi_max).astype(np.float32)

    def interpolate_segment(
        self,
        rssi_a: np.ndarray,
        rssi_b: np.ndarray,
        coord_a: np.ndarray,
        coord_b: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool], List[int]]:
        rssi_frames: List[np.ndarray] = []
        coord_frames: List[np.ndarray] = []
        interpolated_flags: List[bool] = []
        source_indices: List[int] = []

        num_steps = self.cfg.interpolation_steps
        if num_steps <= 0:
            return rssi_frames, coord_frames, interpolated_flags, source_indices

        for step_idx in range(1, num_steps + 1):
            alpha = step_idx / (num_steps + 1)

            rssi_mid = linear_interpolate(rssi_a, rssi_b, alpha).astype(np.float32)
            coord_mid = linear_interpolate(coord_a, coord_b, alpha).astype(np.float32)

            rssi_frames.append(rssi_mid)
            coord_frames.append(coord_mid)
            interpolated_flags.append(True)
            source_indices.append(-1)

        return rssi_frames, coord_frames, interpolated_flags, source_indices

    def expand_anchor_trajectory_to_sequence(
        self,
        anchor_global_indices: List[int],
        all_rssi: np.ndarray,
        all_coords: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rssi_seq: List[np.ndarray] = []
        coord_seq: List[np.ndarray] = []
        interpolated_flags: List[bool] = []
        source_indices: List[int] = []

        for idx, global_idx in enumerate(anchor_global_indices):
            frame_rssi = all_rssi[global_idx].copy().astype(np.float32)
            frame_coord = all_coords[global_idx].copy().astype(np.float32)

            rssi_seq.append(frame_rssi)
            coord_seq.append(frame_coord)
            interpolated_flags.append(False)
            source_indices.append(int(global_idx))

            if idx < len(anchor_global_indices) - 1:
                next_global_idx = anchor_global_indices[idx + 1]
                inter_rssi, inter_coord, inter_flags, inter_sources = self.interpolate_segment(
                    all_rssi[global_idx],
                    all_rssi[next_global_idx],
                    all_coords[global_idx],
                    all_coords[next_global_idx],
                )
                rssi_seq.extend(inter_rssi)
                coord_seq.extend(inter_coord)
                interpolated_flags.extend(inter_flags)
                source_indices.extend(inter_sources)

        return (
            np.stack(rssi_seq, axis=0).astype(np.float32),
            np.stack(coord_seq, axis=0).astype(np.float32),
            np.asarray(interpolated_flags, dtype=np.bool_),
            np.asarray(source_indices, dtype=np.int32),
        )

    def build_motion_targets(self, coord_window: np.ndarray) -> Dict[str, np.ndarray]:
        seq_len = coord_window.shape[0]
        step_seconds = float(self.cfg.synthetic_step_seconds)
        if step_seconds <= 0.0:
            raise ValueError("synthetic_step_seconds must be > 0.")

        displacement = np.zeros_like(coord_window, dtype=np.float32)
        displacement[1:] = coord_window[1:] - coord_window[:-1]

        velocity = np.zeros_like(coord_window, dtype=np.float32)
        velocity[1:] = displacement[1:] / step_seconds

        step_distance = np.linalg.norm(displacement, axis=1, keepdims=True).astype(np.float32)

        speed = np.zeros((seq_len, 1), dtype=np.float32)
        speed[1:] = step_distance[1:] / step_seconds

        heading = np.zeros((seq_len, 1), dtype=np.float32)
        direction_vector = np.zeros_like(coord_window, dtype=np.float32)

        motion_valid = (step_distance > 1e-6).astype(np.bool_)
        moving = motion_valid[:, 0]
        if moving.any():
            heading[moving, 0] = np.arctan2(
                displacement[moving, 1],
                displacement[moving, 0],
            ).astype(np.float32)
            direction_vector[moving] = displacement[moving] / step_distance[moving]

        delta_t = np.full((seq_len, 1), step_seconds, dtype=np.float32)
        delta_t[0, 0] = 0.0

        elapsed_time = (
            np.arange(seq_len, dtype=np.float32).reshape(-1, 1) * step_seconds
        ).astype(np.float32)
        time_index = np.arange(seq_len, dtype=np.int32).reshape(-1, 1)

        return {
            "displacement": displacement,
            "velocity": velocity,
            "step_distance": step_distance,
            "speed": speed,
            "heading": heading,
            "direction_vector": direction_vector,
            "delta_t": delta_t,
            "elapsed_time": elapsed_time,
            "time_index": time_index,
            "motion_valid": motion_valid,
        }

    def build_motion_feature_tensor(
        self,
        motion: Dict[str, np.ndarray],
    ) -> np.ndarray:
        heading = motion["heading"].astype(np.float32)
        heading_sin = np.sin(heading).astype(np.float32)
        heading_cos = np.cos(heading).astype(np.float32)
        motion_valid = motion["motion_valid"].astype(np.float32)

        return np.concatenate(
            [
                motion["velocity"].astype(np.float32),
                motion["speed"].astype(np.float32),
                heading_sin,
                heading_cos,
                motion["delta_t"].astype(np.float32),
                motion_valid,
            ],
            axis=1,
        ).astype(np.float32)

    def cut_sliding_windows(
        self,
        rssi_seq: np.ndarray,
        coord_seq: np.ndarray,
        is_interpolated_seq: np.ndarray,
        source_index_seq: np.ndarray,
        group_key: Tuple[int, int],
        trajectory_id: int,
    ) -> List[Dict[str, np.ndarray]]:
        samples: List[Dict[str, np.ndarray]] = []

        total_len = len(rssi_seq)
        if total_len < self.cfg.seq_len:
            return samples

        for start in range(0, total_len - self.cfg.seq_len + 1):
            end = start + self.cfg.seq_len

            x_window = rssi_seq[start:end].astype(np.float32)
            coord_window = coord_seq[start:end].astype(np.float32)
            motion = self.build_motion_targets(coord_window)

            sample = {
                "X": x_window,
                "y": coord_window,
                "coords": coord_window,
                "y_last": coord_window[-1].copy().astype(np.float32),
                "group": np.asarray(group_key, dtype=np.int32),
                "trajectory_id": np.int32(trajectory_id),
                "is_interpolated": is_interpolated_seq[start:end].astype(np.bool_),
                "source_index": source_index_seq[start:end].astype(np.int32),
            }
            sample.update(motion)
            sample["motion_features"] = self.build_motion_feature_tensor(motion)
            samples.append(sample)

        return samples

    @staticmethod
    def stack_samples(samples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        keys = samples[0].keys()
        return {
            key: np.stack([sample[key] for sample in samples], axis=0)
            for key in keys
        }

    @staticmethod
    def slice_arrays(arrays: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
        return {key: value[indices] for key, value in arrays.items()}

    def save_split(self, path: str, arrays: Dict[str, np.ndarray]) -> None:
        np.savez_compressed(path, **arrays)

    def build_source_window_key(self, sample: Dict[str, np.ndarray]) -> Tuple[object, ...]:
        group = sample["group"].astype(np.int32)
        source_index = sample["source_index"].astype(np.int32)

        key: List[object] = [
            int(group[0]),
            int(group[1]),
            *[int(x) for x in source_index.tolist()],
        ]

        if self.cfg.coord_key_round_decimals >= 0:
            coord_flat = np.round(
                sample["coords"],
                decimals=self.cfg.coord_key_round_decimals,
            ).reshape(-1)
            key.extend(float(x) for x in coord_flat.tolist())

        return tuple(key)

    def deduplicate_and_assign_source_window_ids(
        self,
        samples: List[Dict[str, np.ndarray]],
    ) -> List[Dict[str, np.ndarray]]:
        if not samples:
            return samples

        if self.cfg.deduplicate_source_windows:
            seen: Set[Tuple[object, ...]] = set()
            deduplicated: List[Dict[str, np.ndarray]] = []
            for sample in samples:
                key = self.build_source_window_key(sample)
                if key in seen:
                    continue
                seen.add(key)
                deduplicated.append(sample)
        else:
            deduplicated = samples

        for source_window_id, sample in enumerate(deduplicated):
            sample["source_window_id"] = np.int32(source_window_id)

        return deduplicated

    def split_sample_indices(self, arrays: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        sample_indices = np.arange(arrays["X"].shape[0], dtype=np.int32)

        if self.cfg.split_by_trajectory and "trajectory_id" in arrays:
            trajectory_ids = arrays["trajectory_id"].astype(np.int32)
            unique_trajectory_ids = np.unique(trajectory_ids)

            if len(unique_trajectory_ids) >= 2:
                trajectory_groups: Dict[int, str] = {}
                for trajectory_id in unique_trajectory_ids.tolist():
                    first_index = int(np.where(trajectory_ids == trajectory_id)[0][0])
                    building, floor = arrays["group"][first_index]
                    trajectory_groups[int(trajectory_id)] = f"{int(building)}_{int(floor)}"

                trajectory_labels = np.asarray(
                    [trajectory_groups[int(trajectory_id)] for trajectory_id in unique_trajectory_ids.tolist()],
                    dtype=object,
                )
                stratify = (
                    trajectory_labels
                    if len(set(trajectory_labels.tolist())) > 1
                    else None
                )

                try:
                    train_trajectory_ids, val_trajectory_ids = train_test_split(
                        unique_trajectory_ids,
                        test_size=self.cfg.val_ratio,
                        random_state=self.cfg.random_seed,
                        shuffle=True,
                        stratify=stratify,
                    )
                except ValueError:
                    print("  Stratified trajectory split failed, fallback to random trajectory split.")
                    train_trajectory_ids, val_trajectory_ids = train_test_split(
                        unique_trajectory_ids,
                        test_size=self.cfg.val_ratio,
                        random_state=self.cfg.random_seed,
                        shuffle=True,
                        stratify=None,
                    )

                train_trajectory_set = set(
                    int(trajectory_id) for trajectory_id in train_trajectory_ids.tolist()
                )
                train_mask = np.asarray(
                    [int(trajectory_id) in train_trajectory_set for trajectory_id in trajectory_ids.tolist()],
                    dtype=np.bool_,
                )
                val_mask = ~train_mask

                train_indices = sample_indices[train_mask]
                val_indices = sample_indices[val_mask]

                if len(train_indices) > 0 and len(val_indices) > 0:
                    return train_indices, val_indices

                print("  Trajectory split produced empty subset, fallback to sample split.")

        stratify_labels = np.asarray(
            [f"{building}_{floor}" for building, floor in arrays["group"]],
            dtype=object,
        )
        stratify = stratify_labels if len(set(stratify_labels.tolist())) > 1 else None

        try:
            return train_test_split(
                sample_indices,
                test_size=self.cfg.val_ratio,
                random_state=self.cfg.random_seed,
                shuffle=True,
                stratify=stratify,
            )
        except ValueError:
            print("  Stratified sample split failed, fallback to random sample split.")
            return train_test_split(
                sample_indices,
                test_size=self.cfg.val_ratio,
                random_state=self.cfg.random_seed,
                shuffle=True,
                stratify=None,
            )

    def apply_rssi_augmentation_to_arrays(self, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not (self.cfg.add_rssi_noise or self.cfg.ap_dropout_prob > 0):
            return arrays

        augmented = {
            key: value.copy()
            for key, value in arrays.items()
        }
        x = augmented["X"]
        is_interpolated = augmented["is_interpolated"]

        for sample_idx in range(x.shape[0]):
            for frame_idx in range(x.shape[1]):
                if self.cfg.apply_augmentation_to_real_frames_only and bool(
                    is_interpolated[sample_idx, frame_idx]
                ):
                    continue
                x[sample_idx, frame_idx] = self.augment_rssi_frame(x[sample_idx, frame_idx])

        augmented["X"] = x.astype(np.float32)
        return augmented

    @staticmethod
    def count_shared_source_windows(
        train_arrays: Dict[str, np.ndarray],
        val_arrays: Dict[str, np.ndarray],
    ) -> int:
        train_keys = {
            tuple(int(x) for x in row.tolist())
            for row in train_arrays["source_index"]
        }
        val_keys = {
            tuple(int(x) for x in row.tolist())
            for row in val_arrays["source_index"]
        }
        return int(len(train_keys.intersection(val_keys)))

    def build_metadata(
        self,
        arrays: Dict[str, np.ndarray],
        train_arrays: Dict[str, np.ndarray],
        val_arrays: Dict[str, np.ndarray],
        selected_waps: List[str],
        group_to_indices: Dict[Tuple[int, int], np.ndarray],
        deduplicated_windows_removed: int,
        shared_source_windows_train_val: int,
        shared_trajectories_train_val: int,
    ) -> Dict[str, object]:
        return {
            "config": asdict(self.cfg),
            "selected_waps": selected_waps,
            "num_total_sequences": int(arrays["X"].shape[0]),
            "num_train_sequences": int(train_arrays["X"].shape[0]),
            "num_val_sequences": int(val_arrays["X"].shape[0]),
            "num_unique_trajectories": int(np.unique(arrays["trajectory_id"]).shape[0]),
            "num_unique_source_windows": int(np.unique(arrays["source_window_id"]).shape[0]),
            "deduplicated_windows_removed": int(deduplicated_windows_removed),
            "split_quality": {
                "split_by_trajectory": bool(self.cfg.split_by_trajectory),
                "shared_source_windows_train_val": int(shared_source_windows_train_val),
                "shared_trajectory_ids_train_val": int(shared_trajectories_train_val),
            },
            "sample_shapes": {key: list(value.shape[1:]) for key, value in arrays.items()},
            "sample_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
            "usable_groups": {
                f"{key[0]}_{key[1]}": int(len(indices))
                for key, indices in group_to_indices.items()
            },
            "motion_definition": {
                "trajectories_are_pseudo_temporal": True,
                "coordinate_columns": self.coord_cols,
                "time_step_seconds": self.cfg.synthetic_step_seconds,
                "min_transition_distance": self.cfg.min_transition_distance,
                "motion_features_are_proxy_signals": True,
                "motion_features_note": "Derived from coordinate differences as a proxy for production motion sensors.",
                "velocity_unit": "coordinate_units_per_second",
                "step_distance_unit": "coordinate_units",
                "heading_unit": "radians",
                "heading_zero_when_stationary": True,
                "motion_valid_marks_non_zero_displacement": True,
                "is_interpolated_marks_synthetic_frames": True,
                "source_index_is_minus_one_for_interpolated_frames": True,
                "dataset_split_applies_augmentation_to_train_only": True,
            },
            "array_descriptions": {
                "X": "RSSI sequence after AP filtering and optional normalization.",
                "y": "Full coordinate sequence target for each time step.",
                "coords": "Alias of y for explicit coordinate access.",
                "y_last": "Final coordinate in the window, kept for convenience.",
                "trajectory_id": "Pseudo-trajectory identifier before sliding-window extraction.",
                "source_window_id": "Unique source-window id after optional deduplication.",
                "motion_features": "Packed motion feature sequence aligned with X for production-like model input.",
                "displacement": "Coordinate difference from previous frame.",
                "velocity": "Displacement divided by synthetic_step_seconds.",
                "step_distance": "Euclidean norm of displacement.",
                "speed": "Scalar speed derived from step_distance.",
                "heading": "atan2(dy, dx) in radians; 0 when stationary.",
                "direction_vector": "Unit direction vector; zero when stationary.",
                "delta_t": "Time gap to previous frame; first frame is 0.",
                "elapsed_time": "Elapsed synthetic time from the window start.",
                "time_index": "Integer time index within the window.",
                "motion_valid": "Whether the frame has non-zero displacement.",
                "is_interpolated": "Whether the frame was inserted by interpolation.",
                "source_index": "Original CSV row index for real frames, -1 for interpolated frames.",
                "group": "Building/floor identifier.",
            },
            "motion_feature_names": self.motion_feature_names,
        }

    def build(self) -> None:
        if self.cfg.synthetic_step_seconds <= 0.0:
            raise ValueError("synthetic_step_seconds must be > 0.")
        if not 0.0 < self.cfg.val_ratio < 1.0:
            raise ValueError("val_ratio must be in (0, 1).")

        set_seed(self.cfg.random_seed)
        ensure_dir(self.cfg.output_dir)

        print("[1/8] Loading CSV...")
        df = self.load_dataframe()

        print("[2/8] Selecting useful APs...")
        selected_waps = self.select_useful_aps(df)
        print(f"Selected APs: {len(selected_waps)}")

        print("[3/8] Preprocessing features...")
        all_rssi, all_coords, all_groups = self.preprocess_features(df, selected_waps)

        print("[4/8] Grouping by building/floor...")
        group_to_indices = self.build_group_indices(all_groups)
        print(f"Usable groups: {len(group_to_indices)}")

        all_samples: List[Dict[str, np.ndarray]] = []
        trajectory_counter = 0

        print("[5/8] Generating trajectories and sequences...")
        for group_key, global_indices in group_to_indices.items():
            local_coords = all_coords[global_indices]
            neighbor_indices, neighbor_distances = self.build_knn_graph(local_coords)

            interp = self.cfg.interpolation_steps
            anchors_needed = math.ceil((self.cfg.seq_len + interp) / (interp + 1))

            generated_for_group = 0
            attempts = 0
            max_attempts = self.cfg.trajectories_per_group * 10

            while generated_for_group < self.cfg.trajectories_per_group and attempts < max_attempts:
                attempts += 1

                local_anchor_traj = self.generate_anchor_trajectory(
                    coords_local=local_coords,
                    neighbor_indices=neighbor_indices,
                    neighbor_distances=neighbor_distances,
                    num_anchor_points=anchors_needed,
                )
                if local_anchor_traj is None:
                    continue

                global_anchor_traj = [
                    int(global_indices[local_idx])
                    for local_idx in local_anchor_traj
                ]

                rssi_seq, coord_seq, is_interpolated_seq, source_index_seq = (
                    self.expand_anchor_trajectory_to_sequence(
                        anchor_global_indices=global_anchor_traj,
                        all_rssi=all_rssi,
                        all_coords=all_coords,
                    )
                )

                samples = self.cut_sliding_windows(
                    rssi_seq=rssi_seq,
                    coord_seq=coord_seq,
                    is_interpolated_seq=is_interpolated_seq,
                    source_index_seq=source_index_seq,
                    group_key=group_key,
                    trajectory_id=trajectory_counter,
                )
                trajectory_counter += 1
                if not samples:
                    continue

                all_samples.extend(samples)
                generated_for_group += 1

            print(f"  Group {group_key}: generated {generated_for_group} trajectories")

        if not all_samples:
            raise RuntimeError("No sequences generated. Relax graph constraints or inspect the data.")

        print("[6/8] Deduplicating source windows...")
        num_samples_before_dedup = len(all_samples)
        all_samples = self.deduplicate_and_assign_source_window_ids(all_samples)
        deduplicated_windows_removed = num_samples_before_dedup - len(all_samples)
        print(
            f"  Kept {len(all_samples)} windows, removed {deduplicated_windows_removed} duplicates"
        )

        arrays = self.stack_samples(all_samples)

        print("[7/8] Splitting train/val...")
        train_indices, val_indices = self.split_sample_indices(arrays)
        train_arrays_no_aug = self.slice_arrays(arrays, train_indices)
        val_arrays = self.slice_arrays(arrays, val_indices)
        train_arrays = self.apply_rssi_augmentation_to_arrays(train_arrays_no_aug)

        shared_source_windows_train_val = self.count_shared_source_windows(train_arrays, val_arrays)
        shared_trajectories_train_val = int(
            len(
                set(int(x) for x in train_arrays["trajectory_id"].tolist()).intersection(
                    int(x) for x in val_arrays["trajectory_id"].tolist()
                )
            )
        )
        print(
            "  Shared train/val windows after split:",
            shared_source_windows_train_val,
        )

        print("[8/8] Saving...")
        self.save_split(os.path.join(self.cfg.output_dir, "train_sequences.npz"), train_arrays)
        self.save_split(os.path.join(self.cfg.output_dir, "val_sequences.npz"), val_arrays)

        metadata = self.build_metadata(
            arrays=arrays,
            train_arrays=train_arrays,
            val_arrays=val_arrays,
            selected_waps=selected_waps,
            group_to_indices=group_to_indices,
            deduplicated_windows_removed=deduplicated_windows_removed,
            shared_source_windows_train_val=shared_source_windows_train_val,
            shared_trajectories_train_val=shared_trajectories_train_val,
        )
        with open(os.path.join(self.cfg.output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        print("Done.")
        print(f"Train X: {train_arrays['X'].shape}, y: {train_arrays['y'].shape}")
        print(f"Val   X: {val_arrays['X'].shape}, y: {val_arrays['y'].shape}")
        print(f"Saved to: {self.cfg.output_dir}")


# =========================
# CLI
# =========================


def parse_args() -> Config:
    default_cfg = Config()

    parser = argparse.ArgumentParser(
        description="Build a pseudo-temporal WiFi dataset with full coordinate trajectories.",
    )
    parser.add_argument("--input-csv", type=str, default=default_cfg.input_csv)
    parser.add_argument("--output-dir", type=str, default=default_cfg.output_dir)
    parser.add_argument("--selected-waps-json", type=str, default=default_cfg.selected_waps_json)
    parser.add_argument("--random-seed", type=int, default=default_cfg.random_seed)

    parser.add_argument("--seq-len", type=int, default=default_cfg.seq_len)
    parser.add_argument(
        "--trajectories-per-group",
        type=int,
        default=default_cfg.trajectories_per_group,
    )
    parser.add_argument(
        "--interpolation-steps",
        type=int,
        default=default_cfg.interpolation_steps,
    )

    parser.add_argument("--k-neighbors", type=int, default=default_cfg.k_neighbors)
    parser.add_argument(
        "--max-neighbor-distance",
        type=float,
        default=default_cfg.max_neighbor_distance,
    )
    parser.add_argument(
        "--min-transition-distance",
        type=float,
        default=default_cfg.min_transition_distance,
    )
    parser.add_argument("--min-group-size", type=int, default=default_cfg.min_group_size)

    add_bool_argument(
        parser,
        "enforce_direction_consistency",
        default_cfg.enforce_direction_consistency,
        "Require locally consistent movement direction when sampling anchor points.",
    )
    parser.add_argument(
        "--direction-similarity-threshold",
        type=float,
        default=default_cfg.direction_similarity_threshold,
    )

    parser.add_argument(
        "--min-ap-presence-ratio",
        type=float,
        default=default_cfg.min_ap_presence_ratio,
    )
    parser.add_argument("--top-k-aps", type=int, default=default_cfg.top_k_aps)

    add_bool_argument(
        parser,
        "add_rssi_noise",
        default_cfg.add_rssi_noise,
        "Apply Gaussian RSSI noise augmentation.",
    )
    parser.add_argument(
        "--rssi-noise-std-dbm",
        type=float,
        default=default_cfg.rssi_noise_std_dbm,
    )
    parser.add_argument(
        "--ap-dropout-prob",
        type=float,
        default=default_cfg.ap_dropout_prob,
    )
    add_bool_argument(
        parser,
        "apply_augmentation_to_real_frames_only",
        default_cfg.apply_augmentation_to_real_frames_only,
        "Apply RSSI augmentation only on real anchor frames, not interpolated frames.",
    )

    parser.add_argument("--rssi-min", type=float, default=default_cfg.rssi_min)
    parser.add_argument("--rssi-max", type=float, default=default_cfg.rssi_max)
    add_bool_argument(
        parser,
        "normalize_rssi_to_01",
        default_cfg.normalize_rssi_to_01,
        "Normalize RSSI values into [0, 1].",
    )

    parser.add_argument(
        "--synthetic-step-seconds",
        type=float,
        default=default_cfg.synthetic_step_seconds,
    )
    parser.add_argument("--val-ratio", type=float, default=default_cfg.val_ratio)
    add_bool_argument(
        parser,
        "deduplicate_source_windows",
        default_cfg.deduplicate_source_windows,
        "Drop duplicated source windows before train/val split.",
    )
    add_bool_argument(
        parser,
        "split_by_trajectory",
        default_cfg.split_by_trajectory,
        "Split train/val at trajectory level to avoid leakage.",
    )
    parser.add_argument(
        "--coord-key-round-decimals",
        type=int,
        default=default_cfg.coord_key_round_decimals,
    )

    args = parser.parse_args()

    return Config(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        selected_waps_json=args.selected_waps_json,
        random_seed=args.random_seed,
        seq_len=args.seq_len,
        trajectories_per_group=args.trajectories_per_group,
        interpolation_steps=args.interpolation_steps,
        k_neighbors=args.k_neighbors,
        max_neighbor_distance=args.max_neighbor_distance,
        min_transition_distance=args.min_transition_distance,
        min_group_size=args.min_group_size,
        enforce_direction_consistency=args.enforce_direction_consistency,
        direction_similarity_threshold=args.direction_similarity_threshold,
        min_ap_presence_ratio=args.min_ap_presence_ratio,
        top_k_aps=args.top_k_aps,
        add_rssi_noise=args.add_rssi_noise,
        rssi_noise_std_dbm=args.rssi_noise_std_dbm,
        ap_dropout_prob=args.ap_dropout_prob,
        apply_augmentation_to_real_frames_only=args.apply_augmentation_to_real_frames_only,
        rssi_min=args.rssi_min,
        rssi_max=args.rssi_max,
        normalize_rssi_to_01=args.normalize_rssi_to_01,
        synthetic_step_seconds=args.synthetic_step_seconds,
        val_ratio=args.val_ratio,
        deduplicate_source_windows=args.deduplicate_source_windows,
        split_by_trajectory=args.split_by_trajectory,
        coord_key_round_decimals=args.coord_key_round_decimals,
    )


def main() -> None:
    builder = PseudoTemporalBuilder(parse_args())
    builder.build()


if __name__ == "__main__":
    main()
