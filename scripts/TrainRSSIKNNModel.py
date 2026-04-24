#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


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


def encode_groups(groups: np.ndarray) -> np.ndarray:
    return (groups[:, 0].astype(np.int32) * 10 + groups[:, 1].astype(np.int32)).astype(np.int32)


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


def build_rssi_features(arrays: Dict[str, np.ndarray], feature_set: str) -> np.ndarray:
    x = arrays["X"].astype(np.float32)
    if feature_set == "last":
        return x[:, -1, :].astype(np.float32)
    if feature_set == "mean":
        return x.mean(axis=1).astype(np.float32)
    if feature_set == "flatten":
        return x.reshape(len(x), -1).astype(np.float32)
    if feature_set == "last_mean":
        last = x[:, -1, :]
        mean = x.mean(axis=1)
        return np.concatenate([last, mean], axis=1).astype(np.float32)
    if feature_set == "robust_last_mean":
        last = x[:, -1, :]
        mean = x.mean(axis=1)
        mask_last = (last > 1e-6).astype(np.float32)
        mask_mean = (mean > 1e-6).astype(np.float32)
        cnt_last = mask_last.mean(axis=1, keepdims=True).astype(np.float32)
        cnt_mean = mask_mean.mean(axis=1, keepdims=True).astype(np.float32)
        return np.concatenate(
            [last, mean, mask_last, mask_mean, cnt_last, cnt_mean],
            axis=1,
        ).astype(np.float32)
    raise ValueError(
        f"Unknown feature_set='{feature_set}', expected "
        "last|mean|flatten|last_mean|robust_last_mean."
    )


def parse_int_list(raw: str) -> List[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("k-candidates is empty.")
    return sorted(set(values))


def parse_float_list(raw: str) -> List[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("float list is empty.")
    return sorted(set(values))


@dataclass
class Candidate:
    k: int
    weighted: bool
    group_aware: bool

    @property
    def name(self) -> str:
        return f"k{self.k}_{'w' if self.weighted else 'avg'}_{'ga' if self.group_aware else 'global'}"


@dataclass
class TemporalFilterConfig:
    method: str
    window: int
    ema_alpha: float

    @property
    def name(self) -> str:
        return f"{self.method}_w{self.window}_a{self.ema_alpha:.2f}"


class RSSIKNNLocalizer:
    def __init__(
        self,
        train_features: np.ndarray,
        train_coords: np.ndarray,
        train_group_ids: np.ndarray,
        max_k: int,
    ) -> None:
        self.train_features = train_features.astype(np.float32)
        self.train_coords = train_coords.astype(np.float32)
        self.train_group_ids = train_group_ids.astype(np.int32)
        self.max_k = int(max_k)

        self.global_nn = NearestNeighbors(
            n_neighbors=min(self.max_k, len(self.train_features)),
            metric="euclidean",
        )
        self.global_nn.fit(self.train_features)

        self.group_indices: Dict[int, np.ndarray] = {}
        self.group_nn: Dict[int, NearestNeighbors] = {}
        for group_id in sorted(set(int(x) for x in self.train_group_ids.tolist())):
            idx = np.where(self.train_group_ids == group_id)[0]
            if len(idx) == 0:
                continue
            self.group_indices[group_id] = idx
            nn = NearestNeighbors(
                n_neighbors=min(self.max_k, len(idx)),
                metric="euclidean",
            )
            nn.fit(self.train_features[idx])
            self.group_nn[group_id] = nn

    @staticmethod
    def _weighted_average(coords: np.ndarray, dist: np.ndarray, weighted: bool) -> np.ndarray:
        if not weighted:
            return coords.mean(axis=1).astype(np.float32)
        w = 1.0 / (dist + 1e-6)
        w = w / w.sum(axis=1, keepdims=True)
        return (coords * w[:, :, None]).sum(axis=1).astype(np.float32)

    def _predict_groups(self, neighbor_idx: np.ndarray, neighbor_dist: np.ndarray) -> np.ndarray:
        groups = self.train_group_ids[neighbor_idx]
        weights = 1.0 / (neighbor_dist + 1e-6)
        pred = np.zeros((len(groups),), dtype=np.int32)
        for i in range(len(groups)):
            uniq = np.unique(groups[i])
            score = np.zeros((len(uniq),), dtype=np.float64)
            for j, g in enumerate(uniq):
                score[j] = weights[i][groups[i] == g].sum()
            pred[i] = int(uniq[np.argmax(score)])
        return pred

    def predict(
        self,
        features: np.ndarray,
        k: int,
        weighted: bool,
        group_aware: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        k_use = max(1, min(int(k), self.max_k))
        dist_global, idx_global = self.global_nn.kneighbors(
            features,
            n_neighbors=k_use,
            return_distance=True,
        )
        pred_groups = self._predict_groups(idx_global, dist_global)

        if not group_aware:
            pred_coords = self._weighted_average(self.train_coords[idx_global], dist_global, weighted=weighted)
            return pred_coords, pred_groups

        pred_coords = np.zeros((len(features), 2), dtype=np.float32)
        for group_id in sorted(set(int(x) for x in pred_groups.tolist())):
            sample_idx = np.where(pred_groups == group_id)[0]
            if len(sample_idx) == 0:
                continue
            if group_id not in self.group_nn:
                pred_coords[sample_idx] = self._weighted_average(
                    self.train_coords[idx_global[sample_idx]],
                    dist_global[sample_idx],
                    weighted=weighted,
                )
                continue
            group_nn = self.group_nn[group_id]
            group_map = self.group_indices[group_id]
            dist_local, idx_local = group_nn.kneighbors(
                features[sample_idx],
                n_neighbors=min(k_use, len(group_map)),
                return_distance=True,
            )
            idx_ref = group_map[idx_local]
            pred_coords[sample_idx] = self._weighted_average(
                self.train_coords[idx_ref],
                dist_local,
                weighted=weighted,
            )
        return pred_coords, pred_groups


def build_sequence_groups(arrays: Dict[str, np.ndarray]) -> List[np.ndarray]:
    trajectory = arrays["trajectory_id"].astype(np.int64)
    if "time_index" in arrays:
        order_key = arrays["time_index"][:, -1, 0].astype(np.int64)
    elif "source_index" in arrays:
        order_key = arrays["source_index"][:, -1].astype(np.int64)
    else:
        order_key = np.arange(len(trajectory), dtype=np.int64)

    groups: List[np.ndarray] = []
    for traj_id in np.unique(trajectory):
        idx = np.where(trajectory == traj_id)[0]
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(order_key[idx], kind="mergesort")]
        groups.append(idx.astype(np.int64))
    return groups


def summarize_sequence_groups(sequence_groups: List[np.ndarray]) -> Dict[str, float]:
    if not sequence_groups:
        return {
            "num_groups": 0.0,
            "mean_len": 0.0,
            "max_len": 0.0,
            "min_len": 0.0,
            "groups_ge2": 0.0,
        }
    lengths = np.asarray([len(x) for x in sequence_groups], dtype=np.float32)
    return {
        "num_groups": float(len(sequence_groups)),
        "mean_len": float(lengths.mean()),
        "max_len": float(lengths.max()),
        "min_len": float(lengths.min()),
        "groups_ge2": float((lengths >= 2).sum()),
    }


def _rolling_mean_2d(points: np.ndarray, window: int) -> np.ndarray:
    n = len(points)
    if n == 0:
        return points
    out = np.zeros_like(points, dtype=np.float32)
    half = max(0, window // 2)
    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        out[i] = points[left:right].mean(axis=0).astype(np.float32)
    return out


def _rolling_median_2d(points: np.ndarray, window: int) -> np.ndarray:
    n = len(points)
    if n == 0:
        return points
    out = np.zeros_like(points, dtype=np.float32)
    half = max(0, window // 2)
    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        out[i] = np.median(points[left:right], axis=0).astype(np.float32)
    return out


def _ema_2d(points: np.ndarray, alpha: float) -> np.ndarray:
    n = len(points)
    if n == 0:
        return points
    a = float(np.clip(alpha, 1e-3, 1.0))
    out = np.zeros_like(points, dtype=np.float32)
    out[0] = points[0].astype(np.float32)
    for i in range(1, n):
        out[i] = (a * points[i] + (1.0 - a) * out[i - 1]).astype(np.float32)
    return out


def apply_temporal_filter(
    pred_coords: np.ndarray,
    sequence_groups: List[np.ndarray],
    cfg: TemporalFilterConfig,
) -> np.ndarray:
    method = cfg.method.lower()
    window = max(1, int(cfg.window))
    if window % 2 == 0:
        window += 1
    out = pred_coords.astype(np.float32).copy()

    for idx in sequence_groups:
        seq = out[idx]
        if method == "none":
            filtered = seq
        elif method == "mean":
            filtered = _rolling_mean_2d(seq, window=window)
        elif method == "median":
            filtered = _rolling_median_2d(seq, window=window)
        elif method == "ema":
            filtered = _ema_2d(seq, alpha=cfg.ema_alpha)
        elif method == "hybrid":
            med = _rolling_median_2d(seq, window=window)
            filtered = _ema_2d(med, alpha=cfg.ema_alpha)
        else:
            raise ValueError(
                f"Unknown temporal filter method='{cfg.method}', expected "
                "none|mean|median|ema|hybrid."
            )
        out[idx] = filtered.astype(np.float32)
    return out


def evaluate(
    localizer: RSSIKNNLocalizer,
    features: np.ndarray,
    coords: np.ndarray,
    groups: np.ndarray,
    candidate: Candidate,
) -> Dict[str, object]:
    pred_coords, pred_groups = localizer.predict(
        features=features,
        k=candidate.k,
        weighted=candidate.weighted,
        group_aware=candidate.group_aware,
    )
    reg = regression_metrics(pred_coords, coords)
    cls_acc = float((pred_groups == groups).mean())
    return {
        "regression": reg,
        "classification_accuracy": cls_acc,
        "pred_coords": pred_coords,
        "pred_groups": pred_groups,
    }


def parse_temporal_methods(raw: str) -> List[str]:
    methods = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not methods:
        raise ValueError("temporal methods list is empty.")
    valid = {"none", "mean", "median", "ema", "hybrid"}
    for m in methods:
        if m not in valid:
            raise ValueError(
                f"Unsupported temporal method '{m}', expected one of {sorted(valid)}."
            )
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate RSSI fingerprint KNN indoor localization baseline.",
    )
    parser.add_argument("--train-dir", type=str, default="training_dataset")
    parser.add_argument("--test-dir", type=str, default="test_dataset")
    parser.add_argument("--output-dir", type=str, default="runs/rssi_knn")
    parser.add_argument(
        "--feature-set",
        type=str,
        default="last_mean",
        choices=["last", "mean", "flatten", "last_mean", "robust_last_mean"],
    )
    parser.add_argument("--k-candidates", type=str, default="1,3,5,7,9,11,15,21")
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--group-aware", action="store_true")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--enable-temporal-filter", action="store_true")
    parser.add_argument("--temporal-method", type=str, default="hybrid")
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--temporal-ema-alpha", type=float, default=0.35)
    parser.add_argument("--auto-tune-temporal", action="store_true")
    parser.add_argument("--temporal-method-candidates", type=str, default="median,ema,hybrid")
    parser.add_argument("--temporal-window-candidates", type=str, default="3,5,7")
    parser.add_argument("--temporal-ema-alpha-candidates", type=str, default="0.25,0.35,0.5")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    train_features = build_rssi_features(train_arrays, feature_set=args.feature_set)
    val_features = build_rssi_features(val_arrays, feature_set=args.feature_set)
    test_features = build_rssi_features(test_arrays, feature_set=args.feature_set)

    train_coords = train_arrays["y_last"].astype(np.float32)
    val_coords = val_arrays["y_last"].astype(np.float32)
    test_coords = test_arrays["y_last"].astype(np.float32)

    train_groups = encode_groups(train_arrays["group"])
    val_groups = encode_groups(val_arrays["group"])
    test_groups = encode_groups(test_arrays["group"])

    scaler: Optional[StandardScaler] = None
    if not args.no_scale:
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features).astype(np.float32)
        val_features = scaler.transform(val_features).astype(np.float32)
        test_features = scaler.transform(test_features).astype(np.float32)

    k_list = parse_int_list(args.k_candidates)
    max_k = max(k_list)
    localizer = RSSIKNNLocalizer(
        train_features=train_features,
        train_coords=train_coords,
        train_group_ids=train_groups,
        max_k=max_k,
    )

    candidates: List[Candidate] = []
    for k in k_list:
        candidates.append(
            Candidate(
                k=int(k),
                weighted=bool(args.weighted),
                group_aware=bool(args.group_aware),
            )
        )

    candidate_summaries: List[Dict[str, object]] = []
    best_score = None
    best_candidate = None
    for c in candidates:
        val_out = evaluate(
            localizer=localizer,
            features=val_features,
            coords=val_coords,
            groups=val_groups,
            candidate=c,
        )
        reg = val_out["regression"]
        row = {
            "candidate": c.name,
            "k": c.k,
            "weighted": c.weighted,
            "group_aware": c.group_aware,
            "val_mean_error_m": float(reg["mean_error_m"]),
            "val_p90_error_m": float(reg["p90_error_m"]),
            "val_rmse_m": float(reg["rmse_m"]),
            "val_cls_acc": float(val_out["classification_accuracy"]),
        }
        candidate_summaries.append(row)

        score = (
            float(reg["mean_error_m"]),
            float(reg["p90_error_m"]),
            float(reg["rmse_m"]),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_candidate = c

    if best_candidate is None:
        raise RuntimeError("No KNN candidate evaluated.")

    val_best = evaluate(
        localizer=localizer,
        features=val_features,
        coords=val_coords,
        groups=val_groups,
        candidate=best_candidate,
    )
    test_best = evaluate(
        localizer=localizer,
        features=test_features,
        coords=test_coords,
        groups=test_groups,
        candidate=best_candidate,
    )

    val_seq_groups = build_sequence_groups(val_arrays)
    test_seq_groups = build_sequence_groups(test_arrays)
    val_seq_stats = summarize_sequence_groups(val_seq_groups)
    test_seq_stats = summarize_sequence_groups(test_seq_groups)
    temporal_effective = (
        val_seq_stats["groups_ge2"] > 0.0 and
        test_seq_stats["groups_ge2"] > 0.0
    )

    temporal_cfg = TemporalFilterConfig(
        method=args.temporal_method,
        window=args.temporal_window,
        ema_alpha=args.temporal_ema_alpha,
    )
    temporal_tuned = False
    temporal_tuning_rows: List[Dict[str, object]] = []

    if args.enable_temporal_filter and (not temporal_effective):
        print(
            "Warning: temporal filter enabled but no multi-step trajectories found "
            "(all sequence groups have length 1), so offline metrics will be unchanged.",
            flush=True,
        )

    if args.enable_temporal_filter and temporal_effective:
        if args.auto_tune_temporal:
            temporal_methods = parse_temporal_methods(args.temporal_method_candidates)
            temporal_windows = parse_int_list(args.temporal_window_candidates)
            temporal_alphas = parse_float_list(args.temporal_ema_alpha_candidates)
            best_temporal_score = None
            best_temporal_cfg = temporal_cfg

            for method in temporal_methods:
                for window in temporal_windows:
                    if method in {"ema", "none"}:
                        alpha_iter = temporal_alphas[:1]
                    else:
                        alpha_iter = temporal_alphas
                    for alpha in alpha_iter:
                        cfg = TemporalFilterConfig(
                            method=method,
                            window=int(window),
                            ema_alpha=float(alpha),
                        )
                        val_filtered = apply_temporal_filter(
                            pred_coords=val_best["pred_coords"],  # type: ignore[arg-type]
                            sequence_groups=val_seq_groups,
                            cfg=cfg,
                        )
                        val_reg = regression_metrics(
                            pred=val_filtered,
                            target=val_coords,
                        )
                        row = {
                            "cfg": cfg.name,
                            "method": cfg.method,
                            "window": cfg.window,
                            "ema_alpha": cfg.ema_alpha,
                            "val_mean_error_m": float(val_reg["mean_error_m"]),
                            "val_p90_error_m": float(val_reg["p90_error_m"]),
                            "val_rmse_m": float(val_reg["rmse_m"]),
                        }
                        temporal_tuning_rows.append(row)
                        score = (
                            float(val_reg["mean_error_m"]),
                            float(val_reg["p90_error_m"]),
                            float(val_reg["rmse_m"]),
                        )
                        if best_temporal_score is None or score < best_temporal_score:
                            best_temporal_score = score
                            best_temporal_cfg = cfg

            temporal_cfg = best_temporal_cfg
            temporal_tuned = True

    test_baseline_reg = dict(test_best["regression"])  # type: ignore[arg-type]
    val_temporal_reg = dict(val_best["regression"])  # type: ignore[arg-type]
    test_temporal_reg = dict(test_baseline_reg)
    if args.enable_temporal_filter and temporal_effective:
        val_filtered = apply_temporal_filter(
            pred_coords=val_best["pred_coords"],  # type: ignore[arg-type]
            sequence_groups=val_seq_groups,
            cfg=temporal_cfg,
        )
        test_filtered = apply_temporal_filter(
            pred_coords=test_best["pred_coords"],  # type: ignore[arg-type]
            sequence_groups=test_seq_groups,
            cfg=temporal_cfg,
        )
        val_temporal_reg = regression_metrics(val_filtered, val_coords)
        test_temporal_reg = regression_metrics(test_filtered, test_coords)

    model_bundle = {
        "feature_set": args.feature_set,
        "scale_enabled": (not args.no_scale),
        "scaler_mean": None if scaler is None else scaler.mean_.astype(np.float32),
        "scaler_scale": None if scaler is None else scaler.scale_.astype(np.float32),
        "train_features": train_features.astype(np.float32),
        "train_coords": train_coords.astype(np.float32),
        "train_group_ids": train_groups.astype(np.int32),
        "best_candidate": {
            "name": best_candidate.name,
            "k": best_candidate.k,
            "weighted": best_candidate.weighted,
            "group_aware": best_candidate.group_aware,
        },
        "temporal_filter": {
            "enabled": bool(args.enable_temporal_filter),
            "auto_tuned": bool(temporal_tuned),
            "effective": bool(temporal_effective),
            "method": temporal_cfg.method,
            "window": int(temporal_cfg.window),
            "ema_alpha": float(temporal_cfg.ema_alpha),
        },
        "sequence_stats": {
            "val": val_seq_stats,
            "test": test_seq_stats,
        },
    }
    with (output_dir / "model_bundle.pkl").open("wb") as handle:
        pickle.dump(model_bundle, handle)

    metrics = {
        "device": "cpu",
        "model_type": "rssi_knn",
        "feature_set": args.feature_set,
        "scale_enabled": (not args.no_scale),
        "best_candidate": {
            "name": best_candidate.name,
            "k": best_candidate.k,
            "weighted": best_candidate.weighted,
            "group_aware": best_candidate.group_aware,
        },
        "temporal_filter": {
            "enabled": bool(args.enable_temporal_filter),
            "auto_tuned": bool(temporal_tuned),
            "effective": bool(temporal_effective),
            "method": temporal_cfg.method,
            "window": int(temporal_cfg.window),
            "ema_alpha": float(temporal_cfg.ema_alpha),
        },
        "sequence_stats": {
            "val": val_seq_stats,
            "test": test_seq_stats,
        },
        "val_best_metrics": val_best["regression"],
        "val_temporal_metrics": val_temporal_reg,
        "test_baseline_metrics": test_baseline_reg,
        "test_metrics": test_temporal_reg,
        "test_classification_accuracy": float(test_best["classification_accuracy"]),
        "candidate_summaries": candidate_summaries,
        "temporal_tuning_summaries": temporal_tuning_rows,
        "num_train_samples": int(len(train_features)),
        "num_val_samples": int(len(val_features)),
        "num_test_samples": int(len(test_features)),
        "feature_dim": int(train_features.shape[1]),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "training_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    test_reg = test_temporal_reg
    print("Final RSSI-KNN Results", flush=True)
    print(f"  Best Candidate: {best_candidate.name}", flush=True)
    if args.enable_temporal_filter:
        mode = "auto-tuned" if temporal_tuned else "manual"
        print(
            "  Temporal Filter: "
            f"{temporal_cfg.name} ({mode})",
            flush=True,
        )
        print(
            f"  Temporal Effective: {temporal_effective} "
            f"(val groups>=2: {int(val_seq_stats['groups_ge2'])}, "
            f"test groups>=2: {int(test_seq_stats['groups_ge2'])})",
            flush=True,
        )
        print(
            f"  Baseline Mean : {float(test_baseline_reg['mean_error_m']):.3f} m",
            flush=True,
        )
    print(f"  Mean Error  : {float(test_reg['mean_error_m']):.3f} m", flush=True)
    print(f"  Median Error: {float(test_reg['median_error_m']):.3f} m", flush=True)
    print(f"  P90 Error   : {float(test_reg['p90_error_m']):.3f} m", flush=True)
    print(f"  P95 Error   : {float(test_reg['p95_error_m']):.3f} m", flush=True)
    print(f"  RMSE        : {float(test_reg['rmse_m']):.3f} m", flush=True)
    print(f"  Cls Acc     : {float(test_best['classification_accuracy']):.3f}", flush=True)
    print(f"  Metrics     : {output_dir / 'metrics.json'}", flush=True)
    print(f"  Model Bundle: {output_dir / 'model_bundle.pkl'}", flush=True)


if __name__ == "__main__":
    main()
