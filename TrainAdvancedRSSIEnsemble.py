#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def parse_int_list(raw: str) -> List[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Empty integer list.")
    return sorted(set(values))


def parse_str_list(raw: str) -> List[str]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Empty string list.")
    return values


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    return (x / norm).astype(np.float32)


def build_feature_matrix(arrays: Dict[str, np.ndarray], feature_set: str) -> np.ndarray:
    x = arrays["X"].astype(np.float32)
    first = x[:, 0, :]
    last = x[:, -1, :]
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    xmax = x.max(axis=1)
    xmin = x.min(axis=1)
    slope = x[:, -1, :] - x[:, 0, :]
    q25 = np.quantile(x, 0.25, axis=1)
    q50 = np.quantile(x, 0.50, axis=1)
    q75 = np.quantile(x, 0.75, axis=1)
    delta_abs_mean = np.abs(np.diff(x, axis=1)).mean(axis=1)
    presence = (x > 1e-6).astype(np.float32)
    presence_ratio = presence.mean(axis=1)
    count_ratio = presence_ratio.mean(axis=1, keepdims=True)

    if feature_set == "last":
        return last.astype(np.float32)
    if feature_set == "mean":
        return mean.astype(np.float32)
    if feature_set == "flatten":
        return x.reshape(len(x), -1).astype(np.float32)
    if feature_set == "last_mean":
        return np.concatenate([last, mean], axis=1).astype(np.float32)
    if feature_set == "robust_last_mean":
        return np.concatenate(
            [
                last,
                mean,
                (last > 1e-6).astype(np.float32),
                (mean > 1e-6).astype(np.float32),
                count_ratio,
            ],
            axis=1,
        ).astype(np.float32)
    if feature_set == "stat_stack":
        return np.concatenate(
            [last, mean, std, xmax, xmin, slope, presence_ratio, count_ratio],
            axis=1,
        ).astype(np.float32)
    if feature_set == "last_mean_std":
        return np.concatenate(
            [last, mean, std, slope, presence_ratio, count_ratio],
            axis=1,
        ).astype(np.float32)
    if feature_set == "temporal_signature":
        return np.concatenate(
            [first, last, mean, slope, delta_abs_mean, presence_ratio, count_ratio],
            axis=1,
        ).astype(np.float32)
    if feature_set == "quantile_stack":
        return np.concatenate(
            [q25, q50, q75, slope, delta_abs_mean, presence_ratio, count_ratio],
            axis=1,
        ).astype(np.float32)
    if feature_set == "flatten_stat":
        return np.concatenate(
            [x.reshape(len(x), -1), last, mean, std, slope, presence_ratio, count_ratio],
            axis=1,
        ).astype(np.float32)
    raise ValueError(
        f"Unsupported feature_set='{feature_set}', expected "
        "last|mean|flatten|last_mean|robust_last_mean|stat_stack|last_mean_std|"
        "temporal_signature|quantile_stack|flatten_stat."
    )


def parse_max_features(raw: str):
    token = raw.strip().lower()
    if token in {"none", "null"}:
        return None
    if token in {"sqrt", "log2"}:
        return token
    try:
        value = float(token)
    except ValueError:
        return raw
    if value.is_integer() and "." not in token:
        return int(value)
    return value


def build_classifier(name: str, n_estimators: int, max_features, seed: int, n_jobs: int):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=seed,
            n_jobs=n_jobs,
        )
    raise ValueError(f"Unsupported classifier name: {name}")


@dataclass
class GroupClassifierCandidate:
    classifier_name: str
    n_estimators: int
    max_features: object
    feature_set: str

    @property
    def name(self) -> str:
        return f"{self.classifier_name}_est{self.n_estimators}_mf{self.max_features}_fs{self.feature_set}"


@dataclass
class LocalizerCandidate:
    feature_set: str
    metric: str
    k: int
    aggregator: str
    group_mode: str

    @property
    def name(self) -> str:
        return f"{self.feature_set}_{self.metric}_k{self.k}_{self.aggregator}_{self.group_mode}"


@dataclass
class EnsembleStep:
    candidate_name: str
    alpha: float


class FeatureView:
    def __init__(
        self,
        name: str,
        train_raw: np.ndarray,
        val_raw: np.ndarray,
        test_raw: np.ndarray,
    ) -> None:
        self.name = name
        self.train_raw = train_raw.astype(np.float32)
        self.val_raw = val_raw.astype(np.float32)
        self.test_raw = test_raw.astype(np.float32)
        self.scaler = StandardScaler()
        self.train_scaled = self.scaler.fit_transform(self.train_raw).astype(np.float32)
        self.val_scaled = self.scaler.transform(self.val_raw).astype(np.float32)
        self.test_scaled = self.scaler.transform(self.test_raw).astype(np.float32)

        self.train_cosine = l2_normalize(self.train_scaled)
        self.val_cosine = l2_normalize(self.val_scaled)
        self.test_cosine = l2_normalize(self.test_scaled)


class GroupProbClassifier:
    def __init__(
        self,
        classifier,
        scaler: StandardScaler,
        classes_: np.ndarray,
    ) -> None:
        self.classifier = classifier
        self.scaler = scaler
        self.classes_ = classes_.astype(np.int32)

    def predict_topk(self, features: np.ndarray, topn: int) -> Tuple[np.ndarray, np.ndarray]:
        features_scaled = self.scaler.transform(features).astype(np.float32)
        proba = self.classifier.predict_proba(features_scaled).astype(np.float32)
        topn = max(1, min(int(topn), proba.shape[1]))
        idx = np.argsort(proba, axis=1)[:, -topn:][:, ::-1]
        top_groups = self.classes_[idx]
        top_probs = np.take_along_axis(proba, idx, axis=1)
        return top_groups.astype(np.int32), top_probs.astype(np.float32)

    def top1_accuracy(self, features: np.ndarray, target_groups: np.ndarray) -> float:
        pred_groups, _ = self.predict_topk(features, topn=1)
        return float((pred_groups[:, 0] == target_groups).mean())


class KNNDatabase:
    def __init__(
        self,
        train_features: np.ndarray,
        train_coords: np.ndarray,
        train_groups: np.ndarray,
        metric: str,
        max_k: int,
    ) -> None:
        self.train_features = train_features.astype(np.float32)
        self.train_coords = train_coords.astype(np.float32)
        self.train_groups = train_groups.astype(np.int32)
        self.metric = metric
        self.max_k = int(max_k)

        algo = "brute" if metric == "cosine" else "auto"
        self.global_nn = NearestNeighbors(
            n_neighbors=min(self.max_k, len(self.train_features)),
            metric=metric,
            algorithm=algo,
        )
        self.global_nn.fit(self.train_features)

        self.group_indices: Dict[int, np.ndarray] = {}
        self.group_nn: Dict[int, NearestNeighbors] = {}
        for group_id in sorted(set(int(g) for g in self.train_groups.tolist())):
            idx = np.where(self.train_groups == group_id)[0]
            if len(idx) == 0:
                continue
            nn = NearestNeighbors(
                n_neighbors=min(self.max_k, len(idx)),
                metric=metric,
                algorithm=algo,
            )
            nn.fit(self.train_features[idx])
            self.group_indices[group_id] = idx
            self.group_nn[group_id] = nn

    def query_global(self, features: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k_use = max(1, min(int(k), self.max_k))
        return self.global_nn.kneighbors(features, n_neighbors=k_use, return_distance=True)

    def query_group(self, features: np.ndarray, group_ids: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k_use = max(1, min(int(k), self.max_k))
        dist = np.zeros((len(features), k_use), dtype=np.float32)
        idx = np.zeros((len(features), k_use), dtype=np.int64)

        for group_id in sorted(set(int(g) for g in group_ids.tolist())):
            sample_idx = np.where(group_ids == group_id)[0]
            if len(sample_idx) == 0:
                continue
            if group_id not in self.group_nn:
                d, i = self.query_global(features[sample_idx], k=k_use)
                dist[sample_idx] = d.astype(np.float32)
                idx[sample_idx] = i.astype(np.int64)
                continue

            local_nn = self.group_nn[group_id]
            group_map = self.group_indices[group_id]
            d_local, i_local = local_nn.kneighbors(
                features[sample_idx],
                n_neighbors=min(k_use, len(group_map)),
                return_distance=True,
            )
            global_idx = group_map[i_local]

            if global_idx.shape[1] < k_use:
                # Pad small groups by repeating the furthest available neighbor.
                pad_cols = k_use - global_idx.shape[1]
                global_idx = np.concatenate([global_idx, np.repeat(global_idx[:, -1:], pad_cols, axis=1)], axis=1)
                d_local = np.concatenate([d_local, np.repeat(d_local[:, -1:], pad_cols, axis=1)], axis=1)

            dist[sample_idx] = d_local[:, :k_use].astype(np.float32)
            idx[sample_idx] = global_idx[:, :k_use].astype(np.int64)

        return dist, idx


def aggregate_neighbors(
    query_features: np.ndarray,
    train_features: np.ndarray,
    train_coords: np.ndarray,
    neighbor_idx: np.ndarray,
    neighbor_dist: np.ndarray,
    aggregator: str,
) -> np.ndarray:
    coords = train_coords[neighbor_idx]
    if aggregator == "avg":
        return coords.mean(axis=1).astype(np.float32)

    if aggregator == "idw":
        w = 1.0 / (neighbor_dist + 1e-6)
        w = w / w.sum(axis=1, keepdims=True)
        return (coords * w[:, :, None]).sum(axis=1).astype(np.float32)

    if aggregator == "idw2":
        w = 1.0 / (neighbor_dist ** 2 + 1e-6)
        w = w / w.sum(axis=1, keepdims=True)
        return (coords * w[:, :, None]).sum(axis=1).astype(np.float32)

    if aggregator == "kernel":
        sigma = np.median(neighbor_dist, axis=1, keepdims=True)
        sigma = np.where(sigma < 1e-6, 1.0, sigma)
        w = np.exp(-(neighbor_dist ** 2) / (2.0 * sigma ** 2))
        w = w / np.sum(w, axis=1, keepdims=True)
        return (coords * w[:, :, None]).sum(axis=1).astype(np.float32)

    if aggregator == "softmax":
        temperature = np.maximum(np.median(neighbor_dist, axis=1, keepdims=True), 1e-3)
        shifted = neighbor_dist - neighbor_dist.min(axis=1, keepdims=True)
        w = np.exp(-(shifted / temperature))
        w = w / np.sum(w, axis=1, keepdims=True)
        return (coords * w[:, :, None]).sum(axis=1).astype(np.float32)

    if aggregator == "trimmed_idw":
        pred = np.zeros((len(query_features), 2), dtype=np.float32)
        for i in range(len(query_features)):
            keep = max(1, int(math.ceil(neighbor_dist.shape[1] * 0.7)))
            order = np.argsort(neighbor_dist[i])[:keep]
            d = neighbor_dist[i, order]
            c = coords[i, order]
            w = 1.0 / (d + 1e-6)
            w = w / np.sum(w)
            pred[i] = np.sum(c * w[:, None], axis=0)
        return pred.astype(np.float32)

    if aggregator == "median":
        return np.median(coords, axis=1).astype(np.float32)

    if aggregator == "lle":
        pred = np.zeros((len(query_features), 2), dtype=np.float32)
        ones_cache: Dict[int, np.ndarray] = {}
        eye_cache: Dict[int, np.ndarray] = {}
        for i in range(len(query_features)):
            feat = train_features[neighbor_idx[i]] - query_features[i][None, :]
            c = feat @ feat.T
            k = c.shape[0]
            if k not in ones_cache:
                ones_cache[k] = np.ones((k,), dtype=np.float64)
                eye_cache[k] = np.eye(k, dtype=np.float64)
            trace = float(np.trace(c))
            reg = 1e-3 * trace if trace > 1e-8 else 1e-3
            c = c.astype(np.float64) + eye_cache[k] * reg
            w = np.linalg.solve(c, ones_cache[k])
            w = w / np.sum(w)
            pred[i] = np.sum(coords[i].astype(np.float64) * w[:, None], axis=0).astype(np.float32)
        return pred

    raise ValueError(f"Unsupported aggregator: {aggregator}")


def score_tuple(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    return (
        float(metrics["mean_error_m"]),
        float(metrics["p90_error_m"]),
        float(metrics["rmse_m"]),
    )


def fit_best_group_classifier(
    train_arrays: Dict[str, np.ndarray],
    val_arrays: Dict[str, np.ndarray],
    candidates: Sequence[GroupClassifierCandidate],
    n_jobs: int,
    seed: int,
) -> Tuple[GroupProbClassifier, Dict[str, object], List[Dict[str, object]]]:
    train_groups = encode_groups(train_arrays["group"])
    val_groups = encode_groups(val_arrays["group"])

    rows: List[Dict[str, object]] = []
    best_bundle: Optional[Tuple[GroupProbClassifier, Dict[str, object]]] = None
    best_acc = -1.0

    for cand in candidates:
        train_feat = build_feature_matrix(train_arrays, cand.feature_set)
        val_feat = build_feature_matrix(val_arrays, cand.feature_set)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_feat).astype(np.float32)
        val_scaled = scaler.transform(val_feat).astype(np.float32)

        clf = build_classifier(
            name=cand.classifier_name,
            n_estimators=cand.n_estimators,
            max_features=cand.max_features,
            seed=seed,
            n_jobs=n_jobs,
        )
        clf.fit(train_scaled, train_groups)
        classes_ = np.asarray(clf.classes_, dtype=np.int32)
        bundle = GroupProbClassifier(classifier=clf, scaler=scaler, classes_=classes_)
        acc = bundle.top1_accuracy(val_feat, val_groups)
        row = {
            "candidate": cand.name,
            "feature_set": cand.feature_set,
            "classifier_name": cand.classifier_name,
            "n_estimators": cand.n_estimators,
            "max_features": cand.max_features,
            "val_group_accuracy": float(acc),
        }
        rows.append(row)
        if acc > best_acc:
            best_acc = acc
            best_bundle = (bundle, row)

    if best_bundle is None:
        raise RuntimeError("No group classifier candidate evaluated.")
    return best_bundle[0], best_bundle[1], rows


def build_feature_views(
    train_arrays: Dict[str, np.ndarray],
    val_arrays: Dict[str, np.ndarray],
    test_arrays: Dict[str, np.ndarray],
    feature_sets: Sequence[str],
) -> Dict[str, FeatureView]:
    views: Dict[str, FeatureView] = {}
    for feature_set in feature_sets:
        train_raw = build_feature_matrix(train_arrays, feature_set)
        val_raw = build_feature_matrix(val_arrays, feature_set)
        test_raw = build_feature_matrix(test_arrays, feature_set)
        views[feature_set] = FeatureView(
            name=feature_set,
            train_raw=train_raw,
            val_raw=val_raw,
            test_raw=test_raw,
        )
    return views


def select_metric_features(view: FeatureView, metric: str, split: str) -> np.ndarray:
    if metric == "cosine":
        if split == "train":
            return view.train_cosine
        if split == "val":
            return view.val_cosine
        return view.test_cosine
    if split == "train":
        return view.train_scaled
    if split == "val":
        return view.val_scaled
    return view.test_scaled


def make_knn_databases(
    views: Dict[str, FeatureView],
    train_coords: np.ndarray,
    train_groups: np.ndarray,
    metrics: Sequence[str],
    max_k: int,
) -> Dict[Tuple[str, str], KNNDatabase]:
    out: Dict[Tuple[str, str], KNNDatabase] = {}
    for feature_set, view in views.items():
        for metric in metrics:
            out[(feature_set, metric)] = KNNDatabase(
                train_features=select_metric_features(view, metric, "train"),
                train_coords=train_coords,
                train_groups=train_groups,
                metric=metric,
                max_k=max_k,
            )
    return out


def predict_with_candidate(
    db: KNNDatabase,
    view: FeatureView,
    classifier_view: FeatureView,
    split: str,
    candidate: LocalizerCandidate,
    classifier: GroupProbClassifier,
) -> Tuple[np.ndarray, Dict[str, float]]:
    query_feat = select_metric_features(view, candidate.metric, split)
    global_dist, global_idx = db.query_global(query_feat, k=candidate.k)
    global_pred = aggregate_neighbors(
        query_features=query_feat,
        train_features=db.train_features,
        train_coords=db.train_coords,
        neighbor_idx=global_idx,
        neighbor_dist=global_dist,
        aggregator=candidate.aggregator,
    )

    if candidate.group_mode == "global":
        diag = {"group_mode_used": 0.0}
        return global_pred, diag

    raw_feat = classifier_view.val_raw if split == "val" else classifier_view.test_raw
    top_groups, top_probs = classifier.predict_topk(raw_feat, topn=2 if candidate.group_mode == "top2" else 1)

    if candidate.group_mode == "top1":
        local_dist, local_idx = db.query_group(query_feat, top_groups[:, 0], k=candidate.k)
        local_pred = aggregate_neighbors(
            query_features=query_feat,
            train_features=db.train_features,
            train_coords=db.train_coords,
            neighbor_idx=local_idx,
            neighbor_dist=local_dist,
            aggregator=candidate.aggregator,
        )
        conf = top_probs[:, 0:1]
        conf = np.clip(conf, 0.0, 1.0)
        # Blend local and global to reduce catastrophic group mistakes.
        pred = (conf * local_pred + (1.0 - conf) * global_pred).astype(np.float32)
        return pred, {"group_mode_used": 1.0, "mean_top1_prob": float(top_probs[:, 0].mean())}

    if candidate.group_mode == "top2":
        pred = np.zeros_like(global_pred, dtype=np.float32)
        probs = top_probs / np.maximum(top_probs.sum(axis=1, keepdims=True), 1e-6)
        local_preds = []
        for col in range(top_groups.shape[1]):
            local_dist, local_idx = db.query_group(query_feat, top_groups[:, col], k=candidate.k)
            local_pred = aggregate_neighbors(
                query_features=query_feat,
                train_features=db.train_features,
                train_coords=db.train_coords,
                neighbor_idx=local_idx,
                neighbor_dist=local_dist,
                aggregator=candidate.aggregator,
            )
            local_preds.append(local_pred)
            pred += probs[:, col : col + 1] * local_pred

        top1_conf = top_probs[:, 0:1]
        global_mix = np.clip(0.5 - top1_conf, 0.0, 0.5) * 1.2
        pred = (1.0 - global_mix) * pred + global_mix * global_pred
        return pred.astype(np.float32), {
            "group_mode_used": 2.0,
            "mean_top1_prob": float(top_probs[:, 0].mean()),
            "mean_global_mix": float(global_mix.mean()),
        }

    raise ValueError(f"Unsupported group_mode: {candidate.group_mode}")


def greedy_ensemble_search(
    candidate_rows: Sequence[Dict[str, object]],
    val_predictions: Dict[str, np.ndarray],
    val_target: np.ndarray,
    max_candidates: int,
    max_steps: int,
) -> Tuple[List[EnsembleStep], np.ndarray, Dict[str, float]]:
    top_rows = sorted(candidate_rows, key=lambda row: float(row["val_mean_error_m"]))[:max_candidates]
    if not top_rows:
        raise RuntimeError("No candidate rows available for ensemble search.")

    current_name = str(top_rows[0]["candidate"])
    current_pred = val_predictions[current_name].copy()
    current_metrics = regression_metrics(current_pred, val_target)
    chosen = {current_name}
    steps: List[EnsembleStep] = [EnsembleStep(candidate_name=current_name, alpha=1.0)]

    alpha_grid = [0.10, 0.15, 0.20, 0.25, 0.33, 0.40, 0.50]

    for _ in range(max_steps - 1):
        best_trial_pred: Optional[np.ndarray] = None
        best_trial_metrics: Optional[Dict[str, float]] = None
        best_trial_step: Optional[EnsembleStep] = None

        for row in top_rows:
            cand_name = str(row["candidate"])
            if cand_name in chosen:
                continue
            cand_pred = val_predictions[cand_name]
            for alpha in alpha_grid:
                trial_pred = ((1.0 - alpha) * current_pred + alpha * cand_pred).astype(np.float32)
                trial_metrics = regression_metrics(trial_pred, val_target)
                if best_trial_metrics is None or score_tuple(trial_metrics) < score_tuple(best_trial_metrics):
                    best_trial_pred = trial_pred
                    best_trial_metrics = trial_metrics
                    best_trial_step = EnsembleStep(candidate_name=cand_name, alpha=float(alpha))

        if best_trial_metrics is None or best_trial_step is None or best_trial_pred is None:
            break
        if score_tuple(best_trial_metrics) >= score_tuple(current_metrics):
            break

        current_pred = best_trial_pred
        current_metrics = best_trial_metrics
        steps.append(best_trial_step)
        chosen.add(best_trial_step.candidate_name)

    return steps, current_pred, current_metrics


def apply_ensemble_steps(
    steps: Sequence[EnsembleStep],
    prediction_bank: Dict[str, np.ndarray],
) -> np.ndarray:
    if not steps:
        raise ValueError("Ensemble steps must not be empty.")
    pred = prediction_bank[steps[0].candidate_name].astype(np.float32).copy()
    for step in steps[1:]:
        pred = ((1.0 - step.alpha) * pred + step.alpha * prediction_bank[step.candidate_name]).astype(np.float32)
    return pred


def _setup_matplotlib(output_dir: Path) -> None:
    mplcfg = output_dir / ".mplconfig"
    mplcfg.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mplcfg)
    os.environ["MPLBACKEND"] = "Agg"


def save_predictions_csv(
    out_csv: Path,
    pred: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> None:
    err = np.linalg.norm(pred - target, axis=1)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "building",
                "floor",
                "true_x",
                "true_y",
                "pred_x",
                "pred_y",
                "error_m",
            ]
        )
        for idx in range(len(pred)):
            writer.writerow(
                [
                    idx,
                    int(groups[idx, 0]),
                    int(groups[idx, 1]),
                    float(target[idx, 0]),
                    float(target[idx, 1]),
                    float(pred[idx, 0]),
                    float(pred[idx, 1]),
                    float(err[idx]),
                ]
            )


def plot_predictions(
    out_png: Path,
    pred: np.ndarray,
    target: np.ndarray,
    title: str,
) -> None:
    _setup_matplotlib(out_png.parent)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    error = np.linalg.norm(pred - target, axis=1)
    fig, ax = plt.subplots(figsize=(8.8, 7.0), constrained_layout=True)
    ax.scatter(target[:, 0], target[:, 1], s=10, alpha=0.40, c="#111827", label="True")
    sc = ax.scatter(pred[:, 0], pred[:, 1], s=12, alpha=0.72, c=error, cmap="turbo", label="Pred")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Error (m)")
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_error_hist(out_png: Path, pred: np.ndarray, target: np.ndarray, title: str) -> None:
    _setup_matplotlib(out_png.parent)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    error = np.linalg.norm(pred - target, axis=1)
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.hist(error, bins=40, color="#1d4ed8", alpha=0.85)
    ax.axvline(float(np.mean(error)), color="#dc2626", ls="--", lw=2, label=f"mean={np.mean(error):.2f}")
    ax.axvline(float(np.quantile(error, 0.90)), color="#16a34a", ls=":", lw=2, label=f"p90={np.quantile(error, 0.90):.2f}")
    ax.set_title(title)
    ax.set_xlabel("Error (m)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def parse_group_classifier_candidates(raw: str) -> List[GroupClassifierCandidate]:
    out: List[GroupClassifierCandidate] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid group classifier candidate '{token}', expected classifier:n_estimators:max_features:feature_set"
            )
        out.append(
            GroupClassifierCandidate(
                classifier_name=parts[0],
                n_estimators=int(parts[1]),
                max_features=parse_max_features(parts[2]),
                feature_set=parts[3],
            )
        )
    if not out:
        raise ValueError("No group classifier candidate parsed.")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Advanced RSSI-only localizer: multi-view features + learned group classifier + "
            "group-aware KNN + local interpolation + validation-tuned ensemble."
        )
    )
    parser.add_argument("--train-dir", type=str, default="training_dataset")
    parser.add_argument("--test-dir", type=str, default="test_dataset")
    parser.add_argument("--output-dir", type=str, default="runs/advanced_rssi_ensemble")
    parser.add_argument(
        "--feature-sets",
        type=str,
        default="last,mean,last_mean,robust_last_mean,stat_stack,last_mean_std,temporal_signature,quantile_stack,flatten_stat",
    )
    parser.add_argument("--metrics", type=str, default="euclidean,manhattan,cosine")
    parser.add_argument("--k-candidates", type=str, default="1,3,5,7,11,15,21")
    parser.add_argument("--aggregators", type=str, default="idw,idw2,kernel,softmax,trimmed_idw,median,lle")
    parser.add_argument("--group-modes", type=str, default="global,top1,top2")
    parser.add_argument(
        "--group-classifier-candidates",
        type=str,
        default="rf:500:sqrt:stat_stack,extra_trees:700:sqrt:stat_stack,extra_trees:700:sqrt:robust_last_mean",
    )
    parser.add_argument("--ensemble-max-candidates", type=int, default=8)
    parser.add_argument("--ensemble-max-steps", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

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

    train_coords = train_arrays["y_last"].astype(np.float32)
    val_coords = val_arrays["y_last"].astype(np.float32)
    test_coords = test_arrays["y_last"].astype(np.float32)
    train_groups = train_arrays["group"].astype(np.int32)
    val_groups = val_arrays["group"].astype(np.int32)
    test_groups = test_arrays["group"].astype(np.int32)
    train_group_ids = encode_groups(train_groups)

    feature_sets = parse_str_list(args.feature_sets)
    metrics = parse_str_list(args.metrics)
    k_values = parse_int_list(args.k_candidates)
    aggregators = parse_str_list(args.aggregators)
    group_modes = parse_str_list(args.group_modes)
    group_classifier_candidates = parse_group_classifier_candidates(args.group_classifier_candidates)

    print(
        f"Train/Val/Test samples: {len(train_coords)}/{len(val_coords)}/{len(test_coords)}",
        flush=True,
    )
    print(f"Feature sets: {feature_sets}", flush=True)
    print(f"Search space size: {len(feature_sets) * len(metrics) * len(k_values) * len(aggregators) * len(group_modes)}", flush=True)

    best_group_classifier, best_group_row, group_search_rows = fit_best_group_classifier(
        train_arrays=train_arrays,
        val_arrays=val_arrays,
        candidates=group_classifier_candidates,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )
    print(
        f"Best group classifier: {best_group_row['candidate']} "
        f"(val group acc={float(best_group_row['val_group_accuracy']):.3f})",
        flush=True,
    )

    views = build_feature_views(
        train_arrays=train_arrays,
        val_arrays=val_arrays,
        test_arrays=test_arrays,
        feature_sets=feature_sets,
    )
    dbs = make_knn_databases(
        views=views,
        train_coords=train_coords,
        train_groups=train_group_ids,
        metrics=metrics,
        max_k=max(k_values),
    )

    candidate_rows: List[Dict[str, object]] = []
    val_prediction_bank: Dict[str, np.ndarray] = {}

    total = len(feature_sets) * len(metrics) * len(k_values) * len(aggregators) * len(group_modes)
    idx_counter = 0
    for feature_set in feature_sets:
        for metric in metrics:
            for k in k_values:
                for aggregator in aggregators:
                    for group_mode in group_modes:
                        idx_counter += 1
                        candidate = LocalizerCandidate(
                            feature_set=feature_set,
                            metric=metric,
                            k=k,
                            aggregator=aggregator,
                            group_mode=group_mode,
                        )
                        pred_val, diag = predict_with_candidate(
                            db=dbs[(feature_set, metric)],
                            view=views[feature_set],
                            classifier_view=views[str(best_group_row["feature_set"])],
                            split="val",
                            candidate=candidate,
                            classifier=best_group_classifier,
                        )
                        reg = regression_metrics(pred_val, val_coords)
                        row = {
                            "candidate": candidate.name,
                            "feature_set": feature_set,
                            "metric": metric,
                            "k": int(k),
                            "aggregator": aggregator,
                            "group_mode": group_mode,
                            "val_mean_error_m": float(reg["mean_error_m"]),
                            "val_p90_error_m": float(reg["p90_error_m"]),
                            "val_rmse_m": float(reg["rmse_m"]),
                            **diag,
                        }
                        candidate_rows.append(row)
                        val_prediction_bank[candidate.name] = pred_val.astype(np.float32)
                        if idx_counter % 25 == 0 or idx_counter == total:
                            print(
                                f"[search {idx_counter:03d}/{total}] best so far: "
                                f"{min(candidate_rows, key=lambda r: float(r['val_mean_error_m']))['candidate']} "
                                f"=> {min(float(r['val_mean_error_m']) for r in candidate_rows):.3f} m",
                                flush=True,
                            )

    best_row = min(candidate_rows, key=lambda row: float(row["val_mean_error_m"]))
    print(
        f"Best single candidate: {best_row['candidate']} "
        f"(val mean={float(best_row['val_mean_error_m']):.3f} m)",
        flush=True,
    )

    ensemble_steps, ensemble_val_pred, ensemble_val_metrics = greedy_ensemble_search(
        candidate_rows=candidate_rows,
        val_predictions=val_prediction_bank,
        val_target=val_coords,
        max_candidates=int(args.ensemble_max_candidates),
        max_steps=int(args.ensemble_max_steps),
    )
    print(
        f"Ensemble val mean={float(ensemble_val_metrics['mean_error_m']):.3f} m "
        f"using {[step.candidate_name for step in ensemble_steps]}",
        flush=True,
    )

    # Refit group classifier on train+val for final test inference.
    full_train_arrays = concat_splits([train_arrays, val_arrays])
    full_train_coords = full_train_arrays["y_last"].astype(np.float32)
    full_train_group_ids = encode_groups(full_train_arrays["group"].astype(np.int32))

    best_gc_cand = GroupClassifierCandidate(
        classifier_name=str(best_group_row["classifier_name"]),
        n_estimators=int(best_group_row["n_estimators"]),
        max_features=str(best_group_row["max_features"]),
        feature_set=str(best_group_row["feature_set"]),
    )
    full_gc_train_feat = build_feature_matrix(full_train_arrays, best_gc_cand.feature_set)
    full_gc_scaler = StandardScaler()
    full_gc_train_scaled = full_gc_scaler.fit_transform(full_gc_train_feat).astype(np.float32)
    full_gc = build_classifier(
        name=best_gc_cand.classifier_name,
        n_estimators=best_gc_cand.n_estimators,
        max_features=best_gc_cand.max_features,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    full_gc.fit(full_gc_train_scaled, full_train_group_ids)
    full_group_classifier = GroupProbClassifier(
        classifier=full_gc,
        scaler=full_gc_scaler,
        classes_=np.asarray(full_gc.classes_, dtype=np.int32),
    )

    full_views = build_feature_views(
        train_arrays=full_train_arrays,
        val_arrays=test_arrays,
        test_arrays=test_arrays,
        feature_sets=feature_sets,
    )
    # For final inference, both val/test slots point to the same test data to reuse the split selector.
    full_dbs = make_knn_databases(
        views=full_views,
        train_coords=full_train_coords,
        train_groups=full_train_group_ids,
        metrics=metrics,
        max_k=max(k_values),
    )

    final_prediction_bank: Dict[str, np.ndarray] = {}
    for row in candidate_rows:
        cand = LocalizerCandidate(
            feature_set=str(row["feature_set"]),
            metric=str(row["metric"]),
            k=int(row["k"]),
            aggregator=str(row["aggregator"]),
            group_mode=str(row["group_mode"]),
        )
        pred_test, _ = predict_with_candidate(
            db=full_dbs[(cand.feature_set, cand.metric)],
            view=full_views[cand.feature_set],
            classifier_view=full_views[best_gc_cand.feature_set],
            split="val",
            candidate=cand,
            classifier=full_group_classifier,
        )
        final_prediction_bank[cand.name] = pred_test.astype(np.float32)

    best_single_test_pred = final_prediction_bank[str(best_row["candidate"])]
    best_single_test_metrics = regression_metrics(best_single_test_pred, test_coords)
    ensemble_test_pred = apply_ensemble_steps(ensemble_steps, final_prediction_bank)
    ensemble_test_metrics = regression_metrics(ensemble_test_pred, test_coords)

    np.savez_compressed(
        output_dir / "test_predictions.npz",
        best_single_pred=best_single_test_pred.astype(np.float32),
        ensemble_pred=ensemble_test_pred.astype(np.float32),
        true_coords=test_coords.astype(np.float32),
        groups=test_groups.astype(np.int32),
    )
    save_predictions_csv(output_dir / "test_predictions_best_single.csv", best_single_test_pred, test_coords, test_groups)
    save_predictions_csv(output_dir / "test_predictions_ensemble.csv", ensemble_test_pred, test_coords, test_groups)
    plot_predictions(output_dir / "test_best_single_scatter.png", best_single_test_pred, test_coords, "Advanced RSSI Best Single Candidate")
    plot_predictions(output_dir / "test_ensemble_scatter.png", ensemble_test_pred, test_coords, "Advanced RSSI Ensemble")
    plot_error_hist(output_dir / "test_ensemble_error_hist.png", ensemble_test_pred, test_coords, "Advanced RSSI Ensemble Error Distribution")

    bundle = {
        "best_group_classifier": {
            "candidate": best_group_row["candidate"],
            "classifier_name": best_gc_cand.classifier_name,
            "n_estimators": best_gc_cand.n_estimators,
            "max_features": best_gc_cand.max_features,
            "feature_set": best_gc_cand.feature_set,
            "scaler": full_gc_scaler,
            "classifier": full_gc,
        },
        "train_coords": full_train_coords.astype(np.float32),
        "train_group_ids": full_train_group_ids.astype(np.int32),
        "feature_views": {
            feature_set: {
                "scaler_mean": full_views[feature_set].scaler.mean_.astype(np.float32),
                "scaler_scale": full_views[feature_set].scaler.scale_.astype(np.float32),
                "train_scaled": full_views[feature_set].train_scaled.astype(np.float32),
                "train_cosine": full_views[feature_set].train_cosine.astype(np.float32),
            }
            for feature_set in feature_sets
        },
        "candidate_rows": candidate_rows,
        "best_single_candidate": best_row,
        "ensemble_steps": [step.__dict__ for step in ensemble_steps],
        "feature_sets": feature_sets,
        "metrics": metrics,
        "k_values": k_values,
    }
    with (output_dir / "model_bundle.pkl").open("wb") as handle:
        pickle.dump(bundle, handle)

    metrics_payload = {
        "device": "cpu",
        "model_type": "advanced_rssi_ensemble",
        "num_train_samples": int(len(full_train_coords)),
        "num_test_samples": int(len(test_coords)),
        "group_classifier_search": group_search_rows,
        "best_group_classifier": best_group_row,
        "candidate_summaries": sorted(candidate_rows, key=lambda row: float(row["val_mean_error_m"])),
        "best_single_candidate": best_row,
        "best_single_test_metrics": best_single_test_metrics,
        "ensemble_steps": [step.__dict__ for step in ensemble_steps],
        "ensemble_val_metrics": ensemble_val_metrics,
        "ensemble_test_metrics": ensemble_test_metrics,
        "artifacts": {
            "model_bundle": str(output_dir / "model_bundle.pkl"),
            "test_predictions_npz": str(output_dir / "test_predictions.npz"),
            "best_single_csv": str(output_dir / "test_predictions_best_single.csv"),
            "ensemble_csv": str(output_dir / "test_predictions_ensemble.csv"),
            "best_single_scatter": str(output_dir / "test_best_single_scatter.png"),
            "ensemble_scatter": str(output_dir / "test_ensemble_scatter.png"),
            "ensemble_error_hist": str(output_dir / "test_ensemble_error_hist.png"),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nFinal Advanced RSSI Results", flush=True)
    print(
        f"  Best Single : {str(best_row['candidate'])} "
        f"=> {float(best_single_test_metrics['mean_error_m']):.3f} m",
        flush=True,
    )
    print(
        f"  Ensemble    : {float(ensemble_test_metrics['mean_error_m']):.3f} m "
        f"(rmse={float(ensemble_test_metrics['rmse_m']):.3f} m)",
        flush=True,
    )
    print(f"  Metrics     : {output_dir / 'metrics.json'}", flush=True)
    print(f"  Model Bundle: {output_dir / 'model_bundle.pkl'}", flush=True)


if __name__ == "__main__":
    main()
