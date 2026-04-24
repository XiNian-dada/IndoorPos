#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


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


def load_arrays(train_dir: Path, test_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    train_arrays = load_npz(train_dir / "train_sequences.npz")
    val_arrays = load_npz(train_dir / "val_sequences.npz")
    test_arrays = concat_splits(
        [
            load_npz(test_dir / "train_sequences.npz"),
            load_npz(test_dir / "val_sequences.npz"),
        ]
    )
    return train_arrays, val_arrays, test_arrays


def build_feature_matrix(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    rssi = arrays["X"].astype(np.float32).reshape(len(arrays["X"]), -1)
    motion = arrays["motion_features"].astype(np.float32).reshape(len(arrays["motion_features"]), -1)
    return np.concatenate([rssi, motion], axis=1).astype(np.float32)


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


@dataclass
class Candidate:
    classifier_name: str
    n_estimators: int
    max_features: str
    group_k: int
    global_k: int

    @property
    def name(self) -> str:
        return (
            f"{self.classifier_name}_"
            f"est{self.n_estimators}_mf{self.max_features}_"
            f"gk{self.group_k}_gg{self.global_k}"
        )


class GroupAwareKNNLocalizer:
    def __init__(
        self,
        scaler: StandardScaler,
        group_classifier: object,
        group_k: int,
        global_k: int,
        train_features_scaled: np.ndarray,
        train_coords: np.ndarray,
        train_group_ids: np.ndarray,
    ) -> None:
        self.scaler = scaler
        self.group_classifier = group_classifier
        self.group_k = int(group_k)
        self.global_k = int(global_k)
        self.train_features_scaled = train_features_scaled.astype(np.float32)
        self.train_coords = train_coords.astype(np.float32)
        self.train_group_ids = train_group_ids.astype(np.int32)

        self.global_nn = NearestNeighbors(
            n_neighbors=min(self.global_k, len(self.train_features_scaled)),
            metric="euclidean",
        )
        self.global_nn.fit(self.train_features_scaled)

        self.group_index: Dict[int, np.ndarray] = {}
        self.group_nn: Dict[int, NearestNeighbors] = {}
        for group_id in sorted(set(int(g) for g in self.train_group_ids.tolist())):
            indices = np.where(self.train_group_ids == group_id)[0]
            if len(indices) == 0:
                continue
            k = min(self.group_k, len(indices))
            nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
            nn.fit(self.train_features_scaled[indices])
            self.group_index[group_id] = indices
            self.group_nn[group_id] = nn

    @staticmethod
    def _weighted_average(coords: np.ndarray, dist: np.ndarray) -> np.ndarray:
        w = 1.0 / (dist + 1e-6)
        w = w / w.sum(axis=1, keepdims=True)
        return (coords * w[:, :, None]).sum(axis=1)

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        features_scaled = self.scaler.transform(features).astype(np.float32)
        pred_group_ids = self.group_classifier.predict(features_scaled).astype(np.int32)
        pred_coords = np.zeros((len(features_scaled), 2), dtype=np.float32)
        group_conf = self.group_classifier.predict_proba(features_scaled).max(axis=1).astype(np.float32)

        for group_id in sorted(set(int(g) for g in pred_group_ids.tolist())):
            sample_idx = np.where(pred_group_ids == group_id)[0]
            if len(sample_idx) == 0:
                continue

            if group_id not in self.group_nn:
                dist, nn_idx = self.global_nn.kneighbors(features_scaled[sample_idx])
                pred_coords[sample_idx] = self._weighted_average(self.train_coords[nn_idx], dist)
                continue

            local_nn = self.group_nn[group_id]
            index_map = self.group_index[group_id]
            dist_local, idx_local = local_nn.kneighbors(features_scaled[sample_idx])
            global_idx = index_map[idx_local]
            pred_coords[sample_idx] = self._weighted_average(self.train_coords[global_idx], dist_local)

        return pred_coords, pred_group_ids, group_conf


def build_classifier(
    name: str,
    n_estimators: int,
    max_features: str,
    seed: int,
    n_jobs: int,
):
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
    raise ValueError(f"Unsupported classifier_name: {name}")


def evaluate_candidate(
    candidate: Candidate,
    train_features: np.ndarray,
    train_groups: np.ndarray,
    train_coords: np.ndarray,
    val_features: np.ndarray,
    val_groups: np.ndarray,
    val_coords: np.ndarray,
    seed: int,
    n_jobs: int,
) -> Dict[str, object]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features).astype(np.float32)

    clf = build_classifier(
        name=candidate.classifier_name,
        n_estimators=candidate.n_estimators,
        max_features=candidate.max_features,
        seed=seed,
        n_jobs=n_jobs,
    )
    clf.fit(train_scaled, train_groups)

    localizer = GroupAwareKNNLocalizer(
        scaler=scaler,
        group_classifier=clf,
        group_k=candidate.group_k,
        global_k=candidate.global_k,
        train_features_scaled=train_scaled,
        train_coords=train_coords,
        train_group_ids=train_groups,
    )
    val_pred_coord, val_pred_group, _ = localizer.predict(val_features)
    reg = regression_metrics(val_pred_coord, val_coords)
    cls_acc = float((val_pred_group == val_groups).mean())

    return {
        "candidate": candidate,
        "regression": reg,
        "classification_accuracy": cls_acc,
    }


def parse_candidate_list(raw: str) -> List[Candidate]:
    candidates: List[Candidate] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 5:
            raise ValueError(
                f"Invalid candidate '{token}'. Expected format: classifier:n_estimators:max_features:group_k:global_k"
            )
        classifier_name, n_estimators, max_features, group_k, global_k = parts
        candidates.append(
            Candidate(
                classifier_name=classifier_name,
                n_estimators=int(n_estimators),
                max_features=max_features,
                group_k=int(group_k),
                global_k=int(global_k),
            )
        )
    if not candidates:
        raise ValueError("No valid candidates provided.")
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a high-accuracy group-aware KNN indoor localizer.",
    )
    parser.add_argument("--train-dir", type=str, default="training_dataset")
    parser.add_argument("--test-dir", type=str, default="test_dataset")
    parser.add_argument("--output-dir", type=str, default="runs/high_accuracy_hospital")
    parser.add_argument(
        "--candidates",
        type=str,
        default=(
            "rf:500:sqrt:11:5,"
            "rf:500:sqrt:15:5,"
            "rf:500:sqrt:21:5,"
            "rf:800:sqrt:15:5,"
            "extra_trees:500:sqrt:15:5"
        ),
        help="Comma-separated candidate list: classifier:n_estimators:max_features:group_k:global_k",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def limit_arrays(arrays: Dict[str, np.ndarray], max_samples: int) -> Dict[str, np.ndarray]:
    if max_samples <= 0:
        return arrays
    n = next(iter(arrays.values())).shape[0]
    if max_samples >= n:
        return arrays
    return {k: v[:max_samples] for k, v in arrays.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_arrays, val_arrays, test_arrays = load_arrays(train_dir=train_dir, test_dir=test_dir)
    train_arrays = limit_arrays(train_arrays, args.max_train_samples)
    val_arrays = limit_arrays(val_arrays, args.max_val_samples)
    test_arrays = limit_arrays(test_arrays, args.max_test_samples)

    train_features = build_feature_matrix(train_arrays)
    val_features = build_feature_matrix(val_arrays)
    test_features = build_feature_matrix(test_arrays)

    train_coords = train_arrays["y_last"].astype(np.float32)
    val_coords = val_arrays["y_last"].astype(np.float32)
    test_coords = test_arrays["y_last"].astype(np.float32)

    train_groups = encode_groups(train_arrays["group"])
    val_groups = encode_groups(val_arrays["group"])
    test_groups = encode_groups(test_arrays["group"])

    candidates = parse_candidate_list(args.candidates)
    search_results: List[Dict[str, object]] = []
    best_result: Dict[str, object] | None = None

    print(
        f"Train/Val/Test samples: {len(train_features)}/{len(val_features)}/{len(test_features)}",
        flush=True,
    )
    print(f"Feature dim: {train_features.shape[1]}", flush=True)
    print(f"Candidates: {len(candidates)}", flush=True)

    for idx, candidate in enumerate(candidates, start=1):
        print(f"\n[{idx}/{len(candidates)}] Evaluating {candidate.name} ...", flush=True)
        result = evaluate_candidate(
            candidate=candidate,
            train_features=train_features,
            train_groups=train_groups,
            train_coords=train_coords,
            val_features=val_features,
            val_groups=val_groups,
            val_coords=val_coords,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        reg = result["regression"]  # type: ignore[index]
        cls_acc = result["classification_accuracy"]  # type: ignore[index]
        print(
            f"  val_mean={reg['mean_error_m']:.3f}m "
            f"val_p90={reg['p90_error_m']:.3f}m "
            f"val_rmse={reg['rmse_m']:.3f}m "
            f"val_cls_acc={cls_acc:.3f}",
            flush=True,
        )

        search_results.append(
            {
                "candidate": candidate.name,
                "classifier_name": candidate.classifier_name,
                "n_estimators": candidate.n_estimators,
                "max_features": candidate.max_features,
                "group_k": candidate.group_k,
                "global_k": candidate.global_k,
                "val_metrics": reg,
                "val_classification_accuracy": cls_acc,
            }
        )

        if best_result is None:
            best_result = result
        else:
            best_mean = best_result["regression"]["mean_error_m"]  # type: ignore[index]
            if reg["mean_error_m"] < best_mean:
                best_result = result

    if best_result is None:
        raise RuntimeError("No valid candidate evaluated.")

    best_candidate: Candidate = best_result["candidate"]  # type: ignore[assignment]
    print(f"\nBest candidate on val: {best_candidate.name}", flush=True)

    # Refit on all labeled training data (train+val) for final model.
    full_train_arrays = concat_splits([train_arrays, val_arrays])
    full_train_features = build_feature_matrix(full_train_arrays)
    full_train_coords = full_train_arrays["y_last"].astype(np.float32)
    full_train_groups = encode_groups(full_train_arrays["group"])

    scaler = StandardScaler()
    full_train_scaled = scaler.fit_transform(full_train_features).astype(np.float32)
    clf = build_classifier(
        name=best_candidate.classifier_name,
        n_estimators=best_candidate.n_estimators,
        max_features=best_candidate.max_features,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    clf.fit(full_train_scaled, full_train_groups)

    localizer = GroupAwareKNNLocalizer(
        scaler=scaler,
        group_classifier=clf,
        group_k=best_candidate.group_k,
        global_k=best_candidate.global_k,
        train_features_scaled=full_train_scaled,
        train_coords=full_train_coords,
        train_group_ids=full_train_groups,
    )
    test_pred_coord, test_pred_group, test_group_conf = localizer.predict(test_features)
    test_reg = regression_metrics(test_pred_coord, test_coords)
    test_cls_acc = float((test_pred_group == test_groups).mean())

    payload = {
        "best_candidate": {
            "name": best_candidate.name,
            "classifier_name": best_candidate.classifier_name,
            "n_estimators": best_candidate.n_estimators,
            "max_features": best_candidate.max_features,
            "group_k": best_candidate.group_k,
            "global_k": best_candidate.global_k,
        },
        "val_best_metrics": best_result["regression"],
        "val_best_classification_accuracy": best_result["classification_accuracy"],
        "test_metrics": test_reg,
        "test_classification_accuracy": test_cls_acc,
        "test_group_confidence_mean": float(test_group_conf.mean()),
        "num_train_samples": int(len(full_train_features)),
        "num_test_samples": int(len(test_features)),
        "feature_dim": int(full_train_features.shape[1]),
        "search_results": search_results,
        "seed": args.seed,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model_bundle = {
        "scaler": scaler,
        "group_classifier": clf,
        "group_k": best_candidate.group_k,
        "global_k": best_candidate.global_k,
        "train_features_scaled": full_train_scaled,
        "train_coords": full_train_coords,
        "train_group_ids": full_train_groups,
        "feature_schema": {
            "rssi_shape": list(full_train_arrays["X"].shape[1:]),
            "motion_shape": list(full_train_arrays["motion_features"].shape[1:]),
            "feature_dim": int(full_train_features.shape[1]),
        },
        "meta": payload,
    }
    with (output_dir / "model_bundle.pkl").open("wb") as handle:
        pickle.dump(model_bundle, handle)

    np.savez_compressed(
        output_dir / "test_predictions.npz",
        pred_coords=test_pred_coord.astype(np.float32),
        true_coords=test_coords.astype(np.float32),
        pred_groups=test_pred_group.astype(np.int32),
        true_groups=test_groups.astype(np.int32),
        group_confidence=test_group_conf.astype(np.float32),
    )

    print("\nFinal Test Results", flush=True)
    print(f"  Mean Error  : {test_reg['mean_error_m']:.3f} m", flush=True)
    print(f"  Median Error: {test_reg['median_error_m']:.3f} m", flush=True)
    print(f"  P90 Error   : {test_reg['p90_error_m']:.3f} m", flush=True)
    print(f"  P95 Error   : {test_reg['p95_error_m']:.3f} m", flush=True)
    print(f"  RMSE        : {test_reg['rmse_m']:.3f} m", flush=True)
    print(f"  Cls Acc     : {test_cls_acc:.3f}", flush=True)
    print(f"  Metrics     : {output_dir / 'metrics.json'}", flush=True)
    print(f"  Model Bundle: {output_dir / 'model_bundle.pkl'}", flush=True)


if __name__ == "__main__":
    main()
