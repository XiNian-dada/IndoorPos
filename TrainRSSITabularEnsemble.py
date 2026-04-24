#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from TrainAdvancedRSSIEnsemble import (
    EnsembleStep,
    FeatureView,
    GroupClassifierCandidate,
    GroupProbClassifier,
    _setup_matplotlib,
    apply_ensemble_steps,
    build_feature_matrix,
    fit_best_group_classifier,
    greedy_ensemble_search,
    limit_arrays,
    load_npz,
    concat_splits,
    parse_group_classifier_candidates,
    parse_max_features,
    parse_str_list,
    regression_metrics,
    score_tuple,
    set_seed,
)


os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


def encode_groups(groups: np.ndarray) -> np.ndarray:
    return (groups[:, 0].astype(np.int32) * 10 + groups[:, 1].astype(np.int32)).astype(np.int32)


def build_feature_views(
    train_arrays: Dict[str, np.ndarray],
    val_arrays: Dict[str, np.ndarray],
    test_arrays: Dict[str, np.ndarray],
    feature_sets: Sequence[str],
) -> Dict[str, FeatureView]:
    views: Dict[str, FeatureView] = {}
    for feature_set in feature_sets:
        views[feature_set] = FeatureView(
            name=feature_set,
            train_raw=build_feature_matrix(train_arrays, feature_set),
            val_raw=build_feature_matrix(val_arrays, feature_set),
            test_raw=build_feature_matrix(test_arrays, feature_set),
        )
    return views


@dataclass
class RegressorCandidate:
    regressor_name: str
    n_estimators: int
    max_features: object
    min_samples_leaf: int
    feature_set: str
    group_mode: str

    @property
    def name(self) -> str:
        return (
            f"{self.regressor_name}_est{self.n_estimators}_mf{self.max_features}_"
            f"leaf{self.min_samples_leaf}_{self.feature_set}_{self.group_mode}"
        )


class GroupAwareTabularRegressor:
    def __init__(
        self,
        global_scaler: StandardScaler,
        global_regressor,
        local_scaler: StandardScaler,
        local_regressors: Dict[int, object],
        train_group_ids: np.ndarray,
        group_classifier: GroupProbClassifier,
    ) -> None:
        self.global_scaler = global_scaler
        self.global_regressor = global_regressor
        self.local_scaler = local_scaler
        self.local_regressors = local_regressors
        self.train_group_ids = train_group_ids.astype(np.int32)
        self.group_classifier = group_classifier

    def predict(self, raw_features: np.ndarray, classifier_raw_features: np.ndarray, group_mode: str) -> Tuple[np.ndarray, Dict[str, float]]:
        global_feat = self.global_scaler.transform(raw_features).astype(np.float32)
        global_pred = self.global_regressor.predict(global_feat).astype(np.float32)

        if group_mode == "global":
            return global_pred, {"group_mode_used": 0.0}

        local_feat = self.local_scaler.transform(raw_features).astype(np.float32)
        top_groups, top_probs = self.group_classifier.predict_topk(
            classifier_raw_features,
            topn=2 if group_mode == "top2" else 1,
        )

        if group_mode == "top1":
            local_pred = self._predict_local(local_feat, top_groups[:, 0], global_pred)
            conf = np.clip(top_probs[:, 0:1], 0.0, 1.0)
            pred = (conf * local_pred + (1.0 - conf) * global_pred).astype(np.float32)
            return pred, {"group_mode_used": 1.0, "mean_top1_prob": float(top_probs[:, 0].mean())}

        if group_mode == "top2":
            probs = top_probs / np.maximum(top_probs.sum(axis=1, keepdims=True), 1e-6)
            pred = np.zeros_like(global_pred, dtype=np.float32)
            for col in range(top_groups.shape[1]):
                local_pred = self._predict_local(local_feat, top_groups[:, col], global_pred)
                pred += probs[:, col : col + 1] * local_pred
            top1_conf = top_probs[:, 0:1]
            global_mix = np.clip(0.45 - top1_conf, 0.0, 0.45) * 1.2
            pred = ((1.0 - global_mix) * pred + global_mix * global_pred).astype(np.float32)
            return pred, {
                "group_mode_used": 2.0,
                "mean_top1_prob": float(top_probs[:, 0].mean()),
                "mean_global_mix": float(global_mix.mean()),
            }

        raise ValueError(f"Unsupported group_mode: {group_mode}")

    def _predict_local(self, scaled_features: np.ndarray, group_ids: np.ndarray, global_pred: np.ndarray) -> np.ndarray:
        pred = np.zeros((len(scaled_features), 2), dtype=np.float32)
        for group_id in sorted(set(int(g) for g in group_ids.tolist())):
            sample_idx = np.where(group_ids == group_id)[0]
            if len(sample_idx) == 0:
                continue
            if group_id not in self.local_regressors:
                pred[sample_idx] = global_pred[sample_idx]
                continue
            pred[sample_idx] = self.local_regressors[group_id].predict(scaled_features[sample_idx]).astype(np.float32)
        return pred.astype(np.float32)


def build_regressor(name: str, n_estimators: int, max_features, min_samples_leaf: int, seed: int, n_jobs: int):
    common = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "min_samples_leaf": min_samples_leaf,
        "random_state": seed,
        "n_jobs": n_jobs,
    }
    if name == "extra_trees":
        return ExtraTreesRegressor(**common)
    if name == "rf":
        return RandomForestRegressor(**common)
    raise ValueError(f"Unsupported regressor name: {name}")


def parse_regressor_candidates(raw: str) -> List[RegressorCandidate]:
    out: List[RegressorCandidate] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 6:
            raise ValueError(
                f"Invalid regressor candidate '{token}', expected "
                "regressor:n_estimators:max_features:min_samples_leaf:feature_set:group_mode"
            )
        out.append(
            RegressorCandidate(
                regressor_name=parts[0],
                n_estimators=int(parts[1]),
                max_features=parse_max_features(parts[2]),
                min_samples_leaf=int(parts[3]),
                feature_set=parts[4],
                group_mode=parts[5],
            )
        )
    if not out:
        raise ValueError("No regressor candidate parsed.")
    return out


def fit_group_aware_regressor(
    candidate: RegressorCandidate,
    views: Dict[str, FeatureView],
    train_coords: np.ndarray,
    train_group_ids: np.ndarray,
    group_classifier: GroupProbClassifier,
    seed: int,
    n_jobs: int,
) -> GroupAwareTabularRegressor:
    view = views[candidate.feature_set]
    global_scaler = StandardScaler()
    train_scaled_global = global_scaler.fit_transform(view.train_raw).astype(np.float32)
    global_regressor = build_regressor(
        name=candidate.regressor_name,
        n_estimators=candidate.n_estimators,
        max_features=candidate.max_features,
        min_samples_leaf=candidate.min_samples_leaf,
        seed=seed,
        n_jobs=n_jobs,
    )
    global_regressor.fit(train_scaled_global, train_coords)

    local_scaler = StandardScaler()
    train_scaled_local = local_scaler.fit_transform(view.train_raw).astype(np.float32)
    local_regressors: Dict[int, object] = {}
    for group_id in sorted(set(int(g) for g in train_group_ids.tolist())):
        idx = np.where(train_group_ids == group_id)[0]
        if len(idx) < max(8, candidate.min_samples_leaf * 4):
            continue
        local_regressor = build_regressor(
            name=candidate.regressor_name,
            n_estimators=candidate.n_estimators,
            max_features=candidate.max_features,
            min_samples_leaf=candidate.min_samples_leaf,
            seed=seed + int(group_id),
            n_jobs=n_jobs,
        )
        local_regressor.fit(train_scaled_local[idx], train_coords[idx])
        local_regressors[group_id] = local_regressor

    return GroupAwareTabularRegressor(
        global_scaler=global_scaler,
        global_regressor=global_regressor,
        local_scaler=local_scaler,
        local_regressors=local_regressors,
        train_group_ids=train_group_ids,
        group_classifier=group_classifier,
    )


def save_predictions_csv(out_csv: Path, pred: np.ndarray, target: np.ndarray, groups: np.ndarray) -> None:
    err = np.linalg.norm(pred - target, axis=1)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "building", "floor", "true_x", "true_y", "pred_x", "pred_y", "error_m"])
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


def plot_predictions(out_png: Path, pred: np.ndarray, target: np.ndarray, title: str) -> None:
    _setup_matplotlib(out_png.parent)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    error = np.linalg.norm(pred - target, axis=1)
    fig, ax = plt.subplots(figsize=(8.8, 7.0), constrained_layout=True)
    ax.scatter(target[:, 0], target[:, 1], s=10, alpha=0.40, c="#111827", label="True")
    ax.scatter(pred[:, 0], pred[:, 1], s=12, alpha=0.72, c=error, cmap="turbo", label="Pred")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.25)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure RSSI tabular ensemble: group classifier + ExtraTrees/RF coordinate regressors + greedy blending.",
    )
    parser.add_argument("--train-dir", type=str, default="training_dataset")
    parser.add_argument("--test-dir", type=str, default="test_dataset")
    parser.add_argument("--output-dir", type=str, default="runs/rssi_tabular_ensemble")
    parser.add_argument(
        "--feature-sets",
        type=str,
        default="last_mean,robust_last_mean,stat_stack,last_mean_std,temporal_signature,quantile_stack,flatten_stat",
    )
    parser.add_argument(
        "--group-classifier-candidates",
        type=str,
        default="rf:500:sqrt:stat_stack,extra_trees:700:sqrt:stat_stack,extra_trees:700:sqrt:quantile_stack",
    )
    parser.add_argument(
        "--regressor-candidates",
        type=str,
        default=(
            "extra_trees:600:sqrt:1:stat_stack:global,"
            "extra_trees:600:sqrt:1:stat_stack:top1,"
            "extra_trees:900:0.5:1:flatten_stat:global,"
            "extra_trees:900:0.5:1:flatten_stat:top1,"
            "rf:800:sqrt:1:quantile_stack:global,"
            "rf:800:sqrt:2:temporal_signature:top1,"
            "rf:800:0.5:1:last_mean_std:top2"
        ),
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

    train_arrays = load_npz(Path(args.train_dir) / "train_sequences.npz")
    val_arrays = load_npz(Path(args.train_dir) / "val_sequences.npz")
    test_arrays = concat_splits(
        [
            load_npz(Path(args.test_dir) / "train_sequences.npz"),
            load_npz(Path(args.test_dir) / "val_sequences.npz"),
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
    group_classifier_candidates = parse_group_classifier_candidates(args.group_classifier_candidates)
    regressor_candidates = parse_regressor_candidates(args.regressor_candidates)

    print(f"Train/Val/Test samples: {len(train_coords)}/{len(val_coords)}/{len(test_coords)}", flush=True)
    print(f"Feature sets: {feature_sets}", flush=True)
    print(f"Regressor candidates: {len(regressor_candidates)}", flush=True)

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

    required_feature_sets = list(feature_sets)
    classifier_feature_set = str(best_group_row["feature_set"])
    if classifier_feature_set not in required_feature_sets:
        required_feature_sets.append(classifier_feature_set)

    views = build_feature_views(train_arrays, val_arrays, test_arrays, required_feature_sets)
    classifier_view = views[classifier_feature_set]

    candidate_rows: List[Dict[str, object]] = []
    val_prediction_bank: Dict[str, np.ndarray] = {}
    candidate_models: Dict[str, GroupAwareTabularRegressor] = {}

    for idx, cand in enumerate(regressor_candidates, start=1):
        model = fit_group_aware_regressor(
            candidate=cand,
            views=views,
            train_coords=train_coords,
            train_group_ids=train_group_ids,
            group_classifier=best_group_classifier,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        pred_val, diag = model.predict(
            raw_features=views[cand.feature_set].val_raw,
            classifier_raw_features=classifier_view.val_raw,
            group_mode=cand.group_mode,
        )
        reg = regression_metrics(pred_val, val_coords)
        row = {
            "candidate": cand.name,
            "regressor_name": cand.regressor_name,
            "n_estimators": cand.n_estimators,
            "max_features": cand.max_features,
            "min_samples_leaf": cand.min_samples_leaf,
            "feature_set": cand.feature_set,
            "group_mode": cand.group_mode,
            "val_mean_error_m": float(reg["mean_error_m"]),
            "val_p90_error_m": float(reg["p90_error_m"]),
            "val_rmse_m": float(reg["rmse_m"]),
            **diag,
        }
        candidate_rows.append(row)
        val_prediction_bank[cand.name] = pred_val.astype(np.float32)
        candidate_models[cand.name] = model
        print(
            f"[cand {idx:02d}/{len(regressor_candidates)}] {cand.name} => {float(reg['mean_error_m']):.3f} m",
            flush=True,
        )

    best_row = min(candidate_rows, key=lambda row: float(row["val_mean_error_m"]))
    print(
        f"Best single candidate: {best_row['candidate']} "
        f"(val mean={float(best_row['val_mean_error_m']):.3f} m)",
        flush=True,
    )

    ensemble_steps, _, ensemble_val_metrics = greedy_ensemble_search(
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

    full_train_arrays = concat_splits([train_arrays, val_arrays])
    full_train_coords = full_train_arrays["y_last"].astype(np.float32)
    full_train_group_ids = encode_groups(full_train_arrays["group"].astype(np.int32))

    best_gc_cand = GroupClassifierCandidate(
        classifier_name=str(best_group_row["classifier_name"]),
        n_estimators=int(best_group_row["n_estimators"]),
        max_features=best_group_row["max_features"],
        feature_set=str(best_group_row["feature_set"]),
    )

    full_group_classifier, _, _ = fit_best_group_classifier(
        train_arrays=full_train_arrays,
        val_arrays=val_arrays,
        candidates=[best_gc_cand],
        n_jobs=args.n_jobs,
        seed=args.seed,
    )

    full_views = build_feature_views(full_train_arrays, test_arrays, test_arrays, required_feature_sets)
    full_classifier_view = full_views[best_gc_cand.feature_set]
    final_prediction_bank: Dict[str, np.ndarray] = {}
    final_models: Dict[str, GroupAwareTabularRegressor] = {}

    for cand in regressor_candidates:
        model = fit_group_aware_regressor(
            candidate=cand,
            views=full_views,
            train_coords=full_train_coords,
            train_group_ids=full_train_group_ids,
            group_classifier=full_group_classifier,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        pred_test, _ = model.predict(
            raw_features=full_views[cand.feature_set].val_raw,
            classifier_raw_features=full_classifier_view.val_raw,
            group_mode=cand.group_mode,
        )
        final_prediction_bank[cand.name] = pred_test.astype(np.float32)
        final_models[cand.name] = model

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
    plot_predictions(output_dir / "test_best_single_scatter.png", best_single_test_pred, test_coords, "RSSI Tabular Best Single Candidate")
    plot_predictions(output_dir / "test_ensemble_scatter.png", ensemble_test_pred, test_coords, "RSSI Tabular Ensemble")
    plot_error_hist(output_dir / "test_ensemble_error_hist.png", ensemble_test_pred, test_coords, "RSSI Tabular Ensemble Error Distribution")

    bundle = {
        "best_group_classifier": best_group_row,
        "candidate_summaries": sorted(candidate_rows, key=lambda row: float(row["val_mean_error_m"])),
        "best_single_candidate": best_row,
        "ensemble_steps": [step.__dict__ for step in ensemble_steps],
        "test_metrics": {
            "best_single": best_single_test_metrics,
            "ensemble": ensemble_test_metrics,
        },
    }
    with (output_dir / "model_bundle.pkl").open("wb") as handle:
        pickle.dump(bundle, handle)

    metrics_payload = {
        "device": "cpu",
        "model_type": "rssi_tabular_ensemble",
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

    print("\nFinal RSSI Tabular Ensemble Results", flush=True)
    print(
        f"  Best Single : {str(best_row['candidate'])} => {float(best_single_test_metrics['mean_error_m']):.3f} m",
        flush=True,
    )
    print(
        f"  Ensemble    : {float(ensemble_test_metrics['mean_error_m']):.3f} m "
        f"(rmse={float(ensemble_test_metrics['rmse_m']):.3f} m)",
        flush=True,
    )
    print(f"  Metrics     : {output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
