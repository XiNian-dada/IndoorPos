#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def run_cmd(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def parse_extra_args(raw: str) -> List[str]:
    raw = raw.strip()
    if not raw:
        return []
    return shlex.split(raw)


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return float(path.stat().st_size) / (1024.0 * 1024.0)


def collect_rows(metrics_path: Path, label: str) -> List[Dict[str, object]]:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = []
    model_type = str(data.get("model_type", label))

    def make_row(row_label: str, row_type: str, metrics: Dict[str, object], cls_acc: Optional[float], size_mb: float) -> Dict[str, object]:
        return {
            "model": row_label,
            "type": row_type,
            "mean_m": float(metrics.get("mean_error_m", 0.0)),
            "median_m": float(metrics.get("median_error_m", 0.0)),
            "p90_m": float(metrics.get("p90_error_m", 0.0)),
            "p95_m": float(metrics.get("p95_error_m", 0.0)),
            "rmse_m": float(metrics.get("rmse_m", 0.0)),
            "cls_acc": None if cls_acc is None else float(cls_acc),
            "size_mb": float(size_mb),
        }

    if model_type == "rssi_knn":
        bundle_size = file_size_mb(metrics_path.parent / "model_bundle.pkl")
        rows.append(
            make_row(
                label,
                model_type,
                data["test_metrics"],
                data.get("test_classification_accuracy"),
                bundle_size,
            )
        )
        return rows

    if model_type in {"advanced_rssi_ensemble", "rssi_tabular_ensemble"}:
        bundle_size = file_size_mb(metrics_path.parent / "model_bundle.pkl")
        rows.append(
            make_row(
                f"{label}_best_single",
                f"{model_type}_best_single",
                data["best_single_test_metrics"],
                None,
                bundle_size,
            )
        )
        rows.append(
            make_row(
                f"{label}_ensemble",
                f"{model_type}_ensemble",
                data["ensemble_test_metrics"],
                None,
                bundle_size,
            )
        )
        return rows

    if model_type == "rssi_only_high_accuracy_torch":
        ckpt_size = file_size_mb(metrics_path.parent / "best_rssi_only_high_accuracy_torch.pt")
        rows.append(
            make_row(
                label,
                model_type,
                data["test_metrics"],
                data.get("test_classification_accuracy"),
                ckpt_size,
            )
        )
        return rows

    raise ValueError(f"Unsupported metrics schema for {metrics_path}")


def write_summary(rows: List[Dict[str, object]], output_root: Path) -> None:
    rows_sorted = sorted(rows, key=lambda row: float(row["mean_m"]))
    csv_path = output_root / "summary.csv"
    md_path = output_root / "summary.md"
    json_path = output_root / "summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "type", "mean_m", "median_m", "p90_m", "p95_m", "rmse_m", "cls_acc", "size_mb"],
        )
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)

    md_lines = [
        "# Pure RSSI Benchmark Summary",
        "",
        "| model | type | mean(m) | median(m) | p90(m) | p95(m) | rmse(m) | cls_acc | size(MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows_sorted:
        cls_acc = "-" if row["cls_acc"] is None else f"{float(row['cls_acc']):.3f}"
        md_lines.append(
            "| {model} | {type} | {mean_m:.3f} | {median_m:.3f} | {p90_m:.3f} | {p95_m:.3f} | {rmse_m:.3f} | {cls_acc} | {size_mb:.2f} |".format(
                **row,
                cls_acc=cls_acc,
            )
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure RSSI localization benchmarks and summarize test-set performance.")
    parser.add_argument("--train-dir", type=str, default="training_dataset")
    parser.add_argument("--test-dir", type=str, default="test_dataset")
    parser.add_argument("--output-root", type=str, default="runs/pure_rssi_bench")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--run-knn", action="store_true")
    parser.add_argument("--run-advanced", action="store_true")
    parser.add_argument("--run-tabular", action="store_true")
    parser.add_argument("--run-torch", action="store_true")
    parser.add_argument("--knn-extra-args", type=str, default="--feature-set robust_last_mean --k-candidates 1,3,5,7,9,11,15,21 --weighted")
    parser.add_argument(
        "--advanced-extra-args",
        type=str,
        default=(
            "--feature-sets last,mean,last_mean,robust_last_mean,stat_stack,last_mean_std,temporal_signature,quantile_stack,flatten_stat "
            "--metrics euclidean,manhattan,cosine "
            "--k-candidates 1,3,5,7,9,11,15,21 "
            "--aggregators idw,idw2,kernel,softmax,trimmed_idw,median,lle "
            "--group-modes global,top1,top2 "
            "--group-classifier-candidates rf:500:sqrt:stat_stack,extra_trees:700:sqrt:stat_stack,extra_trees:700:sqrt:quantile_stack "
            "--ensemble-max-candidates 10 --ensemble-max-steps 5 --n-jobs -1"
        ),
    )
    parser.add_argument(
        "--tabular-extra-args",
        type=str,
        default=(
            "--feature-sets last_mean,robust_last_mean,stat_stack,last_mean_std,temporal_signature,quantile_stack,flatten_stat "
            "--group-classifier-candidates rf:500:sqrt:stat_stack,extra_trees:700:sqrt:stat_stack,extra_trees:700:sqrt:quantile_stack "
            "--regressor-candidates "
            "\"extra_trees:600:sqrt:1:stat_stack:global,"
            "extra_trees:600:sqrt:1:stat_stack:top1,"
            "extra_trees:900:0.5:1:flatten_stat:global,"
            "extra_trees:900:0.5:1:flatten_stat:top1,"
            "rf:800:sqrt:1:quantile_stack:global,"
            "rf:800:sqrt:2:temporal_signature:top1,"
            "rf:800:0.5:1:last_mean_std:top2\" "
            "--ensemble-max-candidates 8 --ensemble-max-steps 4 --n-jobs -1"
        ),
    )
    parser.add_argument(
        "--torch-extra-args",
        type=str,
        default=(
            "--device auto --epochs 140 --batch-size 384 --patience 16 "
            "--candidates 192:192:2:256:0.15,256:256:2:320:0.18,320:320:2:384:0.20"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    do_all = not (args.run_knn or args.run_advanced or args.run_tabular or args.run_torch)
    run_knn = args.run_knn or do_all
    run_advanced = args.run_advanced or do_all
    run_tabular = args.run_tabular or do_all
    run_torch = args.run_torch

    rows: List[Dict[str, object]] = []

    jobs = [
        (
            "rssi_knn",
            run_knn,
            output_root / "rssi_knn",
            ["python3", "TrainRSSIKNNModel.py", "--train-dir", args.train_dir, "--test-dir", args.test_dir, "--output-dir", str(output_root / "rssi_knn")] + parse_extra_args(args.knn_extra_args),
        ),
        (
            "advanced_rssi",
            run_advanced,
            output_root / "advanced_rssi",
            ["python3", "TrainAdvancedRSSIEnsemble.py", "--train-dir", args.train_dir, "--test-dir", args.test_dir, "--output-dir", str(output_root / "advanced_rssi")] + parse_extra_args(args.advanced_extra_args),
        ),
        (
            "tabular_rssi",
            run_tabular,
            output_root / "tabular_rssi",
            ["python3", "TrainRSSITabularEnsemble.py", "--train-dir", args.train_dir, "--test-dir", args.test_dir, "--output-dir", str(output_root / "tabular_rssi")] + parse_extra_args(args.tabular_extra_args),
        ),
        (
            "rssi_torch",
            run_torch,
            output_root / "rssi_torch",
            ["python3", "TrainRSSIOnlyHighAccuracyTorch.py", "--train-dir", args.train_dir, "--test-dir", args.test_dir, "--output-dir", str(output_root / "rssi_torch")] + parse_extra_args(args.torch_extra_args),
        ),
    ]

    for label, enabled, out_dir, cmd in jobs:
        if not enabled:
            continue
        metrics_path = out_dir / "metrics.json"
        if args.skip_train and not metrics_path.exists():
            raise FileNotFoundError(f"--skip-train set but missing metrics: {metrics_path}")
        if not args.skip_train:
            run_cmd(cmd)
        rows.extend(collect_rows(metrics_path, label))

    write_summary(rows, output_root)
    print("\n=== Pure RSSI Benchmark Summary ===", flush=True)
    print((output_root / "summary.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
