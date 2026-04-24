#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class TinySpec:
    name: str
    arch: str
    candidate: str
    seed: int


def parse_tiny_specs(raw: str) -> List[TinySpec]:
    specs: List[TinySpec] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token or "@" not in token:
            raise ValueError(
                "Invalid tiny spec "
                f"'{token}'. Expected format: name=[arch|]candidate@seed"
            )
        name, rest = token.split("=", 1)
        candidate_part, seed_str = rest.rsplit("@", 1)
        arch = "dscnn"
        candidate = candidate_part
        if "|" in candidate_part:
            arch_part, candidate = candidate_part.split("|", 1)
            arch = arch_part.strip().lower()
        if arch not in {"dscnn", "gru", "tcn"}:
            raise ValueError(
                f"Invalid tiny arch '{arch}' in '{token}', expected dscnn|gru|tcn."
            )
        specs.append(TinySpec(name=name, arch=arch, candidate=candidate, seed=int(seed_str)))
    if not specs:
        raise ValueError("No tiny specs parsed.")
    return specs


def run_cmd(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def script_path(name: str) -> str:
    return str(SCRIPT_DIR / name)


def tiny_metrics_path(output_dir: Path) -> Path:
    return output_dir / "metrics.json"


def tiny_needs_train(spec: TinySpec, args: argparse.Namespace, output_dir: Path) -> bool:
    if args.skip_train and (output_dir / "metrics.json").exists():
        print(f"[skip] tiny {spec.name}: metrics exists at {output_dir / 'metrics.json'}", flush=True)
        return False
    if args.skip_train:
        raise FileNotFoundError(
            f"--skip-train is set but missing metrics: {output_dir / 'metrics.json'}"
        )
    return True


def build_tiny_cmd(
    spec: TinySpec,
    args: argparse.Namespace,
    output_dir: Path,
) -> List[str]:
    cmd: List[str] = [
        "python3",
        script_path("TrainTinyESP32Model.py"),
        "--train-dir", args.train_dir,
        "--test-dir", args.test_dir,
        "--output-dir", str(output_dir),
        "--epochs", str(args.tiny_epochs),
        "--batch-size", str(args.tiny_batch_size),
        "--patience", str(args.tiny_patience),
        "--candidate-configs", spec.candidate,
        "--model-arch", spec.arch,
        "--seed", str(spec.seed),
        "--no-progress",
    ]
    if args.cpu_only:
        cmd.append("--cpu-only")
    return cmd


def launch_tiny_process(
    spec: TinySpec,
    args: argparse.Namespace,
    output_dir: Path,
    gpu_id: Optional[str],
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_tiny_cmd(spec=spec, args=args, output_dir=output_dir)
    env = os.environ.copy()
    if gpu_id is not None and not args.cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

    log_path = output_dir / "train.log"
    log_handle = log_path.open("w", encoding="utf-8")
    print(
        f"[launch] tiny {spec.name} on GPU {gpu_id if gpu_id is not None else 'CPU'}",
        flush=True,
    )
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return {
        "proc": proc,
        "spec": spec,
        "output_dir": output_dir,
        "gpu_id": gpu_id,
        "log_path": log_path,
        "log_handle": log_handle,
        "start_time": time.time(),
    }


def _read_log_tail(path: Path, n_lines: int = 40) -> str:
    if not path.exists():
        return "<log file not found>"
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-n_lines:])


def run_tiny_parallel(
    specs: List[TinySpec],
    args: argparse.Namespace,
    output_root: Path,
    gpu_ids: List[Optional[str]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    pending: List[Tuple[TinySpec, Path]] = []

    for spec in specs:
        out_dir = output_root / f"{spec.name}_seed{spec.seed}"
        need_train = tiny_needs_train(spec=spec, args=args, output_dir=out_dir)
        if need_train:
            pending.append((spec, out_dir))
        else:
            rows.append(collect_tiny_row(name=spec.name, output_dir=out_dir))

    if not pending:
        return rows

    if args.cpu_only:
        max_parallel = max(1, args.tiny_max_parallel) if args.tiny_max_parallel > 0 else 1
    else:
        default_parallel = max(1, len(gpu_ids))
        max_parallel = default_parallel if args.tiny_max_parallel <= 0 else max(1, args.tiny_max_parallel)
        max_parallel = min(max_parallel, max(1, len(gpu_ids)))

    running: List[Dict[str, object]] = []

    while pending or running:
        # Launch as many new tasks as possible.
        while pending and len(running) < max_parallel:
            spec, out_dir = pending.pop(0)
            if args.cpu_only:
                gpu_id = None
            else:
                used = {str(item["gpu_id"]) for item in running if item["gpu_id"] is not None}
                free = [g for g in gpu_ids if g is not None and str(g) not in used]
                if not free:
                    # No free GPU slot now.
                    pending.insert(0, (spec, out_dir))
                    break
                gpu_id = str(free[0])

            running.append(
                launch_tiny_process(
                    spec=spec,
                    args=args,
                    output_dir=out_dir,
                    gpu_id=gpu_id,
                )
            )

        # Poll running tasks.
        finished: List[Dict[str, object]] = []
        for item in running:
            proc: subprocess.Popen = item["proc"]  # type: ignore[assignment]
            code = proc.poll()
            if code is None:
                continue
            finished.append(item)
            log_handle = item["log_handle"]
            log_handle.close()
            spec: TinySpec = item["spec"]  # type: ignore[assignment]
            out_dir: Path = item["output_dir"]  # type: ignore[assignment]
            gpu_id = item["gpu_id"]
            elapsed_s = time.time() - float(item["start_time"])
            if code != 0:
                tail = _read_log_tail(item["log_path"])  # type: ignore[arg-type]
                # Stop any remaining jobs so failure is explicit and early.
                for r in running:
                    p: subprocess.Popen = r["proc"]  # type: ignore[assignment]
                    if p.poll() is None:
                        p.terminate()
                        r["log_handle"].close()
                raise RuntimeError(
                    f"Tiny job failed: {spec.name} (gpu={gpu_id}, code={code})\n"
                    f"log: {item['log_path']}\n"
                    f"--- tail ---\n{tail}"
                )
            print(
                f"[done] tiny {spec.name} on GPU {gpu_id if gpu_id is not None else 'CPU'} "
                f"in {elapsed_s:.1f}s",
                flush=True,
            )
            rows.append(collect_tiny_row(name=spec.name, output_dir=out_dir))

        if finished:
            running = [x for x in running if x not in finished]
        else:
            time.sleep(2.0)

    return rows


def maybe_run_high_accuracy(args: argparse.Namespace, output_dir: Path) -> None:
    if args.skip_train and (output_dir / "metrics.json").exists():
        print(f"[skip] high-accuracy: metrics exists at {output_dir / 'metrics.json'}", flush=True)
        return
    if args.skip_train:
        raise FileNotFoundError(
            f"--skip-train is set but missing metrics: {output_dir / 'metrics.json'}"
        )

    cmd = [
        "python3",
        script_path("TrainHighAccuracyModel.py"),
        "--train-dir", args.train_dir,
        "--test-dir", args.test_dir,
        "--output-dir", str(output_dir),
        "--candidates", args.high_candidates,
        "--seed", str(args.high_seed),
        "--n-jobs", str(args.high_n_jobs),
    ]
    run_cmd(cmd)


def maybe_run_high_accuracy_torch(args: argparse.Namespace, output_dir: Path) -> None:
    if args.skip_train and (output_dir / "metrics.json").exists():
        print(f"[skip] high-accuracy-torch: metrics exists at {output_dir / 'metrics.json'}", flush=True)
        return
    if args.skip_train:
        raise FileNotFoundError(
            f"--skip-train is set but missing metrics: {output_dir / 'metrics.json'}"
        )

    cmd = [
        "python3",
        script_path("TrainHighAccuracyTorchModel.py"),
        "--train-dir", args.train_dir,
        "--test-dir", args.test_dir,
        "--output-dir", str(output_dir),
        "--candidates", args.high_torch_candidates,
        "--epochs", str(args.high_torch_epochs),
        "--batch-size", str(args.high_torch_batch_size),
        "--seed", str(args.high_torch_seed),
        "--num-workers", str(args.high_torch_num_workers),
        "--no-progress",
    ]
    env = os.environ.copy()
    if (not args.cpu_only) and args.high_torch_gpu_id.strip():
        env["CUDA_VISIBLE_DEVICES"] = args.high_torch_gpu_id.strip()
        print(
            f"[launch] high-accuracy-torch on GPU {args.high_torch_gpu_id.strip()}",
            flush=True,
        )
    else:
        print("[launch] high-accuracy-torch on CPU", flush=True)
        cmd.append("--cpu-only")
    run_cmd(cmd, env=env)


def maybe_run_lightweight_zoo(args: argparse.Namespace, output_dir: Path) -> None:
    if args.skip_train and (output_dir / "metrics.json").exists():
        print(f"[skip] lightweight-zoo: metrics exists at {output_dir / 'metrics.json'}", flush=True)
        return
    if args.skip_train:
        raise FileNotFoundError(
            f"--skip-train is set but missing metrics: {output_dir / 'metrics.json'}"
        )

    cmd = [
        "python3",
        script_path("TrainLightweightSchemeZoo.py"),
        "--train-dir", args.train_dir,
        "--test-dir", args.test_dir,
        "--output-dir", str(output_dir),
        "--candidates", args.lightweight_candidates,
        "--epochs", str(args.lightweight_epochs),
        "--batch-size", str(args.lightweight_batch_size),
        "--patience", str(args.lightweight_patience),
        "--seed", str(args.lightweight_seed),
        "--num-workers", str(args.lightweight_num_workers),
        "--no-progress",
    ]
    env = os.environ.copy()
    if (not args.cpu_only) and args.lightweight_gpu_id.strip():
        env["CUDA_VISIBLE_DEVICES"] = args.lightweight_gpu_id.strip()
        print(
            f"[launch] lightweight-zoo on GPU {args.lightweight_gpu_id.strip()}",
            flush=True,
        )
    else:
        print("[launch] lightweight-zoo on CPU", flush=True)
        cmd.append("--cpu-only")
    run_cmd(cmd, env=env)


def maybe_run_article_model(args: argparse.Namespace, output_dir: Path) -> None:
    if args.skip_train and (output_dir / "metrics.json").exists():
        print(f"[skip] article-model: metrics exists at {output_dir / 'metrics.json'}", flush=True)
        return
    if args.skip_train:
        raise FileNotFoundError(
            f"--skip-train is set but missing metrics: {output_dir / 'metrics.json'}"
        )

    train_dir = args.article_train_dir.strip() or args.train_dir
    test_dir = args.article_test_dir.strip() or args.test_dir
    cmd = [
        "python3",
        script_path("TrainArticleTrajectoryModel.py"),
        "--train-dir", train_dir,
        "--test-dir", test_dir,
        "--output-dir", str(output_dir),
        "--candidates", args.article_candidates,
        "--epochs", str(args.article_epochs),
        "--batch-size", str(args.article_batch_size),
        "--lr", str(args.article_lr),
        "--weight-decay", str(args.article_weight_decay),
        "--patience", str(args.article_patience),
        "--seed", str(args.article_seed),
        "--num-workers", str(args.article_num_workers),
        "--cls-loss-weight", str(args.article_cls_loss_weight),
        "--label-smoothing", str(args.article_label_smoothing),
        "--grid-cell-size", str(args.article_grid_cell_size),
        "--grid-margin", str(args.article_grid_margin),
        "--selection-metric", args.article_selection_metric,
        "--speed-cap-scale", str(args.article_speed_cap_scale),
        "--speed-cap-min", str(args.article_speed_cap_min),
        "--speed-cap-quantile", str(args.article_speed_cap_quantile),
        "--speed-cap-multiplier", str(args.article_speed_cap_multiplier),
        "--kalman-process-var", str(args.article_kalman_process_var),
        "--kalman-measurement-var", str(args.article_kalman_measurement_var),
        "--no-progress",
    ]
    if args.article_disable_postprocess:
        cmd.append("--disable-postprocess")
    env = os.environ.copy()
    if (not args.cpu_only) and args.article_gpu_id.strip():
        env["CUDA_VISIBLE_DEVICES"] = args.article_gpu_id.strip()
        print(
            f"[launch] article-model on GPU {args.article_gpu_id.strip()}",
            flush=True,
        )
    else:
        print("[launch] article-model on CPU", flush=True)
        cmd.append("--cpu-only")
    run_cmd(cmd, env=env)


def maybe_run_rssi_knn(args: argparse.Namespace, output_dir: Path) -> None:
    if args.skip_train and (output_dir / "metrics.json").exists():
        print(f"[skip] rssi-knn: metrics exists at {output_dir / 'metrics.json'}", flush=True)
        return
    if args.skip_train:
        raise FileNotFoundError(
            f"--skip-train is set but missing metrics: {output_dir / 'metrics.json'}"
        )

    cmd = [
        "python3",
        script_path("TrainRSSIKNNModel.py"),
        "--train-dir", args.train_dir,
        "--test-dir", args.test_dir,
        "--output-dir", str(output_dir),
        "--feature-set", args.rssi_knn_feature_set,
        "--k-candidates", args.rssi_knn_k_candidates,
    ]
    if args.rssi_knn_weighted:
        cmd.append("--weighted")
    if args.rssi_knn_group_aware:
        cmd.append("--group-aware")
    if args.rssi_knn_no_scale:
        cmd.append("--no-scale")
    if args.rssi_knn_enable_temporal_filter:
        cmd.append("--enable-temporal-filter")
    if args.rssi_knn_auto_tune_temporal:
        cmd.append("--auto-tune-temporal")
    cmd.extend(
        [
            "--temporal-method", args.rssi_knn_temporal_method,
            "--temporal-window", str(args.rssi_knn_temporal_window),
            "--temporal-ema-alpha", str(args.rssi_knn_temporal_ema_alpha),
            "--temporal-method-candidates", args.rssi_knn_temporal_method_candidates,
            "--temporal-window-candidates", args.rssi_knn_temporal_window_candidates,
            "--temporal-ema-alpha-candidates", args.rssi_knn_temporal_ema_alpha_candidates,
        ]
    )
    run_cmd(cmd)


def maybe_run_tiny_consensus(
    args: argparse.Namespace,
    spec: TinySpec,
    tiny_output_dir: Path,
    gpu_id: Optional[str],
) -> Path:
    consensus_dir = tiny_output_dir / "consensus_eval"
    out_json = consensus_dir / "consensus_metrics.json"

    if args.skip_train and out_json.exists():
        print(
            f"[skip] tiny consensus {spec.name}: metrics exists at {out_json}",
            flush=True,
        )
        return consensus_dir
    if args.skip_train:
        raise FileNotFoundError(
            f"--skip-train is set but missing metrics: {out_json}"
        )

    checkpoint = tiny_output_dir / "best_tiny_esp32_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing tiny checkpoint for consensus: {checkpoint}"
        )

    cmd = [
        "python3",
        script_path("EvaluateTinyConsensus.py"),
        "--checkpoint", str(checkpoint),
        "--test-dir", args.test_dir,
        "--output-dir", str(consensus_dir),
        "--batch-size", str(args.consensus_batch_size),
        "--num-workers", str(args.consensus_num_workers),
        "--n-runs", str(args.consensus_n_runs),
        "--radius-m", str(args.consensus_radius_m),
        "--noise-std", str(args.consensus_noise_std),
    ]
    if args.cpu_only:
        cmd.append("--cpu-only")

    env = os.environ.copy()
    if (gpu_id is not None) and (not args.cpu_only):
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(
        f"[consensus] tiny {spec.name} on GPU {gpu_id if gpu_id is not None else 'CPU'}",
        flush=True,
    )
    run_cmd(cmd, env=env)
    return consensus_dir


def file_size_bytes(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0


def collect_tiny_row(name: str, output_dir: Path) -> Dict[str, object]:
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    int8_npz = output_dir / "esp32_tiny_model_int8.npz"
    ckpt = output_dir / "best_tiny_esp32_model.pt"

    return {
        "model_name": name,
        "model_type": "tiny",
        "output_dir": str(output_dir),
        "test_mean_error_m": float(metrics["test_mean_error_m"]),
        "test_median_error_m": float(metrics["test_median_error_m"]),
        "test_p90_error_m": float(metrics["test_p90_error_m"]),
        "test_p95_error_m": float(metrics["test_p95_error_m"]),
        "test_rmse_m": float(metrics["test_rmse_m"]),
        "test_classification_accuracy": float(metrics["test_classification_accuracy"]),
        "param_count": int(metrics.get("param_count", 0)),
        "artifact_size_bytes": file_size_bytes(int8_npz) or file_size_bytes(ckpt),
    }


def collect_high_row(output_dir: Path) -> Dict[str, object]:
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    model_bundle = output_dir / "model_bundle.pkl"
    test = metrics["test_metrics"]

    return {
        "model_name": "high_accuracy",
        "model_type": "high_accuracy",
        "output_dir": str(output_dir),
        "test_mean_error_m": float(test["mean_error_m"]),
        "test_median_error_m": float(test["median_error_m"]),
        "test_p90_error_m": float(test["p90_error_m"]),
        "test_p95_error_m": float(test["p95_error_m"]),
        "test_rmse_m": float(test["rmse_m"]),
        "test_classification_accuracy": float(metrics["test_classification_accuracy"]),
        "param_count": None,
        "artifact_size_bytes": file_size_bytes(model_bundle),
    }


def collect_high_torch_row(output_dir: Path) -> Dict[str, object]:
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    model_ckpt = output_dir / "best_high_accuracy_torch_model.pt"
    test = metrics["test_metrics"]
    return {
        "model_name": "high_accuracy_torch",
        "model_type": "high_accuracy_torch",
        "output_dir": str(output_dir),
        "test_mean_error_m": float(test["mean_error_m"]),
        "test_median_error_m": float(test["median_error_m"]),
        "test_p90_error_m": float(test["p90_error_m"]),
        "test_p95_error_m": float(test["p95_error_m"]),
        "test_rmse_m": float(test["rmse_m"]),
        "test_classification_accuracy": float(metrics["test_classification_accuracy"]),
        "param_count": int(metrics.get("param_count", 0)),
        "artifact_size_bytes": file_size_bytes(model_ckpt),
    }


def collect_rssi_knn_row(output_dir: Path) -> Dict[str, object]:
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    model_bundle = output_dir / "model_bundle.pkl"
    test = metrics["test_metrics"]
    return {
        "model_name": "rssi_knn",
        "model_type": "rssi_knn",
        "output_dir": str(output_dir),
        "test_mean_error_m": float(test["mean_error_m"]),
        "test_median_error_m": float(test["median_error_m"]),
        "test_p90_error_m": float(test["p90_error_m"]),
        "test_p95_error_m": float(test["p95_error_m"]),
        "test_rmse_m": float(test["rmse_m"]),
        "test_classification_accuracy": float(metrics["test_classification_accuracy"]),
        "param_count": None,
        "artifact_size_bytes": file_size_bytes(model_bundle),
    }


def collect_lightweight_rows(output_dir: Path) -> List[Dict[str, object]]:
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = []
    for item in metrics.get("scheme_results", []):
        test = item["test_metrics"]
        ckpt_path = Path(str(item.get("checkpoint_path", "")))
        rows.append(
            {
                "model_name": str(item["model_name"]),
                "model_type": "lightweight",
                "output_dir": str(output_dir),
                "test_mean_error_m": float(test["mean_error_m"]),
                "test_median_error_m": float(test["median_error_m"]),
                "test_p90_error_m": float(test["p90_error_m"]),
                "test_p95_error_m": float(test["p95_error_m"]),
                "test_rmse_m": float(test["rmse_m"]),
                "test_classification_accuracy": float(item["test_classification_accuracy"]),
                "param_count": int(item.get("param_count", 0)),
                "artifact_size_bytes": file_size_bytes(ckpt_path),
            }
        )
    return rows


def collect_article_rows(
    output_dir: Path,
    include_raw: bool,
) -> List[Dict[str, object]]:
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = []
    for item in metrics.get("scheme_results", []):
        ckpt_path = Path(str(item.get("checkpoint_path", "")))
        post = item["test_metrics_post"]
        rows.append(
            {
                "model_name": f"{item['model_name']}_post",
                "model_type": "article_post",
                "output_dir": str(output_dir),
                "test_mean_error_m": float(post["mean_error_m"]),
                "test_median_error_m": float(post["median_error_m"]),
                "test_p90_error_m": float(post["p90_error_m"]),
                "test_p95_error_m": float(post["p95_error_m"]),
                "test_rmse_m": float(post["rmse_m"]),
                "test_classification_accuracy": float(item.get("test_grid_accuracy", 0.0)),
                "param_count": int(item.get("param_count", 0)),
                "artifact_size_bytes": file_size_bytes(ckpt_path),
            }
        )
        if include_raw:
            raw = item["test_metrics_raw"]
            rows.append(
                {
                    "model_name": f"{item['model_name']}_raw",
                    "model_type": "article_raw",
                    "output_dir": str(output_dir),
                    "test_mean_error_m": float(raw["mean_error_m"]),
                    "test_median_error_m": float(raw["median_error_m"]),
                    "test_p90_error_m": float(raw["p90_error_m"]),
                    "test_p95_error_m": float(raw["p95_error_m"]),
                    "test_rmse_m": float(raw["rmse_m"]),
                    "test_classification_accuracy": float(item.get("test_grid_accuracy", 0.0)),
                    "param_count": int(item.get("param_count", 0)),
                    "artifact_size_bytes": file_size_bytes(ckpt_path),
                }
            )
    return rows


def collect_tiny_consensus_row(
    name: str,
    tiny_output_dir: Path,
    consensus_output_dir: Path,
) -> Dict[str, object]:
    baseline_metrics = json.loads((tiny_output_dir / "metrics.json").read_text(encoding="utf-8"))
    consensus_metrics = json.loads(
        (consensus_output_dir / "consensus_metrics.json").read_text(encoding="utf-8")
    )
    cons = consensus_metrics["consensus"]
    int8_npz = tiny_output_dir / "esp32_tiny_model_int8.npz"
    ckpt = tiny_output_dir / "best_tiny_esp32_model.pt"

    return {
        "model_name": f"{name}_consensus",
        "model_type": "tiny_consensus",
        "output_dir": str(consensus_output_dir),
        "test_mean_error_m": float(cons["mean_error_m"]),
        "test_median_error_m": float(cons["median_error_m"]),
        "test_p90_error_m": float(cons["p90_error_m"]),
        "test_p95_error_m": float(cons["p95_error_m"]),
        "test_rmse_m": float(cons["rmse_m"]),
        "test_classification_accuracy": float(baseline_metrics["test_classification_accuracy"]),
        "param_count": int(baseline_metrics.get("param_count", 0)),
        "artifact_size_bytes": file_size_bytes(int8_npz) or file_size_bytes(ckpt),
    }


def write_summary(rows: List[Dict[str, object]], out_dir: Path) -> None:
    rows = sorted(rows, key=lambda x: float(x["test_mean_error_m"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(
        json.dumps({"models": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = out_dir / "summary.csv"
    fields = [
        "model_name",
        "model_type",
        "output_dir",
        "test_mean_error_m",
        "test_median_error_m",
        "test_p90_error_m",
        "test_p95_error_m",
        "test_rmse_m",
        "test_classification_accuracy",
        "param_count",
        "artifact_size_bytes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# Model Test Summary",
        "",
        "| model | type | mean(m) | median(m) | p90(m) | p95(m) | rmse(m) | cls_acc | size(MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        size_mb = float(r["artifact_size_bytes"]) / (1024 * 1024)
        lines.append(
            f"| {r['model_name']} | {r['model_type']} | "
            f"{float(r['test_mean_error_m']):.3f} | {float(r['test_median_error_m']):.3f} | "
            f"{float(r['test_p90_error_m']):.3f} | {float(r['test_p95_error_m']):.3f} | "
            f"{float(r['test_rmse_m']):.3f} | {float(r['test_classification_accuracy']):.3f} | "
            f"{size_mb:.2f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full indoor-position model zoo benchmark and summarize test-set metrics."
    )
    parser.add_argument("--train-dir", type=str, default="training_dataset")
    parser.add_argument("--test-dir", type=str, default="test_dataset")
    parser.add_argument("--output-root", type=str, default="runs/server_model_zoo")

    parser.add_argument(
        "--tiny-specs",
        type=str,
        default=(
            "tiny_s=16:24@7,"
            "tiny_m=24:32@7,"
            "tiny_l=32:48@7,"
            "tiny_xl=40:64:64@7"
        ),
        help="Comma-separated specs: name=[arch|]candidate@seed (arch: dscnn|gru|tcn)",
    )
    parser.add_argument("--tiny-epochs", type=int, default=100)
    parser.add_argument("--tiny-batch-size", type=int, default=256)
    parser.add_argument("--tiny-patience", type=int, default=12)
    parser.add_argument(
        "--tiny-gpu-ids",
        type=str,
        default="0",
        help="Comma-separated GPU ids used in round-robin for tiny runs, e.g. '0,1'",
    )
    parser.add_argument(
        "--tiny-max-parallel",
        type=int,
        default=0,
        help="Max concurrent tiny jobs. 0 means auto (= number of GPU ids, or 1 on CPU).",
    )

    parser.add_argument("--run-high-accuracy", action="store_true")
    parser.add_argument(
        "--high-candidates",
        type=str,
        default=(
            "rf:500:sqrt:11:5,"
            "rf:500:sqrt:15:5,"
            "rf:500:sqrt:21:5,"
            "rf:800:sqrt:15:5,"
            "extra_trees:500:sqrt:15:5"
        ),
    )
    parser.add_argument("--high-seed", type=int, default=42)
    parser.add_argument("--high-n-jobs", type=int, default=-1)
    parser.add_argument("--run-high-accuracy-torch", action="store_true")
    parser.add_argument(
        "--high-torch-candidates",
        type=str,
        default=(
            "192:192:2:256:0.15,"
            "256:256:2:320:0.18,"
            "320:320:2:384:0.20"
        ),
    )
    parser.add_argument("--high-torch-epochs", type=int, default=140)
    parser.add_argument("--high-torch-batch-size", type=int, default=384)
    parser.add_argument("--high-torch-seed", type=int, default=42)
    parser.add_argument("--high-torch-num-workers", type=int, default=0)
    parser.add_argument(
        "--high-torch-gpu-id",
        type=str,
        default="0",
        help="GPU id for TrainHighAccuracyTorchModel.py when not --cpu-only.",
    )
    parser.add_argument("--run-rssi-knn", action="store_true")
    parser.add_argument(
        "--rssi-knn-feature-set",
        type=str,
        default="last_mean",
        choices=["last", "mean", "flatten", "last_mean", "robust_last_mean"],
    )
    parser.add_argument("--rssi-knn-k-candidates", type=str, default="1,3,5,7,9,11,15,21")
    parser.add_argument("--rssi-knn-weighted", action="store_true")
    parser.add_argument("--rssi-knn-group-aware", action="store_true")
    parser.add_argument("--rssi-knn-no-scale", action="store_true")
    parser.add_argument("--rssi-knn-enable-temporal-filter", action="store_true")
    parser.add_argument("--rssi-knn-auto-tune-temporal", action="store_true")
    parser.add_argument(
        "--rssi-knn-temporal-method",
        type=str,
        default="hybrid",
        choices=["none", "mean", "median", "ema", "hybrid"],
    )
    parser.add_argument("--rssi-knn-temporal-window", type=int, default=5)
    parser.add_argument("--rssi-knn-temporal-ema-alpha", type=float, default=0.35)
    parser.add_argument(
        "--rssi-knn-temporal-method-candidates",
        type=str,
        default="median,ema,hybrid",
    )
    parser.add_argument("--rssi-knn-temporal-window-candidates", type=str, default="3,5,7")
    parser.add_argument(
        "--rssi-knn-temporal-ema-alpha-candidates",
        type=str,
        default="0.25,0.35,0.5",
    )
    parser.add_argument("--run-lightweight-zoo", action="store_true")
    parser.add_argument(
        "--lightweight-candidates",
        type=str,
        default=(
            "set_tcn:12:24:48:64:0.10,"
            "cnn_tcn:12:24:48:64:0.10,"
            "pure_tcn:12:24:48:64:0.10"
        ),
    )
    parser.add_argument("--lightweight-epochs", type=int, default=100)
    parser.add_argument("--lightweight-batch-size", type=int, default=256)
    parser.add_argument("--lightweight-patience", type=int, default=12)
    parser.add_argument("--lightweight-seed", type=int, default=42)
    parser.add_argument("--lightweight-num-workers", type=int, default=0)
    parser.add_argument(
        "--lightweight-gpu-id",
        type=str,
        default="0",
        help="GPU id for TrainLightweightSchemeZoo.py when not --cpu-only.",
    )
    parser.add_argument("--run-article-model", action="store_true")
    parser.add_argument(
        "--article-train-dir",
        type=str,
        default="",
        help="Optional dataset dir for article model. Empty means --train-dir.",
    )
    parser.add_argument(
        "--article-test-dir",
        type=str,
        default="",
        help="Optional dataset dir for article model. Empty means --test-dir.",
    )
    parser.add_argument(
        "--article-candidates",
        type=str,
        default=(
            "set_tcn:16:48:96:128:0.10,"
            "cnn_tcn:16:48:96:128:0.10,"
            "pure_tcn:16:48:96:128:0.10,"
            "cnn_tcn:20:64:128:192:0.08,"
            "pure_tcn:20:64:128:192:0.08"
        ),
    )
    parser.add_argument("--article-epochs", type=int, default=120)
    parser.add_argument("--article-batch-size", type=int, default=256)
    parser.add_argument("--article-lr", type=float, default=6e-4)
    parser.add_argument("--article-weight-decay", type=float, default=5e-5)
    parser.add_argument("--article-patience", type=int, default=14)
    parser.add_argument("--article-seed", type=int, default=11)
    parser.add_argument("--article-num-workers", type=int, default=0)
    parser.add_argument("--article-cls-loss-weight", type=float, default=0.12)
    parser.add_argument("--article-label-smoothing", type=float, default=0.03)
    parser.add_argument("--article-grid-cell-size", type=float, default=24.0)
    parser.add_argument("--article-grid-margin", type=float, default=1.0)
    parser.add_argument(
        "--article-selection-metric",
        type=str,
        default="post",
        choices=["raw", "post"],
    )
    parser.add_argument("--article-disable-postprocess", action="store_true")
    parser.add_argument("--article-speed-cap-scale", type=float, default=1.15)
    parser.add_argument("--article-speed-cap-min", type=float, default=0.2)
    parser.add_argument("--article-speed-cap-quantile", type=float, default=0.995)
    parser.add_argument("--article-speed-cap-multiplier", type=float, default=1.1)
    parser.add_argument("--article-kalman-process-var", type=float, default=6.0)
    parser.add_argument("--article-kalman-measurement-var", type=float, default=16.0)
    parser.add_argument(
        "--article-gpu-id",
        type=str,
        default="0",
        help="GPU id for TrainArticleTrajectoryModel.py when not --cpu-only.",
    )
    parser.add_argument(
        "--article-include-raw-rows",
        action="store_true",
        help="Append article raw (no postprocess) metrics rows to summary in addition to post rows.",
    )
    parser.add_argument(
        "--run-tiny-consensus",
        action="store_true",
        help="Run N-time dense-cluster consensus inference for each tiny model and append results to summary.",
    )
    parser.add_argument("--consensus-n-runs", type=int, default=11)
    parser.add_argument("--consensus-radius-m", type=float, default=4.0)
    parser.add_argument("--consensus-noise-std", type=float, default=0.02)
    parser.add_argument("--consensus-batch-size", type=int, default=512)
    parser.add_argument("--consensus-num-workers", type=int, default=0)

    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    tiny_specs = parse_tiny_specs(args.tiny_specs)
    gpu_ids = [x.strip() for x in args.tiny_gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        gpu_ids = [None]

    rows: List[Dict[str, object]] = run_tiny_parallel(
        specs=tiny_specs,
        args=args,
        output_root=output_root,
        gpu_ids=gpu_ids,
    )

    if args.run_tiny_consensus:
        print("\n[phase] tiny consensus evaluation", flush=True)
        if args.cpu_only:
            consensus_gpu_ids = [None]
        else:
            consensus_gpu_ids = [str(g) for g in gpu_ids if g is not None]
            if not consensus_gpu_ids:
                consensus_gpu_ids = [None]

        for idx, spec in enumerate(tiny_specs):
            tiny_out_dir = output_root / f"{spec.name}_seed{spec.seed}"
            gpu_id = consensus_gpu_ids[idx % len(consensus_gpu_ids)]
            consensus_dir = maybe_run_tiny_consensus(
                args=args,
                spec=spec,
                tiny_output_dir=tiny_out_dir,
                gpu_id=gpu_id,
            )
            rows.append(
                collect_tiny_consensus_row(
                    name=spec.name,
                    tiny_output_dir=tiny_out_dir,
                    consensus_output_dir=consensus_dir,
                )
            )

    if args.run_high_accuracy:
        high_dir = output_root / "high_accuracy"
        maybe_run_high_accuracy(args=args, output_dir=high_dir)
        rows.append(collect_high_row(output_dir=high_dir))

    if args.run_high_accuracy_torch:
        high_torch_dir = output_root / "high_accuracy_torch"
        maybe_run_high_accuracy_torch(args=args, output_dir=high_torch_dir)
        rows.append(collect_high_torch_row(output_dir=high_torch_dir))

    if args.run_rssi_knn:
        rssi_knn_dir = output_root / "rssi_knn"
        maybe_run_rssi_knn(args=args, output_dir=rssi_knn_dir)
        rows.append(collect_rssi_knn_row(output_dir=rssi_knn_dir))

    if args.run_lightweight_zoo:
        lw_dir = output_root / "lightweight_zoo"
        maybe_run_lightweight_zoo(args=args, output_dir=lw_dir)
        rows.extend(collect_lightweight_rows(output_dir=lw_dir))

    if args.run_article_model:
        article_dir = output_root / "article_model"
        maybe_run_article_model(args=args, output_dir=article_dir)
        rows.extend(
            collect_article_rows(
                output_dir=article_dir,
                include_raw=bool(args.article_include_raw_rows),
            )
        )

    summary_dir = output_root / "summary"
    write_summary(rows=rows, out_dir=summary_dir)
    print(f"\nSummary saved to: {summary_dir}", flush=True)
    print(f"  - {summary_dir / 'summary.json'}", flush=True)
    print(f"  - {summary_dir / 'summary.csv'}", flush=True)
    print(f"  - {summary_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
