# Server Training Guide (Multi-Model Benchmark)

This guide is for high-performance Linux servers with NVIDIA GPUs.

## 1) Install Dependencies

```bash
chmod +x setup_server_env.sh
PYTHON_BIN=python3.12 REQ_FILE=requirements-server-py312-cu130.txt ./setup_server_env.sh
source .venv-server/bin/activate
```

Dependency file used:

- `requirements-server-py312-cu130.txt`

## 2) Build Pseudo-Temporal Dataset (Recommended)

If you need trajectory-aware training without collecting new data, rebuild train/test datasets with longer generated trajectories:

```bash
# train split from TrainingData.csv
python3 DatasetProc.py \
  --input-csv archive/TrainingData.csv \
  --output-dir training_dataset_seq \
  --seq-len 5 \
  --trajectory-total-len 13 \
  --window-stride 1 \
  --sequence-generation-mode endpoint_path \
  --paths-per-endpoint 1 \
  --trajectories-per-group 2000 \
  --interpolation-steps 0

python3 - <<'PY'
import json
meta = json.load(open("training_dataset_seq/metadata.json", "r", encoding="utf-8"))
json.dump(meta["selected_waps"], open("training_dataset_seq/selected_waps.json", "w", encoding="utf-8"))
print("wrote training_dataset_seq/selected_waps.json")
PY

# test split from ValidationData.csv (reuse AP list from train metadata)
python3 DatasetProc.py \
  --input-csv archive/ValidationData.csv \
  --output-dir test_dataset_seq \
  --selected-waps-json training_dataset_seq/selected_waps.json \
  --seq-len 5 \
  --trajectory-total-len 13 \
  --window-stride 1 \
  --sequence-generation-mode endpoint_path \
  --paths-per-endpoint 1 \
  --trajectories-per-group 2000 \
  --interpolation-steps 0
```

Then use `training_dataset_seq` / `test_dataset_seq` in training commands.

## 3) Run Multi-Model Training + Evaluation

Train multiple tiny scales/architectures and optionally high-accuracy models, then output a unified test-set report:

Quick one-command launcher (recommended):

```bash
chmod +x run_full_benchmark_server.sh
TINY_GPU_IDS=0 LIGHTWEIGHT_GPU_ID=0 ARTICLE_GPU_ID=0 HIGH_TORCH_GPU_ID=0 \
  ./run_full_benchmark_server.sh training_dataset_seq test_dataset_seq runs/server_model_zoo
```

Manual command (full control):

```bash
python3 RunServerBenchmarks.py \
  --train-dir training_dataset_seq \
  --test-dir test_dataset_seq \
  --output-root runs/server_model_zoo \
  --tiny-specs "tiny_s=dscnn|16:24@7,tiny_m=dscnn|24:32@7,tiny_gru_m=gru|24:32@7,tiny_tcn_l=tcn|32:48@7,tiny_xl=dscnn|40:64:64@7" \
  --tiny-gpu-ids "0,1" \
  --tiny-max-parallel 2 \
  --run-rssi-knn \
  --rssi-knn-feature-set last_mean \
  --rssi-knn-k-candidates "1,3,5,7,9,11,15,21" \
  --rssi-knn-weighted \
  --rssi-knn-enable-temporal-filter \
  --rssi-knn-auto-tune-temporal \
  --run-lightweight-zoo \
  --lightweight-gpu-id 0 \
  --run-article-model \
  --article-train-dir training_dataset_seq \
  --article-test-dir test_dataset_seq \
  --article-gpu-id 1 \
  --article-include-raw-rows \
  --run-tiny-consensus \
  --consensus-n-runs 11 \
  --consensus-radius-m 4.0 \
  --consensus-noise-std 0.02 \
  --run-high-accuracy \
  --run-high-accuracy-torch \
  --high-torch-gpu-id 0
```

Outputs:

- `runs/server_model_zoo/summary/summary.json`
- `runs/server_model_zoo/summary/summary.csv`
- `runs/server_model_zoo/summary/summary.md`

When `--run-tiny-consensus` is enabled, summary will include extra rows with `model_type=tiny_consensus`.

Notes:

- `--run-high-accuracy` runs sklearn RF/ExtraTrees (CPU-oriented).
- `--run-high-accuracy-torch` runs a GPU-accelerated sequence model + kNN refinement.
- `--run-rssi-knn` runs RSSI fingerprint + KNN baseline (CPU-oriented).
- `--run-lightweight-zoo` runs `set_tcn / cnn_tcn / pure_tcn` lightweight sequence schemes.
- `--run-article-model` runs article-style trajectory model (`Δx,Δy + grid + speed-cap + Kalman`).
- `--rssi-knn-enable-temporal-filter` enables online-style trajectory smoothing for RSSI-KNN.
- `--rssi-knn-auto-tune-temporal` picks temporal filter params on validation split, then applies to test.
- If sequence groups in the dataset are single-step only, temporal filtering has no offline effect (still useful for real online continuous scans).
- In `--tiny-specs`, use `name=[arch|]candidate@seed`, where `arch` is `dscnn|gru|tcn`.

## 4) Rebuild Summary Only (No Retraining)

```bash
python3 RunServerBenchmarks.py \
  --output-root runs/server_model_zoo \
  --tiny-specs "tiny_s=16:24@7,tiny_m=24:32@7,tiny_l=32:48@7,tiny_xl=40:64:64@7" \
  --run-high-accuracy \
  --skip-train
```

This requires each model's `metrics.json` to already exist in `output-root`.
