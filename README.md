# IndoorPos

English | [简体中文](README.zh-CN.md)

An archived experimental repository for RSSI-based indoor localization.

This project explored several directions:

- tiny models for ESP32 / ESP32-S3 deployment
- pure RSSI fingerprint baselines
- stronger classical CPU-side localization models
- larger Torch models
- pseudo-trajectory / sequence-based models
- server benchmark tooling

The repo is being archived because the final outcome was useful as an experiment log and benchmark reference, but not strong enough to justify continued active product work.

## Repository Status

Archived. The code is kept for:

- result traceability
- reproduction of past experiments
- comparing different localization families on the same derived datasets

## Project Layout

```text
IndoorPos/
├── README.md
├── README.zh-CN.md
├── scripts/         # training, evaluation, dataset, benchmark entrypoints
├── docs/            # experiment notes, deployment/server guides, HTML visualization
├── requirements/    # dependency sets for local/server/CUDA variants
├── tools/           # helper shell / bat launchers
├── archive/         # original CSV fingerprint datasets
├── training_dataset/
├── test_dataset/
├── training_dataset_fixed/
├── test_dataset_fixed/
└── runs/            # generated experiment outputs
```

## What Finally Worked Best

For **deployable pure RSSI absolute localization**, the best result in this repo came from:

- script: `scripts/TrainRSSITabularEnsemble.py`
- model family: `ExtraTreesRegressor` + building/floor gate
- final candidate:
  `extra_trees_est1200_mf0.5_leaf1_flatten_stat_top1`

Final local pure RSSI test result:

| method | mean (m) | median (m) | p90 (m) | p95 (m) | rmse (m) |
|---|---:|---:|---:|---:|---:|
| tabular winner | 13.580 | 10.319 | 27.549 | 35.058 | 17.923 |
| pure RSSI WKNN baseline | 13.724 | 8.155 | 32.120 | 47.370 | 21.903 |

Interpretation:

- the mean improvement over WKNN is small
- the long-tail improvement is meaningful
- `p90`, `p95`, `rmse`, and `max error` are all better than WKNN
- the median error is still better for WKNN

For a hospital-style navigation setting, the lower long-tail error is the main reason to prefer the tabular model.

Relevant artifacts:

- `runs/local_pure_rssi/tabular_final/metrics.json`
- `runs/local_pure_rssi/tabular_final/test_best_single_scatter.png`
- `runs/local_pure_rssi/tabular_final/test_predictions_best_single.csv`
- `runs/local_pure_rssi/knn_baseline_last_mean/metrics.json`

## Datasets

This repo uses both the original Wi-Fi fingerprint CSVs and several derived sequence datasets.

### Original CSVs

Located in `archive/`:

- `archive/TrainingData.csv`
- `archive/ValidationData.csv`

These are the base Wi-Fi fingerprint tables used to generate the sequence-style training/test sets.

### Non-fixed sequence datasets

Primary experiment datasets:

- `training_dataset/`
- `test_dataset/`

Key properties from metadata:

- source CSVs:
  `archive/TrainingData.csv` and `archive/ValidationData.csv`
- sequence length:
  `5`
- generation mode:
  `endpoint_path`
- selected APs:
  `128`
- train split:
  `15949`
- validation split:
  `3988`
- test split:
  `1111`
- augmentation:
  RSSI noise enabled with `2.0 dBm`

### Fixed sequence datasets

Additional experiment datasets:

- `training_dataset_fixed/`
- `test_dataset_fixed/`

These were used for some later fixed-dataset and hybrid experiments.

### Rebuilding datasets

Main generator:

- `scripts/DatasetProc.py`

Useful helper:

- `scripts/visualize_dataset.py`

Example:

```bash
python3 scripts/DatasetProc.py \
  --input-csv archive/TrainingData.csv \
  --output-dir training_dataset \
  --seq-len 5
```

For larger server-side rebuild workflows, see:

- `docs/SERVER_TRAINING_GUIDE.md`
- `docs/DEPLOY_SERVER.md`

## Algorithms Tried

This section lists the main algorithm families that were actually implemented in this repository.

### 1. Tiny ESP32-friendly models

Script:

- `scripts/TrainTinyESP32Model.py`

Architectures:

- depthwise-separable CNN (`dscnn`)
- GRU (`gru`)
- TCN (`tcn`)

Goal:

- fit into very small on-device footprints
- explore ESP32-S3 deployment feasibility

Related script:

- `scripts/EvaluateTinyConsensus.py`

This script tests repeated noisy inference plus consensus clustering to see whether multiple tiny runs can improve localization robustness.

### 2. Pure RSSI WKNN baseline

Script:

- `scripts/TrainRSSIKNNModel.py`

Method:

- RSSI fingerprint features
- nearest-neighbor matching
- weighted and unweighted KNN
- optional group-aware lookup
- optional temporal filter

This remained one of the strongest and most useful baselines in the repo.

### 3. Advanced pure RSSI retrieval ensemble

Script:

- `scripts/TrainAdvancedRSSIEnsemble.py`

Method:

- multiple RSSI feature views
- learned building/floor classifier
- group-aware KNN
- multiple neighbor aggregation rules:
  `idw`, `idw2`, `kernel`, `softmax`, `trimmed_idw`, `median`, `lle`
- validation-tuned greedy ensemble

This was the strongest non-tree pure RSSI retrieval-style pipeline in the repo.

### 4. Pure RSSI tabular ensemble

Script:

- `scripts/TrainRSSITabularEnsemble.py`

Method:

- building/floor classifier with RF / ExtraTrees
- global and group-aware coordinate regressors
- main regressors:
  `ExtraTreesRegressor` and `RandomForestRegressor`
- validation-driven model selection
- optional ensemble of top candidates

This is the final recommended pure RSSI solution in the repo.

### 5. High-accuracy classical model

Script:

- `scripts/TrainHighAccuracyModel.py`

Method:

- accuracy-first CPU-side pipeline
- learned group classifier
- group-aware KNN localizer

This script was useful as an accuracy-first classical baseline, but it was not the final pure RSSI winner.

Guide:

- `docs/HIGH_ACCURACY_GUIDE.md`

### 6. High-accuracy Torch sequence model

Script:

- `scripts/TrainHighAccuracyTorchModel.py`

Method:

- sequence encoder
- GRU-based temporal modeling
- learned embedding
- kNN refinement

This script targeted server-side high-accuracy experimentation.

### 7. Pure RSSI Torch model

Script:

- `scripts/TrainRSSIOnlyHighAccuracyTorch.py`

Method:

- pure RSSI sequence encoder
- GRU-based temporal model
- embedding + coordinate head
- optional kNN refinement

This was tested on Apple Silicon with MPS and on other hardware, but it did not beat the final tabular solution.

### 8. Absolute RSSI-only neural regressor

Script:

- `scripts/TrainAbsoluteRSSIOnly.py`

Method:

- pure RSSI input
- absolute coordinate regression
- auxiliary grid classification head

This direction was explored because it avoids trajectory recursion, but it did not become the final best solution.

### 9. Lightweight sequence scheme zoo

Script:

- `scripts/TrainLightweightSchemeZoo.py`

Model families:

- `set_tcn`
- `cnn_tcn`
- `pure_tcn`

Goal:

- explore compact sequence models without going all the way down to the tiny ESP32 family

### 10. Hybrid CNN + TCN

Script:

- `scripts/TrainHybridModel.py`

Goal:

- experiment with fixed-dataset hybrid temporal modeling

### 11. Article-style trajectory models

Scripts:

- `scripts/TrainArticleTrajectoryModel.py`
- `scripts/ArticlePureTCNModel.py`
- `scripts/TrainAndVisualizeArticlePureTCN.py`
- `docs/article_model_visualization.html`

Method:

- Top-K AP tokenization / flattening
- TCN-based temporal modeling
- delta-position prediction
- auxiliary grid classification
- optional post-processing

These models sometimes achieved low short-window error, but they were not the final recommended pure RSSI absolute localization approach because the evaluation setting was less deployment-friendly.

## Benchmark Entry Points

### Pure RSSI benchmark

Script:

- `scripts/RunPureRSSIBenchmarks.py`

This is the best entry point for reproducing the final pure RSSI comparison.

Example:

```bash
python3 scripts/RunPureRSSIBenchmarks.py \
  --train-dir training_dataset \
  --test-dir test_dataset \
  --output-root runs/pure_rssi_bench
```

### Multi-model server benchmark

Script:

- `scripts/RunServerBenchmarks.py`

Supporting scripts:

- `tools/run_full_benchmark_server.sh`
- `tools/setup_server_env.sh`
- `docs/SERVER_TRAINING_GUIDE.md`

This path is intended for multi-model server-side benchmarking with stronger hardware.

## Reproducibility

### 1. Install dependencies

For the pure RSSI CPU benchmark flow:

```bash
python3 -m pip install -r requirements/requirements-pure-rssi-bench.txt
```

For optional Torch experiments:

- `requirements/requirements-torch-cu118.txt`
- `requirements/requirements-torch-cu126.txt`
- or a local Apple Silicon PyTorch installation with MPS support

### 2. Reproduce the WKNN baseline

```bash
python3 scripts/TrainRSSIKNNModel.py \
  --train-dir training_dataset \
  --test-dir test_dataset \
  --output-dir runs/local_pure_rssi/knn_baseline_last_mean \
  --feature-set last_mean \
  --k-candidates 1,3,5,7,9,11,15,21 \
  --weighted
```

### 3. Reproduce the final pure RSSI winner

```bash
python3 scripts/TrainRSSITabularEnsemble.py \
  --train-dir training_dataset \
  --test-dir test_dataset \
  --output-dir runs/local_pure_rssi/tabular_final \
  --feature-sets flatten_stat \
  --group-classifier-candidates rf:700:sqrt:stat_stack,extra_trees:900:sqrt:stat_stack,extra_trees:900:sqrt:quantile_stack \
  --regressor-candidates "extra_trees:1200:0.5:1:flatten_stat:global,extra_trees:1200:0.5:1:flatten_stat:top1,extra_trees:1500:0.35:1:flatten_stat:top1" \
  --ensemble-max-candidates 3 \
  --ensemble-max-steps 3 \
  --n-jobs -1
```

### 4. Reproduce MPS comparison on Apple Silicon

```bash
python3 scripts/TrainRSSIOnlyHighAccuracyTorch.py \
  --train-dir training_dataset \
  --test-dir test_dataset \
  --output-dir runs/local_pure_rssi/rssi_torch_mps \
  --device mps \
  --epochs 100 \
  --batch-size 384 \
  --patience 12 \
  --candidates 192:192:2:256:0.15,256:256:2:320:0.18,320:320:2:384:0.20 \
  --num-workers 0
```

## Why the Repository Was Archived

The repository did produce working code and useful benchmark infrastructure, but the final localization quality was still not compelling enough for the intended product expectations.

More specifically:

- pure RSSI methods improved, but not dramatically enough
- some trajectory-style methods looked better in narrower evaluation settings than they would likely be in deployment
- the project accumulated many experiment paths, but not a single clearly satisfying end result

So the most honest final state is:

- keep the repo
- keep the scripts reproducible
- document what was learned
- stop pretending it already became a strong finished system
