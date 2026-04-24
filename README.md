# IndoorPos

An experimental repository for RSSI-based indoor localization.

The original goal of this project was broad:

- train very small models for ESP32 / ESP32-S3 deployment
- try higher-accuracy pure RSSI localization methods
- compare classical fingerprinting, lightweight neural networks, larger neural networks, and trajectory-style models
- benchmark everything on the same dataset splits

In practice, the project produced useful baselines and a few clear lessons, but it did not reach the level of result needed to justify continued active development. This repository is therefore being archived in its current state.

## Dataset

The repo works with the original UJI-style Wi-Fi fingerprint data plus derived sequence datasets:

- `archive/TrainingData.csv`
- `archive/ValidationData.csv`
- `training_dataset/`
- `test_dataset/`

The sequence datasets are generated from the original data and are used by most of the benchmark scripts in this repository.

## Final Takeaways

### 1. Pure RSSI classical methods remained very strong

The strongest deployable pure RSSI result in this repo came from a tree-based tabular model rather than a neural network:

- best model:
  `extra_trees_est1200_mf0.5_leaf1_flatten_stat_top1`
- script:
  `TrainRSSITabularEnsemble.py`
- key idea:
  use a strong floor/building classifier first, then perform coordinate regression inside the predicted local region using a large `ExtraTreesRegressor`

Local final test result:

| method | mean (m) | median (m) | p90 (m) | p95 (m) | rmse (m) |
|---|---:|---:|---:|---:|---:|
| tabular winner | 13.580 | 10.319 | 27.549 | 35.058 | 17.923 |
| pure RSSI WKNN baseline | 13.724 | 8.155 | 32.120 | 47.370 | 21.903 |

Interpretation:

- the mean improvement over WKNN is small
- the tail error improvement is real and meaningful
- for navigation-style use, the reduced `p90`, `p95`, and `rmse` are the main reason to prefer the tabular model
- WKNN still has a better median error on this dataset

### 2. Bigger pure RSSI neural networks did not become the best solution

Several pure RSSI neural models were tried, including MPS-accelerated Torch models on Apple Silicon. They trained correctly, but they did not beat the strongest tabular solution.

This repo therefore does **not** conclude that "more neural capacity" was the missing ingredient. For this dataset, the stronger direction was better fingerprint features plus stronger classical tabular regression.

### 3. Trajectory-style models looked good in a narrower setting, but were not the right final answer

Some trajectory / TCN-based models achieved very low short-window error in evaluation. However, they depended on assumptions that made them less suitable as the primary pure RSSI localization solution:

- they used sequential windows rather than single-shot cold start localization
- some variants depended on proxy motion-like features
- the evaluation setting was not as deployment-friendly as a pure RSSI absolute locator

For that reason, the final recommended direction in this repo is still the pure RSSI tabular model, not the trajectory model.

## Recommended Scripts

### Pure RSSI benchmark entrypoint

`RunPureRSSIBenchmarks.py`

Runs a focused pure RSSI comparison, including:

- `TrainRSSIKNNModel.py`
- `TrainAdvancedRSSIEnsemble.py`
- `TrainRSSITabularEnsemble.py`
- optionally `TrainRSSIOnlyHighAccuracyTorch.py`

### Best pure RSSI final model

`TrainRSSITabularEnsemble.py`

Recommended candidate family:

- feature set: `flatten_stat`
- regressor: `ExtraTrees`
- group mode: `top1`

### Baseline

`TrainRSSIKNNModel.py`

Strong baseline:

- feature set: `last_mean`
- candidate: `k1_w_global`

## Example Commands

### Run the pure RSSI benchmark

```bash
python3 RunPureRSSIBenchmarks.py \
  --train-dir training_dataset \
  --test-dir test_dataset \
  --output-root runs/pure_rssi_bench
```

### Run the final tabular model only

```bash
python3 TrainRSSITabularEnsemble.py \
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

## Important Artifacts

Useful result files from the last local pure RSSI evaluation:

- `runs/local_pure_rssi/tabular_final/metrics.json`
- `runs/local_pure_rssi/tabular_final/test_best_single_scatter.png`
- `runs/local_pure_rssi/tabular_final/test_predictions_best_single.csv`
- `runs/local_pure_rssi/knn_baseline_last_mean/metrics.json`

## Dependencies

For the pure RSSI benchmark flow:

- `requirements-pure-rssi-bench.txt`

For optional Torch experiments:

- `requirements-torch-cu118.txt`
- `requirements-torch-cu126.txt`
- or a local Apple Silicon / MPS-enabled PyTorch install

## Repository Status

Archived.

This repo is being kept as an experiment log and benchmark reference rather than an actively improving product.
