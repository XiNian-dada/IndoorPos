#!/usr/bin/env bash
set -euo pipefail

# Usage example:
#   bash run_full_benchmark_server.sh \
#     training_dataset_seq test_dataset_seq runs/server_full_benchmark

TRAIN_DIR="${1:-training_dataset_seq}"
TEST_DIR="${2:-test_dataset_seq}"
OUTPUT_ROOT="${3:-runs/server_full_benchmark}"
TINY_GPU_IDS="${TINY_GPU_IDS:-0}"
LIGHTWEIGHT_GPU_ID="${LIGHTWEIGHT_GPU_ID:-0}"
ARTICLE_GPU_ID="${ARTICLE_GPU_ID:-0}"
HIGH_TORCH_GPU_ID="${HIGH_TORCH_GPU_ID:-0}"

python3 RunServerBenchmarks.py \
  --train-dir "${TRAIN_DIR}" \
  --test-dir "${TEST_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --tiny-specs "tiny_s=dscnn|16:24@7,tiny_m=dscnn|24:32@7,tiny_gru_m=gru|24:32@7,tiny_tcn_l=tcn|32:48@7,tiny_xl=dscnn|40:64:64@7" \
  --tiny-gpu-ids "${TINY_GPU_IDS}" \
  --tiny-max-parallel 2 \
  --run-rssi-knn \
  --rssi-knn-feature-set last_mean \
  --rssi-knn-k-candidates "1,3,5,7,9,11,15,21" \
  --rssi-knn-weighted \
  --rssi-knn-enable-temporal-filter \
  --rssi-knn-auto-tune-temporal \
  --run-lightweight-zoo \
  --lightweight-gpu-id "${LIGHTWEIGHT_GPU_ID}" \
  --run-article-model \
  --article-train-dir "${TRAIN_DIR}" \
  --article-test-dir "${TEST_DIR}" \
  --article-gpu-id "${ARTICLE_GPU_ID}" \
  --article-include-raw-rows \
  --run-tiny-consensus \
  --consensus-n-runs 11 \
  --consensus-radius-m 4.0 \
  --consensus-noise-std 0.02 \
  --run-high-accuracy \
  --run-high-accuracy-torch \
  --high-torch-gpu-id "${HIGH_TORCH_GPU_ID}"

echo "Benchmark done. Summary: ${OUTPUT_ROOT}/summary/summary.md"
