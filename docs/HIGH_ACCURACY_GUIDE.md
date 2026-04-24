# High Accuracy Indoor Localization (Hospital-Oriented)

This repo now includes a high-accuracy pipeline:

- Training script: `scripts/TrainHighAccuracyModel.py`
- Default dataset: `training_dataset` + `test_dataset` (non-fixed)
- Output directory example: `runs/high_accuracy_hospital_v1`

## 1) Train

```bash
python3 scripts/TrainHighAccuracyModel.py \
  --train-dir training_dataset \
  --test-dir test_dataset \
  --output-dir runs/high_accuracy_hospital_v1
```

## 2) Key Outputs

- `metrics.json`: full candidate-search + final test metrics
- `model_bundle.pkl`: scaler + classifier + KNN reference bank for inference
- `test_predictions.npz`: predicted coordinates / true coordinates on test set
- `comparison_vs_tiny.json`: comparison with the tiny neural baseline

## 3) Notes

- This is accuracy-first, not size-first.
- The model bundle is large (hundreds of MB), intended for server-side inference.
- For on-device ESP32S3 deployment, keep using tiny models from `scripts/TrainTinyESP32Model.py`.
