# Server Deploy

Recommended for a Linux server with an NVIDIA 2080 Ti:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-torch-cu126.txt
```

The packaged code is kept compatible with Python 3.9. A newer Python version
such as 3.10 or 3.11 is still recommended when the server environment allows it.

If `nvidia-smi` reports a CUDA runtime around `12.6`, the `cu126` wheel above is
the best match. If the server environment is different, replace the PyTorch
install line with a wheel index that matches that environment.

Example training command:

```bash
PYTHONUNBUFFERED=1 python3 TrainHybridModel.py \
  --train-dir training_dataset_fixed \
  --test-dir test_dataset_fixed \
  --epochs 150 \
  --batch-size 1024 \
  --patience 12 \
  --output-dir runs/fixed_hybrid_cnn_tcn_motion
```
