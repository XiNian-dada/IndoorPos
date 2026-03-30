@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

set "PYTHONUNBUFFERED=1"
if not defined MPLCONFIGDIR (
    set "MPLCONFIGDIR=%TEMP%\mplconfig"
)

if not exist "runs" (
    mkdir "runs"
)

python TrainHybridModel.py ^
  --train-dir training_dataset_fixed ^
  --test-dir test_dataset_fixed ^
  --epochs 150 ^
  --batch-size 256 ^
  --patience 12 ^
  --output-dir runs\fixed_hybrid_cnn_tcn_motion ^
  %*

if errorlevel 1 (
    echo.
    echo Training failed. Please check your Python environment and dependencies.
    exit /b 1
)

echo.
echo Training completed successfully.
exit /b 0
