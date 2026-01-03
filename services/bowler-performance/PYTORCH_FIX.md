# PyTorch DLL Error Fix for Windows

## Problem
Error: `[WinError 127] The specified procedure could not be found. Error loading "c10_cuda.dll"`

## Root Cause
PyTorch on Windows requires Microsoft Visual C++ Redistributable packages.

## Solution

### Option 1: Install Visual C++ Redistributables (Recommended)
1. Download and install: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Restart your terminal/IDE
3. Restart the server: `python run.py`

### Option 2: Use CPU-only PyTorch
```bash
pip uninstall torch torchvision -y
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

### Option 3: Use Conda (if available)
```bash
conda install pytorch torchvision cpuonly -c pytorch
```

## Verification
Test if PyTorch works:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## Alternative: Run without GPU
If you continue to have issues, the API will still work but YOLO model loading will fail.
You can test other endpoints or run the model on a different machine.
