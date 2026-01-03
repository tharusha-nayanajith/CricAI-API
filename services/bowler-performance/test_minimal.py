"""
Minimal YOLO test - matching your working script pattern
"""

import sys
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"PyTorch import error: {e}")

try:
    from ultralytics import YOLO
    print("Ultralytics imported successfully")
    
    # Try loading model
    print("\nLoading model from: app/models/ml/best.pt")
    model = YOLO('app/models/ml/best.pt')
    print("✅ Model loaded successfully!")
    
    # Check model type
    print(f"Model task: {model.task}")
    print(f"Model names: {model.names}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
