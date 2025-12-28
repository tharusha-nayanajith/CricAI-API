"""
YOLOv8 Cricket Ball Detection Training Script
Optimized for small object detection
"""
import os
import json
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

from features.SHOT_CLASSIFICATION_SYSTEM.utils.config import BALL_DATASET_PATH, MODEL_FOLDER_PATH

# ==================== Configuration ====================
class Config:
    MODEL_NAME = "yolov8_ball_detector"
    
    # Paths
    DATASET_PATH = Path(BALL_DATASET_PATH)
    MODEL_SAVE_PATH = Path(MODEL_FOLDER_PATH) / MODEL_NAME
    EVAL_RESULTS_PATH = MODEL_SAVE_PATH / "eval_results"
    
    # Model selection (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
    YOLO_MODEL = "yolov8s.pt"  # Small model, good balance
    
    # Training parameters
    IMG_SIZE = 640  # YOLOv8 works best at 640
    BATCH_SIZE = 8 if torch.cuda.is_available() else 4  # Lower for CPU
    NUM_EPOCHS = 35
    LEARNING_RATE = 0.01
    
    # Augmentation
    AUGMENT = True
    
    # Device
    DEVICE = 0 if torch.cuda.is_available() else 'cpu'
    
    # Early stopping
    PATIENCE = 20
    
    # Confidence thresholds
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45


def create_yaml_config():
    """Create data.yaml for YOLOv8"""
    yaml_path = Config.DATASET_PATH / "data.yaml"
    
    # Use absolute paths to avoid issues
    data_config = {
        'path': str(Config.DATASET_PATH.absolute()).replace('\\', '/'),  # Convert Windows paths
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 1,  # Number of classes
        'names': ['ball']  # Class names
    }
    
    # Verify directories exist
    train_dir = Config.DATASET_PATH / 'train' / 'images'
    valid_dir = Config.DATASET_PATH / 'valid' / 'images'
    test_dir = Config.DATASET_PATH / 'test' / 'images'
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not valid_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {valid_dir}")
    
    print(f"✓ Found {len(list(train_dir.glob('*.jpg')))} training images")
    print(f"✓ Found {len(list(valid_dir.glob('*.jpg')))} validation images")
    if test_dir.exists():
        print(f"✓ Found {len(list(test_dir.glob('*.jpg')))} test images")
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created data.yaml at {yaml_path}")
    print(f"✓ Dataset path: {data_config['path']}")
    
    return yaml_path


def train_model():
    """Train YOLOv8 model"""
    print("="*80)
    print("YOLOv8 Cricket Ball Detection Training")
    print("="*80)
    print(f"Device: {Config.DEVICE}")
    print(f"Dataset: {Config.DATASET_PATH}")
    print(f"Model: {Config.YOLO_MODEL}")
    print(f"Input size: {Config.IMG_SIZE}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print("="*80)
    
    # Create directories
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(Config.EVAL_RESULTS_PATH, exist_ok=True)
    
    # Create data.yaml
    yaml_path = create_yaml_config()
    
    # Initialize model
    print("\nInitializing YOLOv8 model...")
    model = YOLO(Config.YOLO_MODEL)
    
    # Training arguments
    train_args = {
        'data': str(yaml_path),
        'epochs': Config.NUM_EPOCHS,
        'imgsz': Config.IMG_SIZE,
        'batch': Config.BATCH_SIZE,
        'lr0': Config.LEARNING_RATE,
        'device': Config.DEVICE,
        'project': str(Config.MODEL_SAVE_PATH),
        'name': 'train',
        'exist_ok': True,
        'patience': Config.PATIENCE,
        'save': True,
        'save_period': 10,
        'verbose': True,
        'cache': False,  # Don't cache on disk to save space
        'workers': 2 if torch.cuda.is_available() else 1,
        
        # Augmentation settings (optimized for small objects)
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,    # HSV-Saturation augmentation
        'hsv_v': 0.4,    # HSV-Value augmentation
        'degrees': 5.0,  # Slight rotation for cricket shots
        'translate': 0.1,  # Translation
        'scale': 0.5,    # Scale
        'shear': 0.0,    # Shear
        'perspective': 0.0,  # Perspective
        'flipud': 0.0,   # Flip up-down
        'fliplr': 0.5,   # Flip left-right
        'mosaic': 1.0,   # Mosaic augmentation
        'mixup': 0.0,    # Mixup augmentation
        'copy_paste': 0.0,  # Copy-paste augmentation
        
        # Small object optimization - CRITICAL FOR TINY BALLS
        'box': 7.5,      # Box loss gain (higher for small objects)
        'cls': 0.5,      # Class loss gain
        'dfl': 1.5,      # DFL loss gain
        'iou': 0.7,      # IoU training threshold
        
        # Optimizer
        'optimizer': 'AdamW',
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'momentum': 0.937,
        'lrf': 0.01,     # Final learning rate factor
        
        # Other
        'close_mosaic': 10,  # Disable mosaic in last N epochs
        'amp': torch.cuda.is_available(),  # AMP only on GPU
        'fraction': 1.0,  # Use 100% of dataset
        'plots': True,    # Save training plots
        'val': True,      # Validate during training
    }
    
    # Train
    print("\nStarting training...")
    print("="*80)
    results = model.train(**train_args)
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    
    return model, results


def evaluate_model(model):
    """Evaluate trained model"""
    print("\nEvaluating model...")
    
    # Validate on validation set
    val_metrics = model.val(
        data=str(Config.DATASET_PATH / "data.yaml"),
        imgsz=Config.IMG_SIZE,
        batch=Config.BATCH_SIZE,
        conf=Config.CONF_THRESHOLD,
        iou=Config.IOU_THRESHOLD,
        device=Config.DEVICE
    )
    
    # Validate on test set
    test_metrics = model.val(
        data=str(Config.DATASET_PATH / "data.yaml"),
        split='test',
        imgsz=Config.IMG_SIZE,
        batch=Config.BATCH_SIZE,
        conf=Config.CONF_THRESHOLD,
        iou=Config.IOU_THRESHOLD,
        device=Config.DEVICE
    )
    
    # Extract metrics
    results = {
        'validation': {
            'precision': float(val_metrics.results_dict.get('metrics/precision(B)', 0)),
            'recall': float(val_metrics.results_dict.get('metrics/recall(B)', 0)),
            'mAP50': float(val_metrics.results_dict.get('metrics/mAP50(B)', 0)),
            'mAP50-95': float(val_metrics.results_dict.get('metrics/mAP50-95(B)', 0))
        },
        'test': {
            'precision': float(test_metrics.results_dict.get('metrics/precision(B)', 0)),
            'recall': float(test_metrics.results_dict.get('metrics/recall(B)', 0)),
            'mAP50': float(test_metrics.results_dict.get('metrics/mAP50(B)', 0)),
            'mAP50-95': float(test_metrics.results_dict.get('metrics/mAP50-95(B)', 0))
        },
        'config': {
            'model': Config.YOLO_MODEL,
            'img_size': Config.IMG_SIZE,
            'batch_size': Config.BATCH_SIZE,
            'epochs': Config.NUM_EPOCHS,
            'learning_rate': Config.LEARNING_RATE
        }
    }
    
    print("\n" + "="*80)
    print("Validation Results:")
    print(f"Precision: {results['validation']['precision']:.4f}")
    print(f"Recall: {results['validation']['recall']:.4f}")
    print(f"mAP50: {results['validation']['mAP50']:.4f}")
    print(f"mAP50-95: {results['validation']['mAP50-95']:.4f}")
    
    print("\nTest Results:")
    print(f"Precision: {results['test']['precision']:.4f}")
    print(f"Recall: {results['test']['recall']:.4f}")
    print(f"mAP50: {results['test']['mAP50']:.4f}")
    print(f"mAP50-95': {results['test']['mAP50-95']:.4f}")
    print("="*80)
    
    # Save results
    results_path = Config.EVAL_RESULTS_PATH / 'eval_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Evaluation results saved to {results_path}")
    
    return results


def export_model(model):
    """Export model for inference"""
    print("\nExporting model...")
    
    # Export to ONNX format (optional)
    try:
        model.export(
            format='onnx',
            imgsz=Config.IMG_SIZE,
            dynamic=False,
            simplify=True
        )
        print("✓ Model exported to ONNX format")
    except Exception as e:
        print(f"Note: ONNX export failed: {e}")
    
    # Copy best weights to standard location
    best_weights = Config.MODEL_SAVE_PATH / 'train' / 'weights' / 'best.pt'
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, Config.MODEL_SAVE_PATH / 'best_model.pt')
        print(f"✓ Best model saved to {Config.MODEL_SAVE_PATH / 'best_model.pt'}")


def main():
    """Main function"""
    try:
        # Train model
        model, results = train_model()
        
        # Load best model
        best_model_path = Config.MODEL_SAVE_PATH / 'train' / 'weights' / 'best.pt'
        model = YOLO(str(best_model_path))
        
        # Evaluate
        eval_results = evaluate_model(model)
        
        # Export
        export_model(model)
        
        print("\n" + "="*80)
        print("✓ Training pipeline completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()