"""
CRICKET SHOT CLASSIFICATION - TRAINING SCRIPT
==============================================

This script trains the EfficientNetB4 + GRU video classifier.
No ball-bat detection, no contact frame extraction - just 30 frames of pure video classification.

Usage:
    python train_video_classifier.py --dataset <path_to_dataset> --shots cut,drive,flick,misc,pull,slog,sweep
    
Example:
    python train_video_classifier.py --dataset "C:\\Users\\User\\Downloads\\Cricket-Shots" --shots cut,drive,flick,misc,pull,slog,sweep
"""

import os
import sys
import argparse
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent  # Navigate to ShotSense-AI root
sys.path.insert(0, str(project_root))
# Also add current working directory
sys.path.insert(0, os.getcwd())

from features.SHOT_CLASSIFICATION_SYSTEM.model_training.video_classifier_trainer import VideoClassifierTrainer
from features.SHOT_CLASSIFICATION_SYSTEM.utils.config import SHOT_TYPES, MODEL_FOLDER_PATH


def main():
    parser = argparse.ArgumentParser(
        description='Train EfficientNetB4 + GRU video classifier for cricket shots'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset root directory (contains shot type folders)',
        default='C:\\Users\\User\\Downloads\\Cricket-Shots'
    )
    
    parser.add_argument(
        '--shots',
        type=str,
        help='Comma-separated list of shot types to train',
        default=','.join(SHOT_TYPES)
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        help='Model output directory',
        default=MODEL_FOLDER_PATH
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs',
        default=20
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size',
        default=16
    )
    
    args = parser.parse_args()
    
    # Parse shot types
    shot_types = [s.strip() for s in args.shots.split(',')]
    
    # Verify dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {args.dataset}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print(" CRICKET SHOT CLASSIFICATION - VIDEO CLASSIFIER TRAINING")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Shot types: {shot_types}")
    print(f"Model directory: {args.model_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("="*80 + "\n")
    
    # Initialize trainer
    trainer = VideoClassifierTrainer(model_dir=args.model_dir)
    
    # Update trainer parameters
    trainer.EPOCHS = args.epochs
    trainer.BATCH_SIZE = args.batch_size
    
    # Run training pipeline
    try:
        trainer.train_pipeline(
            dataset_path=str(dataset_path),
            shot_types=shot_types
        )
        
        print("\n" + "="*80)
        print(" ✅ TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Models saved to: {args.model_dir}")
        print("\nNext step: Use the new model in your API")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()