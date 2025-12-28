"""
Model Retraining Script
CLI tool for retraining ensemble models with new data
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from features.SHOT_CLASSIFICATION_SYSTEM.model_training.ensemble_trainer import EnsembleTrainer
from features.SHOT_CLASSIFICATION_SYSTEM.model_training.model_registry import ModelRegistry


def retrain_with_new_data(new_data_path: str, merge_existing: bool = True):
    """
    Retrain models with new data
    
    Args:
        new_data_path: Path to new dataset directory
        merge_existing: Whether to merge with existing data
    """
    print("\n" + "="*70)
    print(" "*15 + "MODEL RETRAINING PIPELINE")
    print("="*70)
    
    SHOT_TYPES = ['cut', 'drive', 'flick', 'misc', 'pull', 'slog', 'sweep']
    
    trainer = EnsembleTrainer()
    
    if merge_existing:
        print("\n📂 Mode: MERGE with existing data")
        trainer.retrain_models(new_data_path=new_data_path, shot_types=SHOT_TYPES)
    else:
        print("\n📂 Mode: FRESH training (ignoring existing data)")
        X, y = trainer.prepare_dataset(new_data_path, SHOT_TYPES, use_augmentation=True)
        results = trainer.train_all_models(X, y)
    
    # Register new version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    registry = ModelRegistry()
    
    registry.register_model(timestamp, {
        'performance': results if not merge_existing else {},
        'config': {
            'data_path': new_data_path,
            'merged': merge_existing,
            'shot_types': SHOT_TYPES
        },
        'dataset_info': {
            'source': 'retrained',
            'timestamp': timestamp
        }
    })
    
    # Set as active
    registry.set_active_version(timestamp)
    
    print("\n" + "="*70)
    print("✅ RETRAINING COMPLETE")
    print("="*70)
    print(f"\nNew model version: {timestamp}")
    print("Status: Active and deployed")
    print("\nNext steps:")
    print("  1. Run evaluation: python evaluate_model.py")
    print("  2. Test API: python main.py")
    print("  3. If needed, rollback: python retrain.py --rollback <version>")


def retrain_from_scratch(data_path: str):
    """Train completely fresh models"""
    print("\n" + "="*70)
    print(" "*10 + "TRAINING FROM SCRATCH (NO EXISTING DATA)")
    print("="*70)
    
    SHOT_TYPES = ['cut', 'drive', 'flick', 'misc', 'pull', 'slog', 'sweep']
    
    trainer = EnsembleTrainer()
    
    # Prepare dataset
    print("\nPreparing dataset...")
    X, y = trainer.prepare_dataset(data_path, SHOT_TYPES, use_augmentation=True)
    
    # Train models
    results = trainer.train_all_models(X, y)
    
    # Register
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    registry = ModelRegistry()
    
    registry.register_model(timestamp, {
        'performance': results,
        'config': {
            'data_path': data_path,
            'shot_types': SHOT_TYPES
        },
        'dataset_info': {
            'source': 'fresh_training',
            'timestamp': timestamp
        }
    })
    
    registry.set_active_version(timestamp)
    
    print("\n✅ Training complete!")


def list_model_versions():
    """List all model versions"""
    registry = ModelRegistry()
    registry.print_summary()


def rollback_model(version_id: str):
    """Rollback to a previous model version"""
    print(f"\n⚠️  Rolling back to version: {version_id}")
    
    registry = ModelRegistry()
    registry.rollback_to_version(version_id)
    
    print("\n✅ Rollback complete! Restart your API server.")


def compare_versions(version1: str, version2: str):
    """Compare two model versions"""
    registry = ModelRegistry()
    comparison = registry.compare_versions(version1, version2)
    
    print("\n" + "="*70)
    print(" "*20 + "VERSION COMPARISON")
    print("="*70)
    
    print(f"\nVersion 1: {version1}")
    print(f"  Timestamp: {comparison['version1']['timestamp']}")
    if 'performance' in comparison['version1']:
        perf = comparison['version1']['performance']
        print(f"  Test Accuracy: {perf.get('test_acc', 'N/A')}")
    
    print(f"\nVersion 2: {version2}")
    print(f"  Timestamp: {comparison['version2']['timestamp']}")
    if 'performance' in comparison['version2']:
        perf = comparison['version2']['performance']
        print(f"  Test Accuracy: {perf.get('test_acc', 'N/A')}")
    
    if 'improvements' in comparison:
        print("\n📈 Improvements:")
        for metric, values in comparison['improvements'].items():
            symbol = "📈" if values['absolute'] > 0 else "📉"
            print(f"  {symbol} {metric}: {values['absolute']:+.4f} ({values['percentage']:+.2f}%)")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Model Retraining & Management Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Retrain with new data (merge with existing)
  python retrain.py --new-data /path/to/new/videos
  
  # Train from scratch (ignore existing data)
  python retrain.py --from-scratch /path/to/dataset
  
  # List all model versions
  python retrain.py --list
  
  # Rollback to previous version
  python retrain.py --rollback 20250120_143022
  
  # Compare two versions
  python retrain.py --compare 20250120_143022 20250121_091500
        """
    )
    
    parser.add_argument('--new-data', type=str, 
                       help='Path to new dataset for retraining (merges with existing)')
    
    parser.add_argument('--from-scratch', type=str,
                       help='Path to dataset for training from scratch (ignores existing)')
    
    parser.add_argument('--list', action='store_true',
                       help='List all model versions')
    
    parser.add_argument('--rollback', type=str,
                       help='Rollback to specific model version (version ID)')
    
    parser.add_argument('--compare', nargs=2, metavar=('VERSION1', 'VERSION2'),
                       help='Compare two model versions')
    
    parser.add_argument('--no-merge', action='store_true',
                       help='Don\'t merge with existing data (use with --new-data)')
    
    args = parser.parse_args()
    
    # Handle commands
    if args.list:
        list_model_versions()
    
    elif args.rollback:
        rollback_model(args.rollback)
    
    elif args.compare:
        compare_versions(args.compare[0], args.compare[1])
    
    elif args.from_scratch:
        if not os.path.exists(args.from_scratch):
            print(f"❌ Error: Dataset path not found: {args.from_scratch}")
            sys.exit(1)
        retrain_from_scratch(args.from_scratch)
    
    elif args.new_data:
        if not os.path.exists(args.new_data):
            print(f"❌ Error: Dataset path not found: {args.new_data}")
            sys.exit(1)
        
        merge = not args.no_merge
        retrain_with_new_data(args.new_data, merge_existing=merge)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()