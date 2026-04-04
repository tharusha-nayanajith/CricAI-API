"""
Evaluate EfficientNetB4 + GRU Video Classifier

- Reproduces the exact split logic used in training:
  train_test_split(..., test_size=VALIDATION_SPLIT, random_state=42, stratify=y_encoded)
- Evaluates both train and test sets without leakage
- Reports accuracy, precision, recall, confusion matrix, and MCC

"""

import os
import sys
import json
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    matthews_corrcoef,
    classification_report,
)

# Add paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, os.getcwd())

from features.SHOT_CLASSIFICATION_SYSTEM.model_training.video_classifier_trainer import VideoClassifierTrainer
from features.SHOT_CLASSIFICATION_SYSTEM.utils.config import SHOT_TYPES, MODEL_FOLDER_PATH, DATASET_PATH


def evaluate_split(
    trainer: VideoClassifierTrainer,
    model,
    paths: np.ndarray,
    labels: np.ndarray,
    split_name: str,
) -> Dict[str, Any]:
    ds = trainer.build_tf_dataset(paths, labels, shuffle=False)
    probs = model.predict(ds, verbose=1)
    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(labels, preds)
    precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    recall_macro = recall_score(labels, preds, average="macro", zero_division=0)
    precision_weighted = precision_score(labels, preds, average="weighted", zero_division=0)
    recall_weighted = recall_score(labels, preds, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    cm = confusion_matrix(labels, preds)

    print("\n" + "=" * 80)
    print(f"{split_name.upper()} METRICS")
    print("=" * 80)
    print(f"Accuracy          : {acc:.4f}")
    print(f"Precision (macro) : {precision_macro:.4f}")
    print(f"Recall (macro)    : {recall_macro:.4f}")
    print(f"Precision (wtd)   : {precision_weighted:.4f}")
    print(f"Recall (wtd)      : {recall_weighted:.4f}")
    print(f"MCC               : {mcc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    report = classification_report(
        labels,
        preds,
        target_names=[str(c) for c in trainer.label_encoder.classes_],
        output_dict=True,
        zero_division=0,
    )

    return {
        "split": split_name,
        "num_samples": int(len(labels)),
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "mcc": float(mcc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def load_weights_with_fallback(model, model_dir: str) -> str:
    candidates = [
        os.path.join(model_dir, "video_classifier", "best_model.weights.h5"),
        os.path.join(model_dir, "video_classifier", "model.weights.h5"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                model.load_weights(path)
            except ValueError:
                # Keras 3 may require legacy '.h5' extension for by_name loading.
                with tempfile.TemporaryDirectory() as tmp_dir:
                    legacy_path = os.path.join(tmp_dir, "legacy_weights.h5")
                    shutil.copyfile(path, legacy_path)
                    model.load_weights(legacy_path, by_name=True, skip_mismatch=True)
            return path
    raise FileNotFoundError(
        "No model weights found. Expected best_model.weights.h5 or model.weights.h5 "
        f"in {os.path.join(model_dir, 'video_classifier')}"
    )


def load_cached_model(model_dir: str):
    cache_path = os.path.join(model_dir, "video_classifier", "model_complete.keras")
    if not os.path.exists(cache_path):
        return None, cache_path

    try:
        model = keras.models.load_model(cache_path)
        print(f"✓ Loaded cached complete model: {cache_path}")
        return model, cache_path
    except Exception as e:
        print(f"⚠ Failed to load cached model ({e}); will rebuild from weights")
        return None, cache_path


def ensure_model_built(model, frame_count: int, frame_size: tuple[int, int]):
    expected_shape = (None, frame_count, frame_size[0], frame_size[1], 3)

    try:
        if not model.built:
            model.build(expected_shape)
    except Exception:
        pass

    if not getattr(model, "inputs", None):
        dummy_input = tf.zeros((1, frame_count, frame_size[0], frame_size[1], 3), dtype=tf.float32)
        _ = model(dummy_input, training=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate EfficientNetB4 + GRU video classifier")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH, help="Dataset root containing shot folders")
    parser.add_argument("--shots", type=str, default=",".join(SHOT_TYPES), help="Comma-separated shot types")
    parser.add_argument("--model-dir", type=str, default=MODEL_FOLDER_PATH, help="Model directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results/video_classifier_evaluation_report.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    shot_types = [s.strip() for s in args.shots.split(",") if s.strip()]
    dataset_path = args.dataset

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    print("\n" + "=" * 80)
    print("EVALUATING EFFICIENTNETB4 + GRU VIDEO CLASSIFIER")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Shots: {shot_types}")
    print(f"Model dir: {args.model_dir}")
    print("=" * 80)

    trainer = VideoClassifierTrainer(model_dir=args.model_dir)
    trainer.BATCH_SIZE = args.batch_size

    video_paths, y = trainer.prepare_dataset(dataset_path, shot_types)

    # Exact training split logic reproduction.
    y_encoded = trainer.label_encoder.fit_transform(y)
    paths_train, paths_test, y_train, y_test = train_test_split(
        video_paths,
        y_encoded,
        test_size=trainer.VALIDATION_SPLIT,
        random_state=42,
        stratify=y_encoded,
    )

    print(f"\nTrain samples: {len(paths_train)}")
    print(f"Test samples : {len(paths_test)}")

    model, cache_path = load_cached_model(args.model_dir)
    loaded_from = cache_path if model is not None else None

    if model is None:
        model = trainer.build_model(num_classes=len(trainer.label_encoder.classes_))
        loaded_from = load_weights_with_fallback(model, args.model_dir)
        print(f"Loaded weights: {loaded_from}")

        # Save complete model (architecture + loaded weights) for faster future loads
        try:
            trainer.model = model
            trainer.save_compiled_model(len(trainer.label_encoder.classes_))
        except Exception as e:
            print(f"⚠ Could not cache model: {e}")
    else:
        print(f"Using cached complete model: {loaded_from}")

    ensure_model_built(model, trainer.FRAME_COUNT, trainer.FRAME_SIZE)

    train_metrics = evaluate_split(trainer, model, paths_train, y_train, "train")
    test_metrics = evaluate_split(trainer, model, paths_test, y_test, "test")

    report = {
        "model_type": "video_classifier_efficientnetb4_gru",
        "weights_path": loaded_from,
        "dataset": dataset_path,
        "shot_types": shot_types,
        "random_state": 42,
        "test_size": trainer.VALIDATION_SPLIT,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()