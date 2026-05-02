"""EfficientNetB4 + GRU video classifier evaluation.

Bash (from project root):
    uv run python app/modules/shot_classifier/assets/development/model_training/evaluate_video_classifier.py `
        --dataset "C:\\Users\\User\\Downloads\\Cricket-Shots" `
        --model-dir app/modules/shot_classifier/assets/trained_models
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from tensorflow import keras


current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))
sys.path.insert(0, os.getcwd())

from video_classifier_trainer import VideoClassifierTrainer
from app.modules.shot_classifier.assets.utils.config import SHOT_TYPES, MODEL_FOLDER_PATH, DATASET_PATH


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    tick_positions = np.arange(len(class_names))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = cm.max() / 2.0 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = cm[row, col]
            ax.text(
                col,
                row,
                format(value, "d"),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
) -> Dict[str, float]:
    num_classes = len(class_names)
    labels_binarized = label_binarize(labels, classes=np.arange(num_classes))

    fig, ax = plt.subplots(figsize=(9, 7))
    per_class_auc: Dict[str, float] = {}

    for class_index, class_name in enumerate(class_names):
        y_true = labels_binarized[:, class_index]
        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, probabilities[:, class_index])
        roc_auc = auc(fpr, tpr)
        per_class_auc[class_name] = float(roc_auc)
        ax.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC = {roc_auc:.3f})")

    try:
        macro_auc = roc_auc_score(labels_binarized, probabilities, average="macro")
        weighted_auc = roc_auc_score(labels_binarized, probabilities, average="weighted")
    except ValueError:
        macro_auc = float("nan")
        weighted_auc = float("nan")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "roc_auc_macro": float(macro_auc),
        "roc_auc_weighted": float(weighted_auc),
        "roc_auc_per_class": per_class_auc,
    }


def evaluate_split(
    trainer: VideoClassifierTrainer,
    model,
    paths: np.ndarray,
    labels: np.ndarray,
    split_name: str,
    output_dir: Path,
    class_names: list[str],
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
    output_dir.mkdir(parents=True, exist_ok=True)
    confusion_matrix_path = output_dir / f"{split_name}_confusion_matrix.png"
    roc_curve_path = output_dir / f"{split_name}_roc_curves.png"
    plot_confusion_matrix(
        cm,
        class_names,
        confusion_matrix_path,
        f"{split_name.title()} Set Confusion Matrix",
    )
    roc_auc_metrics = plot_roc_curves(
        labels,
        probs,
        class_names,
        roc_curve_path,
        f"{split_name.title()} Set ROC Curves",
    )

    print("\n" + "=" * 80)
    print(f"{split_name.upper()} METRICS")
    print("=" * 80)
    print(f"Accuracy          : {acc:.4f}")
    print(f"Precision (macro) : {precision_macro:.4f}")
    print(f"Recall (macro)    : {recall_macro:.4f}")
    print(f"Precision (wtd)   : {precision_weighted:.4f}")
    print(f"Recall (wtd)      : {recall_weighted:.4f}")
    print(f"MCC               : {mcc:.4f}")
    print(f"ROC AUC (macro)   : {roc_auc_metrics['roc_auc_macro']:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"Confusion matrix image: {confusion_matrix_path}")
    print(f"ROC curves image      : {roc_curve_path}")

    report = classification_report(
        labels,
        preds,
        target_names=class_names,
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
        "roc_auc_macro": roc_auc_metrics["roc_auc_macro"],
        "roc_auc_weighted": roc_auc_metrics["roc_auc_weighted"],
        "roc_auc_per_class": roc_auc_metrics["roc_auc_per_class"],
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_image": str(confusion_matrix_path),
        "roc_curve_image": str(roc_curve_path),
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
        default="app/modules/shot_classifier/assets/evaluation_results_shot_classifier/video_classifier_evaluation_report.json",
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

    class_names = [str(c) for c in trainer.label_encoder.classes_]
    output_dir = Path(args.output).parent

    model, cache_path = load_cached_model(args.model_dir)
    loaded_from = cache_path if model is not None else None

    if model is None:
        model = trainer.build_model(num_classes=len(trainer.label_encoder.classes_))
        loaded_from = load_weights_with_fallback(model, args.model_dir)
        print(f"Loaded weights: {loaded_from}")

        try:
            trainer.model = model
            trainer.save_compiled_model(len(trainer.label_encoder.classes_))
        except Exception as e:
            print(f"⚠ Could not cache model: {e}")
    else:
        print(f"Using cached complete model: {loaded_from}")

    ensure_model_built(model, trainer.FRAME_COUNT, trainer.FRAME_SIZE)

    train_metrics = evaluate_split(
        trainer,
        model,
        paths_train,
        y_train,
        "train",
        output_dir,
        class_names,
    )
    test_metrics = evaluate_split(
        trainer,
        model,
        paths_test,
        y_test,
        "test",
        output_dir,
        class_names,
    )

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