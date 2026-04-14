from __future__ import annotations

import asyncio
import os
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from loguru import logger

from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts

from .models import ShotClassifierResult

FRAME_COUNT = 30
MIN_FRAME_COUNT = 20
FRAME_SIZE = (224, 224)
FEATURE_DIM = 128

SHOT_CLASS_LABELS = [
    "cover",
    "cut",
    "drive",
    "flick",
    "glance",
    "misc",
    "pull",
    "slog",
    "straight",
    "sweep",
]

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_PATH = ASSETS_DIR / "model_weights.h5"
EXTERNAL_MODEL_PATH = (
    Path(__file__).resolve().parents[3].parent / "CricketShotClassification" / "model_weights.h5"
)

_model: Any | None = None
_feature_extractor: Any | None = None
_prototypes: dict[str, dict] | None = None


def _trained_models_dir() -> Path:
    return ASSETS_DIR / "trained_models"


def _video_classifier_dir() -> Path:
    return _trained_models_dir() / "video_classifier"


def _prototypes_path() -> Path:
    return _trained_models_dir() / "prototypes" / "shot_prototypes.pkl"


def _load_tensorflow_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import EfficientNetB4
        from tensorflow.keras.applications.efficientnet import preprocess_input
    except ImportError as exc:
        raise FeatureError("tensorflow is required for the shot_classifier module.") from exc

    return tf, keras, layers, models, EfficientNetB4, preprocess_input


class ShotClassifierService:
    async def run(
        self,
        artifacts: VideoArtifacts,
        video_path: Path,
        video_url: str | None = None,
    ) -> ShotClassifierResult:
        start_frame_idx, trigger_source = _resolve_start_frame(artifacts)
        logger.info(
            "Starting shot_classifier analysis start_frame_idx={} trigger_source={} video_url={}",
            start_frame_idx,
            trigger_source,
            video_url,
        )
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                partial(
                    self._run_sync,
                    artifacts,
                    video_path,
                    start_frame_idx,
                    trigger_source,
                    video_url,
                ),
            )
        except FeatureError:
            raise
        except Exception as exc:
            raise FeatureError("Shot classifier analysis failed unexpectedly") from exc

        logger.info(
            "Completed shot_classifier analysis predicted_shot={} confidence={:.3f}",
            result.predicted_shot,
            result.confidence,
        )
        return result

    def _run_sync(
        self,
        artifacts: VideoArtifacts,
        video_path: Path,
        start_frame_idx: int,
        trigger_source: str,
        video_url: str | None,
    ) -> ShotClassifierResult:
        _, _, _, _, _, preprocess_input = _load_tensorflow_modules()
        model = _load_model()
        frames = _read_clip_frames(video_path, start_frame_idx, FRAME_COUNT, FRAME_SIZE)
        frames_normalized = preprocess_input(frames.astype(np.float32))
        batch = np.expand_dims(frames_normalized, axis=0)

        try:
            predictions = model.predict(batch, verbose=0)
        except Exception as exc:
            logger.error("TensorFlow inference failed: {}", exc)
            raise FeatureError("TensorFlow inference failed for shot_classifier.") from exc

        scores = np.asarray(predictions, dtype=np.float32).reshape(-1)
        if scores.shape[0] != len(SHOT_CLASS_LABELS):
            raise FeatureError(
                "Shot classifier output size mismatch: got "
                f"{scores.shape[0]}, expected {len(SHOT_CLASS_LABELS)}"
            )

        predicted_idx = int(np.argmax(scores))
        probabilities = {
            label: round(float(score), 6)
            for label, score in zip(SHOT_CLASS_LABELS, scores, strict=False)
        }

        features = _extract_features(frames_normalized)
        predicted_shot = SHOT_CLASS_LABELS[predicted_idx]
        analysis_result = _run_mistake_analysis(predicted_shot, features)
        mistakes = analysis_result.get("mistakes", [])
        coaching_feedback = _generate_ai_feedback(
            predicted_shot,
            float(scores[predicted_idx]),
            mistakes,
        )
        critical_count = len([m for m in mistakes if m.get("severity") == "critical"])
        correction_summary = f"Critical ({critical_count})" if mistakes else "No issues detected"

        return ShotClassifierResult(
            predicted_shot=predicted_shot,
            confidence=float(scores[predicted_idx]),
            probabilities=probabilities,
            frames_used=FRAME_COUNT,
            frame_start_index=start_frame_idx,
            frame_end_index=start_frame_idx + FRAME_COUNT - 1,
            roi_entry_frame_index=artifacts.batter_roi_entry_frame_idx,
            trigger_source=trigger_source,
            video_url=video_url,
            mistake_analysis=mistakes,
            coaching_feedback=coaching_feedback,
            correction_summary=correction_summary,
        )


def _resolve_start_frame(
    artifacts: VideoArtifacts,
) -> tuple[int, str]:
    if artifacts.batter_roi_entry_frame_idx is not None:
        return max(0, artifacts.batter_roi_entry_frame_idx), "batter_roi_entry"
    if artifacts.bat_contact is not None:
        fallback_start = max(0, artifacts.bat_contact.contact_frame_idx - FRAME_COUNT + 1)
        return fallback_start, "bat_contact_fallback"
    raise FeatureError(
        "Shot classifier requires a batter ROI entry frame or bat-contact fallback frame."
    )


def _read_clip_frames(
    video_path: Path,
    start_frame_idx: int,
    frame_count: int,
    frame_size: tuple[int, int],
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise FeatureError(f"Unable to open video file for shot classification: {video_path}")

    frames: list[np.ndarray] = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < MIN_FRAME_COUNT:
            raise FeatureError(
                f"Video too short: {total_frames} frames (need minimum {MIN_FRAME_COUNT})"
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        for _ in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                if frames:
                    frames.append(np.zeros_like(frames[0]))
                else:
                    frames.append(np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8))
                continue
            frames.append(_format_frame(frame, frame_size))
    finally:
        cap.release()

    return np.asarray(frames, dtype=np.uint8)


def _format_frame(frame_bgr: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    target_height, target_width = frame_size
    source_height, source_width = frame_bgr.shape[:2]
    if source_height <= 0 or source_width <= 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    top = (target_height - resized_height) // 2
    left = (target_width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _load_model() -> Any:
    global _model
    if _model is not None:
        return _model

    _, _, layers, models, EfficientNetB4, _ = _load_tensorflow_modules()
    model_path = _resolve_model_path()

    try:
        logger.info("Loading EfficientNetB4+GRU shot classifier model from {}", model_path)
        base_model = EfficientNetB4(
            include_top=False,
            weights=None,
            input_shape=(FRAME_SIZE[0], FRAME_SIZE[1], 3),
        )
        base_model.trainable = False

        _model = models.Sequential([
            layers.Input(shape=(FRAME_COUNT, FRAME_SIZE[0], FRAME_SIZE[1], 3)),
            layers.TimeDistributed(base_model),
            layers.TimeDistributed(layers.GlobalAveragePooling2D()),
            layers.GRU(256, return_sequences=True, dropout=0.3, unroll=True),
            layers.GRU(128, dropout=0.3, unroll=True),
            layers.Dense(1024, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(len(SHOT_CLASS_LABELS), activation="softmax"),
        ])

        _model.load_weights(str(model_path))
        logger.info("EfficientNetB4+GRU model loaded successfully")
    except FileNotFoundError as exc:
        logger.error("Model weights file not found: {}", exc)
        raise FeatureError(f"Shot classifier model weights not found: {model_path}") from exc
    except FeatureError:
        raise
    except Exception as exc:
        logger.error("Failed to load shot_classifier model: {}", exc)
        raise FeatureError("Failed to load shot_classifier model") from exc

    return _model


def _resolve_model_path() -> Path:
    configured_path = os.getenv("SHOT_CLASSIFIER_MODEL_PATH")
    if configured_path:
        model_path = Path(configured_path)
        if model_path.exists():
            logger.info("Using model from environment variable: {}", model_path)
            return model_path
        logger.warning("Configured model path does not exist: {}", model_path)

    model_paths_to_check = [
        MODEL_PATH,
        _video_classifier_dir() / "model.weights.h5",
        _video_classifier_dir() / "best_model.weights.h5",
        _video_classifier_dir() / "model_complete.keras",
        EXTERNAL_MODEL_PATH,
    ]

    for path in model_paths_to_check:
        if path.exists():
            logger.info("Found shot_classifier model at: {}", path)
            return path

    fallback_assets = sorted(ASSETS_DIR.glob("*.h5"))
    if fallback_assets:
        logger.info("Falling back to shot_classifier asset at {}", fallback_assets[0])
        return fallback_assets[0]

    logger.error(
        "Shot classifier model not found. Checked paths: {}. "
        "Set SHOT_CLASSIFIER_MODEL_PATH environment variable.",
        model_paths_to_check,
    )
    checked_paths = [str(path) for path in model_paths_to_check]
    raise FeatureError(
        "Missing shot_classifier model file. Set SHOT_CLASSIFIER_MODEL_PATH or place "
        f"model_weights.h5 at one of: {checked_paths}"
    )


def _extract_features(frames_normalized: np.ndarray) -> np.ndarray:
    global _feature_extractor

    model = _load_model()
    if not hasattr(model, "layers"):
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    if _feature_extractor is None:
        _, keras, _, models, _, _ = _load_tensorflow_modules()
        feature_input = keras.Input(shape=(FRAME_COUNT, FRAME_SIZE[0], FRAME_SIZE[1], 3))
        feature_output = feature_input
        for layer in model.layers[:5]:
            feature_output = layer(feature_output)
        _feature_extractor = models.Model(feature_input, feature_output)

    batch = np.expand_dims(frames_normalized, axis=0)
    features = _feature_extractor.predict(batch, verbose=0)[0]
    return features.astype(np.float32)


def get_prototypes() -> dict[str, dict]:
    global _prototypes

    if _prototypes is not None:
        return _prototypes

    prototypes_path = _prototypes_path()
    if not prototypes_path.exists():
        logger.warning("Prototypes not found at {}", prototypes_path)
        return {}

    try:
        _prototypes = joblib.load(prototypes_path)
        logger.info("Loaded {} shot prototypes for mistake analysis", len(_prototypes))
        return _prototypes
    except Exception as exc:
        logger.error("Failed to load prototypes: {}", exc)
        return {}


def _run_mistake_analysis(predicted_shot: str, features: np.ndarray) -> dict:
    try:
        prototypes = get_prototypes()
        if not prototypes:
            return {}
        if predicted_shot not in prototypes:
            return {}

        prototype_data = prototypes[predicted_shot]
        prototype_mean = prototype_data.get("mean", np.zeros(FEATURE_DIM))
        prototype_std = prototype_data.get("std", np.ones(FEATURE_DIM))
        deviations = np.abs(features - prototype_mean) / (prototype_std + 1e-6)
        top_indices = np.argsort(deviations)[-5:][::-1]

        mistakes = []
        for idx in top_indices:
            if deviations[idx] > 0.5:
                mistakes.append(
                    {
                        "joint_id": "body_position",
                        "body_part": "Body Position",
                        "feature_name": f"embedding_{idx:03d}",
                        "severity": "critical" if deviations[idx] > 1.0 else "warning",
                        "actual_value": float(features[idx]),
                        "expected_value": float(prototype_mean[idx]),
                        "deviation": float(deviations[idx]),
                        "explanation": (
                            f"Your movement embedding component {idx} was higher than expected "
                            f"for a {predicted_shot}."
                        ),
                        "recommendation": (
                            f"Repeat {predicted_shot} drills to align with the learned prototype."
                        ),
                    }
                )

        return {
            "mistakes": mistakes,
            "prototype_samples": prototype_data.get("samples", 0),
            "analysis_method": "efficientnetb4_gru_embedding",
        }
    except Exception as exc:
        logger.warning("Mistake analysis failed: {}", exc)
        return {}


def _generate_ai_feedback(predicted_shot: str, confidence: float, mistakes: list) -> str:
    try:
        from .assets.utils.ai_feedback_generator import AIFeedbackGenerator

        generator = AIFeedbackGenerator()
        if generator.client:
            return generator.generate_feedback(
                predicted_shot=predicted_shot,
                confidence=confidence,
                mistakes=mistakes,
            )
    except Exception as exc:
        logger.warning("AI feedback generation failed: {}", exc)

    if confidence < 0.7:
        return f"Low confidence prediction for {predicted_shot}. Review form and positioning."
    if mistakes:
        return (
            f"Significant deviation from correct {predicted_shot} form. "
            "Focus on improving the identified body positions."
        )
    return f"Good {predicted_shot} execution. Continue practicing and refining technique."
