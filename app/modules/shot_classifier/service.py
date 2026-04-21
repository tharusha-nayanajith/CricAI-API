from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from loguru import logger
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input

from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts

from .models import ShotClassifierResult

FRAME_COUNT = 30
MIN_FRAME_COUNT = 20
FRAME_SIZE = (224, 224)
FEATURE_DIM = 128

SHOT_CLASS_LABELS = [
    "cut", "drive", "flick", "pull", "slog", "sweep", "misc"
]

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TRAINED_MODELS_DIR = ASSETS_DIR / "trained_models"
VIDEO_CLASSIFIER_DIR = TRAINED_MODELS_DIR / "video_classifier"
PROTOTYPES_PATH = TRAINED_MODELS_DIR / "prototypes" / "shot_prototypes.pkl"

EXTERNAL_MODEL_PATH = (
    Path(__file__).resolve().parents[3].parent / "CricketShotClassification" / "model_weights.h5"
)

_model: Any | None = None
_feature_extractor: Any | None = None
_prototypes: dict[str, dict[str, Any]] | None = None
_logged_missing_genai = False


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
        try:
            result = self._run_sync(
                artifacts,
                video_path,
                start_frame_idx,
                trigger_source,
                video_url,
            )
        except FeatureError:
            raise
        except Exception as exc:
            logger.exception("Shot classifier analysis failed unexpectedly")
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
        timings_ms: dict[str, float] = {}

        t0 = time.perf_counter()
        logger.info("shot_classifier step=model_ready:start video_url={}", video_url)
        model = _load_model()
        timings_ms["model_ready"] = _elapsed_ms(t0)
        logger.info("shot_classifier step=model_ready:done took_ms={:.1f} video_url={}", timings_ms["model_ready"], video_url)

        t0 = time.perf_counter()
        logger.info("shot_classifier step=frame_read:start video_url={}", video_url)
        frames = _read_clip_frames(video_path, start_frame_idx, FRAME_COUNT, FRAME_SIZE)
        timings_ms["frame_read"] = _elapsed_ms(t0)
        logger.info("shot_classifier step=frame_read:done took_ms={:.1f} video_url={}", timings_ms["frame_read"], video_url)

        t0 = time.perf_counter()
        logger.info("shot_classifier step=frame_normalize:start video_url={}", video_url)
        frames_normalized = preprocess_input(frames.astype(np.float32))
        batch = np.expand_dims(frames_normalized, axis=0)
        timings_ms["frame_normalize"] = _elapsed_ms(t0)
        logger.info("shot_classifier step=frame_normalize:done took_ms={:.1f} video_url={}", timings_ms["frame_normalize"], video_url)

        t0 = time.perf_counter()
        logger.info("shot_classifier step=predict:start video_url={}", video_url)
        try:
            predictions = model.predict(batch, verbose=0)
        except Exception as exc:
            logger.error("TensorFlow inference failed: {}", exc)
            raise FeatureError("TensorFlow inference failed for shot_classifier.") from exc
        timings_ms["predict"] = _elapsed_ms(t0)
        logger.info("shot_classifier step=predict:done took_ms={:.1f} video_url={}", timings_ms["predict"], video_url)

        scores = np.asarray(predictions, dtype=np.float32).reshape(-1)
        if scores.shape[0] != len(SHOT_CLASS_LABELS):
            raise FeatureError(
                f"Shot classifier output size mismatch: got {scores.shape[0]}, expected {len(SHOT_CLASS_LABELS)}"
            )

        predicted_idx = int(np.argmax(scores))
        probabilities = {
            label: round(float(score), 6)
            for label, score in zip(SHOT_CLASS_LABELS, scores, strict=False)
        }
        predicted_shot = SHOT_CLASS_LABELS[predicted_idx]

        t0 = time.perf_counter()
        logger.info("shot_classifier step=feature_extract:start video_url={}", video_url)
        features = _extract_features(frames_normalized)
        timings_ms["feature_extract"] = _elapsed_ms(t0)
        logger.info("shot_classifier step=feature_extract:done took_ms={:.1f} video_url={}", timings_ms["feature_extract"], video_url)

        t0 = time.perf_counter()
        logger.info("shot_classifier step=mistake_analysis:start video_url={}", video_url)
        analysis_result = _run_mistake_analysis(predicted_shot, features)
        timings_ms["mistake_analysis"] = _elapsed_ms(t0)
        logger.info("shot_classifier step=mistake_analysis:done took_ms={:.1f} video_url={}", timings_ms["mistake_analysis"], video_url)
        mistakes = analysis_result.get("mistakes", [])

        t0 = time.perf_counter()
        logger.info("shot_classifier step=feedback:start video_url={}", video_url)
        coaching_feedback = _generate_ai_feedback(predicted_shot, float(scores[predicted_idx]), mistakes)
        timings_ms["feedback"] = _elapsed_ms(t0)
        logger.info("shot_classifier step=feedback:done took_ms={:.1f} video_url={}", timings_ms["feedback"], video_url)

        critical_count = sum(1 for mistake in mistakes if mistake.get("severity") == "critical")
        correction_summary = f"Critical ({critical_count})" if mistakes else "No issues detected"

        logger.info(
            "shot_classifier timings_ms model_ready={:.1f} frame_read={:.1f} frame_normalize={:.1f} predict={:.1f} feature_extract={:.1f} mistake_analysis={:.1f} feedback={:.1f}",
            timings_ms["model_ready"],
            timings_ms["frame_read"],
            timings_ms["frame_normalize"],
            timings_ms["predict"],
            timings_ms["feature_extract"],
            timings_ms["mistake_analysis"],
            timings_ms["feedback"],
        )

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


def _elapsed_ms(start_time: float) -> float:
    return (time.perf_counter() - start_time) * 1000.0


def warmup_shot_classifier() -> None:
    start_time = time.perf_counter()
    try:
        _load_model()
        _load_feature_extractor()
        get_prototypes()
        logger.info(
            "Shot classifier warmup completed in {:.1f} ms",
            _elapsed_ms(start_time),
        )
    except Exception as exc:
        logger.warning("Shot classifier warmup failed: {}", exc)


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
        VIDEO_CLASSIFIER_DIR / "model_complete.keras",
        VIDEO_CLASSIFIER_DIR / "best_model_complete.keras",
        VIDEO_CLASSIFIER_DIR / "model_complete.keras",
        ASSETS_DIR / "model_weights.h5",
        EXTERNAL_MODEL_PATH,
    ]

    for path in model_paths_to_check:
        if path.exists():
            logger.info("Found shot_classifier model at: {}", path)
            return path

    logger.error(
        "Shot classifier model not found. Checked paths: {}. Set SHOT_CLASSIFIER_MODEL_PATH environment variable.",
        model_paths_to_check,
    )
    raise FeatureError(
        f"Missing shot_classifier model file. Set SHOT_CLASSIFIER_MODEL_PATH or place model_weights.h5 at one of: {[str(p) for p in model_paths_to_check]}"
    )


def _load_feature_extractor() -> Any:
    global _feature_extractor
    if _feature_extractor is not None:
        return _feature_extractor

    model = _load_model()
    feature_input = keras.Input(shape=(FRAME_COUNT, FRAME_SIZE[0], FRAME_SIZE[1], 3))
    feature_output = feature_input

    for layer in model.layers[:4]:
        feature_output = layer(feature_output)

    _feature_extractor = models.Model(feature_input, feature_output)
    return _feature_extractor


def _extract_features(frames_normalized: np.ndarray) -> np.ndarray:
    extractor = _load_feature_extractor()
    batch = np.expand_dims(frames_normalized, axis=0)
    features = np.asarray(extractor.predict(batch, verbose=0)[0], dtype=np.float32).reshape(-1)
    if features.shape[0] != FEATURE_DIM:
        raise FeatureError(
            f"Shot classifier embedding size mismatch: got {features.shape[0]}, expected {FEATURE_DIM}"
        )
    return features


def get_prototypes() -> dict[str, dict[str, Any]]:
    global _prototypes
    if _prototypes is not None:
        return _prototypes

    if not PROTOTYPES_PATH.exists():
        logger.warning("Prototypes not found at {}", PROTOTYPES_PATH)
        return {}

    try:
        raw_prototypes = joblib.load(PROTOTYPES_PATH)
    except Exception as exc:
        logger.error("Failed to load prototypes: {}", exc)
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in raw_prototypes.items():
        key = str(raw_key)
        if key.isdigit():
            key = SHOT_CLASS_LABELS[int(key)]

        if not isinstance(raw_value, dict):
            continue

        mean = np.asarray(raw_value.get("mean", np.zeros(FEATURE_DIM)), dtype=np.float32).reshape(-1)
        std = np.asarray(raw_value.get("std", np.ones(FEATURE_DIM)), dtype=np.float32).reshape(-1)
        if mean.shape[0] != FEATURE_DIM or std.shape[0] != FEATURE_DIM:
            logger.warning(
                "Skipping prototype {} due to embedding size mismatch mean={} std={}",
                key,
                mean.shape,
                std.shape,
            )
            continue

        normalized[key] = {
            "mean": mean,
            "std": std,
            "samples": int(raw_value.get("samples", 0) or 0),
        }

    _prototypes = normalized
    logger.info("Loaded {} shot prototypes for mistake analysis", len(_prototypes))
    return _prototypes


def _run_mistake_analysis(predicted_shot: str, features: np.ndarray) -> dict[str, Any]:
    try:
        prototypes = get_prototypes()
        if not prototypes:
            return {}
        if predicted_shot not in prototypes:
            return {}

        prototype_data = prototypes[predicted_shot]
        prototype_mean = prototype_data.get("mean", np.zeros(FEATURE_DIM, dtype=np.float32))
        prototype_std = prototype_data.get("std", np.ones(FEATURE_DIM, dtype=np.float32))

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
                        "explanation": f"Your movement embedding component {idx} was higher than expected for a {predicted_shot}.",
                        "recommendation": f"Repeat {predicted_shot} drills to align with the learned prototype.",
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


def _generate_ai_feedback(predicted_shot: str, confidence: float, mistakes: list[dict[str, Any]]) -> str:
    global _logged_missing_genai

    has_api_key = bool(os.getenv("GEMINI_API_KEY"))
    has_genai_sdk = importlib.util.find_spec("google.genai") is not None

    if has_api_key and has_genai_sdk:
        try:
            from .assets.utils.ai_feedback_generator import AIFeedbackGenerator

            generator = AIFeedbackGenerator()
            if generator.client and hasattr(generator, "generate_feedback"):
                feedback = generator.generate_feedback(
                    predicted_shot=predicted_shot,
                    confidence=confidence,
                    mistakes=mistakes,
                )
                if isinstance(feedback, str) and feedback.strip():
                    return feedback
        except Exception as exc:
            logger.warning("AI feedback generation failed: {}", exc)
    elif not _logged_missing_genai:
        logger.info(
            "AI feedback disabled for shot_classifier: {}{}",
            "missing GEMINI_API_KEY" if not has_api_key else "",
            " and missing google-genai SDK" if not has_genai_sdk and not has_api_key else "missing google-genai SDK" if not has_genai_sdk else "",
        )
        _logged_missing_genai = True

    if confidence < 0.7:
        return f"Low confidence prediction for {predicted_shot}. Review form and positioning."
    if mistakes:
        return f"Significant deviation from correct {predicted_shot} form. Focus on improving the identified body positions."
    return f"Good {predicted_shot} execution. Continue practicing and refining technique."
