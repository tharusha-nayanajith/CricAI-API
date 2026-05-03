from __future__ import annotations

import math
import os
import signal
import threading
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

from app.ai.google import get_google_genai_status
from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts
from app.modules.preprocessor.models import BatterHandedness

from .models import SHOT_CLASS_LABELS, ShotClassifierResult, normalize_shot_label

FRAME_COUNT = 30
MIN_FRAME_COUNT = 20
FRAME_SIZE = (224, 224)
FEATURE_DIM = 128
POSE_LANDMARK_IDS = tuple(range(33))

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TRAINED_MODELS_DIR = ASSETS_DIR / "trained_models"
VIDEO_CLASSIFIER_DIR = TRAINED_MODELS_DIR / "video_classifier"
PROTOTYPES_PATH = TRAINED_MODELS_DIR / "prototypes" / "shot_prototypes.pkl"
POSE_LANDMARKER_TASK_PATH = (
    Path(__file__).resolve().parents[1] / "action_legality" / "assets" / "pose_landmarker.task"
)

EXTERNAL_MODEL_PATH = (
    Path(__file__).resolve().parents[3].parent / "CricketShotClassification" / "model_weights.h5"
)

_model: Any | None = None
_feature_extractor: Any | None = None
_prototypes: dict[str, dict[str, Any]] | None = None
_logged_missing_genai = False
_pose_landmarker: Any | None = None
_pose_landmarker_lock = threading.Lock()
_pose_landmarker_initialized = False
_pose_landmarker_cleanup_registered = False


class ShotClassifierService:
    async def run(
        self,
        artifacts: VideoArtifacts,
        video_path: Path,
        video_url: str | None = None,
        intended_shot: str | None = None,
    ) -> ShotClassifierResult:
        start_frame_idx, trigger_source = _resolve_start_frame(artifacts)
        normalized_intended_shot = _validate_intended_shot(intended_shot)
        logger.info(
            "Starting shot_classifier analysis start_frame_idx={} trigger_source={} video_url={} intended_shot={}",
            start_frame_idx,
            trigger_source,
            video_url,
            normalized_intended_shot,
        )
        try:
            result = self._run_sync(
                artifacts,
                video_path,
                start_frame_idx,
                trigger_source,
                video_url,
                normalized_intended_shot,
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
        intended_shot: str | None,
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
        confidence = float(scores[predicted_idx])
        logger.info(
            "shot_classifier prediction predicted_shot={} confidence={:.3f} probabilities={}",
            predicted_shot,
            confidence,
            probabilities,
        )
        intent_match = predicted_shot == intended_shot if intended_shot is not None else None
        intended_shot_score = probabilities.get(intended_shot) if intended_shot is not None else None
        reference_shot = intended_shot or predicted_shot
        analysis_basis = "intended_shot" if intended_shot is not None else "predicted_shot"

        t0 = time.perf_counter()
        logger.info("shot_classifier step=feature_extract:start video_url={}", video_url)
        features = _extract_features(frames_normalized)
        timings_ms["feature_extract"] = _elapsed_ms(t0)
        logger.info("shot_classifier step=feature_extract:done took_ms={:.1f} video_url={}", timings_ms["feature_extract"], video_url)

        t0 = time.perf_counter()
        logger.info(
            "shot_classifier step=mistake_analysis:start video_url={} basis={} reference_shot={}",
            video_url,
            analysis_basis,
            reference_shot,
        )
        analysis_result = _run_mistake_analysis(reference_shot, features)
        timings_ms["mistake_analysis"] = _elapsed_ms(t0)
        logger.info("shot_classifier step=mistake_analysis:done took_ms={:.1f} video_url={}", timings_ms["mistake_analysis"], video_url)
        mistakes = analysis_result.get("mistakes", [])
        logger.info(
            "shot_classifier mistake_analysis result predicted_shot={} intended_shot={} basis={} mistakes_count={} analysis_result={}",
            predicted_shot,
            intended_shot,
            analysis_basis,
            len(mistakes),
            analysis_result,
        )

        t0 = time.perf_counter()
        logger.info("shot_classifier step=technique_map:start video_url={}", video_url)
        technique_frame_bgr = _resolve_technique_frame_bgr(artifacts, frames)
        technique_map, technique_details = _build_technique_map(
            technique_frame_bgr=technique_frame_bgr,
            handedness=artifacts.batter_handedness,
        )
        timings_ms["technique_map"] = _elapsed_ms(t0)
        logger.info(
            "shot_classifier step=technique_map:done took_ms={:.1f} video_url={} technique_map={}",
            timings_ms["technique_map"],
            video_url,
            technique_map,
        )

        t0 = time.perf_counter()
        logger.info("shot_classifier step=feedback:start video_url={}", video_url)
        coaching_feedback = _generate_ai_feedback(
            predicted_shot=predicted_shot,
            confidence=confidence,
            mistakes=mistakes,
            reference_shot=reference_shot,
            intended_shot=intended_shot,
        )
        timings_ms["feedback"] = _elapsed_ms(t0)
        logger.info("shot_classifier step=feedback:done took_ms={:.1f} video_url={}", timings_ms["feedback"], video_url)

        critical_count = sum(1 for mistake in mistakes if mistake.get("severity") == "critical")
        correction_summary = f"Critical ({critical_count})" if mistakes else "No issues detected"

        logger.info(
            "shot_classifier timings_ms model_ready={:.1f} frame_read={:.1f} frame_normalize={:.1f} predict={:.1f} feature_extract={:.1f} mistake_analysis={:.1f} technique_map={:.1f} feedback={:.1f}",
            timings_ms["model_ready"],
            timings_ms["frame_read"],
            timings_ms["frame_normalize"],
            timings_ms["predict"],
            timings_ms["feature_extract"],
            timings_ms["mistake_analysis"],
            timings_ms["technique_map"],
            timings_ms["feedback"],
        )

        result = ShotClassifierResult(
            predicted_shot=predicted_shot,
            confidence=confidence,
            probabilities=probabilities,
            frames_used=FRAME_COUNT,
            frame_start_index=start_frame_idx,
            frame_end_index=start_frame_idx + FRAME_COUNT - 1,
            roi_entry_frame_index=artifacts.batter_roi_entry_frame_idx,
            trigger_source=trigger_source,
            video_url=video_url,
            intended_shot=intended_shot,
            intent_match=intent_match,
            intended_shot_score=intended_shot_score,
            mistake_analysis_basis=analysis_basis,
            mistake_analysis_reference_shot=reference_shot,
            technique_map=technique_map,
            technique_map_basis="pose_landmark_heuristic",
            technique_details=technique_details,
            visual_feedback={
                "mistakes": mistakes,
                "technique_map": technique_map,
                "technique_details": technique_details,
            },
            mistake_analysis=mistakes,
            coaching_feedback=coaching_feedback,
            correction_summary=correction_summary,
        )
        logger.info(
            "shot_classifier output predicted_shot={} intended_shot={} basis={} confidence={:.3f} mistakes_count={} correction_summary={} coaching_feedback={}",
            result.predicted_shot,
            result.intended_shot,
            result.mistake_analysis_basis,
            result.confidence,
            len(result.mistake_analysis or []),
            result.correction_summary,
            result.coaching_feedback,
        )
        return result


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
    if artifacts.ball_path:
        last_ball_frame_idx = int(artifacts.ball_path[-1].frame_idx)
        fallback_start = max(0, last_ball_frame_idx - FRAME_COUNT + 1)
        logger.info(
            "shot_classifier start fallback trigger=ball_path_end last_ball_frame_idx={} start_frame_idx={}",
            last_ball_frame_idx,
            fallback_start,
        )
        return fallback_start, "ball_path_end_fallback"
    raise FeatureError(
        "Shot classifier requires a batter ROI entry frame, bat-contact fallback frame, or ball-path fallback frame."
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
        logger.info("shot_classifier prototypes cache_hit count={}", len(_prototypes))
        return _prototypes

    if not PROTOTYPES_PATH.exists():
        logger.warning("shot_classifier prototypes missing path={}", PROTOTYPES_PATH)
        return {}

    try:
        raw_prototypes = joblib.load(PROTOTYPES_PATH)
    except Exception as exc:
        logger.error("shot_classifier prototypes load_failed error={}", exc)
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
    logger.info("shot_classifier prototypes loaded count={} labels={}", len(_prototypes), sorted(_prototypes.keys()))
    return _prototypes


def _run_mistake_analysis(reference_shot: str, features: np.ndarray) -> dict[str, Any]:
    try:
        prototypes = get_prototypes()
        if not prototypes:
            logger.info("shot_classifier mistake_analysis skipped reason=no_prototypes reference_shot={}", reference_shot)
            return {}
        if reference_shot not in prototypes:
            logger.info(
                "shot_classifier mistake_analysis skipped reason=missing_prototype reference_shot={} available_labels={}",
                reference_shot,
                sorted(prototypes.keys()),
            )
            return {}

        prototype_data = prototypes[reference_shot]
        prototype_mean = prototype_data.get("mean", np.zeros(FEATURE_DIM, dtype=np.float32))
        prototype_std = prototype_data.get("std", np.ones(FEATURE_DIM, dtype=np.float32))
        logger.info(
            "shot_classifier mistake_analysis prototype_ready reference_shot={} samples={} feature_dim={}",
            reference_shot,
            int(prototype_data.get("samples", 0) or 0),
            int(features.shape[0]),
        )

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
                        "explanation": f"Your movement embedding component {idx} was higher than expected for a {reference_shot}.",
                        "recommendation": f"Repeat {reference_shot} drills to align with the learned prototype.",
                    }
                )

        result = {
            "mistakes": mistakes,
            "prototype_samples": prototype_data.get("samples", 0),
            "analysis_method": "efficientnetb4_gru_embedding",
        }
        logger.info(
            "shot_classifier mistake_analysis completed reference_shot={} mistakes_count={} top_deviation={:.3f}",
            reference_shot,
            len(mistakes),
            float(np.max(deviations)) if deviations.size else 0.0,
        )
        return result
    except Exception as exc:
        logger.warning("shot_classifier mistake_analysis failed reference_shot={} error={}", reference_shot, exc)
        return {}


def _generate_ai_feedback(
    predicted_shot: str,
    confidence: float,
    mistakes: list[dict[str, Any]],
    reference_shot: str,
    intended_shot: str | None,
) -> str:
    global _logged_missing_genai

    status = get_google_genai_status()

    if status.enabled:
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
            "AI feedback disabled for shot_classifier: {}",
            status.reason or "not configured",
        )
        _logged_missing_genai = True

    if intended_shot is not None and intended_shot != predicted_shot:
        if mistakes:
            return (
                f"You intended {intended_shot}, but the clip was classified as {predicted_shot}. "
                f"Focus on aligning your technique with {reference_shot} fundamentals."
            )
        return (
            f"You intended {intended_shot}, but the clip was classified as {predicted_shot}. "
            f"Repeat the shot with clearer {reference_shot} mechanics."
        )
    if confidence < 0.7:
        return f"Low confidence prediction for {predicted_shot}. Review form and positioning."
    if mistakes:
        return f"Significant deviation from correct {reference_shot} form. Focus on improving the identified body positions."
    return f"Good {predicted_shot} execution. Continue practicing and refining technique."


def _validate_intended_shot(intended_shot: str | None) -> str | None:
    normalized_intended_shot = normalize_shot_label(intended_shot)
    if normalized_intended_shot is None:
        return None
    if normalized_intended_shot not in SHOT_CLASS_LABELS:
        raise FeatureError(
            f"Unsupported intended_shot '{intended_shot}'. Expected one of: {', '.join(SHOT_CLASS_LABELS)}"
        )
    return normalized_intended_shot


def _resolve_technique_frame_bgr(artifacts: VideoArtifacts, frames_rgb: np.ndarray) -> np.ndarray:
    if artifacts.bat_contact_frame is not None and artifacts.bat_contact_frame.size > 0:
        return artifacts.bat_contact_frame
    if frames_rgb.size == 0:
        raise FeatureError("Technique map requires at least one classifier frame.")
    return cv2.cvtColor(frames_rgb[-1], cv2.COLOR_RGB2BGR)


def _extract_pose_landmarks(frame_bgr: np.ndarray) -> dict[int, tuple[float, float, float, float]]:
    if frame_bgr.size == 0:
        return {}
    try:
        import mediapipe as mp
    except ImportError as exc:
        logger.warning("shot_classifier technique_map skipped: mediapipe unavailable ({})", exc)
        return {}
    try:
        landmarker = _get_pose_landmarker()
    except FeatureError as exc:
        logger.warning("shot_classifier technique_map skipped: {}", exc)
        return {}

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
    )
    results = landmarker.detect(image)
    if not results.pose_landmarks:
        return {}

    landmarks: dict[int, tuple[float, float, float, float]] = {}
    pose_landmarks = results.pose_landmarks[0]
    for landmark_idx in POSE_LANDMARK_IDS:
        landmark = pose_landmarks[landmark_idx]
        landmarks[landmark_idx] = (
            float(landmark.x),
            float(landmark.y),
            float(landmark.z),
            float(getattr(landmark, "visibility", 1.0)),
        )
    return landmarks


def _get_pose_landmarker() -> Any:
    global _pose_landmarker
    global _pose_landmarker_initialized
    if _pose_landmarker is not None:
        return _pose_landmarker
    if _pose_landmarker_initialized:
        raise FeatureError("PoseLandmarker initialization failed earlier.")

    model_path = os.getenv("MEDIAPIPE_POSE_TASK_PATH")
    if not model_path:
        model_path = str(POSE_LANDMARKER_TASK_PATH)
    model_file = Path(model_path)
    if not model_file.exists():
        raise FeatureError(
            "mediapipe PoseLandmarker model is missing. "
            "Set MEDIAPIPE_POSE_TASK_PATH or place the task file at "
            f"{model_file}."
        )

    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
    except Exception as exc:
        raise FeatureError(
            "mediapipe Tasks API is unavailable. Install the full mediapipe package."
        ) from exc

    with _pose_landmarker_lock:
        if _pose_landmarker is not None:
            return _pose_landmarker
        if _pose_landmarker_initialized:
            raise FeatureError("PoseLandmarker initialization failed earlier.")

        delegate_name = os.getenv("MEDIAPIPE_POSE_DELEGATE", "cpu").strip().lower()
        delegate = BaseOptions.Delegate.GPU if delegate_name == "gpu" else BaseOptions.Delegate.CPU
        if delegate is BaseOptions.Delegate.CPU:
            logger.info("shot_classifier technique_map initializing PoseLandmarker with CPU delegate")
        try:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(model_file),
                    delegate=delegate,
                ),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
            )
            _pose_landmarker = PoseLandmarker.create_from_options(options)
        except OSError as cpu_exc:
            raise FeatureError(
                "MediaPipe PoseLandmarker could not be initialized because a required "
                "system library is missing: "
                f"{cpu_exc}. Install the OpenGL runtime package that provides "
                "libGLESv2.so.2 in the Linux environment."
            ) from cpu_exc
        except Exception as exc:
            if delegate is BaseOptions.Delegate.GPU:
                logger.warning(
                    "shot_classifier technique_map PoseLandmarker GPU delegate failed ({}). Falling back to CPU.",
                    exc,
                )
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(model_file),
                        delegate=BaseOptions.Delegate.CPU,
                    ),
                    running_mode=RunningMode.IMAGE,
                    num_poses=1,
                )
                try:
                    _pose_landmarker = PoseLandmarker.create_from_options(options)
                except OSError as cpu_exc:
                    raise FeatureError(
                        "MediaPipe PoseLandmarker could not be initialized because a required "
                        "system library is missing: "
                        f"{cpu_exc}. Install the OpenGL runtime package that provides "
                        "libGLESv2.so.2 in the Linux environment."
                    ) from cpu_exc
                except Exception as cpu_exc:
                    raise FeatureError(
                        f"MediaPipe PoseLandmarker CPU fallback failed: {cpu_exc}"
                    ) from cpu_exc
            else:
                raise FeatureError(
                    f"MediaPipe PoseLandmarker CPU initialization failed: {exc}"
                ) from exc

        _pose_landmarker_initialized = True
        _register_landmarker_cleanup()
        return _pose_landmarker


def _register_landmarker_cleanup() -> None:
    global _pose_landmarker_cleanup_registered
    if _pose_landmarker_cleanup_registered:
        return
    if threading.current_thread() is not threading.main_thread():
        return

    def _cleanup_landmarker(*_args: Any) -> None:
        global _pose_landmarker
        if _pose_landmarker is None:
            return
        try:
            _pose_landmarker.close()
        except Exception as exc:
            logger.debug("shot_classifier PoseLandmarker close failed: {}", exc)
        finally:
            _pose_landmarker = None

    signal.signal(signal.SIGTERM, _cleanup_landmarker)
    signal.signal(signal.SIGINT, _cleanup_landmarker)
    _pose_landmarker_cleanup_registered = True


def _build_technique_map(
    technique_frame_bgr: np.ndarray,
    handedness: BatterHandedness | None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    default_map = {
        "torso": 100.0,
        "front_elbow": 100.0,
        "back_elbow": 100.0,
        "back_knee": 100.0,
        "shoulders": 100.0,
    }
    landmarks = _extract_pose_landmarks(technique_frame_bgr)
    if not landmarks:
        return default_map, []

    front_side, back_side = _resolve_front_and_back_sides(handedness, landmarks)

    metric_specs = [
        {
            "key": "torso",
            "label": "Torso",
            "score": _score_torso(landmarks),
            "metric": "spine_alignment",
        },
        {
            "key": "front_elbow",
            "label": "Front Elbow",
            "score": _score_elbow(landmarks, front_side),
            "metric": f"{front_side}_elbow_angle",
        },
        {
            "key": "back_elbow",
            "label": "Back Elbow",
            "score": _score_elbow(landmarks, back_side),
            "metric": f"{back_side}_elbow_angle",
        },
        {
            "key": "back_knee",
            "label": "Back Knee",
            "score": _score_knee(landmarks, back_side),
            "metric": f"{back_side}_knee_angle",
        },
        {
            "key": "shoulders",
            "label": "Shoulders",
            "score": _score_shoulders(landmarks),
            "metric": "shoulder_tilt",
        },
    ]

    technique_map = {
        spec["key"]: round(float(spec["score"]), 1)
        for spec in metric_specs
    }
    details = [
        {
            "body_part": spec["label"],
            "score": round(float(spec["score"]), 1),
            "metric": spec["metric"],
        }
        for spec in metric_specs
    ]
    return technique_map, details


def _resolve_front_and_back_sides(
    handedness: BatterHandedness | None,
    landmarks: dict[int, tuple[float, float, float, float]],
) -> tuple[str, str]:
    if handedness == BatterHandedness.LEFT:
        return "right", "left"
    if handedness == BatterHandedness.RIGHT:
        return "left", "right"

    left_visibility = _landmark_visibility(landmarks, 11) + _landmark_visibility(landmarks, 13)
    right_visibility = _landmark_visibility(landmarks, 12) + _landmark_visibility(landmarks, 14)
    if right_visibility > left_visibility:
        return "right", "left"
    return "left", "right"


def _score_torso(landmarks: dict[int, tuple[float, float, float, float]]) -> float:
    torso_visibility = min(
        _landmark_visibility(landmarks, 11),
        _landmark_visibility(landmarks, 12),
        _landmark_visibility(landmarks, 23),
        _landmark_visibility(landmarks, 24),
    )
    if torso_visibility < 0.35:
        return 100.0

    left_shoulder = _xy(landmarks, 11)
    right_shoulder = _xy(landmarks, 12)
    left_hip = _xy(landmarks, 23)
    right_hip = _xy(landmarks, 24)
    if None in {left_shoulder, right_shoulder, left_hip, right_hip}:
        return 100.0

    assert left_shoulder is not None
    assert right_shoulder is not None
    assert left_hip is not None
    assert right_hip is not None

    shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2.0, (left_shoulder[1] + right_shoulder[1]) / 2.0)
    hip_mid = ((left_hip[0] + right_hip[0]) / 2.0, (left_hip[1] + right_hip[1]) / 2.0)
    spine_dx = shoulder_mid[0] - hip_mid[0]
    spine_dy = hip_mid[1] - shoulder_mid[1]
    if abs(spine_dy) < 1e-6:
        spine_dy = 1e-6
    spine_lean = abs(math.degrees(math.atan2(spine_dx, spine_dy)))
    hip_tilt = abs(_tilt_degrees(left_hip, right_hip))
    return _combine_scores(
        _score_from_delta(spine_lean, threshold=12.0, hard_limit=40.0, floor=20.0),
        _score_from_delta(hip_tilt, threshold=10.0, hard_limit=28.0, floor=20.0),
    )


def _score_shoulders(landmarks: dict[int, tuple[float, float, float, float]]) -> float:
    if min(_landmark_visibility(landmarks, 11), _landmark_visibility(landmarks, 12)) < 0.35:
        return 100.0

    left_shoulder = _xy(landmarks, 11)
    right_shoulder = _xy(landmarks, 12)
    if left_shoulder is None or right_shoulder is None:
        return 100.0
    shoulder_tilt = abs(_tilt_degrees(left_shoulder, right_shoulder))
    return _score_from_delta(shoulder_tilt, threshold=10.0, hard_limit=35.0, floor=20.0)


def _score_elbow(landmarks: dict[int, tuple[float, float, float, float]], side: str) -> float:
    if side == "left":
        points = (11, 13, 15)
    else:
        points = (12, 14, 16)
    angle = _angle_for_landmarks(landmarks, *points)
    if angle is None:
        return 100.0
    return _score_from_delta(abs(angle - 145.0), threshold=18.0, hard_limit=65.0)


def _score_knee(landmarks: dict[int, tuple[float, float, float, float]], side: str) -> float:
    if side == "left":
        points = (23, 25, 27)
    else:
        points = (24, 26, 28)
    angle = _angle_for_landmarks(landmarks, *points)
    if angle is None:
        return 100.0
    return _score_from_delta(abs(angle - 150.0), threshold=15.0, hard_limit=55.0)


def _landmark_visibility(landmarks: dict[int, tuple[float, float, float, float]], idx: int) -> float:
    landmark = landmarks.get(idx)
    if landmark is None:
        return 0.0
    return float(landmark[3])


def _xy(
    landmarks: dict[int, tuple[float, float, float, float]],
    idx: int,
    min_visibility: float = 0.2,
) -> tuple[float, float] | None:
    landmark = landmarks.get(idx)
    if landmark is None or landmark[3] < min_visibility:
        return None
    return float(landmark[0]), float(landmark[1])


def _angle_for_landmarks(
    landmarks: dict[int, tuple[float, float, float, float]],
    a_idx: int,
    b_idx: int,
    c_idx: int,
) -> float | None:
    a = _xy(landmarks, a_idx)
    b = _xy(landmarks, b_idx)
    c = _xy(landmarks, c_idx)
    if a is None or b is None or c is None:
        return None
    return _angle_between_points(a, b, c)


def _angle_between_points(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    ba = np.asarray([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
    bc = np.asarray([c[0] - b[0], c[1] - b[1]], dtype=np.float32)
    ba_norm = float(np.linalg.norm(ba))
    bc_norm = float(np.linalg.norm(bc))
    if ba_norm < 1e-6 or bc_norm < 1e-6:
        return 180.0
    cos_theta = float(np.dot(ba, bc) / (ba_norm * bc_norm))
    return math.degrees(math.acos(float(np.clip(cos_theta, -1.0, 1.0))))


def _tilt_degrees(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.degrees(math.atan2(dy, dx))


def _score_from_delta(delta: float, threshold: float, hard_limit: float, floor: float = 0.0) -> float:
    if delta <= threshold:
        return 100.0
    if delta >= hard_limit:
        return floor
    ratio = (delta - threshold) / max(hard_limit - threshold, 1e-6)
    return max(float(floor), 100.0 * (1.0 - ratio))


def _combine_scores(*scores: float) -> float:
    valid_scores = [score for score in scores if score is not None]
    if not valid_scores:
        return 100.0
    return float(sum(valid_scores) / len(valid_scores))
