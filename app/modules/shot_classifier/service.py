from __future__ import annotations

import asyncio
import os
import joblib
from functools import partial
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from loguru import logger

from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts

from .models import ShotClassifierResult

# Configuration - Mode 1 (EfficientNetB4 + GRU)
FRAME_COUNT = 30
MIN_FRAME_COUNT = 20
FRAME_SIZE = (224, 224)
FEATURE_DIM = 128

# Shot type labels - matches trained model
SHOT_CLASS_LABELS = [
    "cut", "drive", "flick", "pull", "slog", "sweep", "misc"
]

# Path resolution for trained models - self-contained in assets
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TRAINED_MODELS_DIR = ASSETS_DIR / "trained_models"
VIDEO_CLASSIFIER_DIR = TRAINED_MODELS_DIR / "video_classifier"
PROTOTYPES_PATH = TRAINED_MODELS_DIR / "prototypes" / "shot_prototypes.pkl"

# Fallback external path if needed
EXTERNAL_MODEL_PATH = (
    Path(__file__).resolve().parents[3].parent / "CricketShotClassification" / "model_weights.h5"
)

# Global caches
_model: Any | None = None
_feature_extractor: Any | None = None
_prototypes: Optional[dict[str, dict]] = None


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
        model = _load_model()
        frames = _read_clip_frames(video_path, start_frame_idx, FRAME_COUNT, FRAME_SIZE)
        
        # Normalize frames for EfficientNetB4 input
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
                f"Shot classifier output size mismatch: got {scores.shape[0]}, "
                f"expected {len(SHOT_CLASS_LABELS)}"
            )

        predicted_idx = int(np.argmax(scores))
        probabilities = {
            label: round(float(score), 6)
            for label, score in zip(SHOT_CLASS_LABELS, scores, strict=False)
        }
        
        # Extract 128-dim feature vector for mistake analysis
        features = _extract_features(frames_normalized)
        
        return ShotClassifierResult(
            predicted_shot=SHOT_CLASS_LABELS[predicted_idx],
            confidence=float(scores[predicted_idx]),
            probabilities=probabilities,
            frames_used=FRAME_COUNT,
            frame_start_index=start_frame_idx,
            frame_end_index=start_frame_idx + FRAME_COUNT - 1,
            roi_entry_frame_index=artifacts.batter_roi_entry_frame_idx,
            trigger_source=trigger_source,
            video_url=video_url,
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
    """Extract frames from video starting at specific index."""
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
    """
    Load EfficientNetB4 + GRU model.
    
    Architecture:
    - EfficientNetB4 (backbone, non-trainable)
    - TimeDistributed GlobalAveragePooling2D
    - GRU(256, return_sequences=True, dropout=0.3)
    - GRU(128, dropout=0.3)
    - Dense(1024, relu) + Dropout(0.3)
    - Dense(512, relu) + Dropout(0.2)
    - Dense(num_shots, softmax)
    """
    global _model
    if _model is not None:
        return _model
    
    model_path = _resolve_model_path()
    
    try:
        logger.info("Loading EfficientNetB4+GRU shot classifier model from {}", model_path)
        
        # Build EfficientNetB4 backbone
        base_model = EfficientNetB4(
            include_top=False,
            weights=None,
            input_shape=(FRAME_SIZE[0], FRAME_SIZE[1], 3),
        )
        base_model.trainable = False
        
        # Build full model with GRU layers
        _model = models.Sequential([
            layers.Input(shape=(FRAME_COUNT, FRAME_SIZE[0], FRAME_SIZE[1], 3)),
            layers.TimeDistributed(base_model),
            layers.TimeDistributed(layers.GlobalAveragePooling2D()),
            layers.GRU(256, return_sequences=True, dropout=0.3, unroll=True),
            layers.GRU(128, dropout=0.3, unroll=True),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(SHOT_CLASS_LABELS), activation='softmax'),
        ])
        
        # Load weights
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
    """
    Resolve model weights path with priority:
    1. Environment variable SHOT_CLASSIFIER_MODEL_PATH
    2. User's trained_models/video_classifier directory
    3. Local assets directory
    4. External fallback path
    """
    # Check environment variable first
    configured_path = os.getenv("SHOT_CLASSIFIER_MODEL_PATH")
    if configured_path:
        model_path = Path(configured_path)
        if model_path.exists():
            logger.info("Using model from environment variable: {}", model_path)
            return model_path
        logger.warning("Configured model path does not exist: {}", model_path)

    # Check video_classifier directory (priority order)
    model_paths_to_check = [
        VIDEO_CLASSIFIER_DIR / "model.weights.h5",
        VIDEO_CLASSIFIER_DIR / "best_model.weights.h5",
        VIDEO_CLASSIFIER_DIR / "model_complete.keras",
        ASSETS_DIR / "model_weights.h5",
        EXTERNAL_MODEL_PATH,
    ]
    
    for path in model_paths_to_check:
        if path.exists():
            logger.info("Found shot_classifier model at: {}", path)
            return path
    
    # Error if nothing found
    logger.error(
        "Shot classifier model not found. Checked paths: {}. "
        "Set SHOT_CLASSIFIER_MODEL_PATH environment variable.",
        model_paths_to_check
    )
    raise FeatureError(
        f"Missing shot_classifier model file. Set SHOT_CLASSIFIER_MODEL_PATH or place "
        f"model_weights.h5 at one of: {[str(p) for p in model_paths_to_check]}"
    )

def _extract_features(frames_normalized: np.ndarray) -> np.ndarray:
    """
    Extract 128-dim feature vector from GRU layer.
    Features come from the second GRU(128) layer output.
    """
    global _feature_extractor
    
    if _feature_extractor is None:
        model = _load_model()
        
        # Create feature extractor up to second GRU layer
        feature_input = keras.Input(
            shape=(FRAME_COUNT, FRAME_SIZE[0], FRAME_SIZE[1], 3)
        )
        feature_output = feature_input
        
        # Process through layers up to and including second GRU
        for layer in model.layers[:5]:
            feature_output = layer(feature_output)
        
        _feature_extractor = models.Model(feature_input, feature_output)
    
    batch = np.expand_dims(frames_normalized, axis=0)
    features = _feature_extractor.predict(batch, verbose=0)[0]
    return features.astype(np.float32)


def get_prototypes() -> dict[str, dict]:
    """
    Load shot prototypes for mistake analysis.
    Prototypes contain mean and std of 128-dim features for each shot type.
    """
    global _prototypes
    
    if _prototypes is not None:
        return _prototypes
    
    if not PROTOTYPES_PATH.exists():
        logger.warning("Prototypes not found at {}", PROTOTYPES_PATH)
        return {}
    
    try:
        _prototypes = joblib.load(PROTOTYPES_PATH)
        logger.info("Loaded {} shot prototypes for mistake analysis", len(_prototypes))
        return _prototypes
    except Exception as exc:
        logger.error("Failed to load prototypes: {}", exc)
        return {}
