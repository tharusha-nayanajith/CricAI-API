from __future__ import annotations

import asyncio
import os
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts

from .models import ShotClassifierResult

FRAME_COUNT = 30
FRAME_SIZE = (224, 224)
SHOT_CLASS_LABELS = [
    "cover",
    "defense",
    "flick",
    "hook",
    "late_cut",
    "lofted",
    "pull",
    "square_cut",
    "straight",
    "sweep",
]
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_PATH = ASSETS_DIR / "model_weights.h5"
EXTERNAL_MODEL_PATH = (
    Path(__file__).resolve().parents[3].parent / "CricketShotClassification" / "model_weights.h5"
)

_model: Any | None = None


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
        batch = np.expand_dims(frames, axis=0)
        try:
            predictions = model.predict(batch, verbose=0)
        except Exception as exc:
            raise FeatureError("TensorFlow inference failed for shot_classifier.") from exc

        scores = np.asarray(predictions, dtype=np.float32).reshape(-1)
        if scores.shape[0] != len(SHOT_CLASS_LABELS):
            raise FeatureError("Shot classifier output size does not match configured labels.")

        predicted_idx = int(np.argmax(scores))
        probabilities = {
            label: round(float(score), 6)
            for label, score in zip(SHOT_CLASS_LABELS, scores, strict=False)
        }
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
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise FeatureError(f"Unable to open video file for shot classification: {video_path}")

    frames: list[np.ndarray] = []
    try:
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
    if _model is None:
        model_path = _resolve_model_path()
        try:
            from tensorflow.keras import layers, models
            from tensorflow.keras.applications import EfficientNetB0
        except ImportError as exc:
            raise FeatureError("tensorflow is required for the shot_classifier module.") from exc

        logger.info("Loading shot_classifier TensorFlow model from {}", model_path)
        try:
            base_model = EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=(FRAME_SIZE[0], FRAME_SIZE[1], 3),
            )
            base_model.trainable = False
            model = models.Sequential(
                [
                    layers.TimeDistributed(
                        base_model,
                        input_shape=(None, FRAME_SIZE[0], FRAME_SIZE[1], 3),
                    ),
                    layers.TimeDistributed(layers.GlobalAveragePooling2D()),
                    layers.GRU(256, return_sequences=True),
                    layers.GRU(128),
                    layers.Dense(1024, activation="relu"),
                    layers.Dropout(0.5),
                    layers.Dense(len(SHOT_CLASS_LABELS), activation="softmax"),
                ]
            )
            model.load_weights(model_path)
        except FeatureError:
            raise
        except Exception as exc:
            raise FeatureError("Failed to load the shot_classifier TensorFlow model.") from exc
        _model = model
    return _model


def _resolve_model_path() -> Path:
    configured_path = os.getenv("SHOT_CLASSIFIER_MODEL_PATH")
    if configured_path:
        model_path = Path(configured_path)
        if model_path.exists():
            return model_path
        raise FeatureError(f"Configured shot_classifier model file was not found: {model_path}")

    if MODEL_PATH.exists():
        return MODEL_PATH
    if EXTERNAL_MODEL_PATH.exists():
        return EXTERNAL_MODEL_PATH
    raise FeatureError(
        "Missing shot_classifier model file. Set SHOT_CLASSIFIER_MODEL_PATH or place "
        f"model_weights.h5 at {MODEL_PATH} or {EXTERNAL_MODEL_PATH}."
    )
