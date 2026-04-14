from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import threading
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts
from app.modules.preprocessor.models import ReleasePoint

from .models import ActionLegalityMetadata, ActionLegalityResult, ActionLegalityScaler

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_PATH = ASSETS_DIR / "bowler_model.h5"
META_PATH = ASSETS_DIR / "meta.json"
SCALER_PATH = ASSETS_DIR / "scaler.json"

_model: Any | None = None
_metadata: ActionLegalityMetadata | None = None
_scaler: ActionLegalityScaler | None = None
_pose_landmarker: Any | None = None
_pose_landmarker_lock = threading.Lock()
_pose_landmarker_initialized = False
_pose_landmarker_cleanup_registered = False


class ActionLegalityService:
    async def run(
        self,
        artifacts: VideoArtifacts,
        video_url: str | None = None,
    ) -> ActionLegalityResult:
        release_frame, used_annotated = _resolve_release_frame(artifacts)
        release_point = artifacts.release_point
        release_frame_idx = release_point.frame_idx if release_point is not None else None

        logger.info(
            "Starting action_legality analysis release_frame_idx={} used_annotated_frame={}",
            release_frame_idx,
            used_annotated,
        )
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                partial(
                    self._run_sync,
                    release_frame,
                    release_point,
                    video_url,
                    used_annotated,
                ),
            )
        except FeatureError:
            raise
        except Exception as exc:
            logger.exception("Unexpected action_legality failure")
            raise FeatureError(
                f"Bowling action legality analysis failed unexpectedly: {exc}"
            ) from exc

        logger.info(
            "Completed action_legality analysis verdict={} illegal_probability={:.3f}",
            result.verdict,
            result.illegal_probability,
        )
        return result

    def _run_sync(
        self,
        release_frame: np.ndarray,
        release_point: ReleasePoint | None,
        video_url: str | None,
        used_annotated: bool,
    ) -> ActionLegalityResult:
        metadata = _load_metadata()
        scaler = _load_scaler()
        model = _load_model()

        keypoints = _extract_keypoints_from_frame(release_frame, metadata.select_landmarks)
        if keypoints is None:
            raise FeatureError("Pose landmarks were not detected in the release frame.")

        normalized_keypoints = _normalize_keypoints_by_torso(keypoints, metadata.select_landmarks)
        standardized_keypoints = _standardize_keypoints(normalized_keypoints, scaler)
        if standardized_keypoints.shape[0] != metadata.feature_dim:
            raise FeatureError(
                "Action legality feature vector size does not match the trained model."
            )

        try:
            prediction = model.predict(standardized_keypoints.reshape(1, -1), verbose=0)
        except Exception as exc:
            raise FeatureError("TensorFlow inference failed for action_legality.") from exc

        illegal_probability = float(np.asarray(prediction, dtype=np.float32).reshape(-1)[0])
        illegal_probability = float(np.clip(illegal_probability, 0.0, 1.0))
        legal_probability = float(1.0 - illegal_probability)
        verdict = "illegal" if illegal_probability >= 0.5 else "legal"
        confidence = illegal_probability if verdict == "illegal" else legal_probability

        return ActionLegalityResult(
            verdict=verdict,
            illegal_probability=illegal_probability,
            legal_probability=legal_probability,
            confidence=confidence,
            release_frame_index=(
                release_point.frame_idx if release_point is not None else None
            ),
            release_timestamp_s=(
                release_point.timestamp_s if release_point is not None else None
            ),
            release_confidence=(
                release_point.confidence if release_point is not None else None
            ),
            selected_landmarks=metadata.select_landmarks,
            normalized_keypoints=[float(value) for value in normalized_keypoints.tolist()],
            video_url=video_url,
            used_annotated_release_frame=used_annotated,
        )


def _resolve_release_frame(artifacts: VideoArtifacts) -> tuple[np.ndarray, bool]:
    release_point = artifacts.release_point
    if release_point is not None and release_point.raw_frame is not None:
        return release_point.raw_frame, False
    if artifacts.release_frame.size == 0:
        raise FeatureError("Release frame is unavailable for action legality analysis.")
    return artifacts.release_frame, True


def _load_metadata() -> ActionLegalityMetadata:
    global _metadata
    if _metadata is None:
        try:
            _metadata = ActionLegalityMetadata.model_validate_json(META_PATH.read_text())
        except FileNotFoundError as exc:
            raise FeatureError(f"Missing action_legality metadata file: {META_PATH}") from exc
        except json.JSONDecodeError as exc:
            raise FeatureError("Action legality metadata JSON is invalid.") from exc
    return _metadata


def _load_scaler() -> ActionLegalityScaler:
    global _scaler
    if _scaler is None:
        try:
            _scaler = ActionLegalityScaler.model_validate_json(SCALER_PATH.read_text())
        except FileNotFoundError as exc:
            raise FeatureError(f"Missing action_legality scaler file: {SCALER_PATH}") from exc
        except json.JSONDecodeError as exc:
            raise FeatureError("Action legality scaler JSON is invalid.") from exc
    return _scaler


def _load_model() -> Any:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FeatureError(f"Missing action_legality model file: {MODEL_PATH}")
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise FeatureError("tensorflow is required for the action_legality module.") from exc

        logger.info("Loading action_legality TensorFlow model from {}", MODEL_PATH)
        try:
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as exc:
            raise FeatureError("Failed to load the action_legality TensorFlow model.") from exc
    return _model


def _extract_keypoints_from_frame(
    frame_bgr: np.ndarray,
    selected_landmarks: list[int],
) -> np.ndarray | None:
    if frame_bgr.size == 0:
        return None
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise FeatureError("mediapipe is required for the action_legality module.") from exc

    pose_cls = _resolve_mediapipe_pose(mp)
    if pose_cls is not None:
        with pose_cls(static_image_mode=True, model_complexity=1) as pose:
            results = pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None

        features: list[float] = []
        for landmark_idx in selected_landmarks:
            landmark = results.pose_landmarks.landmark[landmark_idx]
            features.extend([landmark.x, landmark.y, landmark.z])
        return np.asarray(features, dtype=np.float32)

    return _extract_keypoints_with_tasks(mp, frame_bgr, selected_landmarks)


def _resolve_mediapipe_pose(mp: Any):
    if hasattr(mp, "solutions"):
        pose = getattr(mp.solutions, "pose", None)
        if pose is not None and hasattr(pose, "Pose"):
            return pose.Pose
    try:
        from mediapipe.python.solutions import pose as mp_pose
    except Exception:  # pragma: no cover - depends on local mediapipe build
        return None
    return mp_pose.Pose


def _extract_keypoints_with_tasks(
    mp: Any,
    frame_bgr: np.ndarray,
    selected_landmarks: list[int],
) -> np.ndarray | None:
    landmarker = _get_pose_landmarker()
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
    )
    result = landmarker.detect(image)
    if not result.pose_landmarks:
        return None

    pose_landmarks = result.pose_landmarks[0]
    features: list[float] = []
    for landmark_idx in selected_landmarks:
        landmark = pose_landmarks[landmark_idx]
        features.extend([landmark.x, landmark.y, landmark.z])
    return np.asarray(features, dtype=np.float32)


def _get_pose_landmarker():
    global _pose_landmarker
    global _pose_landmarker_initialized
    if _pose_landmarker is not None:
        return _pose_landmarker
    if _pose_landmarker_initialized:
        raise FeatureError("PoseLandmarker initialization failed earlier.")

    model_path = os.getenv("MEDIAPIPE_POSE_TASK_PATH")
    if not model_path:
        model_path = str(ASSETS_DIR / "pose_landmarker.task")
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
    except Exception as exc:  # pragma: no cover - depends on local mediapipe build
        raise FeatureError(
            "mediapipe Tasks API is unavailable. Install the full mediapipe package."
        ) from exc

    with _pose_landmarker_lock:
        if _pose_landmarker is not None:
            return _pose_landmarker
        if _pose_landmarker_initialized:
            raise FeatureError("PoseLandmarker initialization failed earlier.")

        try:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(model_file),
                    delegate=BaseOptions.Delegate.GPU,
                ),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
            )
            _pose_landmarker = PoseLandmarker.create_from_options(options)
        except Exception as exc:
            logger.warning(
                "PoseLandmarker GPU delegate failed ({}). Falling back to CPU.",
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

        _pose_landmarker_initialized = True
        _register_landmarker_cleanup()
        return _pose_landmarker


def _register_landmarker_cleanup() -> None:
    global _pose_landmarker_cleanup_registered
    if _pose_landmarker_cleanup_registered:
        return
    if threading.current_thread() is not threading.main_thread():
        return

    def _cleanup_landmarker(*_args) -> None:
        global _pose_landmarker
        if _pose_landmarker is None:
            return
        try:
            _pose_landmarker.close()
        except Exception as exc:
            logger.debug("PoseLandmarker close failed: {}", exc)
        finally:
            _pose_landmarker = None

    signal.signal(signal.SIGTERM, _cleanup_landmarker)
    signal.signal(signal.SIGINT, _cleanup_landmarker)
    _pose_landmarker_cleanup_registered = True


def _normalize_keypoints_by_torso(
    features: np.ndarray,
    selected_landmarks: list[int],
) -> np.ndarray:
    landmark_offsets = {
        landmark_idx: position * 3 for position, landmark_idx in enumerate(selected_landmarks)
    }
    left_shoulder_offset = landmark_offsets.get(11)
    right_shoulder_offset = landmark_offsets.get(12)
    if left_shoulder_offset is None or right_shoulder_offset is None:
        return features

    left_shoulder_x = float(features[left_shoulder_offset])
    left_shoulder_y = float(features[left_shoulder_offset + 1])
    right_shoulder_x = float(features[right_shoulder_offset])
    right_shoulder_y = float(features[right_shoulder_offset + 1])
    shoulder_distance = math.hypot(
        right_shoulder_x - left_shoulder_x,
        right_shoulder_y - left_shoulder_y,
    )
    if shoulder_distance < 1e-6:
        shoulder_distance = 1e-6

    normalized = features.astype(np.float32, copy=True)
    for idx in range(0, len(normalized), 3):
        normalized[idx] = (normalized[idx] - left_shoulder_x) / shoulder_distance
        normalized[idx + 1] = (normalized[idx + 1] - left_shoulder_y) / shoulder_distance
        normalized[idx + 2] = normalized[idx + 2] / shoulder_distance
    return normalized


def _standardize_keypoints(
    keypoints: np.ndarray,
    scaler: ActionLegalityScaler,
) -> np.ndarray:
    mean = np.asarray(scaler.mean, dtype=np.float32)
    scale = np.asarray(scaler.scale, dtype=np.float32)
    if keypoints.shape[0] != mean.shape[0] or keypoints.shape[0] != scale.shape[0]:
        raise FeatureError("Action legality scaler shape does not match the pose feature vector.")

    scale = np.where(scale < 1e-6, 1.0, scale)
    return (keypoints.astype(np.float32) - mean) / scale
