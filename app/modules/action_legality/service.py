from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import threading
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from app.ai.google import get_google_genai_client, get_google_genai_status
from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts
from app.modules.preprocessor.models import ReleasePoint
from app.storage.artifacts import write_image_artifact

from .models import (
    ActionLegalityMetadata,
    ActionLegalityResult,
    ActionLegalityScaler,
    JointAnalysisEntry,
    PoseLandmark2D,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_PATH = ASSETS_DIR / "bowler_model.h5"
META_PATH = ASSETS_DIR / "meta.json"
SCALER_PATH = ASSETS_DIR / "scaler.json"
ACTION_LEGALITY_FEATURE_NAME = "action_legality"
POSE_CONNECTION_CANDIDATES = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
]
LANDMARK_NAMES = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
}

_model: Any | None = None
_metadata: ActionLegalityMetadata | None = None
_scaler: ActionLegalityScaler | None = None
_pose_landmarker: Any | None = None
_pose_landmarker_lock = threading.Lock()
_pose_landmarker_initialized = False
_pose_landmarker_cleanup_registered = False


@dataclass(slots=True)
class _PoseVisualizationPoint:
    id: int
    x: float
    y: float
    z: float
    visibility: float


class ActionLegalityService:
    async def run(
        self,
        artifacts: VideoArtifacts,
        video_url: str | None = None,
        job_id: str | None = None,
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
                    job_id,
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
        job_id: str | None,
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

        release_frame_height, release_frame_width = release_frame.shape[:2]
        pose_landmarks_2d = _extract_pose_landmarks_2d(release_frame, metadata.select_landmarks)
        pose_connections = _build_pose_connections(pose_landmarks_2d)
        joint_analysis = _build_joint_analysis(pose_landmarks_2d)
        summary_explanation = _build_summary_explanation(verdict, confidence, joint_analysis)
        coaching_feedback = _build_coaching_feedback(verdict, joint_analysis)
        joint_analysis, summary_explanation, coaching_feedback = _enhance_legality_feedback_with_ai(
            verdict=verdict,
            confidence=confidence,
            illegal_probability=illegal_probability,
            legal_probability=legal_probability,
            joint_analysis=joint_analysis,
            used_annotated_release_frame=used_annotated,
        )

        release_frame_image_url = None
        overlay_image_url = None
        if job_id is not None:
            release_frame_image_url = write_image_artifact(
                job_id,
                ACTION_LEGALITY_FEATURE_NAME,
                "release_frame.jpg",
                release_frame,
            )
            overlay_image = _render_pose_overlay(
                release_frame,
                pose_landmarks_2d,
                pose_connections,
                joint_analysis,
                verdict,
            )
            if overlay_image is not None:
                overlay_image_url = write_image_artifact(
                    job_id,
                    ACTION_LEGALITY_FEATURE_NAME,
                    "overlay.jpg",
                    overlay_image,
                )

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
            release_frame_image_url=release_frame_image_url,
            release_frame_width=release_frame_width,
            release_frame_height=release_frame_height,
            pose_landmarks_2d=[
                PoseLandmark2D(
                    id=landmark.id,
                    name=LANDMARK_NAMES.get(landmark.id, f"landmark_{landmark.id}"),
                    x=landmark.x,
                    y=landmark.y,
                    visibility=landmark.visibility,
                )
                for landmark in pose_landmarks_2d
            ],
            pose_connections=[list(connection) for connection in pose_connections],
            overlay_image_url=overlay_image_url,
            joint_analysis=joint_analysis,
            summary_explanation=summary_explanation,
            coaching_feedback=coaching_feedback,
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


def _extract_pose_landmarks_2d(
    frame_bgr: np.ndarray,
    selected_landmarks: list[int],
) -> list[_PoseVisualizationPoint]:
    try:
        import mediapipe as mp
    except ImportError:
        logger.warning("mediapipe missing while building action_legality visualization")
        return []

    pose_cls = _resolve_mediapipe_pose(mp)
    height, width = frame_bgr.shape[:2]
    if pose_cls is not None:
        with pose_cls(static_image_mode=True, model_complexity=1) as pose:
            results = pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return []
        return [
            _PoseVisualizationPoint(
                id=landmark_idx,
                x=float(results.pose_landmarks.landmark[landmark_idx].x * width),
                y=float(results.pose_landmarks.landmark[landmark_idx].y * height),
                z=float(results.pose_landmarks.landmark[landmark_idx].z),
                visibility=float(getattr(results.pose_landmarks.landmark[landmark_idx], "visibility", 1.0)),
            )
            for landmark_idx in selected_landmarks
        ]

    try:
        return _extract_pose_landmarks_with_tasks(mp, frame_bgr, selected_landmarks)
    except Exception as exc:
        logger.warning("action_legality pose visualization extraction failed: {}", exc)
        return []


def _resolve_mediapipe_pose(mp: Any):
    if hasattr(mp, "solutions"):
        pose = getattr(mp.solutions, "pose", None)
        if pose is not None and hasattr(pose, "Pose"):
            return pose.Pose
    try:
        from mediapipe.python.solutions import pose as mp_pose
    except Exception:
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


def _extract_pose_landmarks_with_tasks(
    mp: Any,
    frame_bgr: np.ndarray,
    selected_landmarks: list[int],
) -> list[_PoseVisualizationPoint]:
    landmarker = _get_pose_landmarker()
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
    )
    result = landmarker.detect(image)
    if not result.pose_landmarks:
        return []

    height, width = frame_bgr.shape[:2]
    pose_landmarks = result.pose_landmarks[0]
    visualization = []
    for landmark_idx in selected_landmarks:
        landmark = pose_landmarks[landmark_idx]
        visualization.append(
            _PoseVisualizationPoint(
                id=landmark_idx,
                x=float(landmark.x * width),
                y=float(landmark.y * height),
                z=float(landmark.z),
                visibility=float(getattr(landmark, "visibility", 1.0)),
            )
        )
    return visualization


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
            logger.info("Initializing PoseLandmarker with CPU delegate")
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


def _build_pose_connections(
    pose_landmarks: list[_PoseVisualizationPoint],
) -> list[tuple[int, int]]:
    available = {landmark.id for landmark in pose_landmarks}
    return [
        connection
        for connection in POSE_CONNECTION_CANDIDATES
        if connection[0] in available and connection[1] in available
    ]


def _build_joint_analysis(
    pose_landmarks: list[_PoseVisualizationPoint],
) -> list[JointAnalysisEntry]:
    landmark_map = {landmark.id: landmark for landmark in pose_landmarks}
    analysis: list[JointAnalysisEntry] = []

    for joint_id, label, points, threshold in [
        ("left_elbow", "Left Elbow", (11, 13, 15), 165.0),
        ("right_elbow", "Right Elbow", (12, 14, 16), 165.0),
    ]:
        shoulder, elbow, wrist = points
        if shoulder in landmark_map and elbow in landmark_map and wrist in landmark_map:
            measured = _angle_between_points(
                landmark_map[shoulder],
                landmark_map[elbow],
                landmark_map[wrist],
            )
            analysis.append(
                _joint_entry_for_upper_threshold(
                    joint_id=joint_id,
                    label=label,
                    measured_value=measured,
                    threshold_value=threshold,
                    explanation_template="{label} extension at release is close to the legality threshold.",
                    recommendation="Review elbow extension at release with a coach.",
                )
            )

    if 11 in landmark_map and 12 in landmark_map:
        shoulder_tilt = _horizontal_tilt_degrees(landmark_map[11], landmark_map[12])
        analysis.append(
            _joint_entry_for_upper_threshold(
                joint_id="shoulder_alignment",
                label="Shoulder Alignment",
                measured_value=shoulder_tilt,
                threshold_value=12.0,
                explanation_template="Shoulder tilt at release suggests lateral imbalance.",
                recommendation="Keep the shoulders more level through release.",
            )
        )

    if 23 in landmark_map and 24 in landmark_map:
        hip_tilt = _horizontal_tilt_degrees(landmark_map[23], landmark_map[24])
        analysis.append(
            _joint_entry_for_upper_threshold(
                joint_id="hip_alignment",
                label="Hip Alignment",
                measured_value=hip_tilt,
                threshold_value=10.0,
                explanation_template="Hip tilt at release suggests the base is not fully balanced.",
                recommendation="Stabilize the hips before release to keep the action repeatable.",
            )
        )

    analysis.sort(key=lambda item: (item.status != "critical", item.status != "warning", item.score))
    return analysis


def _joint_entry_for_upper_threshold(
    *,
    joint_id: str,
    label: str,
    measured_value: float,
    threshold_value: float,
    explanation_template: str,
    recommendation: str,
) -> JointAnalysisEntry:
    excess = max(0.0, measured_value - threshold_value)
    if excess >= 8.0:
        status = "critical"
    elif excess > 0.0:
        status = "warning"
    else:
        status = "ok"
    score = float(np.clip(1.0 - (excess / max(threshold_value * 0.25, 1.0)), 0.0, 1.0))
    explanation = (
        explanation_template.format(label=label)
        if status != "ok"
        else f"{label} is within the expected release range."
    )
    return JointAnalysisEntry(
        joint_id=joint_id,
        label=label,
        status=status,
        score=score,
        measured_value=float(measured_value),
        threshold_value=float(threshold_value),
        explanation=explanation,
        recommendation=(recommendation if status != "ok" else f"Maintain the current {label.lower()} mechanics."),
    )


def _build_summary_explanation(
    verdict: str,
    confidence: float,
    joint_analysis: list[JointAnalysisEntry],
) -> str:
    flagged = [entry for entry in joint_analysis if entry.status != "ok"]
    if verdict == "illegal":
        if flagged:
            return (
                f"Release frame appears illegal with {confidence:.0%} confidence. "
                f"Primary concern: {flagged[0].label}."
            )
        return f"Release frame appears illegal with {confidence:.0%} confidence."
    if flagged:
        return (
            f"Release frame appears legal with {confidence:.0%} confidence, "
            f"but {flagged[0].label.lower()} should be monitored."
        )
    return f"Release frame appears legal with {confidence:.0%} confidence."


def _build_coaching_feedback(
    verdict: str,
    joint_analysis: list[JointAnalysisEntry],
) -> str:
    flagged = [entry for entry in joint_analysis if entry.status != "ok"]
    if verdict == "illegal":
        if flagged:
            return f"Action is flagged illegal. Focus first on {flagged[0].label.lower()} at release."
        return "Action is flagged illegal. Review the release mechanics with a coach."
    if flagged:
        return f"Action is currently legal. Maintain it, but tighten {flagged[0].label.lower()} at release."
    return "Action is legal. Maintain current release mechanics."



def _enhance_legality_feedback_with_ai(
    *,
    verdict: str,
    confidence: float,
    illegal_probability: float,
    legal_probability: float,
    joint_analysis: list[JointAnalysisEntry],
    used_annotated_release_frame: bool,
) -> tuple[list[JointAnalysisEntry], str, str]:
    base_summary = _build_summary_explanation(verdict, confidence, joint_analysis)
    base_coaching = _build_coaching_feedback(verdict, joint_analysis)
    status = get_google_genai_status()
    if not status.enabled:
        return joint_analysis, base_summary, base_coaching

    client = get_google_genai_client()
    if client is None:
        return joint_analysis, base_summary, base_coaching

    prompt = _build_legality_ai_prompt(
        verdict=verdict,
        confidence=confidence,
        illegal_probability=illegal_probability,
        legal_probability=legal_probability,
        joint_analysis=joint_analysis,
        used_annotated_release_frame=used_annotated_release_frame,
    )
    try:
        response = client.models.generate_content(
            model=status.model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={
                "max_output_tokens": 900,
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "summary_explanation": {"type": "string"},
                        "coaching_feedback": {"type": "string"},
                        "joint_analysis": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "joint_id": {"type": "string"},
                                    "explanation": {"type": "string"},
                                    "recommendation": {"type": "string"},
                                },
                                "required": ["joint_id", "explanation", "recommendation"],
                            },
                        },
                    },
                    "required": ["summary_explanation", "coaching_feedback", "joint_analysis"],
                },
            },
        )
        payload = json.loads(response.text)
    except Exception as exc:
        logger.warning("action_legality Vertex AI feedback failed: {}", exc)
        return joint_analysis, base_summary, base_coaching

    ai_summary = payload.get("summary_explanation")
    ai_coaching = payload.get("coaching_feedback")
    ai_joint_updates = payload.get("joint_analysis")
    merged_joint_analysis = _merge_ai_joint_analysis(joint_analysis, ai_joint_updates)
    return (
        merged_joint_analysis,
        ai_summary.strip() if isinstance(ai_summary, str) and ai_summary.strip() else base_summary,
        ai_coaching.strip() if isinstance(ai_coaching, str) and ai_coaching.strip() else base_coaching,
    )


def _build_legality_ai_prompt(
    *,
    verdict: str,
    confidence: float,
    illegal_probability: float,
    legal_probability: float,
    joint_analysis: list[JointAnalysisEntry],
    used_annotated_release_frame: bool,
) -> str:
    joint_payload = [
        {
            "joint_id": entry.joint_id,
            "label": entry.label,
            "status": entry.status,
            "score": round(entry.score, 4),
            "measured_value": round(entry.measured_value, 2),
            "threshold_value": round(entry.threshold_value, 2),
            "explanation": entry.explanation,
            "recommendation": entry.recommendation,
        }
        for entry in joint_analysis
    ]
    return f"""You are a cricket bowling coach writing UI-safe legality feedback.

Classifier output:
- verdict: {verdict}
- confidence: {confidence:.3f}
- illegal_probability: {illegal_probability:.3f}
- legal_probability: {legal_probability:.3f}
- used_annotated_release_frame: {str(used_annotated_release_frame).lower()}

Joint analysis:
{json.dumps(joint_payload, indent=2)}

Return valid JSON only.
Rules:
- Keep the legality verdict aligned with the classifier output. Do not change it.
- summary_explanation must be one short sentence.
- coaching_feedback must be one short coaching-oriented sentence.
- For each input joint, return one item with the same joint_id.
- explanation should be semantic and easy for an athlete to understand.
- recommendation should be specific and brief.
- Do not invent new joint_ids or numeric measurements.
"""


def _merge_ai_joint_analysis(
    joint_analysis: list[JointAnalysisEntry],
    ai_joint_updates: Any,
) -> list[JointAnalysisEntry]:
    if not isinstance(ai_joint_updates, list):
        return joint_analysis
    updates: dict[str, dict[str, str]] = {}
    for item in ai_joint_updates:
        if not isinstance(item, dict):
            continue
        joint_id = item.get("joint_id")
        explanation = item.get("explanation")
        recommendation = item.get("recommendation")
        if not isinstance(joint_id, str):
            continue
        updates[joint_id] = {
            "explanation": explanation if isinstance(explanation, str) else "",
            "recommendation": recommendation if isinstance(recommendation, str) else "",
        }

    merged: list[JointAnalysisEntry] = []
    for entry in joint_analysis:
        update = updates.get(entry.joint_id)
        if not update:
            merged.append(entry)
            continue
        merged.append(
            entry.model_copy(
                update={
                    "explanation": update["explanation"].strip() or entry.explanation,
                    "recommendation": update["recommendation"].strip() or entry.recommendation,
                }
            )
        )
    return merged

def _render_pose_overlay(
    frame_bgr: np.ndarray,
    pose_landmarks: list[_PoseVisualizationPoint],
    pose_connections: list[tuple[int, int]],
    joint_analysis: list[JointAnalysisEntry],
    verdict: str,
) -> np.ndarray | None:
    if frame_bgr.size == 0:
        return None
    overlay = frame_bgr.copy()
    landmark_map = {landmark.id: landmark for landmark in pose_landmarks}
    for start_id, end_id in pose_connections:
        start = landmark_map.get(start_id)
        end = landmark_map.get(end_id)
        if start is None or end is None:
            continue
        cv2.line(
            overlay,
            (int(round(start.x)), int(round(start.y))),
            (int(round(end.x)), int(round(end.y))),
            (0, 220, 255),
            2,
            lineType=cv2.LINE_AA,
        )
    for landmark in pose_landmarks:
        color = (0, 255, 0) if landmark.visibility >= 0.75 else (0, 200, 255)
        cv2.circle(
            overlay,
            (int(round(landmark.x)), int(round(landmark.y))),
            5,
            color,
            -1,
            lineType=cv2.LINE_AA,
        )
    return overlay


def _angle_between_points(
    point_a: _PoseVisualizationPoint,
    point_b: _PoseVisualizationPoint,
    point_c: _PoseVisualizationPoint,
) -> float:
    vector_ba = np.asarray([point_a.x - point_b.x, point_a.y - point_b.y], dtype=np.float32)
    vector_bc = np.asarray([point_c.x - point_b.x, point_c.y - point_b.y], dtype=np.float32)
    norm_product = float(np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc))
    if norm_product < 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(vector_ba, vector_bc) / norm_product, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _horizontal_tilt_degrees(
    point_a: _PoseVisualizationPoint,
    point_b: _PoseVisualizationPoint,
) -> float:
    radians = math.atan2(point_b.y - point_a.y, point_b.x - point_a.x)
    degrees = abs(math.degrees(radians))
    return float(min(degrees, abs(180.0 - degrees)))
