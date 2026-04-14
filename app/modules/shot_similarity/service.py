from __future__ import annotations

import asyncio
import json
import math
from functools import partial
from pathlib import Path

import numpy as np
from loguru import logger

from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts
from app.modules.action_legality.service import _extract_keypoints_from_frame

from .models import PoseKeypoint, ShotReference, ShotSimilarityResult

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
REFERENCE_LIBRARY_PATH = ASSETS_DIR / "golden_frames.json"

_reference_library: dict[str, dict[str, ShotReference]] | None = None


class ShotSimilarityService:
    async def run(
        self,
        artifacts: VideoArtifacts,
        video_url: str | None = None,
    ) -> ShotSimilarityResult:
        frame = _resolve_comparison_frame(artifacts)
        logger.info("Starting shot_similarity analysis video_url={}", video_url)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                partial(self._run_sync, frame, video_url),
            )
        except FeatureError:
            raise
        except Exception as exc:
            logger.exception("Unexpected shot_similarity failure")
            raise FeatureError(f"Shot similarity analysis failed unexpectedly: {exc}") from exc

        logger.info(
            "Completed shot_similarity analysis matched_player={} shot_type={} similarity={:.2f}",
            result.matched_player,
            result.shot_type,
            result.similarity_percentage,
        )
        return result

    def _run_sync(
        self,
        frame_bgr: np.ndarray,
        video_url: str | None,
    ) -> ShotSimilarityResult:
        reference_library = _load_reference_library()
        keypoints_array = _extract_keypoints_from_frame(frame_bgr, list(range(33)))
        if keypoints_array is None:
            raise FeatureError("Pose landmarks were not detected in the shot comparison frame.")

        user_keypoints = _array_to_keypoints(keypoints_array)
        match = _find_best_match(user_keypoints, reference_library)
        if match is None:
            raise FeatureError("No reference shots are available for similarity comparison.")

        avg_visibility = float(
            np.mean([keypoint.visibility for keypoint in user_keypoints]) * 100.0
        )
        return ShotSimilarityResult(
            similarity_percentage=round(match["similarity"], 2),
            matched_player=match["player"],
            shot_type=match["shot"],
            keypoints_detected=len(user_keypoints),
            confidence=round(avg_visibility, 2),
            feedback=match["feedback"][:5],
            compared_frame="bat_contact_frame",
            video_url=video_url,
        )


def _resolve_comparison_frame(artifacts: VideoArtifacts) -> np.ndarray:
    if artifacts.bat_contact_frame is None:
        raise FeatureError(
            "Shot similarity requires a batter contact frame. "
            "Provide a batter shot video with detectable bat-ball contact."
        )
    return artifacts.bat_contact_frame


def _load_reference_library() -> dict[str, dict[str, ShotReference]]:
    global _reference_library
    if _reference_library is None:
        if not REFERENCE_LIBRARY_PATH.exists():
            raise FeatureError(
                "Shot similarity reference library is missing. "
                f"Add golden shot keypoints at {REFERENCE_LIBRARY_PATH}."
            )
        try:
            raw = json.loads(REFERENCE_LIBRARY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FeatureError("Shot similarity reference library JSON is invalid.") from exc

        parsed: dict[str, dict[str, ShotReference]] = {}
        for player_name, shots in raw.items():
            parsed[player_name] = {}
            for shot_type, shot_payload in shots.items():
                parsed[player_name][shot_type] = ShotReference.model_validate(shot_payload)
        _reference_library = parsed
    return _reference_library


def _array_to_keypoints(keypoints_array: np.ndarray) -> list[PoseKeypoint]:
    keypoints: list[PoseKeypoint] = []
    for idx in range(0, len(keypoints_array), 3):
        keypoints.append(
            PoseKeypoint(
                x=float(keypoints_array[idx]),
                y=float(keypoints_array[idx + 1]),
                z=float(keypoints_array[idx + 2]),
                visibility=1.0,
            )
        )
    return keypoints


def _normalize_keypoints(keypoints: list[PoseKeypoint]) -> np.ndarray:
    kp_array = np.asarray([[kp.x, kp.y, kp.z] for kp in keypoints], dtype=np.float32)
    hip_center = (kp_array[23] + kp_array[24]) / 2.0
    kp_normalized = kp_array - hip_center
    scale = float(np.max(np.abs(kp_normalized)))
    if scale > 0:
        kp_normalized = kp_normalized / scale
    return kp_normalized


def _calculate_angle(p1: PoseKeypoint, p2: PoseKeypoint, p3: PoseKeypoint) -> float:
    v1 = np.asarray([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z], dtype=np.float32)
    v2 = np.asarray([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z], dtype=np.float32)
    v1_norm = float(np.linalg.norm(v1))
    v2_norm = float(np.linalg.norm(v2))
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0
    cos_theta = float(np.dot(v1, v2) / (v1_norm * v2_norm))
    angle = math.acos(float(np.clip(cos_theta, -1.0, 1.0)))
    return math.degrees(angle)


def _calculate_similarity(
    user_keypoints: list[PoseKeypoint],
    golden_keypoints: list[PoseKeypoint],
) -> dict[str, float | list[str]]:
    user_norm = _normalize_keypoints(user_keypoints)
    golden_norm = _normalize_keypoints(golden_keypoints)

    joint_weights = {
        11: 2.0,
        12: 2.0,
        13: 2.5,
        14: 2.5,
        15: 3.0,
        16: 3.0,
        23: 2.0,
        24: 2.0,
        25: 1.5,
        26: 1.5,
    }
    total_similarity = 0.0
    total_weight = 0.0
    feedback: list[str] = []

    for idx in range(len(user_norm)):
        weight = joint_weights.get(idx, 1.0)
        distance = float(np.linalg.norm(user_norm[idx] - golden_norm[idx]))
        similarity = max(0.0, 1.0 - distance)
        total_similarity += similarity * weight
        total_weight += weight

    key_angles = {
        "left_elbow": (11, 13, 15),
        "right_elbow": (12, 14, 16),
        "left_shoulder": (13, 11, 23),
        "right_shoulder": (14, 12, 24),
        "left_hip": (11, 23, 25),
        "right_hip": (12, 24, 26),
        "left_knee": (23, 25, 27),
        "right_knee": (24, 26, 28),
    }
    angle_feedback_messages = {
        "left_elbow": "Bend your left elbow more.",
        "right_elbow": "Keep your right arm straighter.",
        "left_shoulder": "Open up your left shoulder.",
        "right_shoulder": "Rotate your right shoulder more.",
        "left_hip": "Engage your left hip.",
        "right_hip": "Drive through with your right hip.",
        "left_knee": "Bend your left knee more for stability.",
        "right_knee": "Ensure your right knee is stable.",
    }

    for angle_name, (p1_idx, p2_idx, p3_idx) in key_angles.items():
        user_angle = _calculate_angle(
            user_keypoints[p1_idx],
            user_keypoints[p2_idx],
            user_keypoints[p3_idx],
        )
        golden_angle = _calculate_angle(
            golden_keypoints[p1_idx],
            golden_keypoints[p2_idx],
            golden_keypoints[p3_idx],
        )
        angle_diff = abs(user_angle - golden_angle)
        angle_similarity = max(0.0, 1.0 - (angle_diff / 180.0))
        total_similarity += angle_similarity * 1.5
        total_weight += 1.5
        if angle_diff > 20.0:
            feedback.append(
                angle_feedback_messages.get(
                    angle_name,
                    f"Check your {angle_name.replace('_', ' ')} angle.",
                )
            )

    overall_similarity = (total_similarity / total_weight) * 100.0 if total_weight else 0.0
    return {
        "similarity": overall_similarity,
        "feedback": feedback,
    }


def _find_best_match(
    user_keypoints: list[PoseKeypoint],
    reference_library: dict[str, dict[str, ShotReference]],
) -> dict[str, str | float | list[str]] | None:
    best_match: dict[str, str | float | list[str]] | None = None
    best_similarity = -1.0
    for player_name, shots in reference_library.items():
        for shot_type, shot_reference in shots.items():
            result = _calculate_similarity(user_keypoints, shot_reference.keypoints)
            similarity = float(result["similarity"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    "player": player_name,
                    "shot": shot_type,
                    "similarity": similarity,
                    "feedback": result["feedback"],
                }
    return best_match
