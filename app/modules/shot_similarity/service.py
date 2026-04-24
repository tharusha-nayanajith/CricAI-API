from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from app.ai.google import get_google_genai_client, get_google_genai_status
from app.config import get_settings
from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts
from app.modules.action_legality.service import _extract_keypoints_from_frame

from .models import PoseKeypoint, ShotReference, ShotSimilarityResult

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
REFERENCE_LIBRARY_PATH = ASSETS_DIR / "golden_frames.json"

_reference_library: list["_LoadedShotReference"] | None = None


@dataclass(slots=True)
class _LoadedShotReference:
    player_name: str
    shot_label: str
    canonical_shot_type: str
    frames: list[list[PoseKeypoint]]


class ShotSimilarityService:
    async def run(
        self,
        artifacts: VideoArtifacts,
        video_url: str | None = None,
        classified_shot_type: str | None = None,
    ) -> ShotSimilarityResult:
        frame = _resolve_comparison_frame(artifacts)
        logger.info(
            "Starting shot_similarity analysis video_url={} classified_shot_type={}",
            video_url,
            classified_shot_type,
        )
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                partial(self._run_sync, frame, video_url, classified_shot_type),
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
        classified_shot_type: str | None,
    ) -> ShotSimilarityResult:
        reference_library = _load_reference_library()
        keypoints_array = _extract_keypoints_from_frame(frame_bgr, list(range(33)))
        if keypoints_array is None:
            raise FeatureError("Pose landmarks were not detected in the shot comparison frame.")

        user_keypoints = _array_to_keypoints(keypoints_array)
        match = _find_best_match(user_keypoints, reference_library, classified_shot_type)
        if match is None:
            if classified_shot_type is not None:
                raise FeatureError(
                    "No reference shots are available for classified shot type "
                    f"'{classified_shot_type}'."
                )
            raise FeatureError("No reference shots are available for similarity comparison.")

        similarity_percentage = round(float(match["similarity"]), 2)
        avg_visibility = float(
            np.mean([keypoint.visibility for keypoint in user_keypoints]) * 100.0
        )
        feedback = list(match["feedback"])
        if match["reference_shot"] != match["canonical_shot_type"]:
            feedback.insert(
                0,
                f"Compared against the {match['reference_shot']} reference from {match['player']}.",
            )
        feedback = feedback[:5]
        shot_type = classified_shot_type or str(match["canonical_shot_type"])
        confidence = round(avg_visibility, 2)
        ai_feedback = _generate_similarity_ai_feedback(
            similarity_percentage=similarity_percentage,
            matched_player=str(match["player"]),
            shot_type=shot_type,
            reference_shot=str(match["reference_shot"]),
            confidence=confidence,
            heuristic_feedback=feedback,
        )
        return ShotSimilarityResult(
            similarity_percentage=similarity_percentage,
            matched_player=str(match["player"]),
            shot_type=shot_type,
            keypoints_detected=len(user_keypoints),
            confidence=confidence,
            feedback=feedback,
            ai_feedback=ai_feedback,
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


def _load_reference_library() -> list[_LoadedShotReference]:
    global _reference_library
    if _reference_library is None:
        references: list[_LoadedShotReference] = []
        settings = get_settings()
        external_reference_dir = settings.shot_similarity_reference_dir
        if external_reference_dir:
            references.extend(_load_directory_reference_library(Path(external_reference_dir)))
        if REFERENCE_LIBRARY_PATH.exists():
            references.extend(_load_legacy_reference_library(REFERENCE_LIBRARY_PATH))
        if not references:
            raise FeatureError(
                "Shot similarity reference library is missing. Configure "
                "SHOT_SIMILARITY_REFERENCE_DIR or add references at "
                f"{REFERENCE_LIBRARY_PATH}."
            )
        _reference_library = references
    return _reference_library


def _load_directory_reference_library(root: Path) -> list[_LoadedShotReference]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FeatureError(
            f"Shot similarity reference directory does not exist: {root}"
        )

    references: list[_LoadedShotReference] = []
    settings = get_settings()
    configured_player_name = settings.shot_similarity_reference_player_name
    for json_path in sorted(root.rglob("*.json")):
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FeatureError(
                f"Shot similarity reference file is invalid JSON: {json_path}"
            ) from exc

        frames_payload = raw.get("frames")
        if not isinstance(frames_payload, list):
            continue

        frames = [
            frame
            for frame in (_parse_frame_payload(frame_payload) for frame_payload in frames_payload)
            if frame
        ]
        if not frames:
            continue

        canonical_shot_type = _canonical_shot_type(json_path.stem)
        if canonical_shot_type is None:
            logger.warning("Skipping unsupported shot similarity reference file {}", json_path)
            continue

        player_path = json_path.parent.relative_to(root)
        player_name = player_path.parts[0] if player_path.parts else root.name
        display_player_name = configured_player_name or _humanize_label(player_name)
        references.append(
            _LoadedShotReference(
                player_name=display_player_name,
                shot_label=_humanize_label(json_path.stem),
                canonical_shot_type=canonical_shot_type,
                frames=frames,
            )
        )

    return references


def _load_legacy_reference_library(path: Path) -> list[_LoadedShotReference]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FeatureError("Shot similarity reference library JSON is invalid.") from exc

    references: list[_LoadedShotReference] = []
    for player_name, shots in raw.items():
        if not isinstance(shots, dict):
            continue
        for shot_type, shot_payload in shots.items():
            shot_reference = ShotReference.model_validate(shot_payload)
            canonical_shot_type = _canonical_shot_type(shot_type)
            if canonical_shot_type is None:
                logger.warning(
                    "Skipping unsupported legacy shot similarity reference {} / {}",
                    player_name,
                    shot_type,
                )
                continue
            references.append(
                _LoadedShotReference(
                    player_name=_humanize_label(player_name),
                    shot_label=_humanize_label(shot_type),
                    canonical_shot_type=canonical_shot_type,
                    frames=[shot_reference.keypoints],
                )
            )
    return references


def _parse_frame_payload(frame_payload: Any) -> list[PoseKeypoint]:
    if not isinstance(frame_payload, list):
        return []
    try:
        return [PoseKeypoint.model_validate(point) for point in frame_payload]
    except Exception:
        return []


def _humanize_label(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title()


def _canonical_shot_type(raw_shot_type: str | None) -> str | None:
    if not raw_shot_type:
        return None
    normalized = raw_shot_type.strip().lower().replace("-", "_").replace(" ", "_")
    alias_groups = {
        "cut": ("cut", "square_cut", "cut_shot"),
        "drive": ("drive", "cover_drive", "straight_drive", "off_drive", "drive_shot"),
        "flick": ("flick", "on_drive", "clip", "leg_glance", "flick_shot"),
        "pull": ("pull", "hook", "pull_shot", "hook_shot"),
        "slog": ("slog", "slog_shot", "lofted_drive", "slog_sweep"),
        "sweep": ("sweep", "sweep_shot", "reverse_sweep", "paddle_sweep"),
        "misc": ("misc", "miscellaneous", "other"),
    }
    for canonical, aliases in alias_groups.items():
        if normalized in aliases:
            return canonical
        if canonical != "misc" and normalized.startswith(f"{canonical}_"):
            return canonical
    return None


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


def _coerce_reference_library(reference_library: Any) -> list[_LoadedShotReference]:
    if isinstance(reference_library, list):
        return reference_library

    coerced: list[_LoadedShotReference] = []
    if isinstance(reference_library, dict):
        for player_name, shots in reference_library.items():
            if not isinstance(shots, dict):
                continue
            for shot_type, shot_reference in shots.items():
                if not isinstance(shot_reference, ShotReference):
                    shot_reference = ShotReference.model_validate(shot_reference)
                canonical_shot_type = _canonical_shot_type(shot_type)
                if canonical_shot_type is None:
                    continue
                coerced.append(
                    _LoadedShotReference(
                        player_name=str(player_name),
                        shot_label=str(shot_type),
                        canonical_shot_type=canonical_shot_type,
                        frames=[shot_reference.keypoints],
                    )
                )
    return coerced


def _find_best_match(
    user_keypoints: list[PoseKeypoint],
    reference_library: Any,
    classified_shot_type: str | None,
) -> dict[str, str | float | list[str]] | None:
    candidate_references = _coerce_reference_library(reference_library)
    canonical_shot_type = _canonical_shot_type(classified_shot_type)
    if canonical_shot_type is not None:
        candidate_references = [
            reference
            for reference in candidate_references
            if reference.canonical_shot_type == canonical_shot_type
        ]
    if not candidate_references:
        return None

    best_match: dict[str, str | float | list[str]] | None = None
    best_similarity = -1.0
    for reference in candidate_references:
        for frame_keypoints in reference.frames:
            result = _calculate_similarity(user_keypoints, frame_keypoints)
            similarity = float(result["similarity"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    "player": reference.player_name,
                    "shot": reference.canonical_shot_type,
                    "reference_shot": reference.shot_label,
                    "canonical_shot_type": reference.canonical_shot_type,
                    "similarity": similarity,
                    "feedback": list(result["feedback"]),
                }
    return best_match


def _generate_similarity_ai_feedback(
    similarity_percentage: float,
    matched_player: str,
    shot_type: str,
    reference_shot: str,
    confidence: float,
    heuristic_feedback: list[str],
) -> str:
    status = get_google_genai_status()
    if not status.enabled:
        return _rule_based_similarity_feedback(
            similarity_percentage=similarity_percentage,
            matched_player=matched_player,
            shot_type=shot_type,
            reference_shot=reference_shot,
            confidence=confidence,
            heuristic_feedback=heuristic_feedback,
        )

    client = get_google_genai_client()
    if client is None:
        return _rule_based_similarity_feedback(
            similarity_percentage=similarity_percentage,
            matched_player=matched_player,
            shot_type=shot_type,
            reference_shot=reference_shot,
            confidence=confidence,
            heuristic_feedback=heuristic_feedback,
        )

    prompt = _build_similarity_feedback_prompt(
        similarity_percentage=similarity_percentage,
        matched_player=matched_player,
        shot_type=shot_type,
        reference_shot=reference_shot,
        confidence=confidence,
        heuristic_feedback=heuristic_feedback,
    )

    try:
        response = client.models.generate_content(
            model=status.model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={"max_output_tokens": 220},
        )
    except Exception as exc:
        logger.warning("Shot similarity AI feedback generation failed: {}", exc)
        return _rule_based_similarity_feedback(
            similarity_percentage=similarity_percentage,
            matched_player=matched_player,
            shot_type=shot_type,
            reference_shot=reference_shot,
            confidence=confidence,
            heuristic_feedback=heuristic_feedback,
        )

    feedback = getattr(response, "text", "")
    if isinstance(feedback, str) and feedback.strip():
        return feedback.strip()

    return _rule_based_similarity_feedback(
        similarity_percentage=similarity_percentage,
        matched_player=matched_player,
        shot_type=shot_type,
        reference_shot=reference_shot,
        confidence=confidence,
        heuristic_feedback=heuristic_feedback,
    )


def _build_similarity_feedback_prompt(
    similarity_percentage: float,
    matched_player: str,
    shot_type: str,
    reference_shot: str,
    confidence: float,
    heuristic_feedback: list[str],
) -> str:
    feedback_lines = (
        "\n".join(f"- {item}" for item in heuristic_feedback)
        or "- No major pose deviations were detected."
    )
    return f"""You are an expert cricket batting coach comparing the player's clip against Virat Kohli reference shots.

Comparison data:
- Classified shot type: {shot_type}
- Best matched player reference: {matched_player}
- Reference shot label: {reference_shot}
- Similarity score: {similarity_percentage:.2f}%
- Pose detection confidence: {confidence:.2f}%
- Heuristic improvement notes:
{feedback_lines}

Write 2-3 concise coaching sentences for the player.
- Make the feedback feel like a Virat Kohli comparison, especially his balance, head position, bat path, and timing through contact.
- Mention how close the player is to Virat's shape at contact.
- Mention the single most important correction if one exists.
- Keep it practical and specific.
- Do not use markdown or bullet points."""


def _rule_based_similarity_feedback(
    similarity_percentage: float,
    matched_player: str,
    shot_type: str,
    reference_shot: str,
    confidence: float,
    heuristic_feedback: list[str],
) -> str:
    closeness = (
        "very close to Virat's"
        if similarity_percentage >= 85
        else "reasonably close to Virat's"
        if similarity_percentage >= 70
        else "still some distance away from Virat's"
    )
    reference_phrase = reference_shot.replace("_", " ")
    if heuristic_feedback:
        primary_fix = heuristic_feedback[0].rstrip(".").lower()
        return (
            f"This {shot_type} is {closeness} {reference_phrase} shape at {similarity_percentage:.1f}% similarity. "
            f"To get more of that Virat Kohli feel, focus first on {primary_fix} while keeping the head still and the bat path clean through contact."
        )
    if confidence < 60:
        return (
            f"The clip was matched against Virat Kohli's {reference_phrase} reference, but the pose confidence was only "
            f"{confidence:.1f}%. Record a clearer contact frame so the Virat-style comparison is more reliable."
        )
    return (
        f"This {shot_type} is {closeness} {reference_phrase} shape at {similarity_percentage:.1f}% similarity. "
        f"Keep building Virat-like balance, head position, and timing through contact so the movement pattern looks cleaner and more repeatable."
    )
