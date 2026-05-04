from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from app.config import get_settings
from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts
from app.modules.action_legality.service import (
    _extract_keypoints_from_frame,
    _extract_pose_landmarks_2d,
)
from app.modules.preprocessor.models import BatterROI
from app.storage.artifacts import build_video_artifact_path, write_json_artifact

from .models import PoseKeypoint, ShotReference, ShotSimilarityResult

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
REFERENCE_LIBRARY_PATH = ASSETS_DIR / "golden_frames.json"
REFERENCE_SHOTS_DIR = ASSETS_DIR / "shots"
REFERENCE_SHOTS_PLAYER_NAME = "Virat Koli"
SHOT_SIMILARITY_FEATURE_NAME = "shot_similarity"
TARGET_FRAME_COUNT = 30
LANDMARK_COUNT = 33
MIN_VALID_POSE_FRAME_RATIO = 0.35
MIN_AVG_POSE_VISIBILITY = 20.0
MIN_VALID_CORE_LANDMARKS = 8
MIN_BODY_SCALE = 0.03
CORE_LANDMARKS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)

_reference_library: list[_LoadedShotReference] | None = None


@dataclass(slots=True)
class _LoadedShotReference:
    player_name: str
    shot_label: str
    canonical_shot_type: str
    frames: list[list[PoseKeypoint]]


@dataclass(slots=True)
class _MatchDetails:
    player: str
    reference_shot: str
    canonical_shot_type: str
    similarity: float
    distance: float
    feedback: list[str]
    reference_frames: list[list[PoseKeypoint]]
    normalized_user_frames: list[list[PoseKeypoint]]
    normalized_reference_frames: list[list[PoseKeypoint]]
    alignment_path: list[tuple[int, int]]


class ShotSimilarityService:
    async def run(
        self,
        artifacts: VideoArtifacts,
        video_url: str | None = None,
        classified_shot_type: str | None = None,
        job_id: str | None = None,
    ) -> ShotSimilarityResult:
        _validate_comparison_source(artifacts)
        logger.info(
            "Starting shot_similarity analysis video_url={} classified_shot_type={}",
            video_url,
            classified_shot_type,
        )
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                partial(self._run_sync, artifacts, video_url, classified_shot_type, job_id),
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
        artifacts: VideoArtifacts,
        video_url: str | None,
        classified_shot_type: str | None,
        job_id: str | None,
    ) -> ShotSimilarityResult:
        reference_library = _load_reference_library()
        user_frames_bgr = _extract_evenly_spaced_frames(artifacts)
        user_sequence = _extract_pose_sequence(user_frames_bgr, artifacts.batter_roi)
        if not user_sequence:
            raise FeatureError("Pose landmarks were not detected in the shot video.")
        _validate_pose_quality(user_sequence)

        match = _find_best_match(user_sequence, reference_library, classified_shot_type)
        if match is None:
            if classified_shot_type is not None:
                raise FeatureError(
                    "No reference shots are available for classified shot type "
                    f"'{classified_shot_type}'."
                )
            raise FeatureError("No reference shots are available for similarity comparison.")

        avg_visibility = _sequence_visibility(user_sequence)
        feedback = list(match.feedback)
        if match.reference_shot != match.canonical_shot_type:
            feedback.insert(
                0,
                f"Compared against the {match.reference_shot} reference from {match.player}.",
            )
        normalized_user_url = None
        normalized_reference_url = None
        visualization_video_url = None
        if job_id is not None:
            normalized_user_url = write_json_artifact(
                job_id,
                SHOT_SIMILARITY_FEATURE_NAME,
                "normalized_user.json",
                _sequence_to_payload(match.normalized_user_frames),
            )
            normalized_reference_url = write_json_artifact(
                job_id,
                SHOT_SIMILARITY_FEATURE_NAME,
                "normalized_reference.json",
                _sequence_to_payload(match.normalized_reference_frames),
            )
            visualization_video_url = _write_comparison_video(
                job_id,
                user_frames_bgr,
                user_sequence,
                match.reference_frames,
                match.alignment_path,
                artifacts.batter_roi,
            )

        shot_type = classified_shot_type or match.canonical_shot_type
        return ShotSimilarityResult(
            similarity_percentage=round(match.similarity, 2),
            matched_player=match.player,
            shot_type=shot_type,
            keypoints_detected=_detected_keypoint_count(user_sequence),
            confidence=round(avg_visibility, 2),
            feedback=feedback[:5],
            compared_frame="30_frame_pose_sequence",
            video_url=video_url,
            ai_feedback=_build_ai_feedback(feedback),
            visualization_video_url=visualization_video_url,
            normalized_user_url=normalized_user_url,
            normalized_reference_url=normalized_reference_url,
        )


def _validate_comparison_source(artifacts: VideoArtifacts) -> None:
    if (
        artifacts.standardized_video_path is None
        or not artifacts.standardized_video_path.exists()
    ) and artifacts.bat_contact_frame is None:
        raise FeatureError(
            "Shot similarity requires a standardized batting video or a batter contact frame."
        )


def _extract_evenly_spaced_frames(artifacts: VideoArtifacts) -> list[np.ndarray]:
    video_path = artifacts.standardized_video_path
    if video_path is None or not video_path.exists():
        if artifacts.bat_contact_frame is None:
            return []
        return [artifacts.bat_contact_frame.copy()] * TARGET_FRAME_COUNT

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FeatureError(f"Unable to open video file for shot similarity: {video_path}")

    frames: list[np.ndarray] = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise FeatureError("Unable to read total frame count for shot similarity.")

        frame_indices = np.linspace(0, total_frames - 1, TARGET_FRAME_COUNT, dtype=np.int32)
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise FeatureError("Unable to extract frames for shot similarity.")
    while len(frames) < TARGET_FRAME_COUNT:
        frames.append(frames[-1].copy())
    return frames[:TARGET_FRAME_COUNT]


def _extract_pose_sequence(
    frames_bgr: list[np.ndarray],
    batter_roi: BatterROI | None = None,
) -> list[list[PoseKeypoint]]:
    return [_extract_pose_frame_keypoints(frame, batter_roi) for frame in frames_bgr]


def _extract_pose_frame_keypoints(
    frame_bgr: np.ndarray,
    batter_roi: BatterROI | None = None,
) -> list[PoseKeypoint]:
    selected_landmarks = list(range(LANDMARK_COUNT))
    height, width = frame_bgr.shape[:2]
    roi = _clamp_batter_roi(batter_roi, width, height)
    pose_frame = frame_bgr
    offset_x = 0
    offset_y = 0
    if roi is not None:
        pose_frame = frame_bgr[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width]
        offset_x = roi.x
        offset_y = roi.y
        if pose_frame.size == 0:
            pose_frame = frame_bgr
            offset_x = 0
            offset_y = 0
            roi = None

    landmarks_2d = _extract_pose_landmarks_2d(pose_frame, selected_landmarks)
    if landmarks_2d:
        return [
            PoseKeypoint(
                x=float((offset_x + point.x) / width) if width else float(point.x),
                y=float((offset_y + point.y) / height) if height else float(point.y),
                z=float(point.z),
                visibility=float(np.clip(point.visibility, 0.0, 1.0)),
            )
            for point in landmarks_2d
        ]

    keypoints_array = _extract_keypoints_from_frame(pose_frame, selected_landmarks)
    if keypoints_array is not None:
        keypoints = _array_to_keypoints(keypoints_array)
        return _map_roi_keypoints_to_frame(keypoints, roi, width, height)
    return _zero_pose_frame()


def _clamp_batter_roi(
    batter_roi: BatterROI | None,
    frame_width: int,
    frame_height: int,
) -> BatterROI | None:
    if batter_roi is None or frame_width <= 0 or frame_height <= 0:
        return None
    x = int(np.clip(batter_roi.x, 0, frame_width - 1))
    y = int(np.clip(batter_roi.y, 0, frame_height - 1))
    width = int(np.clip(batter_roi.width, 1, frame_width - x))
    height = int(np.clip(batter_roi.height, 1, frame_height - y))
    if width <= 1 or height <= 1:
        return None
    return BatterROI(x=x, y=y, width=width, height=height)


def _map_roi_keypoints_to_frame(
    keypoints: list[PoseKeypoint],
    batter_roi: BatterROI | None,
    frame_width: int,
    frame_height: int,
) -> list[PoseKeypoint]:
    if batter_roi is None or frame_width <= 0 or frame_height <= 0:
        return keypoints
    return [
        PoseKeypoint(
            x=float((batter_roi.x + (point.x * batter_roi.width)) / frame_width),
            y=float((batter_roi.y + (point.y * batter_roi.height)) / frame_height),
            z=point.z,
            visibility=point.visibility,
        )
        for point in keypoints
    ]


def _zero_pose_frame() -> list[PoseKeypoint]:
    return [
        PoseKeypoint(x=0.0, y=0.0, z=0.0, visibility=0.0)
        for _ in range(LANDMARK_COUNT)
    ]


def _load_reference_library() -> list[_LoadedShotReference]:
    global _reference_library
    if _reference_library is None:
        references: list[_LoadedShotReference] = []
        settings = get_settings()
        external_reference_dir = settings.shot_similarity_reference_dir
        loaded_reference_roots: set[Path] = set()
        if external_reference_dir:
            root = Path(external_reference_dir).expanduser().resolve()
            references.extend(
                _load_directory_reference_library(
                    root,
                    default_player_name=(
                        REFERENCE_SHOTS_PLAYER_NAME
                        if root == REFERENCE_SHOTS_DIR.expanduser().resolve()
                        else None
                    ),
                )
            )
            loaded_reference_roots.add(root)
        bundled_root = REFERENCE_SHOTS_DIR.expanduser().resolve()
        if REFERENCE_SHOTS_DIR.exists() and bundled_root not in loaded_reference_roots:
            references.extend(
                _load_directory_reference_library(
                    REFERENCE_SHOTS_DIR,
                    default_player_name=REFERENCE_SHOTS_PLAYER_NAME,
                )
            )
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


def _load_directory_reference_library(
    root: Path,
    default_player_name: str | None = None,
) -> list[_LoadedShotReference]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FeatureError(
            f"Shot similarity reference directory does not exist: {root}"
        )

    references: list[_LoadedShotReference] = []
    for json_path in sorted(root.rglob('*.json')):
        try:
            raw = json.loads(json_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            raise FeatureError(
                f"Shot similarity reference file is invalid JSON: {json_path}"
            ) from exc

        frames_payload = raw.get('frames')
        if not isinstance(frames_payload, list):
            continue

        frames = [
            frame
            for frame in (
                _parse_frame_payload(frame_payload)
                for frame_payload in frames_payload
            )
            if frame
        ]
        if not frames:
            continue

        canonical_shot_type = _canonical_shot_type(json_path.stem)
        if canonical_shot_type is None:
            logger.warning('Skipping unsupported shot similarity reference file {}', json_path)
            continue

        player_path = json_path.parent.relative_to(root)
        player_name = (
            player_path.parts[0]
            if player_path.parts
            else default_player_name or root.name
        )
        references.append(
            _LoadedShotReference(
                player_name=_humanize_label(player_name),
                shot_label=_humanize_label(json_path.stem),
                canonical_shot_type=canonical_shot_type,
                frames=frames,
            )
        )

    return references


def _load_legacy_reference_library(path: Path) -> list[_LoadedShotReference]:
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise FeatureError('Shot similarity reference library JSON is invalid.') from exc

    references: list[_LoadedShotReference] = []
    for player_name, shots in raw.items():
        if not isinstance(shots, dict):
            continue
        for shot_type, shot_payload in shots.items():
            shot_reference = ShotReference.model_validate(shot_payload)
            canonical_shot_type = _canonical_shot_type(shot_type)
            if canonical_shot_type is None:
                logger.warning(
                    'Skipping unsupported legacy shot similarity reference {} / {}',
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
    return value.replace('_', ' ').replace('-', ' ').strip().title()


def _canonical_shot_type(raw_shot_type: str | None) -> str | None:
    if not raw_shot_type:
        return None
    normalized = raw_shot_type.strip().lower().replace('-', '_').replace(' ', '_')
    alias_groups = {
        'cut': ('cut', 'square_cut', 'cut_shot'),
        'drive': ('drive', 'cover_drive', 'straight_drive', 'off_drive', 'drive_shot'),
        'flick': ('flick', 'on_drive', 'clip', 'leg_glance', 'flick_shot'),
        'pull': ('pull', 'hook', 'pull_shot', 'hook_shot'),
        'slog': ('slog', 'slog_shot', 'lofted_drive', 'slog_sweep'),
        'sweep': ('sweep', 'sweep_shot', 'reverse_sweep', 'paddle_sweep'),
        'misc': ('misc', 'miscellaneous', 'other'),
    }
    for canonical, aliases in alias_groups.items():
        if normalized in aliases:
            return canonical
        if canonical != 'misc' and normalized.startswith(f'{canonical}_'):
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
    return np.asarray(
        [[kp.x, kp.y, kp.z, kp.visibility] for kp in _normalize_pose_frame(keypoints)],
        dtype=np.float32,
    )


def _normalize_pose_sequence(
    frames: list[list[PoseKeypoint]],
) -> list[list[PoseKeypoint]]:
    return [_normalize_pose_frame(frame) for frame in frames]


def _normalize_pose_frame(keypoints: list[PoseKeypoint]) -> list[PoseKeypoint]:
    if len(keypoints) < LANDMARK_COUNT:
        return _zero_pose_frame()

    points = np.asarray([[kp.x, kp.y, kp.z] for kp in keypoints], dtype=np.float32)
    visibility = np.asarray([kp.visibility for kp in keypoints], dtype=np.float32)
    if float(np.max(visibility)) <= 0.0:
        return _zero_pose_frame()

    hip_center = (points[23] + points[24]) / 2.0
    centered = points - hip_center

    shoulder_vector = points[12, :2] - points[11, :2]
    shoulder_distance = float(np.linalg.norm(shoulder_vector))
    if shoulder_distance < 1e-6:
        shoulder_distance = float(np.max(np.abs(centered[:, :2]))) or 1.0
    normalized = centered / shoulder_distance

    shoulder_vector_norm = normalized[12, :2] - normalized[11, :2]
    angle = math.atan2(float(shoulder_vector_norm[1]), float(shoulder_vector_norm[0]))
    cos_theta = math.cos(-angle)
    sin_theta = math.sin(-angle)
    rotation = np.asarray(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
        dtype=np.float32,
    )
    normalized[:, :2] = normalized[:, :2] @ rotation.T

    return [
        PoseKeypoint(
            x=float(normalized[idx, 0]),
            y=float(normalized[idx, 1]),
            z=float(normalized[idx, 2]),
            visibility=float(np.clip(visibility[idx], 0.0, 1.0)),
        )
        for idx in range(LANDMARK_COUNT)
    ]


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
    user_frames: list[list[PoseKeypoint]],
    golden_frames: list[list[PoseKeypoint]],
) -> dict[str, float | list[str] | list[list[PoseKeypoint]] | list[tuple[int, int]]]:
    normalized_user = _normalize_pose_sequence(user_frames)
    normalized_golden = _normalize_pose_sequence(golden_frames)
    distance, path = _dtw_distance(normalized_user, normalized_golden)
    similarity = 100.0 * (1.0 / (1.0 + distance))
    feedback = _build_rule_feedback(normalized_user, normalized_golden, path)

    overall_similarity = float(np.clip(similarity, 0.0, 100.0))
    return {
        'similarity': overall_similarity,
        'distance': distance,
        'feedback': feedback,
        'normalized_user_frames': normalized_user,
        'normalized_reference_frames': normalized_golden,
        'alignment_path': path,
    }


def _joint_weights() -> np.ndarray:
    weights = np.ones(LANDMARK_COUNT, dtype=np.float32)
    for idx, weight in {
        11: 2.0,
        12: 2.0,
        13: 2.7,
        14: 2.7,
        15: 3.0,
        16: 3.0,
        23: 2.2,
        24: 2.2,
        25: 1.7,
        26: 1.7,
        27: 1.3,
        28: 1.3,
    }.items():
        weights[idx] = weight
    return weights


def _frame_distance(
    user_frame: list[PoseKeypoint],
    reference_frame: list[PoseKeypoint],
) -> float:
    user_xy = np.asarray([[point.x, point.y] for point in user_frame], dtype=np.float32)
    ref_xy = np.asarray([[point.x, point.y] for point in reference_frame], dtype=np.float32)
    user_vis = np.asarray([point.visibility for point in user_frame], dtype=np.float32)
    ref_vis = np.asarray([point.visibility for point in reference_frame], dtype=np.float32)
    visibility = np.minimum(user_vis, ref_vis)
    weights = _joint_weights() * np.clip(visibility, 0.0, 1.0)
    if float(np.sum(weights)) <= 1e-6:
        return 1.0
    distances = np.linalg.norm(user_xy - ref_xy, axis=1)
    return float(np.sum(distances * weights) / np.sum(weights))


def _dtw_distance(
    user_frames: list[list[PoseKeypoint]],
    reference_frames: list[list[PoseKeypoint]],
) -> tuple[float, list[tuple[int, int]]]:
    user_count = len(user_frames)
    reference_count = len(reference_frames)
    if user_count == 0 or reference_count == 0:
        return 1.0, []

    cost = np.full((user_count + 1, reference_count + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0
    for user_idx in range(1, user_count + 1):
        for ref_idx in range(1, reference_count + 1):
            frame_cost = _frame_distance(
                user_frames[user_idx - 1],
                reference_frames[ref_idx - 1],
            )
            cost[user_idx, ref_idx] = frame_cost + min(
                cost[user_idx - 1, ref_idx],
                cost[user_idx, ref_idx - 1],
                cost[user_idx - 1, ref_idx - 1],
            )

    path: list[tuple[int, int]] = []
    user_idx = user_count
    ref_idx = reference_count
    while user_idx > 0 and ref_idx > 0:
        path.append((user_idx - 1, ref_idx - 1))
        choices = (
            cost[user_idx - 1, ref_idx - 1],
            cost[user_idx - 1, ref_idx],
            cost[user_idx, ref_idx - 1],
        )
        step = int(np.argmin(choices))
        if step == 0:
            user_idx -= 1
            ref_idx -= 1
        elif step == 1:
            user_idx -= 1
        else:
            ref_idx -= 1
    path.reverse()

    path_length = max(len(path), 1)
    return float(cost[user_count, reference_count] / path_length), path


def _build_rule_feedback(
    user_frames: list[list[PoseKeypoint]],
    reference_frames: list[list[PoseKeypoint]],
    alignment_path: list[tuple[int, int]],
) -> list[str]:
    if not alignment_path:
        return ["Pose timing could not be aligned clearly with the reference shot."]

    angle_rules = {
        "left_elbow": ((11, 13, 15), "Raise and extend your left elbow through contact."),
        "right_elbow": ((12, 14, 16), "Keep your right elbow closer to the reference shape."),
        "left_knee": ((23, 25, 27), "Bend your left knee more for balance."),
        "right_knee": ((24, 26, 28), "Keep your right knee more stable during the shot."),
        "left_shoulder": ((13, 11, 23), "Improve left shoulder rotation into the shot."),
        "right_shoulder": ((14, 12, 24), "Rotate your right shoulder more like the reference."),
    }
    issue_scores: dict[str, float] = {}
    for points, message in angle_rules.values():
        diffs: list[float] = []
        for user_idx, ref_idx in alignment_path:
            user_frame = user_frames[user_idx]
            reference_frame = reference_frames[ref_idx]
            if not _angle_visible(user_frame, points) or not _angle_visible(
                reference_frame,
                points,
            ):
                continue
            user_angle = _calculate_angle(
                user_frame[points[0]],
                user_frame[points[1]],
                user_frame[points[2]],
            )
            reference_angle = _calculate_angle(
                reference_frame[points[0]],
                reference_frame[points[1]],
                reference_frame[points[2]],
            )
            diffs.append(abs(user_angle - reference_angle))
        if diffs:
            issue_scores[message] = float(np.mean(diffs))

    selected = [
        message
        for message, diff in sorted(issue_scores.items(), key=lambda item: item[1], reverse=True)
        if diff > 18.0
    ]
    if selected:
        return selected[:3]
    return ["Good pose alignment across the shot sequence."]


def _angle_visible(frame: list[PoseKeypoint], points: tuple[int, int, int]) -> bool:
    return all(frame[idx].visibility >= 0.25 for idx in points)


def _sequence_visibility(frames: list[list[PoseKeypoint]]) -> float:
    values = [
        point.visibility
        for frame in frames
        for point in frame
    ]
    return float(np.mean(values) * 100.0) if values else 0.0


def _validate_pose_quality(frames: list[list[PoseKeypoint]]) -> None:
    avg_visibility = _sequence_visibility(frames)
    valid_frame_count = sum(1 for frame in frames if _is_valid_pose_frame(frame))
    valid_frame_ratio = valid_frame_count / max(len(frames), 1)
    if (
        avg_visibility >= MIN_AVG_POSE_VISIBILITY
        and valid_frame_ratio >= MIN_VALID_POSE_FRAME_RATIO
    ):
        return

    raise FeatureError(
        "Pose detection quality is too low for shot similarity. "
        "Please record the batter with the full body visible, good lighting, "
        "and minimal camera shake."
    )


def _is_valid_pose_frame(frame: list[PoseKeypoint]) -> bool:
    if len(frame) < LANDMARK_COUNT:
        return False
    visible_core_landmarks = sum(
        1
        for idx in CORE_LANDMARKS
        if frame[idx].visibility >= 0.25
    )
    if visible_core_landmarks < MIN_VALID_CORE_LANDMARKS:
        return False
    return _body_scale(frame) >= MIN_BODY_SCALE


def _body_scale(frame: list[PoseKeypoint]) -> float:
    shoulder_width = _point_distance(frame[11], frame[12])
    hip_width = _point_distance(frame[23], frame[24])
    torso_length = _point_distance(
        _midpoint(frame[11], frame[12]),
        _midpoint(frame[23], frame[24]),
    )
    return max(shoulder_width, hip_width, torso_length)


def _point_distance(first: PoseKeypoint, second: PoseKeypoint) -> float:
    return float(math.hypot(first.x - second.x, first.y - second.y))


def _midpoint(first: PoseKeypoint, second: PoseKeypoint) -> PoseKeypoint:
    return PoseKeypoint(
        x=(first.x + second.x) / 2.0,
        y=(first.y + second.y) / 2.0,
        z=(first.z + second.z) / 2.0,
        visibility=min(first.visibility, second.visibility),
    )


def _detected_keypoint_count(frames: list[list[PoseKeypoint]]) -> int:
    return sum(1 for frame in frames for point in frame if point.visibility > 0.0)


def _sequence_to_payload(
    frames: list[list[PoseKeypoint]],
) -> dict[str, list[list[dict[str, float]]]]:
    return {
        "frames": [
            [point.model_dump() for point in frame]
            for frame in frames
        ]
    }


def _build_ai_feedback(feedback: list[str]) -> str:
    if not feedback:
        return "Good shot similarity. Keep repeating the same movement pattern."
    return " ".join(feedback[:3])


POSE_CONNECTIONS = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
)


def _write_comparison_video(
    job_id: str,
    user_frames_bgr: list[np.ndarray],
    user_sequence: list[list[PoseKeypoint]],
    reference_sequence: list[list[PoseKeypoint]],
    alignment_path: list[tuple[int, int]],
    batter_roi: BatterROI | None = None,
) -> str | None:
    if not user_frames_bgr or not alignment_path:
        return None

    smoothed_user_sequence = _smooth_pose_sequence(user_sequence)
    smoothed_reference_sequence = _smooth_pose_sequence(reference_sequence)
    output_path, artifact_url = build_video_artifact_path(
        job_id,
        SHOT_SIMILARITY_FEATURE_NAME,
        "comparison.mp4",
    )
    canvas_width = 1280
    canvas_height = 720
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        15.0,
        (canvas_width, canvas_height),
    )
    if not writer.isOpened():
        return None

    try:
        selected_pairs = _sample_alignment_path(alignment_path, TARGET_FRAME_COUNT)
        user_panel = (56, 112, 544, 520)
        reference_panel = (680, 112, 544, 520)
        for user_idx, ref_idx in selected_pairs:
            frame = _build_comparison_canvas(canvas_width, canvas_height)
            _draw_pose_panel(
                frame,
                smoothed_user_sequence[min(user_idx, len(smoothed_user_sequence) - 1)],
                user_panel,
                (40, 255, 40),
                line_thickness=6,
                point_radius=8,
            )
            _draw_pose_panel(
                frame,
                smoothed_reference_sequence[min(ref_idx, len(smoothed_reference_sequence) - 1)],
                reference_panel,
                (40, 40, 255),
                line_thickness=6,
                point_radius=8,
            )
            writer.write(frame)
    finally:
        writer.release()

    return artifact_url if output_path.exists() else None


def _build_comparison_canvas(width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), (248, 248, 245), dtype=np.uint8)
    cv2.line(canvas, (width // 2, 86), (width // 2, height - 52), (210, 214, 218), 2)
    cv2.putText(
        canvas,
        "USER",
        (250, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (28, 120, 52),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "REFERENCE",
        (835, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (40, 40, 190),
        3,
        cv2.LINE_AA,
    )
    return canvas


def _sample_alignment_path(
    alignment_path: list[tuple[int, int]],
    target_count: int,
) -> list[tuple[int, int]]:
    if len(alignment_path) <= target_count:
        return alignment_path
    indices = np.linspace(0, len(alignment_path) - 1, target_count, dtype=np.int32)
    return [alignment_path[int(idx)] for idx in indices]


def _draw_pose_panel(
    frame_bgr: np.ndarray,
    keypoints: list[PoseKeypoint],
    panel: tuple[int, int, int, int],
    color: tuple[int, int, int],
    *,
    line_thickness: int,
    point_radius: int,
) -> None:
    x, y, width, height = panel
    cv2.rectangle(
        frame_bgr,
        (x, y),
        (x + width, y + height),
        (228, 232, 235),
        2,
        cv2.LINE_AA,
    )
    points = _pose_points_for_panel(keypoints, panel)
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx in points and end_idx in points:
            cv2.line(
                frame_bgr,
                points[start_idx],
                points[end_idx],
                (20, 20, 20),
                line_thickness + 4,
                lineType=cv2.LINE_AA,
            )
            cv2.line(
                frame_bgr,
                points[start_idx],
                points[end_idx],
                color,
                line_thickness,
                lineType=cv2.LINE_AA,
            )
    for point in points.values():
        cv2.circle(
            frame_bgr,
            point,
            point_radius + 3,
            (20, 20, 20),
            -1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(frame_bgr, point, point_radius, color, -1, lineType=cv2.LINE_AA)


def _pose_points_for_panel(
    keypoints: list[PoseKeypoint],
    panel: tuple[int, int, int, int],
) -> dict[int, tuple[int, int]]:
    visible = [
        (idx, point)
        for idx, point in enumerate(keypoints)
        if point.visibility >= 0.25
    ]
    if not visible:
        return {}

    xs = [float(point.x) for _, point in visible]
    ys = [float(point.y) for _, point in visible]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    pose_width = max(max_x - min_x, 1e-6)
    pose_height = max(max_y - min_y, 1e-6)

    panel_x, panel_y, panel_width, panel_height = panel
    padding = 46
    drawable_width = max(panel_width - (padding * 2), 1)
    drawable_height = max(panel_height - (padding * 2), 1)
    scale = min(drawable_width / pose_width, drawable_height / pose_height)
    scaled_width = pose_width * scale
    scaled_height = pose_height * scale
    offset_x = panel_x + ((panel_width - scaled_width) / 2.0)
    offset_y = panel_y + ((panel_height - scaled_height) / 2.0)

    return {
        idx: (
            int(offset_x + ((point.x - min_x) * scale)),
            int(offset_y + ((point.y - min_y) * scale)),
        )
        for idx, point in visible
    }


def _draw_pose(
    frame_bgr: np.ndarray,
    keypoints: list[PoseKeypoint],
    color: tuple[int, int, int],
    width: int,
    height: int,
    *,
    line_thickness: int,
    point_radius: int,
    target_roi: BatterROI | None = None,
) -> None:
    points: dict[int, tuple[int, int]] = {}
    for idx, point in enumerate(keypoints):
        if point.visibility < 0.25:
            continue
        if target_roi is not None:
            x = target_roi.x + int(np.clip(point.x, 0.0, 1.0) * target_roi.width)
            y = target_roi.y + int(np.clip(point.y, 0.0, 1.0) * target_roi.height)
        else:
            x = int(np.clip(point.x, 0.0, 1.0) * width)
            y = int(np.clip(point.y, 0.0, 1.0) * height)
        points[idx] = (x, y)
        cv2.circle(
            frame_bgr,
            (x, y),
            point_radius + 2,
            (0, 0, 0),
            -1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(frame_bgr, (x, y), point_radius, color, -1, lineType=cv2.LINE_AA)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx in points and end_idx in points:
            cv2.line(
                frame_bgr,
                points[start_idx],
                points[end_idx],
                (0, 0, 0),
                line_thickness + 3,
                lineType=cv2.LINE_AA,
            )
            cv2.line(
                frame_bgr,
                points[start_idx],
                points[end_idx],
                color,
                line_thickness,
                lineType=cv2.LINE_AA,
            )


def _draw_batter_roi(frame_bgr: np.ndarray, batter_roi: BatterROI) -> None:
    top_left = (batter_roi.x, batter_roi.y)
    bottom_right = (
        batter_roi.x + batter_roi.width,
        batter_roi.y + batter_roi.height,
    )
    cv2.rectangle(frame_bgr, top_left, bottom_right, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.rectangle(frame_bgr, top_left, bottom_right, (255, 255, 255), 2, cv2.LINE_AA)


def _smooth_pose_sequence(frames: list[list[PoseKeypoint]]) -> list[list[PoseKeypoint]]:
    if not frames:
        return []

    smoothed = [_copy_pose_frame(frame) for frame in frames]
    for landmark_idx in range(LANDMARK_COUNT):
        smoothed = _interpolate_landmark(smoothed, landmark_idx)

    window_radius = 1
    result: list[list[PoseKeypoint]] = []
    for frame_idx, frame in enumerate(smoothed):
        next_frame: list[PoseKeypoint] = []
        start = max(0, frame_idx - window_radius)
        end = min(len(smoothed), frame_idx + window_radius + 1)
        for landmark_idx in range(LANDMARK_COUNT):
            candidates = [
                smoothed[idx][landmark_idx]
                for idx in range(start, end)
                if smoothed[idx][landmark_idx].visibility >= 0.25
            ]
            if not candidates:
                next_frame.append(frame[landmark_idx])
                continue
            next_frame.append(
                PoseKeypoint(
                    x=float(np.mean([point.x for point in candidates])),
                    y=float(np.mean([point.y for point in candidates])),
                    z=float(np.mean([point.z for point in candidates])),
                    visibility=float(np.mean([point.visibility for point in candidates])),
                )
            )
        result.append(next_frame)
    return result


def _copy_pose_frame(frame: list[PoseKeypoint]) -> list[PoseKeypoint]:
    return [PoseKeypoint.model_validate(point.model_dump()) for point in frame]


def _interpolate_landmark(
    frames: list[list[PoseKeypoint]],
    landmark_idx: int,
) -> list[list[PoseKeypoint]]:
    visible_indices = [
        idx
        for idx, frame in enumerate(frames)
        if frame[landmark_idx].visibility >= 0.25
    ]
    if not visible_indices:
        return frames

    for frame_idx, frame in enumerate(frames):
        if frame[landmark_idx].visibility >= 0.25:
            continue
        previous_indices = [idx for idx in visible_indices if idx < frame_idx]
        next_indices = [idx for idx in visible_indices if idx > frame_idx]
        previous_idx = previous_indices[-1] if previous_indices else None
        next_idx = next_indices[0] if next_indices else None
        replacement = None
        if previous_idx is not None and next_idx is not None:
            amount = (frame_idx - previous_idx) / (next_idx - previous_idx)
            previous = frames[previous_idx][landmark_idx]
            next_point = frames[next_idx][landmark_idx]
            replacement = PoseKeypoint(
                x=float(previous.x + ((next_point.x - previous.x) * amount)),
                y=float(previous.y + ((next_point.y - previous.y) * amount)),
                z=float(previous.z + ((next_point.z - previous.z) * amount)),
                visibility=float(min(previous.visibility, next_point.visibility) * 0.85),
            )
        elif previous_idx is not None:
            replacement = frames[previous_idx][landmark_idx]
        elif next_idx is not None:
            replacement = frames[next_idx][landmark_idx]
        if replacement is not None:
            frame[landmark_idx] = PoseKeypoint.model_validate(replacement.model_dump())
    return frames


def _coerce_reference_library(
    reference_library: Any,
) -> list[_LoadedShotReference]:
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
    user_frames: list[list[PoseKeypoint]],
    reference_library: Any,
    classified_shot_type: str | None,
) -> _MatchDetails | None:
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

    best_match: _MatchDetails | None = None
    best_similarity = -1.0
    for reference in candidate_references:
        if not reference.frames:
            continue
        result = _calculate_similarity(user_frames, reference.frames)
        similarity = float(result['similarity'])
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = _MatchDetails(
                player=reference.player_name,
                reference_shot=reference.shot_label,
                canonical_shot_type=reference.canonical_shot_type,
                similarity=similarity,
                distance=float(result["distance"]),
                feedback=list(result["feedback"]),
                reference_frames=reference.frames,
                normalized_user_frames=list(result["normalized_user_frames"]),
                normalized_reference_frames=list(result["normalized_reference_frames"]),
                alignment_path=list(result["alignment_path"]),
            )
    return best_match
