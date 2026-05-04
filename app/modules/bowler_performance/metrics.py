from __future__ import annotations

import numpy as np
from loguru import logger

from app.modules.bowler_performance.models import (
    BouncePoint,
    BowlerPerformanceResult,
    LengthClass,
    classify_length,
)
from app.modules.bowler_performance.pitch_coordinates import (
    BOWLING_STUMP_Z_METRES,
    PitchPoint,
)
from app.modules.preprocessor.models import BallDetection

WorldPoint = tuple[BallDetection, np.ndarray]

DEFAULT_PROXY_RELEASE_EXTENSION_METRES = 1.5
DEFAULT_PROXY_RELEASE_TO_BOUNCE_DISTANCE_METRES = 16.0
MAX_PLAUSIBLE_BOUNCE_FRAME_OFFSET = 8


def compute_speed(
    world_points: list[WorldPoint],
    n_frames: int = 5,
) -> tuple[float, float]:
    points = world_points[: max(2, n_frames)]
    speed_samples: list[float] = []

    for (left_detection, left_point), (right_detection, right_point) in zip(
        points,
        points[1:],
        strict=False,
    ):
        dt = right_detection.timestamp_s - left_detection.timestamp_s
        if dt <= 0.0:
            continue
        distance = float(np.linalg.norm(right_point - left_point))
        speed_samples.append(distance / dt)

    if not speed_samples:
        return 0.0, 0.0

    speed_ms = float(np.median(np.asarray(speed_samples, dtype=np.float64)))
    return speed_ms, speed_ms * 3.6


def compute_swing(
    pitch_points: list[PitchPoint],
    bounce_frame: int | None,
) -> float:
    if not pitch_points:
        return 0.0

    release_x = float(pitch_points[0][1][0])
    if bounce_frame is None:
        target_x = float(pitch_points[-1][1][0])
    else:
        target_detection, target_point = min(
            pitch_points,
            key=lambda point: abs(point[0].frame_idx - bounce_frame),
        )
        _ = target_detection
        target_x = float(target_point[0])

    return target_x - release_x


def compute_bounce_and_length(
    pitch_points: list[PitchPoint],
    bounce_frame: int | None,
) -> tuple[BouncePoint | None, LengthClass | None]:
    if bounce_frame is None or not pitch_points:
        return None, None

    _detection, pitch_point = min(
        pitch_points,
        key=lambda point: abs(point[0].frame_idx - bounce_frame),
    )
    bounce_x = float(pitch_point[0])
    bounce_z = float(pitch_point[2])
    length_class = classify_length(bounce_z)
    return BouncePoint(x_metres=bounce_x, z_metres=bounce_z), length_class


def _find_nearest_plausible_bounce_point(
    pitch_points: list[PitchPoint],
    bounce_frame: int | None,
    max_frame_offset: int = MAX_PLAUSIBLE_BOUNCE_FRAME_OFFSET,
) -> BouncePoint | None:
    if bounce_frame is None:
        return None

    plausible_candidates: list[tuple[int, BouncePoint]] = []
    for detection, pitch_point in pitch_points:
        bounce_z = float(pitch_point[2])
        if not 0.0 <= bounce_z <= BOWLING_STUMP_Z_METRES:
            continue
        frame_offset = abs(int(detection.frame_idx) - bounce_frame)
        if frame_offset > max_frame_offset:
            continue
        plausible_candidates.append(
            (
                frame_offset,
                BouncePoint(
                    x_metres=float(pitch_point[0]),
                    z_metres=bounce_z,
                ),
            )
        )

    if not plausible_candidates:
        return None

    plausible_candidates.sort(key=lambda item: (item[0], item[1].z_metres))
    return plausible_candidates[0][1]


def _is_plausible_canonical_bounce_point(
    canonical_bounce_point: BouncePoint,
    raw_bounce_point: BouncePoint | None,
) -> bool:
    canonical_z = float(canonical_bounce_point.z_metres)
    if canonical_z < 0.0 or canonical_z > BOWLING_STUMP_Z_METRES:
        return False

    if raw_bounce_point is None:
        return True

    raw_z = float(raw_bounce_point.z_metres)
    return abs(canonical_z - raw_z) <= 2.0


def compute_proxy_speed(
    inliers: list[BallDetection],
    bounce_frame: int | None,
    release_timestamp_s: float | None,
    bounce_point: BouncePoint | None,
) -> tuple[float | None, float | None]:
    if release_timestamp_s is None or bounce_frame is None or not inliers:
        return None, None

    bounce_detection = min(
        inliers,
        key=lambda detection: abs(detection.frame_idx - bounce_frame),
    )
    dt = float(bounce_detection.timestamp_s - release_timestamp_s)
    if dt <= 0.0:
        return None, None

    bounce_z = bounce_point.z_metres if bounce_point is not None else None
    if bounce_z is not None and 0.0 <= bounce_z <= BOWLING_STUMP_Z_METRES:
        distance_metres = (
            BOWLING_STUMP_Z_METRES
            - bounce_z
            + DEFAULT_PROXY_RELEASE_EXTENSION_METRES
        )
    else:
        distance_metres = DEFAULT_PROXY_RELEASE_TO_BOUNCE_DISTANCE_METRES

    speed_ms = float(distance_metres / dt)
    return speed_ms, speed_ms * 3.6


def build_result(
    world_points: list[WorldPoint],
    pitch_points: list[PitchPoint],
    inliers: list[BallDetection],
    bounce_frame: int | None,
    release_timestamp_s: float | None = None,
    trajectory_reliable: bool = True,
    trajectory_warning: str | None = None,
    canonical_bounce_point: BouncePoint | None = None,
) -> BowlerPerformanceResult:
    _ = world_points
    bounce_point, length_class = compute_bounce_and_length(pitch_points, bounce_frame)
    raw_bounce_point = bounce_point
    if length_class is None:
        fallback_bounce_point = _find_nearest_plausible_bounce_point(
            pitch_points,
            bounce_frame,
        )
        if fallback_bounce_point is not None:
            logger.warning(
                "Replacing implausible bounce point raw_bounce={} fallback_bounce={}",
                (
                    {
                        "x": round(float(raw_bounce_point.x_metres), 3),
                        "z": round(float(raw_bounce_point.z_metres), 3),
                    }
                    if raw_bounce_point is not None
                    else None
                ),
                {
                    "x": round(float(fallback_bounce_point.x_metres), 3),
                    "z": round(float(fallback_bounce_point.z_metres), 3),
                },
            )
            bounce_point = fallback_bounce_point
            length_class = classify_length(fallback_bounce_point.z_metres)
    if canonical_bounce_point is not None:
        if _is_plausible_canonical_bounce_point(canonical_bounce_point, raw_bounce_point):
            bounce_point = canonical_bounce_point
            length_class = classify_length(canonical_bounce_point.z_metres)
        else:
            logger.warning(
                "Ignoring implausible canonical bounce point canonical_bounce={} raw_bounce={}",
                {
                    "x": round(float(canonical_bounce_point.x_metres), 3),
                    "z": round(float(canonical_bounce_point.z_metres), 3),
                },
                (
                    {
                        "x": round(float(raw_bounce_point.x_metres), 3),
                        "z": round(float(raw_bounce_point.z_metres), 3),
                    }
                    if raw_bounce_point is not None
                    else None
                ),
            )
    speed_ms, speed_kmh = compute_proxy_speed(
        inliers,
        bounce_frame,
        release_timestamp_s,
        bounce_point,
    )
    swing_metres = (
        compute_swing(pitch_points, bounce_frame)
        if trajectory_reliable
        else None
    )
    confidence = float(
        np.mean([detection.confidence for detection in inliers], dtype=np.float64)
    ) if inliers else 0.0

    return BowlerPerformanceResult(
        speed_kmh=speed_kmh,
        swing_metres=swing_metres,
        bounce_point=bounce_point,
        length_class=length_class,
        confidence=confidence,
        inlier_count=len(inliers),
        raw_speed_ms=speed_ms,
        trajectory_reliable=trajectory_reliable,
        trajectory_warning=trajectory_warning,
    )
