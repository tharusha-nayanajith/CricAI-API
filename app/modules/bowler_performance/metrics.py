from __future__ import annotations

import numpy as np

from app.modules.bowler_performance.models import (
    BouncePoint,
    BowlerPerformanceResult,
    LengthClass,
    classify_length,
)
from app.modules.bowler_performance.pitch_coordinates import PitchPoint
from app.modules.preprocessor.models import BallDetection

WorldPoint = tuple[BallDetection, np.ndarray]


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


def build_result(
    world_points: list[WorldPoint],
    pitch_points: list[PitchPoint],
    inliers: list[BallDetection],
    bounce_frame: int | None,
) -> BowlerPerformanceResult:
    speed_ms, speed_kmh = compute_speed(world_points)
    swing_metres = compute_swing(pitch_points, bounce_frame)
    bounce_point, length_class = compute_bounce_and_length(pitch_points, bounce_frame)
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
    )
