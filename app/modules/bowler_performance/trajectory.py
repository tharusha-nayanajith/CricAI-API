from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.models.artifacts import VideoArtifacts
from app.modules.bowler_performance.camera import unproject_to_ground, unproject_to_plane_z
from app.modules.bowler_performance.pitch_coordinates import (
    BATTING_STUMP_Z_METRES,
    PitchFrame,
    world_to_pitch,
)
from app.modules.preprocessor.models import BallDetection

BOWLING_CREASE_Z_METRES = 8.84
POST_STUMPS_Z_EXTENSION_METRES = 2.0
POST_BOUNCE_EXTENSION_FRAMES = 6
MAX_LATERAL_METRES = 2.5
MIN_RELEASE_HEIGHT_METRES = 1.2
MAX_RELEASE_HEIGHT_METRES = 2.6
MIN_TARGET_HEIGHT_METRES = 0.2
MAX_TARGET_HEIGHT_METRES = 1.8


@dataclass(slots=True)
class AnchorTrajectory:
    frame_values: np.ndarray
    world_points: np.ndarray
    pitch_points: np.ndarray
    release_anchor: np.ndarray
    bounce_anchor: np.ndarray
    target_anchor: np.ndarray


def build_anchor_trajectory(
    artifacts: VideoArtifacts,
    detections: list[BallDetection],
    bounce_frame: int | None,
    K: np.ndarray,
    RT: np.ndarray,
    pitch_frame: PitchFrame,
) -> AnchorTrajectory | None:
    if artifacts.release_point is None or not detections or bounce_frame is None:
        return None

    release_anchor = unproject_to_plane_z(
        artifacts.release_point.hand_position[0],
        artifacts.release_point.hand_position[1],
        K,
        RT,
        BOWLING_CREASE_Z_METRES,
    )
    if release_anchor is None:
        return None
    release_anchor = _clamp_anchor(
        release_anchor,
        min_height=MIN_RELEASE_HEIGHT_METRES,
        max_height=MAX_RELEASE_HEIGHT_METRES,
    )

    bounce_detection = min(
        detections,
        key=lambda detection: abs(detection.frame_idx - bounce_frame),
    )
    bounce_anchor = unproject_to_ground(
        bounce_detection.x,
        bounce_detection.y,
        K,
        RT,
    )
    if bounce_anchor is None:
        return None
    bounce_anchor = _clamp_anchor(bounce_anchor, min_height=0.0, max_height=0.0)

    post_bounce_detections = [
        detection
        for detection in detections
        if detection.frame_idx >= bounce_frame
    ]
    if not post_bounce_detections:
        return None

    target_frame = (
        int(post_bounce_detections[-1].frame_idx) + POST_BOUNCE_EXTENSION_FRAMES
    )
    target_pixel = _extrapolated_pixel(post_bounce_detections, target_frame)
    target_anchor = unproject_to_plane_z(
        target_pixel[0],
        target_pixel[1],
        K,
        RT,
        BATTING_STUMP_Z_METRES - POST_STUMPS_Z_EXTENSION_METRES,
    )
    if target_anchor is None:
        return None
    target_anchor = _clamp_anchor(
        target_anchor,
        min_height=MIN_TARGET_HEIGHT_METRES,
        max_height=MAX_TARGET_HEIGHT_METRES,
    )

    release_frame = int(artifacts.release_point.frame_idx)
    ordered_detections = _dedupe_detections_by_frame(detections)
    ordered_detections = [
        detection
        for detection in ordered_detections
        if detection.frame_idx > release_frame
    ]
    if not ordered_detections:
        return None

    end_frame = target_frame
    if release_frame >= bounce_frame or bounce_frame >= end_frame:
        return None

    frame_values = np.arange(release_frame, end_frame + 1, dtype=np.float64)
    if frame_values.size < 3:
        return None

    sampled_pixels = _interpolated_pixels(
        release_frame,
        artifacts.release_point.hand_position,
        ordered_detections,
        target_frame,
        target_pixel,
        frame_values,
    )
    world_points = np.asarray(
        [
            _piecewise_projected_point(
                frame_idx=float(frame_value),
                pixel_x=float(pixel[0]),
                pixel_y=float(pixel[1]),
                release_frame=release_frame,
                bounce_frame=bounce_frame,
                end_frame=end_frame,
                release_anchor=release_anchor,
                bounce_anchor=bounce_anchor,
                target_anchor=target_anchor,
                K=K,
                RT=RT,
            )
            for frame_value, pixel in zip(frame_values, sampled_pixels, strict=True)
        ],
        dtype=np.float64,
    )
    bounce_index = int(bounce_frame - release_frame)
    world_points[0] = release_anchor
    world_points[bounce_index] = bounce_anchor
    world_points[-1] = target_anchor
    world_points = _smooth_piecewise_world_points(
        world_points,
        bounce_index,
        release_anchor,
        bounce_anchor,
        target_anchor,
    )
    pitch_points = np.asarray(
        [world_to_pitch(point, pitch_frame) for point in world_points],
        dtype=np.float64,
    )
    return AnchorTrajectory(
        frame_values=frame_values,
        world_points=world_points,
        pitch_points=pitch_points,
        release_anchor=release_anchor,
        bounce_anchor=bounce_anchor,
        target_anchor=target_anchor,
    )


def _dedupe_detections_by_frame(detections: list[BallDetection]) -> list[BallDetection]:
    best_by_frame: dict[int, BallDetection] = {}
    for detection in detections:
        previous = best_by_frame.get(detection.frame_idx)
        if previous is None or detection.confidence > previous.confidence:
            best_by_frame[detection.frame_idx] = detection
    return [best_by_frame[frame_idx] for frame_idx in sorted(best_by_frame)]


def _interpolated_pixels(
    release_frame: int,
    release_pixel: tuple[float, float],
    detections: list[BallDetection],
    target_frame: int,
    target_pixel: tuple[float, float],
    frame_values: np.ndarray,
) -> np.ndarray:
    control_frames = [float(release_frame)]
    control_x = [float(release_pixel[0])]
    control_y = [float(release_pixel[1])]

    for detection in detections:
        control_frames.append(float(detection.frame_idx))
        control_x.append(float(detection.x))
        control_y.append(float(detection.y))

    control_frames.append(float(target_frame))
    control_x.append(float(target_pixel[0]))
    control_y.append(float(target_pixel[1]))

    interpolated_x = np.interp(frame_values, control_frames, control_x)
    interpolated_y = np.interp(frame_values, control_frames, control_y)
    return np.column_stack([interpolated_x, interpolated_y]).astype(np.float64)


def _extrapolated_pixel(
    detections: list[BallDetection],
    target_frame: int,
) -> tuple[float, float]:
    sample = detections[-min(4, len(detections)) :]
    if len(sample) == 1:
        return float(sample[0].x), float(sample[0].y)

    frame_values = np.asarray(
        [float(detection.frame_idx) for detection in sample],
        dtype=np.float64,
    )
    x_values = np.asarray([float(detection.x) for detection in sample], dtype=np.float64)
    y_values = np.asarray([float(detection.y) for detection in sample], dtype=np.float64)
    coeff_x = np.polyfit(frame_values, x_values, deg=1)
    coeff_y = np.polyfit(frame_values, y_values, deg=1)
    target_x = float(np.polyval(coeff_x, float(target_frame)))
    target_y = float(np.polyval(coeff_y, float(target_frame)))
    return target_x, target_y


def _piecewise_projected_point(
    *,
    frame_idx: float,
    pixel_x: float,
    pixel_y: float,
    release_frame: int,
    bounce_frame: int,
    end_frame: int,
    release_anchor: np.ndarray,
    bounce_anchor: np.ndarray,
    target_anchor: np.ndarray,
    K: np.ndarray,
    RT: np.ndarray,
) -> np.ndarray:
    if frame_idx <= float(bounce_frame):
        if bounce_frame == release_frame:
            t_value = 1.0
        else:
            t_value = (frame_idx - float(release_frame)) / float(bounce_frame - release_frame)
        target_z = _lerp_scalar(release_anchor[2], bounce_anchor[2], t_value)
        projected = unproject_to_plane_z(pixel_x, pixel_y, K, RT, target_z)
        if projected is None:
            projected = _lerp_point(release_anchor, bounce_anchor, t_value)
        return _clamp_projected_point(
            projected,
            min_height=0.0,
            max_height=MAX_RELEASE_HEIGHT_METRES,
        )

    if end_frame == bounce_frame:
        t_value = 1.0
    else:
        t_value = (frame_idx - float(bounce_frame)) / float(end_frame - bounce_frame)
    target_z = _lerp_scalar(bounce_anchor[2], target_anchor[2], t_value)
    projected = unproject_to_plane_z(pixel_x, pixel_y, K, RT, target_z)
    if projected is None:
        projected = _lerp_point(bounce_anchor, target_anchor, t_value)
    return _clamp_projected_point(
        projected,
        min_height=0.0,
        max_height=max(MAX_TARGET_HEIGHT_METRES, float(target_anchor[1]) + 0.3),
    )


def _lerp_scalar(start: float, end: float, t_value: float) -> float:
    return float((1.0 - t_value) * start + t_value * end)


def _lerp_point(start: np.ndarray, end: np.ndarray, t_value: float) -> np.ndarray:
    return ((1.0 - t_value) * start + t_value * end).astype(np.float64)


def _clamp_anchor(
    point: np.ndarray,
    *,
    min_height: float,
    max_height: float,
) -> np.ndarray:
    clamped = np.asarray(point, dtype=np.float64).copy()
    clamped[0] = float(np.clip(clamped[0], -MAX_LATERAL_METRES, MAX_LATERAL_METRES))
    clamped[1] = float(np.clip(clamped[1], min_height, max_height))
    return clamped


def _clamp_projected_point(
    point: np.ndarray,
    *,
    min_height: float,
    max_height: float,
) -> np.ndarray:
    clamped = np.asarray(point, dtype=np.float64).copy()
    clamped[0] = float(np.clip(clamped[0], -MAX_LATERAL_METRES, MAX_LATERAL_METRES))
    clamped[1] = float(np.clip(clamped[1], min_height, max_height))
    return clamped


def _smooth_piecewise_world_points(
    world_points: np.ndarray,
    bounce_index: int,
    release_anchor: np.ndarray,
    bounce_anchor: np.ndarray,
    target_anchor: np.ndarray,
) -> np.ndarray:
    smoothed = np.asarray(world_points, dtype=np.float64).copy()
    pre_segment = _fit_quadratic_bezier_segment(
        smoothed[: bounce_index + 1],
        release_anchor,
        bounce_anchor,
    )
    post_segment = _fit_quadratic_bezier_segment(
        smoothed[bounce_index:],
        bounce_anchor,
        target_anchor,
    )
    smoothed[: bounce_index + 1] = pre_segment
    smoothed[bounce_index:] = post_segment
    return smoothed


def _fit_quadratic_bezier_segment(
    observed_points: np.ndarray,
    start_point: np.ndarray,
    end_point: np.ndarray,
) -> np.ndarray:
    point_count = len(observed_points)
    if point_count <= 2:
        return np.asarray(observed_points, dtype=np.float64)

    t_values = np.linspace(0.0, 1.0, point_count, dtype=np.float64)
    coeff = (2.0 * (1.0 - t_values) * t_values).reshape(-1, 1)
    base = (
        ((1.0 - t_values) ** 2).reshape(-1, 1) * start_point.reshape(1, 3)
        + (t_values**2).reshape(-1, 1) * end_point.reshape(1, 3)
    )

    valid_mask = coeff[:, 0] > 1e-6
    if np.count_nonzero(valid_mask) < 1:
        return np.asarray(observed_points, dtype=np.float64)

    control = np.empty(3, dtype=np.float64)
    for axis in range(3):
        numerator = observed_points[valid_mask, axis] - base[valid_mask, axis]
        denominator = coeff[valid_mask, 0]
        control[axis] = float(np.mean(numerator / denominator))

    bezier = (
        ((1.0 - t_values) ** 2).reshape(-1, 1) * start_point.reshape(1, 3)
        + coeff * control.reshape(1, 3)
        + (t_values**2).reshape(-1, 1) * end_point.reshape(1, 3)
    )
    bezier[0] = start_point
    bezier[-1] = end_point
    return bezier.astype(np.float64)
