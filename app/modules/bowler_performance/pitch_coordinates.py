from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.models.calibration import CalibrationData
from app.modules.preprocessor.models import BallDetection

PITCH_LENGTH_METRES = 20.118
PITCH_WIDTH_METRES = 3.05
BATTING_BASE_CHANNELS = (0, 2, 4)
BOWLING_BASE_CHANNELS = (6, 8, 10)
MIN_PITCH_KEYPOINTS = 2

WorldPoint = tuple[BallDetection, np.ndarray]
PitchPoint = tuple[BallDetection, np.ndarray]


@dataclass(slots=True)
class PitchFrame:
    batting_origin_world: np.ndarray
    x_axis_world: np.ndarray
    z_axis_world: np.ndarray
    scale: float
    measured_pitch_length: float


def build_pitch_frame(
    calibration: CalibrationData,
    K: np.ndarray,
    RT: np.ndarray,
) -> PitchFrame | None:
    from app.modules.bowler_performance.camera import unproject_to_ground

    batting_points = _unproject_channels(
        calibration,
        BATTING_BASE_CHANNELS,
        K,
        RT,
        unproject_to_ground,
    )
    bowling_points = _unproject_channels(
        calibration,
        BOWLING_BASE_CHANNELS,
        K,
        RT,
        unproject_to_ground,
    )
    if len(batting_points) < MIN_PITCH_KEYPOINTS or len(bowling_points) < MIN_PITCH_KEYPOINTS:
        batting_center = (
            np.mean(np.asarray(batting_points, dtype=np.float64), axis=0)
            if len(batting_points) >= MIN_PITCH_KEYPOINTS
            else None
        )
        bowling_center = (
            np.mean(np.asarray(bowling_points, dtype=np.float64), axis=0)
            if len(bowling_points) >= MIN_PITCH_KEYPOINTS
            else None
        )
    else:
        batting_center = np.mean(np.asarray(batting_points, dtype=np.float64), axis=0)
        bowling_center = np.mean(np.asarray(bowling_points, dtype=np.float64), axis=0)

    if batting_center is None and bowling_center is None:
        return None

    camera_ground = np.array(
        [calibration.position[0], 0.0, calibration.position[2]],
        dtype=np.float64,
    )
    if batting_center is None:
        return None
    if bowling_center is None:
        bowling_center = camera_ground

    raw_z_axis = bowling_center - batting_center
    raw_z_axis[1] = 0.0
    measured_pitch_length = float(np.linalg.norm(raw_z_axis))
    if measured_pitch_length < 1e-6:
        return None
    z_axis_world = raw_z_axis / measured_pitch_length

    raw_x_axis = _estimate_lateral_axis(batting_points, bowling_points)
    if raw_x_axis is None:
        return None

    x_axis_world = raw_x_axis - np.dot(raw_x_axis, z_axis_world) * z_axis_world
    x_axis_world[1] = 0.0
    x_norm = float(np.linalg.norm(x_axis_world))
    if x_norm < 1e-6:
        return None
    x_axis_world /= x_norm

    return PitchFrame(
        batting_origin_world=batting_center,
        x_axis_world=x_axis_world,
        z_axis_world=z_axis_world,
        scale=PITCH_LENGTH_METRES / measured_pitch_length,
        measured_pitch_length=measured_pitch_length,
    )


def world_to_pitch(point_world: np.ndarray, frame: PitchFrame) -> np.ndarray:
    relative = np.asarray(point_world, dtype=np.float64) - frame.batting_origin_world
    pitch_x = float(np.dot(relative, frame.x_axis_world) * frame.scale)
    pitch_z = float(np.dot(relative, frame.z_axis_world) * frame.scale)
    return np.array([pitch_x, 0.0, pitch_z], dtype=np.float64)


def world_points_to_pitch_points(
    world_points: list[WorldPoint],
    frame: PitchFrame,
) -> list[PitchPoint]:
    return [
        (detection, world_to_pitch(world_point, frame))
        for detection, world_point in world_points
    ]


def _unproject_channels(
    calibration: CalibrationData,
    channels: tuple[int, ...],
    K: np.ndarray,
    RT: np.ndarray,
    unproject_fn: Callable[[float, float, np.ndarray, np.ndarray], np.ndarray | None],
) -> list[np.ndarray]:
    keypoints = [
        keypoint
        for keypoint in calibration.keypoints
        if keypoint.channel_index in channels
    ]
    world_points: list[np.ndarray] = []
    for keypoint in keypoints:
        point_world = unproject_fn(keypoint.x, keypoint.y, K, RT)
        if point_world is not None:
            world_points.append(np.asarray(point_world, dtype=np.float64))
    return world_points


def _estimate_lateral_axis(
    batting_points: list[np.ndarray],
    bowling_points: list[np.ndarray],
) -> np.ndarray | None:
    candidates: list[np.ndarray] = []

    if len(batting_points) >= 2:
        candidates.append(batting_points[-1] - batting_points[0])
    if len(bowling_points) >= 2:
        candidates.append(bowling_points[-1] - bowling_points[0])
    if not candidates:
        return None

    lateral = np.mean(np.asarray(candidates, dtype=np.float64), axis=0)
    lateral[1] = 0.0
    if float(np.linalg.norm(lateral)) < 1e-6:
        return None
    return lateral
