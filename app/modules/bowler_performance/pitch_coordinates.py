from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.models.calibration import CalibrationData
from app.modules.preprocessor.models import BallDetection

STUMP_HALF_WIDTH_METRES = 0.0954
STUMP_HEIGHT_METRES = 0.711
BATTING_STUMP_Z_METRES = -10.059
BOWLING_STUMP_Z_METRES = 10.059
PITCH_LENGTH_METRES = BOWLING_STUMP_Z_METRES - BATTING_STUMP_Z_METRES
PITCH_WIDTH_METRES = 3.05
PITCH_LENGTH_TOLERANCE_METRES = 2.0

STUMP_WORLD_BY_CHANNEL: dict[int, tuple[float, float, float]] = {
    0: (-STUMP_HALF_WIDTH_METRES, 0.0, BATTING_STUMP_Z_METRES),
    1: (-STUMP_HALF_WIDTH_METRES, STUMP_HEIGHT_METRES, BATTING_STUMP_Z_METRES),
    2: (0.0, 0.0, BATTING_STUMP_Z_METRES),
    3: (0.0, STUMP_HEIGHT_METRES, BATTING_STUMP_Z_METRES),
    4: (STUMP_HALF_WIDTH_METRES, 0.0, BATTING_STUMP_Z_METRES),
    5: (STUMP_HALF_WIDTH_METRES, STUMP_HEIGHT_METRES, BATTING_STUMP_Z_METRES),
    6: (-STUMP_HALF_WIDTH_METRES, 0.0, BOWLING_STUMP_Z_METRES),
    7: (-STUMP_HALF_WIDTH_METRES, STUMP_HEIGHT_METRES, BOWLING_STUMP_Z_METRES),
    8: (0.0, 0.0, BOWLING_STUMP_Z_METRES),
    9: (0.0, STUMP_HEIGHT_METRES, BOWLING_STUMP_Z_METRES),
    10: (STUMP_HALF_WIDTH_METRES, 0.0, BOWLING_STUMP_Z_METRES),
    11: (STUMP_HALF_WIDTH_METRES, STUMP_HEIGHT_METRES, BOWLING_STUMP_Z_METRES),
}

BATTING_BASE_CHANNELS = (0, 2, 4)
BOWLING_BASE_CHANNELS = (6, 8, 10)

WorldPoint = tuple[BallDetection, np.ndarray]
PitchPoint = tuple[BallDetection, np.ndarray]


@dataclass(slots=True)
class PitchFrame:
    batting_origin_world: np.ndarray
    x_axis_world: np.ndarray
    z_axis_world: np.ndarray
    scale: float
    measured_pitch_length: float | None
    measured_batting_center_world: np.ndarray | None
    measured_bowling_center_world: np.ndarray | None
    length_reliable: bool


def build_pitch_frame(
    calibration: CalibrationData,
    K: np.ndarray,
    RT: np.ndarray,
) -> PitchFrame:
    from app.modules.bowler_performance.camera import unproject_to_ground

    measured_batting_center = _unproject_center(
        calibration,
        BATTING_BASE_CHANNELS,
        K,
        RT,
        unproject_to_ground,
    )
    measured_bowling_center = _unproject_center(
        calibration,
        BOWLING_BASE_CHANNELS,
        K,
        RT,
        unproject_to_ground,
    )
    measured_pitch_length = _measured_pitch_length(
        measured_batting_center,
        measured_bowling_center,
    )
    length_reliable = measured_pitch_length is not None and (
        abs(measured_pitch_length - PITCH_LENGTH_METRES) <= PITCH_LENGTH_TOLERANCE_METRES
    )

    return PitchFrame(
        batting_origin_world=np.array(
            [0.0, 0.0, BATTING_STUMP_Z_METRES],
            dtype=np.float64,
        ),
        x_axis_world=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        z_axis_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        scale=1.0,
        measured_pitch_length=measured_pitch_length,
        measured_batting_center_world=measured_batting_center,
        measured_bowling_center_world=measured_bowling_center,
        length_reliable=length_reliable,
    )


def world_to_pitch(point_world: np.ndarray, frame: PitchFrame) -> np.ndarray:
    relative = np.asarray(point_world, dtype=np.float64) - frame.batting_origin_world
    # Match the stadium/pitch-map convention so the rendered lateral direction
    # lines up with the real video without a manual left-right flip.
    pitch_x = float(-np.dot(relative, frame.x_axis_world))
    pitch_z = float(np.dot(relative, frame.z_axis_world))
    return np.array([pitch_x, 0.0, pitch_z], dtype=np.float64)


def world_points_to_pitch_points(
    world_points: list[WorldPoint],
    frame: PitchFrame,
) -> list[PitchPoint]:
    return [
        (detection, world_to_pitch(world_point, frame))
        for detection, world_point in world_points
    ]


def _unproject_center(
    calibration: CalibrationData,
    channels: tuple[int, ...],
    K: np.ndarray,
    RT: np.ndarray,
    unproject_fn: Callable[[float, float, np.ndarray, np.ndarray], np.ndarray | None],
) -> np.ndarray | None:
    world_points: list[np.ndarray] = []
    for keypoint in calibration.keypoints:
        if keypoint.channel_index not in channels:
            continue
        point_world = unproject_fn(keypoint.x, keypoint.y, K, RT)
        if point_world is None:
            continue
        world_points.append(np.asarray(point_world, dtype=np.float64))

    if not world_points:
        return None
    return np.mean(np.asarray(world_points, dtype=np.float64), axis=0)


def _measured_pitch_length(
    batting_center: np.ndarray | None,
    bowling_center: np.ndarray | None,
) -> float | None:
    if batting_center is None or bowling_center is None:
        return None

    delta = np.asarray(bowling_center - batting_center, dtype=np.float64)
    delta[1] = 0.0
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return None
    return length
