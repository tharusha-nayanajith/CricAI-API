from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from loguru import logger

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
    lateral_sign = _lateral_axis_sign(calibration)

    return PitchFrame(
        batting_origin_world=np.array(
            [0.0, 0.0, BATTING_STUMP_Z_METRES],
            dtype=np.float64,
        ),
        x_axis_world=np.array([lateral_sign, 0.0, 0.0], dtype=np.float64),
        z_axis_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        scale=1.0,
        measured_pitch_length=measured_pitch_length,
        measured_batting_center_world=measured_batting_center,
        measured_bowling_center_world=measured_bowling_center,
        length_reliable=length_reliable,
    )


def world_to_pitch(point_world: np.ndarray, frame: PitchFrame) -> np.ndarray:
    relative = np.asarray(point_world, dtype=np.float64) - frame.batting_origin_world
    # FullTrack's canonical world frame already defines x < 0 as off side and
    # x > 0 as leg side, so preserve the lateral sign directly.
    pitch_x = float(np.dot(relative, frame.x_axis_world))
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


def _lateral_axis_sign(calibration: CalibrationData) -> float:
    batting_debug = _channel_order_debug(
        calibration,
        off_channels=(0, 1),
        leg_channels=(4, 5),
    )
    batting_sign = batting_debug["sign"]
    if batting_sign is not None:
        logger.info(
            "Pitch-frame sign debug source=batting rotation_z={} "
            "off_channels={} leg_channels={} off_x={} leg_x={} delta={} sign={}",
            float(calibration.rotation[2]),
            batting_debug["off_channels"],
            batting_debug["leg_channels"],
            batting_debug["off_x_values"],
            batting_debug["leg_x_values"],
            batting_debug["delta"],
            batting_sign,
        )
        return batting_sign

    bowling_debug = _channel_order_debug(
        calibration,
        off_channels=(6, 7),
        leg_channels=(10, 11),
    )
    bowling_sign = bowling_debug["sign"]
    if bowling_sign is not None:
        logger.info(
            "Pitch-frame sign debug source=bowling rotation_z={} "
            "off_channels={} leg_channels={} off_x={} leg_x={} delta={} sign={}",
            float(calibration.rotation[2]),
            bowling_debug["off_channels"],
            bowling_debug["leg_channels"],
            bowling_debug["off_x_values"],
            bowling_debug["leg_x_values"],
            bowling_debug["delta"],
            bowling_sign,
        )
        return bowling_sign

    logger.info(
        "Pitch-frame sign debug source=default rotation_z={} "
        "batting_debug={} bowling_debug={} sign=1.0",
        float(calibration.rotation[2]),
        batting_debug,
        bowling_debug,
    )
    return 1.0


def _signed_channel_order(
    calibration: CalibrationData,
    *,
    off_channels: tuple[int, ...],
    leg_channels: tuple[int, ...],
) -> float | None:
    debug = _channel_order_debug(
        calibration,
        off_channels=off_channels,
        leg_channels=leg_channels,
    )
    sign = debug["sign"]
    return float(sign) if sign is not None else None


def _channel_order_debug(
    calibration: CalibrationData,
    *,
    off_channels: tuple[int, ...],
    leg_channels: tuple[int, ...],
) -> dict[str, object]:
    off_x_values = _channel_x_values(calibration, off_channels)
    leg_x_values = _channel_x_values(calibration, leg_channels)
    off_mean = float(np.mean(np.asarray(off_x_values, dtype=np.float64))) if off_x_values else None
    leg_mean = float(np.mean(np.asarray(leg_x_values, dtype=np.float64))) if leg_x_values else None
    delta = (leg_mean - off_mean) if off_mean is not None and leg_mean is not None else None
    sign = None
    if delta is not None and abs(delta) >= 1e-6:
        sign = 1.0 if delta > 0.0 else -1.0
    return {
        "off_channels": list(off_channels),
        "leg_channels": list(leg_channels),
        "off_x_values": off_x_values,
        "leg_x_values": leg_x_values,
        "off_mean": off_mean,
        "leg_mean": leg_mean,
        "delta": delta,
        "sign": sign,
    }


def _mean_channel_x(
    calibration: CalibrationData,
    channels: tuple[int, ...],
) -> float | None:
    x_values = _channel_x_values(calibration, channels)
    if not x_values:
        return None
    return float(np.mean(np.asarray(x_values, dtype=np.float64)))


def _channel_x_values(
    calibration: CalibrationData,
    channels: tuple[int, ...],
) -> list[float]:
    best_by_channel: dict[int, tuple[float, float]] = {}
    for keypoint in calibration.keypoints:
        channel_index = keypoint.channel_index
        if channel_index not in channels:
            continue
        candidate = (float(keypoint.score), float(keypoint.x))
        previous = best_by_channel.get(channel_index)
        if previous is None or candidate[0] > previous[0]:
            best_by_channel[channel_index] = candidate

    return [
        best_by_channel[channel_index][1]
        for channel_index in channels
        if channel_index in best_by_channel
    ]
