import numpy as np
import pytest

from app.modules.bowler_performance.metrics import (
    build_result,
    compute_bounce_and_length,
    compute_proxy_speed,
    compute_speed,
    compute_swing,
)
from app.modules.bowler_performance.models import BouncePoint, LengthClass, classify_length
from app.modules.bowler_performance.pitch_coordinates import BOWLING_STUMP_Z_METRES
from app.modules.preprocessor.models import BallDetection


def _world_point(
    frame_idx: int,
    timestamp_s: float,
    world_xyz: tuple[float, float, float],
    confidence: float = 0.9,
) -> tuple[BallDetection, np.ndarray]:
    detection = BallDetection(
        frame_idx=frame_idx,
        timestamp_s=timestamp_s,
        x=world_xyz[0],
        y=world_xyz[1],
        confidence=confidence,
    )
    return detection, np.array(world_xyz, dtype=np.float64)


def test_compute_speed_returns_expected_kmh_for_known_distances() -> None:
    world_points = [
        _world_point(0, 0.0, (0.0, 0.0, 0.0)),
        _world_point(1, 0.1, (2.0, 0.0, 0.0)),
        _world_point(2, 0.2, (4.0, 0.0, 0.0)),
        _world_point(3, 0.3, (6.0, 0.0, 0.0)),
    ]

    speed_ms, speed_kmh = compute_speed(world_points)

    assert speed_ms == pytest.approx(20.0)
    assert speed_kmh == pytest.approx(72.0)


def test_compute_speed_uses_median_instead_of_outlier_mean() -> None:
    world_points = [
        _world_point(0, 0.0, (0.0, 0.0, 0.0)),
        _world_point(1, 1.0, (1.0, 0.0, 0.0)),
        _world_point(2, 2.0, (2.0, 0.0, 0.0)),
        _world_point(3, 3.0, (102.0, 0.0, 0.0)),
    ]

    speed_ms, speed_kmh = compute_speed(world_points)

    assert speed_ms == pytest.approx(1.0)
    assert speed_kmh == pytest.approx(3.6)


def test_compute_proxy_speed_uses_release_to_bounce_timing() -> None:
    detections = [
        BallDetection(frame_idx=0, timestamp_s=0.0, x=10.0, y=10.0, confidence=0.8),
        BallDetection(frame_idx=6, timestamp_s=0.5, x=20.0, y=20.0, confidence=0.9),
    ]

    speed_ms, speed_kmh = compute_proxy_speed(
        detections,
        bounce_frame=6,
        release_timestamp_s=0.0,
        bounce_point=BouncePoint(x_metres=0.1, z_metres=6.0),
    )

    assert speed_ms == pytest.approx((BOWLING_STUMP_Z_METRES - 6.0 + 1.5) / 0.5)
    assert speed_kmh == pytest.approx(speed_ms * 3.6)


def test_compute_proxy_speed_falls_back_to_default_distance_when_bounce_depth_invalid() -> None:
    detections = [
        BallDetection(frame_idx=2, timestamp_s=0.2, x=10.0, y=10.0, confidence=0.8),
        BallDetection(frame_idx=8, timestamp_s=0.6, x=20.0, y=20.0, confidence=0.9),
    ]

    speed_ms, speed_kmh = compute_proxy_speed(
        detections,
        bounce_frame=8,
        release_timestamp_s=0.1,
        bounce_point=BouncePoint(x_metres=-0.5, z_metres=-12.0),
    )

    assert speed_ms == pytest.approx(16.0 / 0.5)
    assert speed_kmh == pytest.approx(speed_ms * 3.6)


def test_compute_swing_returns_positive_value_for_positive_x_motion() -> None:
    world_points = [
        _world_point(0, 0.0, (0.0, 0.0, 0.0)),
        _world_point(1, 0.1, (0.5, 0.0, 1.0)),
        _world_point(2, 0.2, (1.5, 0.0, 2.0)),
    ]

    swing = compute_swing(world_points, bounce_frame=None)

    assert swing == pytest.approx(1.5)


def test_compute_swing_returns_negative_value_for_negative_x_motion() -> None:
    world_points = [
        _world_point(0, 0.0, (0.0, 0.0, 0.0)),
        _world_point(1, 0.1, (-0.5, 0.0, 1.0)),
        _world_point(2, 0.2, (-1.2, 0.0, 2.0)),
    ]

    swing = compute_swing(world_points, bounce_frame=None)

    assert swing == pytest.approx(-1.2)


def test_compute_bounce_and_length_returns_none_when_no_bounce_frame() -> None:
    world_points = [_world_point(0, 0.0, (0.0, 0.0, 0.0))]

    bounce_point, length_class = compute_bounce_and_length(world_points, bounce_frame=None)

    assert bounce_point is None
    assert length_class is None


def test_classify_length_returns_expected_bands_for_boundaries() -> None:
    assert classify_length(1.9) is LengthClass.YORKER
    assert classify_length(2.0) is LengthClass.YORKER
    assert classify_length(2.1) is LengthClass.FULL
    assert classify_length(3.9) is LengthClass.FULL
    assert classify_length(4.0) is LengthClass.FULL
    assert classify_length(4.1) is LengthClass.GOOD_LENGTH
    assert classify_length(6.9) is LengthClass.GOOD_LENGTH
    assert classify_length(7.0) is LengthClass.GOOD_LENGTH
    assert classify_length(7.1) is LengthClass.SHORT_OF_LENGTH
    assert classify_length(8.9) is LengthClass.SHORT_OF_LENGTH
    assert classify_length(9.0) is LengthClass.SHORT_OF_LENGTH
    assert classify_length(9.1) is LengthClass.SHORT


def test_build_result_uses_canonical_bounce_point_when_provided() -> None:
    detections = [
        BallDetection(frame_idx=0, timestamp_s=0.0, x=0.0, y=0.0, confidence=0.8),
        BallDetection(frame_idx=5, timestamp_s=0.5, x=1.0, y=1.0, confidence=0.9),
    ]
    world_points = [_world_point(0, 0.0, (0.0, 1.8, 15.0)), _world_point(5, 0.5, (0.1, 0.2, 11.0))]
    pitch_points = [
        (detections[0], np.array([0.0, 0.0, 8.0], dtype=np.float64)),
        (detections[1], np.array([0.1, 0.0, 6.0], dtype=np.float64)),
    ]

    result = build_result(
        world_points,
        pitch_points,
        detections,
        bounce_frame=5,
        release_timestamp_s=0.0,
        canonical_bounce_point=BouncePoint(x_metres=-0.25, z_metres=2.5),
    )

    assert result.bounce_point == BouncePoint(x_metres=-0.25, z_metres=2.5)
    assert result.length_class is LengthClass.FULL
