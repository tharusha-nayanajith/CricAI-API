import numpy as np
import pytest

from app.modules.bowler_performance.metrics import (
    compute_bounce_and_length,
    compute_speed,
    compute_swing,
)
from app.modules.bowler_performance.models import LengthClass, classify_length
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
