import numpy as np
import pytest

from app.modules.bowler_performance.camera import (
    _euler_to_rotation,
    build_extrinsic_matrix,
    build_intrinsic_matrix,
    pixels_to_world_points,
    unproject_to_ground,
)
from app.modules.preprocessor.models import BallDetection
from tests.conftest import CalibrationDataFactory


def _camera_rt(camera_position: tuple[float, float, float]) -> np.ndarray:
    rotation = np.eye(3, dtype=np.float64)
    camera = np.array(camera_position, dtype=np.float64)
    translation = (-rotation @ camera).reshape(3, 1)
    return np.hstack([rotation, translation])


def test_euler_to_rotation_returns_identity_for_zero_vector() -> None:
    rotation = _euler_to_rotation(np.zeros(3, dtype=np.float64))

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    assert np.allclose(rotation, expected)


def test_euler_to_rotation_returns_expected_matrix_for_known_rotation() -> None:
    rotation = _euler_to_rotation(np.array([0.0, 0.0, np.pi / 2.0], dtype=np.float64))
    expected = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    assert np.allclose(rotation, expected, atol=1e-9)


def test_build_intrinsic_matrix_uses_image_height_and_fov() -> None:
    calibration = CalibrationDataFactory().model_copy(
        update={
            "image_size": (1920, 1080),
            "fov": 90.0,
            "principal_point": (960.0, 540.0),
        }
    )

    intrinsic = build_intrinsic_matrix(calibration)

    assert intrinsic[0, 0] == pytest.approx(540.0)
    assert intrinsic[1, 1] == pytest.approx(540.0)
    assert intrinsic[0, 2] == pytest.approx(960.0)
    assert intrinsic[1, 2] == pytest.approx(540.0)


def test_build_extrinsic_matrix_uses_euler_rotation_convention() -> None:
    calibration = CalibrationDataFactory().model_copy(
        update={
            "rotation": (0.0, 0.0, np.pi / 2.0),
            "position": (1.0, 2.0, 3.0),
        }
    )

    extrinsic = build_extrinsic_matrix(calibration)

    expected_rotation = _euler_to_rotation(np.array([0.0, 0.0, np.pi / 2.0], dtype=np.float64))
    expected_translation = -expected_rotation @ np.array([1.0, 2.0, 3.0], dtype=np.float64)

    assert np.allclose(extrinsic[:, :3], expected_rotation)
    assert np.allclose(extrinsic[:, 3], expected_translation)


def test_unproject_to_ground_returns_none_for_parallel_ray() -> None:
    intrinsic = np.eye(3, dtype=np.float64)
    extrinsic = _camera_rt((0.0, 1.0, 0.0))

    world_point = unproject_to_ground(1.0, 0.0, intrinsic, extrinsic)

    assert world_point is None


def test_unproject_to_ground_returns_none_for_intersection_behind_camera() -> None:
    intrinsic = np.eye(3, dtype=np.float64)
    extrinsic = _camera_rt((0.0, 1.0, 0.0))

    world_point = unproject_to_ground(0.0, 1.0, intrinsic, extrinsic)

    assert world_point is None


def test_unproject_to_ground_returns_point_on_ground_plane_for_valid_ray() -> None:
    intrinsic = np.eye(3, dtype=np.float64)
    extrinsic = _camera_rt((0.0, 1.0, 0.0))

    world_point = unproject_to_ground(0.0, -1.0, intrinsic, extrinsic)

    assert world_point is not None
    assert world_point[1] == pytest.approx(0.0, abs=1e-9)


def test_pixels_to_world_points_skips_invalid_intersections() -> None:
    intrinsic = np.eye(3, dtype=np.float64)
    extrinsic = _camera_rt((0.0, 1.0, 0.0))
    detections = [
        BallDetection(frame_idx=1, timestamp_s=0.1, x=1.0, y=0.0, confidence=0.9),
        BallDetection(frame_idx=2, timestamp_s=0.2, x=0.0, y=1.0, confidence=0.9),
        BallDetection(frame_idx=3, timestamp_s=0.3, x=0.0, y=-1.0, confidence=0.9),
    ]

    world_points = pixels_to_world_points(detections, intrinsic, extrinsic)

    assert len(world_points) == 1
    assert world_points[0][0].frame_idx == 3
    assert world_points[0][1][1] == pytest.approx(0.0, abs=1e-9)
