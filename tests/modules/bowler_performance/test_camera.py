import numpy as np
import pytest

from app.models.calibration import Keypoint
from app.modules.bowler_performance.camera import (
    _euler_to_rotation,
    assess_world_points,
    build_extrinsic_matrix,
    build_intrinsic_matrix,
    decompose_extrinsic_matrix,
    filter_world_point_outliers,
    pixels_to_world_points,
    refine_extrinsic_matrix,
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


def test_decompose_extrinsic_matrix_round_trips_pose() -> None:
    calibration = CalibrationDataFactory().model_copy(
        update={
            "rotation": (0.2, -0.1, 0.3),
            "position": (1.0, 2.5, -3.0),
        }
    )

    extrinsic = build_extrinsic_matrix(calibration)
    position, rotation = decompose_extrinsic_matrix(extrinsic)

    assert np.allclose(position, np.array(calibration.position, dtype=np.float64), atol=1e-6)
    assert np.allclose(rotation, np.array(calibration.rotation, dtype=np.float64), atol=1e-6)


def test_refine_extrinsic_matrix_uses_stump_correspondences_to_correct_pose() -> None:
    true_calibration = CalibrationDataFactory().model_copy(
        update={
            "fov": 45.0,
            "principal_point": (960.0, 540.0),
            "rotation": (0.1, 0.05, 0.02),
            "position": (0.2, 1.6, 0.0),
        }
    )
    intrinsic = build_intrinsic_matrix(true_calibration)
    true_extrinsic = build_extrinsic_matrix(true_calibration)

    stump_world = {
        0: (-0.0954, 0.0, -10.059),
        1: (-0.0954, 0.711, -10.059),
        2: (0.0, 0.0, -10.059),
        3: (0.0, 0.711, -10.059),
        4: (0.0954, 0.0, -10.059),
        5: (0.0954, 0.711, -10.059),
        6: (-0.0954, 0.0, 10.059),
        7: (-0.0954, 0.711, 10.059),
        8: (0.0, 0.0, 10.059),
        9: (0.0, 0.711, 10.059),
        10: (0.0954, 0.0, 10.059),
        11: (0.0954, 0.711, 10.059),
    }

    keypoints = []
    for channel_index, point_world in stump_world.items():
        point = np.array([*point_world, 1.0], dtype=np.float64).reshape(4, 1)
        camera = true_extrinsic @ point
        pixel = intrinsic @ camera
        keypoints.append(
            Keypoint(
                x=float(pixel[0, 0] / pixel[2, 0]),
                y=float(pixel[1, 0] / pixel[2, 0]),
                score=0.99,
                channel_index=channel_index,
            )
        )

    noisy_calibration = true_calibration.model_copy(
        update={
            "rotation": (0.15, 0.0, 0.08),
            "position": (0.5, 1.9, -0.6),
            "keypoints": keypoints,
            "detected_channels": len(keypoints),
            "total_detections": len(keypoints),
        }
    )
    initial_extrinsic = build_extrinsic_matrix(noisy_calibration)

    refined = refine_extrinsic_matrix(noisy_calibration, intrinsic, initial_extrinsic)
    expected_position = np.array(true_calibration.position, dtype=np.float64)
    expected_rotation = np.array(true_calibration.rotation, dtype=np.float64)

    assert refined.refined is True
    assert refined.correspondence_count == len(keypoints)
    assert refined.reprojection_error_px == pytest.approx(0.0, abs=1e-3)
    assert np.allclose(refined.position, expected_position, atol=1e-2)
    assert np.allclose(refined.rotation_euler, expected_rotation, atol=1e-2)


def test_refine_extrinsic_matrix_falls_back_when_too_few_keypoints() -> None:
    calibration = CalibrationDataFactory().model_copy(
        update={
            "keypoints": [
                Keypoint(x=10.0, y=20.0, score=0.9, channel_index=0),
                Keypoint(x=11.0, y=21.0, score=0.9, channel_index=2),
                Keypoint(x=12.0, y=22.0, score=0.9, channel_index=4),
            ],
            "detected_channels": 3,
            "total_detections": 3,
        }
    )
    intrinsic = build_intrinsic_matrix(calibration)
    initial_extrinsic = build_extrinsic_matrix(calibration)

    refined = refine_extrinsic_matrix(calibration, intrinsic, initial_extrinsic)

    assert refined.refined is False
    assert refined.correspondence_count == 3
    assert np.allclose(refined.extrinsic, initial_extrinsic)


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


def test_filter_world_point_outliers_removes_catastrophic_depth_spike() -> None:
    detections = [
        BallDetection(frame_idx=index, timestamp_s=index / 30.0, x=0.0, y=0.0, confidence=0.9)
        for index in range(5)
    ]
    world_points = [
        (detections[0], np.array([0.0, 1.8, 12.0], dtype=np.float64)),
        (detections[1], np.array([0.1, 1.4, 11.2], dtype=np.float64)),
        (detections[2], np.array([0.2, 1.0, -1765.0], dtype=np.float64)),
        (detections[3], np.array([0.3, 0.6, 10.1], dtype=np.float64)),
        (detections[4], np.array([0.35, 0.3, 9.6], dtype=np.float64)),
    ]

    filtered = filter_world_point_outliers(world_points)

    assert [detection.frame_idx for detection, _point in filtered.points] == [0, 1, 3, 4]
    assert filtered.removed_frame_indices == [2]


def test_assess_world_points_flags_ground_collapsed_and_implausible_path() -> None:
    detections = [
        BallDetection(frame_idx=0, timestamp_s=0.0, x=0.0, y=0.0, confidence=0.9),
        BallDetection(frame_idx=1, timestamp_s=1.0, x=1.0, y=1.0, confidence=0.9),
        BallDetection(frame_idx=2, timestamp_s=2.0, x=2.0, y=2.0, confidence=0.9),
    ]
    world_points = [
        (detections[0], np.array([0.0, 0.0, -400.0], dtype=np.float64)),
        (detections[1], np.array([0.1, 0.0, -100.0], dtype=np.float64)),
        (detections[2], np.array([0.2, 0.0, -5.0], dtype=np.float64)),
    ]

    sanity = assess_world_points(world_points)

    assert sanity.all_points_on_ground is True
    assert sanity.implausible_depth_range is True
    assert sanity.trajectory_reliable is False


def test_assess_world_points_accepts_airborne_path_with_plausible_steps() -> None:
    detections = [
        BallDetection(frame_idx=0, timestamp_s=0.0, x=0.0, y=0.0, confidence=0.9),
        BallDetection(frame_idx=1, timestamp_s=1.0, x=1.0, y=1.0, confidence=0.9),
        BallDetection(frame_idx=2, timestamp_s=2.0, x=2.0, y=2.0, confidence=0.9),
    ]
    world_points = [
        (detections[0], np.array([0.0, 1.8, 15.0], dtype=np.float64)),
        (detections[1], np.array([0.2, 1.2, 14.0], dtype=np.float64)),
        (detections[2], np.array([0.3, 0.5, 13.2], dtype=np.float64)),
    ]

    sanity = assess_world_points(world_points)

    assert sanity.all_points_on_ground is False
    assert sanity.implausible_depth_range is False
    assert sanity.implausible_step_jump is False
    assert sanity.trajectory_reliable is True
