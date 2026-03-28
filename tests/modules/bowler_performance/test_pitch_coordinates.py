import numpy as np

from app.models.calibration import CalibrationData, Keypoint
from app.modules.bowler_performance.pitch_coordinates import (
    PITCH_LENGTH_METRES,
    build_pitch_frame,
    world_points_to_pitch_points,
)
from app.modules.preprocessor.models import BallDetection


def _calibration() -> CalibrationData:
    return CalibrationData(
        image_size=(720, 1280),
        fov=40.0,
        yaw=0.0,
        position=(0.0, 1.6, 15.0),
        principal_point=(360.0, 640.0),
        rotation=(0.0, 0.0, 0.0),
        score=1.0,
        detected_channels=6,
        total_detections=6,
        keypoints=[
            Keypoint(x=300.0, y=800.0, score=0.9, channel_index=0),
            Keypoint(x=320.0, y=800.0, score=0.9, channel_index=2),
            Keypoint(x=340.0, y=800.0, score=0.9, channel_index=4),
            Keypoint(x=300.0, y=200.0, score=0.9, channel_index=6),
            Keypoint(x=320.0, y=200.0, score=0.9, channel_index=8),
            Keypoint(x=340.0, y=200.0, score=0.9, channel_index=10),
        ],
    )


def test_build_pitch_frame_returns_standardized_axes(monkeypatch) -> None:
    calibration = _calibration()

    lookup = {
        (300.0, 800.0): np.array([-0.1, 0.0, 0.0], dtype=np.float64),
        (320.0, 800.0): np.array([0.0, 0.0, 0.0], dtype=np.float64),
        (340.0, 800.0): np.array([0.1, 0.0, 0.0], dtype=np.float64),
        (300.0, 200.0): np.array([-0.1, 0.0, 10.0], dtype=np.float64),
        (320.0, 200.0): np.array([0.0, 0.0, 10.0], dtype=np.float64),
        (340.0, 200.0): np.array([0.1, 0.0, 10.0], dtype=np.float64),
    }

    monkeypatch.setattr(
        "app.modules.bowler_performance.camera.unproject_to_ground",
        lambda x_val, y_val, K, RT: lookup[(x_val, y_val)],
    )

    frame = build_pitch_frame(calibration, np.eye(3), np.hstack([np.eye(3), np.zeros((3, 1))]))

    assert frame is not None
    assert frame.measured_pitch_length == 10.0
    assert frame.scale == PITCH_LENGTH_METRES / 10.0
    assert frame.batting_origin_world[2] == 0.0


def test_world_points_to_pitch_points_projects_along_pitch_axis() -> None:
    from app.modules.bowler_performance.pitch_coordinates import PitchFrame

    frame = PitchFrame(
        batting_origin_world=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        x_axis_world=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        z_axis_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        scale=2.0,
        measured_pitch_length=10.0,
    )
    detection = BallDetection(frame_idx=1, timestamp_s=0.1, x=0.0, y=0.0, confidence=0.9)
    world_points = [(detection, np.array([0.5, 0.0, 3.0], dtype=np.float64))]

    pitch_points = world_points_to_pitch_points(world_points, frame)

    assert pitch_points[0][1][0] == 1.0
    assert pitch_points[0][1][2] == 6.0
