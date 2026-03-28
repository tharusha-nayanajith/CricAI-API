import numpy as np

from app.models.artifacts import VideoArtifacts
from app.modules.bowler_performance.trajectory import (
    BOWLING_CREASE_Z_METRES,
    POST_BOUNCE_EXTENSION_FRAMES,
    POST_STUMPS_Z_EXTENSION_METRES,
    build_anchor_trajectory,
)
from app.modules.preprocessor.models import BallDetection, ReleasePoint


def _artifacts() -> VideoArtifacts:
    detections = [
        BallDetection(frame_idx=75, timestamp_s=1.25, x=240.0, y=420.0, confidence=0.9),
        BallDetection(frame_idx=100, timestamp_s=1.67, x=300.0, y=520.0, confidence=0.9),
        BallDetection(frame_idx=130, timestamp_s=2.17, x=350.0, y=650.0, confidence=0.9),
    ]
    return VideoArtifacts(
        release_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        ball_path=detections,
        bat_contact_frame=None,
        release_point=ReleasePoint(
            frame_idx=70,
            timestamp_s=1.17,
            hand_position=(220.0, 360.0),
            confidence=0.95,
            annotated_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        ),
    )


def test_build_anchor_trajectory_creates_release_bounce_target_path(monkeypatch) -> None:
    artifacts = _artifacts()
    detections = artifacts.ball_path
    pitch_frame = type(
        "PitchFrameStub",
        (),
        {
            "batting_origin_world": np.array([0.0, 0.0, -10.059]),
            "x_axis_world": np.array([1.0, 0.0, 0.0]),
            "z_axis_world": np.array([0.0, 0.0, 1.0]),
        },
    )()

    monkeypatch.setattr(
        "app.modules.bowler_performance.trajectory.unproject_to_plane_z",
        lambda x_val, y_val, K, RT, target_z: np.array(
            [
                0.4 if target_z == BOWLING_CREASE_Z_METRES else -0.2,
                1.9 if target_z == BOWLING_CREASE_Z_METRES else 0.8,
                target_z,
            ],
            dtype=np.float64,
        ),
    )
    monkeypatch.setattr(
        "app.modules.bowler_performance.trajectory.unproject_to_ground",
        lambda x_val, y_val, K, RT: np.array([0.1, 0.0, 2.5], dtype=np.float64),
    )

    trajectory = build_anchor_trajectory(
        artifacts,
        detections,
        bounce_frame=100,
        K=np.eye(3),
        RT=np.hstack([np.eye(3), np.zeros((3, 1))]),
        pitch_frame=pitch_frame,
    )

    assert trajectory is not None
    assert trajectory.frame_values[0] == 70.0
    assert trajectory.frame_values[-1] == 130.0 + POST_BOUNCE_EXTENSION_FRAMES
    assert trajectory.release_anchor[2] == BOWLING_CREASE_Z_METRES
    assert trajectory.bounce_anchor[1] == 0.0
    assert trajectory.target_anchor[2] == -10.059 - POST_STUMPS_Z_EXTENSION_METRES
    assert trajectory.world_points.shape[0] == 61 + POST_BOUNCE_EXTENSION_FRAMES
    assert np.allclose(trajectory.pitch_points[:, 1], 0.0)
