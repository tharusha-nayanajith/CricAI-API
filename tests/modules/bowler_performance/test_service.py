import numpy as np
import pytest

import app.modules.bowler_performance.service as bowler_service
from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts
from app.models.calibration import Keypoint
from app.modules.bowler_performance.models import (
    BowlerPerformanceResult,
    WicketRiskBand,
    WicketRiskPrediction,
)
from app.modules.bowler_performance.ransac import Parabola, RANSACResult
from app.modules.bowler_performance.service import BowlerPerformanceAnalyzer
from app.modules.bowler_performance.trajectory import AnchorTrajectory
from app.modules.preprocessor.models import (
    BallDetection,
    BatterMode,
    ReleasePoint,
)
from tests.conftest import CalibrationDataFactory


def _detection(
    frame_idx: int,
    timestamp_s: float,
    confidence: float = 0.9,
) -> BallDetection:
    return BallDetection(
        frame_idx=frame_idx,
        timestamp_s=timestamp_s,
        x=float(frame_idx),
        y=float(frame_idx),
        confidence=confidence,
    )


def _artifacts() -> VideoArtifacts:
    detections = [
        _detection(0, 0.0, 0.8),
        _detection(1, 0.1, 0.85),
        _detection(2, 0.2, 0.9),
        _detection(3, 0.3, 0.95),
    ]
    return VideoArtifacts(
        release_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        ball_path=detections,
        bat_contact_frame=None,
        release_point=ReleasePoint(
            frame_idx=0,
            timestamp_s=0.0,
            hand_position=(300.0, 300.0),
            confidence=0.95,
            annotated_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        ),
        batter_mode=BatterMode.PRESENT,
        bat_contact=None,
    )


def _clean_result(inliers: list[BallDetection]) -> RANSACResult:
    return RANSACResult(
        selected_track=inliers,
        inliers=inliers,
        para_x=Parabola(a=0.0, b=0.0, c=0.0),
        para_y=Parabola(a=0.0, b=0.0, c=0.0),
        bounce_frame=2,
        bounce_t=0.2,
    )


def _world_points(inliers: list[BallDetection]) -> list[tuple[BallDetection, np.ndarray]]:
    world_xyz = [
        np.array([0.0, 1.8, 15.0], dtype=np.float64),
        np.array([0.2, 1.4, 14.0], dtype=np.float64),
        np.array([0.35, 0.9, 13.0], dtype=np.float64),
        np.array([0.45, 0.3, 12.2], dtype=np.float64),
    ]
    return list(zip(inliers, world_xyz, strict=True))


def _invalid_world_points(
    inliers: list[BallDetection],
) -> list[tuple[BallDetection, np.ndarray]]:
    world_xyz = [
        np.array([0.0, 0.0, -18.0], dtype=np.float64),
        np.array([0.1, 0.0, -16.0], dtype=np.float64),
        np.array([0.2, 0.0, -14.0], dtype=np.float64),
        np.array([0.3, 0.0, -12.0], dtype=np.float64),
    ]
    return list(zip(inliers, world_xyz, strict=True))


def _duplicate_channel_calibration():
    calibration = CalibrationDataFactory()
    return calibration.model_copy(
        update={
            "detected_channels": 3,
            "total_detections": 4,
            "keypoints": [
                Keypoint(x=10.0, y=20.0, score=0.6, channel_index=0),
                Keypoint(x=99.0, y=88.0, score=0.2, channel_index=0),
                Keypoint(x=30.0, y=40.0, score=0.9, channel_index=2),
                Keypoint(x=50.0, y=60.0, score=0.8, channel_index=4),
            ],
        }
    )


def _anchor_trajectory() -> AnchorTrajectory:
    frame_values = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    world_points = np.array(
        [
            [0.0, 1.8, 15.0],
            [0.2, 1.1, 14.0],
            [0.35, 0.2, 13.0],
            [0.45, 0.8, -10.059],
        ],
        dtype=np.float64,
    )
    pitch_points = np.array(
        [
            [0.0, 0.0, 25.059],
            [0.2, 0.0, 24.059],
            [0.35, 0.0, 23.059],
            [0.45, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return AnchorTrajectory(
        frame_values=frame_values,
        world_points=world_points,
        pitch_points=pitch_points,
        release_anchor=world_points[0],
        bounce_anchor=np.array([0.35, 0.0, 3.0], dtype=np.float64),
        target_anchor=world_points[-1],
    )


@pytest.mark.asyncio
async def test_run_raises_feature_error_when_ransac_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzer = BowlerPerformanceAnalyzer()
    artifacts = _artifacts()

    monkeypatch.setattr(analyzer._cleaner, "clean", lambda raw_path, fps: None)

    with pytest.raises(FeatureError, match="RANSAC found too few inliers in ball path"):
        await analyzer.run(artifacts, CalibrationDataFactory(), fps=30.0)


@pytest.mark.asyncio
async def test_run_raises_feature_error_when_world_points_are_all_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzer = BowlerPerformanceAnalyzer()
    artifacts = _artifacts()
    clean_result = _clean_result(artifacts.ball_path)

    monkeypatch.setattr(analyzer._cleaner, "clean", lambda raw_path, fps: clean_result)
    monkeypatch.setattr(bowler_service, "build_intrinsic_matrix", lambda calibration: np.eye(3))
    monkeypatch.setattr(
        bowler_service,
        "build_extrinsic_matrix",
        lambda calibration: np.hstack([np.eye(3), np.zeros((3, 1))]),
    )
    monkeypatch.setattr(bowler_service, "pixels_to_world_points", lambda inliers, K, RT, fps: [])

    with pytest.raises(FeatureError, match="Too few valid 3D world points after outlier filtering"):
        await analyzer.run(artifacts, CalibrationDataFactory(), fps=30.0)


@pytest.mark.asyncio
async def test_run_returns_bowler_performance_result_with_valid_mocked_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzer = BowlerPerformanceAnalyzer()
    artifacts = _artifacts()
    clean_result = _clean_result(artifacts.ball_path)

    monkeypatch.setattr(analyzer._cleaner, "clean", lambda raw_path, fps: clean_result)
    monkeypatch.setattr(bowler_service, "build_intrinsic_matrix", lambda calibration: np.eye(3))
    monkeypatch.setattr(
        bowler_service,
        "build_extrinsic_matrix",
        lambda calibration: np.hstack([np.eye(3), np.zeros((3, 1))]),
    )
    monkeypatch.setattr(
        bowler_service,
        "pixels_to_world_points",
        lambda inliers, K, RT, fps: _world_points(inliers),
    )
    monkeypatch.setattr(
        bowler_service,
        "build_pitch_frame",
        lambda calibration, K, RT: object(),
    )
    monkeypatch.setattr(
        bowler_service,
        "world_points_to_pitch_points",
        lambda world_points, frame: world_points,
    )
    monkeypatch.setattr(
        bowler_service,
        "build_anchor_trajectory",
        lambda artifacts, detections, bounce_frame, K, RT, pitch_frame: _anchor_trajectory(),
    )
    monkeypatch.setattr(
        bowler_service,
        "predict_wicket_risk",
        lambda delivery_features: WicketRiskPrediction(
            probability=0.74,
            percentage=74.0,
            risk_band=WicketRiskBand.HIGH,
            model_name="test-model",
            model_version="v1",
        ),
    )

    result = await analyzer.run(
        artifacts,
        CalibrationDataFactory(),
        fps=30.0,
        video_url="sample.mp4",
    )

    assert isinstance(result, BowlerPerformanceResult)
    assert result.bounce_point is not None
    assert result.length_class is not None
    assert result.speed_kmh == pytest.approx(result.raw_speed_ms * 3.6)
    assert result.video_url == "sample.mp4"
    assert result.ball_track is not None
    assert result.ball_track.parameter_x_array
    assert result.ball_track.parameter_y_array
    assert result.ball_track.parameter_z_array
    assert result.camera_calibration is not None
    assert result.camera_calibration.dimensions == [1920, 1080]
    assert result.delivery_features is not None
    assert result.delivery_features.batter_mode == "present"
    assert result.delivery_features.line_bucket == "right_wide"
    assert result.wicket_risk is not None
    assert result.wicket_risk.risk_band is WicketRiskBand.HIGH
    assert result.flutter_payload
    assert result.flutter_payload[0].video_url == "sample.mp4"
    assert result.flutter_payload[0].delivery_features is result.delivery_features
    assert result.flutter_payload[0].wicket_risk is result.wicket_risk
    assert result.flutter_payload[0].ball_track is result.ball_track
    assert result.flutter_payload[0].camera_calibration is result.camera_calibration
    payload = result.model_dump(by_alias=True)
    assert payload["deliveryFeatures"] is not None
    assert payload["deliveryFeatures"]["batterMode"] == "present"
    assert payload["wicketRisk"] is not None
    assert payload["wicketRisk"]["riskBand"] == "high"
    assert payload["videoURL"] == "sample.mp4"
    assert payload["ballTrack"] is not None
    assert payload["cameraCalibration"] is not None
    assert payload["flutterPayload"]
    assert payload["flutterPayload"][0]["videoURL"] == "sample.mp4"
    assert payload["flutterPayload"][0]["deliveryFeatures"] is not None
    assert payload["flutterPayload"][0]["wicketRisk"] is not None
    assert payload["flutterPayload"][0]["ballTrack"] is not None
    assert payload["flutterPayload"][0]["cameraCalibration"] is not None
    assert result.trajectory_reliable is True
    assert result.trajectory_warning is None
    assert result.ball_track.trajectory_mode == "anchor_fitted"


@pytest.mark.asyncio
async def test_run_filters_world_point_outlier_before_assessment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzer = BowlerPerformanceAnalyzer()
    artifacts = _artifacts()
    clean_result = _clean_result(artifacts.ball_path)

    def outlier_world_points(
        inliers: list[BallDetection],
    ) -> list[tuple[BallDetection, np.ndarray]]:
        values = _world_points(inliers)
        values[2] = (
            values[2][0],
            np.array([0.2, 0.8, -1765.0], dtype=np.float64),
        )
        return values

    monkeypatch.setattr(analyzer._cleaner, "clean", lambda raw_path, fps: clean_result)
    monkeypatch.setattr(bowler_service, "build_intrinsic_matrix", lambda calibration: np.eye(3))
    monkeypatch.setattr(
        bowler_service,
        "build_extrinsic_matrix",
        lambda calibration: np.hstack([np.eye(3), np.zeros((3, 1))]),
    )
    monkeypatch.setattr(
        bowler_service,
        "pixels_to_world_points",
        lambda inliers, K, RT, fps: outlier_world_points(inliers),
    )
    monkeypatch.setattr(
        bowler_service,
        "build_pitch_frame",
        lambda calibration, K, RT: object(),
    )
    monkeypatch.setattr(
        bowler_service,
        "world_points_to_pitch_points",
        lambda world_points, frame: world_points,
    )
    monkeypatch.setattr(
        bowler_service,
        "build_anchor_trajectory",
        lambda artifacts, detections, bounce_frame, K, RT, pitch_frame: _anchor_trajectory(),
    )
    monkeypatch.setattr(bowler_service, "predict_wicket_risk", lambda delivery_features: None)

    result = await analyzer.run(
        artifacts,
        CalibrationDataFactory(),
        fps=30.0,
        video_url="sample.mp4",
    )

    assert result.trajectory_reliable is True
    assert result.swing_metres is not None
    assert result.ball_track is not None
    assert result.ball_track.trajectory_mode == "anchor_fitted"


@pytest.mark.asyncio
async def test_run_marks_trajectory_unavailable_when_world_reconstruction_fails_sanity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzer = BowlerPerformanceAnalyzer()
    artifacts = _artifacts()
    clean_result = _clean_result(artifacts.ball_path)

    monkeypatch.setattr(analyzer._cleaner, "clean", lambda raw_path, fps: clean_result)
    monkeypatch.setattr(bowler_service, "build_intrinsic_matrix", lambda calibration: np.eye(3))
    monkeypatch.setattr(
        bowler_service,
        "build_extrinsic_matrix",
        lambda calibration: np.hstack([np.eye(3), np.zeros((3, 1))]),
    )
    monkeypatch.setattr(
        bowler_service,
        "pixels_to_world_points",
        lambda inliers, K, RT, fps: _invalid_world_points(inliers),
    )
    monkeypatch.setattr(
        bowler_service,
        "build_pitch_frame",
        lambda calibration, K, RT: object(),
    )
    monkeypatch.setattr(
        bowler_service,
        "world_points_to_pitch_points",
        lambda world_points, frame: world_points,
    )
    monkeypatch.setattr(
        bowler_service,
        "build_anchor_trajectory",
        lambda artifacts, detections, bounce_frame, K, RT, pitch_frame: _anchor_trajectory(),
    )
    monkeypatch.setattr(bowler_service, "predict_wicket_risk", lambda delivery_features: None)

    result = await analyzer.run(
        artifacts,
        CalibrationDataFactory(),
        fps=30.0,
        video_url="sample.mp4",
    )

    assert result.trajectory_reliable is False
    assert result.trajectory_warning is not None
    assert result.speed_kmh is not None
    assert result.raw_speed_ms is not None
    assert result.speed_kmh == pytest.approx(result.raw_speed_ms * 3.6)
    assert result.swing_metres is None
    assert result.ball_track is not None
    assert result.delivery_features is not None
    assert result.wicket_risk is None
    assert result.ball_track.trajectory_mode == "anchor_fitted"
    assert result.ball_track.trajectory_points_3d
    assert result.camera_calibration is not None
    assert result.flutter_payload
    assert result.flutter_payload[0].ball_track is result.ball_track
    assert result.flutter_payload[0].camera_calibration is result.camera_calibration


@pytest.mark.asyncio
async def test_run_sanitizes_calibration_keypoints_before_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzer = BowlerPerformanceAnalyzer()
    artifacts = _artifacts()
    clean_result = _clean_result(artifacts.ball_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr(analyzer._cleaner, "clean", lambda raw_path, fps: clean_result)

    def fake_intrinsic(calibration):
        captured["intrinsic_calibration"] = calibration
        return np.eye(3)

    def fake_extrinsic(calibration):
        captured["extrinsic_calibration"] = calibration
        return np.hstack([np.eye(3), np.zeros((3, 1))])

    def fake_pitch_frame(calibration, K, RT):
        captured["pitch_frame_calibration"] = calibration
        return object()

    monkeypatch.setattr(bowler_service, "build_intrinsic_matrix", fake_intrinsic)
    monkeypatch.setattr(bowler_service, "build_extrinsic_matrix", fake_extrinsic)
    monkeypatch.setattr(
        bowler_service,
        "pixels_to_world_points",
        lambda inliers, K, RT, fps: _world_points(inliers),
    )
    monkeypatch.setattr(bowler_service, "build_pitch_frame", fake_pitch_frame)
    monkeypatch.setattr(
        bowler_service,
        "world_points_to_pitch_points",
        lambda world_points, frame: world_points,
    )
    monkeypatch.setattr(
        bowler_service,
        "build_anchor_trajectory",
        lambda artifacts, detections, bounce_frame, K, RT, pitch_frame: _anchor_trajectory(),
    )
    monkeypatch.setattr(bowler_service, "predict_wicket_risk", lambda delivery_features: None)

    await analyzer.run(
        artifacts,
        _duplicate_channel_calibration(),
        fps=30.0,
        video_url="sample.mp4",
    )

    for key in ("intrinsic_calibration", "extrinsic_calibration", "pitch_frame_calibration"):
        calibration = captured[key]
        assert len(calibration.keypoints) == 3
        assert [keypoint.channel_index for keypoint in calibration.keypoints] == [0, 2, 4]
        assert calibration.keypoints[0].x == 10.0
        assert calibration.keypoints[0].score == 0.6



