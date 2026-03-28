import numpy as np
import pytest

import app.modules.bowler_performance.service as bowler_service
from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts
from app.modules.bowler_performance.models import BowlerPerformanceResult
from app.modules.bowler_performance.ransac import Parabola, RANSACResult
from app.modules.bowler_performance.service import BowlerPerformanceAnalyzer
from app.modules.preprocessor.models import BallDetection
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
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.6, 0.0, 1.5], dtype=np.float64),
        np.array([1.0, 0.0, 4.5], dtype=np.float64),
        np.array([1.2, 0.0, 6.0], dtype=np.float64),
    ]
    return list(zip(inliers, world_xyz, strict=True))


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
    monkeypatch.setattr(bowler_service, "pixels_to_world_points", lambda inliers, K, RT: [])

    with pytest.raises(FeatureError, match="Too few valid 3D world points after unprojection"):
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
        lambda inliers, K, RT: _world_points(inliers),
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

    result = await analyzer.run(artifacts, CalibrationDataFactory(), fps=30.0)

    assert isinstance(result, BowlerPerformanceResult)
    assert result.bounce_point is not None
    assert result.length_class is not None
    assert result.speed_kmh == pytest.approx(result.raw_speed_ms * 3.6)
