from pathlib import Path

import cv2
import numpy as np
import pytest

from app.modules.preprocessor.bat_contact_detector import BatContactDetector
from app.modules.preprocessor.models import (
    BallDetection,
    BatterMode,
    BatterROI,
    ContactMethod,
    ReleasePoint,
)
from app.modules.preprocessor.service import VideoPreprocessor
from tests.conftest import CalibrationDataFactory


class _FakeCapture:
    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame
        self._pos = 0

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: int) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._pos = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        _ = self._pos
        return True, self._frame

    def release(self) -> None:
        return None


def _ball_path() -> list[BallDetection]:
    return [
        BallDetection(frame_idx=8, timestamp_s=8 / 30, x=20.0, y=20.0, confidence=0.8),
        BallDetection(frame_idx=9, timestamp_s=9 / 30, x=40.0, y=40.0, confidence=0.82),
        BallDetection(frame_idx=10, timestamp_s=10 / 30, x=60.0, y=60.0, confidence=0.84),
        BallDetection(frame_idx=11, timestamp_s=11 / 30, x=150.0, y=150.0, confidence=0.85),
        BallDetection(frame_idx=12, timestamp_s=12 / 30, x=154.0, y=154.0, confidence=0.83),
    ]


def test_detect_uses_ball_velocity_when_enough_detections_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "app.modules.preprocessor.bat_contact_detector.cv2.VideoCapture",
        lambda _: _FakeCapture(frame),
    )
    detector = BatContactDetector()
    roi = BatterROI(x=100, y=100, width=80, height=80)
    monkeypatch.setattr(
        detector,
        "_detect_impact_frame",
        lambda video_path, fps: {"impact_frame": 10, "impact_time": 10 / fps},
    )

    result = detector.detect(Path("video.mp4"), 30.0, roi, _ball_path())

    assert result is not None
    assert result.method is ContactMethod.BALL_VELOCITY
    assert result.contact_frame_idx == 12
    assert result.detection_score is not None
    assert result.detection_score > 0.0


def test_detect_falls_back_to_audio_frame_when_ball_window_is_too_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "app.modules.preprocessor.bat_contact_detector.cv2.VideoCapture",
        lambda _: _FakeCapture(frame),
    )
    detector = BatContactDetector()
    roi = BatterROI(x=100, y=100, width=80, height=80)
    monkeypatch.setattr(
        detector,
        "_detect_impact_frame",
        lambda video_path, fps: {"impact_frame": 30, "impact_time": 1.0},
    )

    result = detector.detect(Path("video.mp4"), 30.0, roi, _ball_path())

    assert result is not None
    assert result.method is ContactMethod.AUDIO_FALLBACK
    assert result.contact_frame_idx == 30
    assert result.detection_score is None


def test_detect_returns_none_when_audio_detection_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        "app.modules.preprocessor.bat_contact_detector.logger.warning",
        lambda message, *args: warnings.append(message.format(*args)),
    )
    detector = BatContactDetector()
    roi = BatterROI(x=100, y=100, width=80, height=80)
    monkeypatch.setattr(detector, "_detect_impact_frame", lambda video_path, fps: None)

    result = detector.detect(Path("video.mp4"), 30.0, roi, _ball_path())

    assert result is None
    assert any("audio impact detection failed" in warning.lower() for warning in warnings)


def test_refine_impact_frame_uses_audio_window() -> None:
    detector = BatContactDetector()

    frame_idx, method, detection_score = detector._refine_impact_frame(_ball_path(), 10)

    assert frame_idx == 12
    assert method is ContactMethod.BALL_VELOCITY
    assert detection_score is not None


def test_find_impact_peak_returns_peak_time() -> None:
    detector = BatContactDetector()
    sr = 100
    envelope = np.zeros(200, dtype=np.float32)
    envelope[125] = 5.0

    peak_idx, peak_time = detector._find_impact_peak(envelope, sr)

    assert peak_idx == 125
    assert peak_time == pytest.approx(1.25)


def test_smooth_signal_returns_non_negative_envelope() -> None:
    detector = BatContactDetector()
    signal = np.array([-1.0, 0.5, -0.25, 0.75], dtype=np.float32)

    smoothed = detector._smooth_signal(signal)

    assert smoothed.shape == signal.shape
    assert np.all(smoothed >= 0.0)


@pytest.mark.asyncio
async def test_run_sets_bat_contact_frame_to_none_when_batter_mode_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocessor = VideoPreprocessor()
    calibration = CalibrationDataFactory()

    async def _fake_standardize(video_path: Path) -> Path:
        return video_path

    async def _fake_detect_release(ctx) -> ReleasePoint:
        _ = ctx
        return ReleasePoint(
            frame_idx=12,
            timestamp_s=0.4,
            hand_position=(100.0, 200.0),
            confidence=0.9,
            annotated_frame=np.zeros((32, 32, 3), dtype=np.uint8),
        )

    class _FakeBatterDetector:
        def detect(self, video_path: Path, calibration_data):
            _ = video_path
            _ = calibration_data
            return BatterMode.NONE, None

    class _FakeBallTracker:
        def track(self, *args, **kwargs):
            _ = args
            _ = kwargs
            return [
                BallDetection(frame_idx=1, timestamp_s=0.1, x=10.0, y=20.0, confidence=0.9),
                BallDetection(frame_idx=2, timestamp_s=0.2, x=15.0, y=25.0, confidence=0.8),
                BallDetection(frame_idx=3, timestamp_s=0.3, x=20.0, y=30.0, confidence=0.7),
            ]

    class _FakeVideoCaptureForFps:
        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            return 0.0

        def release(self) -> None:
            return None

    def _unexpected_bat_contact_detector():
        raise AssertionError("bat contact detector should not run for BatterMode.NONE")

    monkeypatch.setattr(preprocessor, "standardize_video", _fake_standardize)
    monkeypatch.setattr(preprocessor, "_detect_release", _fake_detect_release)
    monkeypatch.setattr(
        "app.modules.preprocessor.service.get_batter_detector",
        lambda: _FakeBatterDetector(),
    )
    monkeypatch.setattr(
        "app.modules.preprocessor.service.get_ball_tracker",
        lambda: _FakeBallTracker(),
    )
    monkeypatch.setattr(
        "app.modules.preprocessor.service.get_bat_contact_detector",
        _unexpected_bat_contact_detector,
    )
    monkeypatch.setattr(
        "app.modules.preprocessor.service.cv2.VideoCapture",
        lambda _: _FakeVideoCaptureForFps(),
    )

    artifacts = await preprocessor.run(Path("input.mp4"), calibration)

    assert artifacts.bat_contact_frame is None
