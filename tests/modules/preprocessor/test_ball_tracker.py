from pathlib import Path

import cv2
import numpy as np
import pytest

from app.exceptions import PreprocessingError
from app.models.calibration import CalibrationData
from app.modules.preprocessor.ball_tracker import BallTracker
from app.modules.preprocessor.constants import (
    BALL_CONF_RAW_THRESHOLD,
    BALL_EARLY_STOP_CONF,
    BALL_EARLY_STOP_MIN_FRAME,
    STANDARDIZED_HEIGHT,
    STANDARDIZED_WIDTH,
)
from app.modules.preprocessor.models import BallDetection, BatterMode, BatterROI, ReleasePoint
from app.modules.preprocessor.service import VideoPreprocessor
from tests.conftest import CalibrationDataFactory


class _FakeOutput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    def __init__(self) -> None:
        self._scores = np.zeros((1, 1, 640, 360), dtype=np.float32)
        self._scores[0, 0, 320, 180] = 0.85

    def get_outputs(self) -> list[_FakeOutput]:
        return [_FakeOutput("scores")]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        _ = output_names
        _ = input_feed
        return [self._scores]


class _FakeCapture:
    def __init__(self, total_frames: int = 120) -> None:
        self._current_frame = 0
        self._total_frames = total_frames

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: int) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._current_frame = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        if self._current_frame >= self._total_frames:
            return False, np.zeros((STANDARDIZED_HEIGHT, STANDARDIZED_WIDTH, 3), dtype=np.uint8)
        frame = np.zeros((STANDARDIZED_HEIGHT, STANDARDIZED_WIDTH, 3), dtype=np.uint8)
        self._current_frame += 1
        return True, frame

    def release(self) -> None:
        return None


def _make_tracker(monkeypatch: pytest.MonkeyPatch) -> BallTracker:
    monkeypatch.setattr(
        "app.modules.preprocessor.ball_tracker.ort.InferenceSession",
        lambda *args, **kwargs: _FakeSession(),
    )
    return BallTracker(Path("ballDetection.onnx"))


def test_preprocess_returns_expected_shape_and_range(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = _make_tracker(monkeypatch)
    frame = np.random.randint(0, 256, size=(500, 400, 3), dtype=np.uint8)

    output = tracker._preprocess(frame)

    assert output.shape == (1, 3, 1280, 720)
    assert np.min(output) >= -1.0
    assert np.max(output) <= 1.0


def test_infer_returns_none_until_three_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = _make_tracker(monkeypatch)
    frame = np.zeros((STANDARDIZED_HEIGHT, STANDARDIZED_WIDTH, 3), dtype=np.uint8)

    first = tracker._infer(frame)
    second = tracker._infer(frame)

    assert first == (None, None, 0.0)
    assert second == (None, None, 0.0)


def test_infer_returns_peak_coordinates_when_buffer_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _make_tracker(monkeypatch)
    frame = np.zeros((STANDARDIZED_HEIGHT, STANDARDIZED_WIDTH, 3), dtype=np.uint8)

    tracker._infer(frame)
    tracker._infer(frame)
    x_val, y_val, confidence = tracker._infer(frame)

    assert x_val is not None and 0.0 <= x_val <= 720.0
    assert y_val is not None and 0.0 <= y_val <= 1280.0
    assert confidence == pytest.approx(0.85)


def test_reset_clears_frame_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = _make_tracker(monkeypatch)
    frame = np.zeros((STANDARDIZED_HEIGHT, STANDARDIZED_WIDTH, 3), dtype=np.uint8)

    tracker._infer(frame)
    tracker._infer(frame)
    tracker._infer(frame)
    tracker.reset()

    assert tracker._infer(frame) == (None, None, 0.0)


def test_ball_in_roi_returns_true_inside(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = _make_tracker(monkeypatch)

    assert tracker._ball_in_roi(50.0, 60.0, BatterROI(x=10, y=20, width=100, height=100)) is True


def test_ball_in_roi_returns_false_outside(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = _make_tracker(monkeypatch)

    assert tracker._ball_in_roi(500.0, 600.0, BatterROI(x=10, y=20, width=100, height=100)) is False


def test_track_stops_early_in_bowler_only_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = _make_tracker(monkeypatch)
    sequence = [(None, None, 0.0), (None, None, 0.0)] + [
        (100.0, 500.0, 0.9) for _ in range(BALL_EARLY_STOP_MIN_FRAME + 1)
    ] + [(50.0, 100.0, BALL_EARLY_STOP_CONF - 0.05)]
    iterator = iter(sequence)

    monkeypatch.setattr(
        "app.modules.preprocessor.ball_tracker.cv2.VideoCapture",
        lambda _: _FakeCapture(),
    )
    monkeypatch.setattr(tracker, "_infer", lambda frame: next(iterator))

    detections = tracker.track(
        Path("video.mp4"),
        release_frame_idx=20,
        fps=30.0,
        batter_mode=BatterMode.NONE,
        batter_roi=None,
    )

    assert len(detections) == BALL_EARLY_STOP_MIN_FRAME + 1


def test_track_marks_roi_entry_but_keeps_tracking_in_batter_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _make_tracker(monkeypatch)
    roi = BatterROI(x=100, y=100, width=120, height=120)
    sequence = [
        (None, None, 0.0),
        (None, None, 0.0),
        (50.0, 50.0, 0.9),
        (110.0, 110.0, 0.95),
        (140.0, 140.0, 0.92),
    ]
    iterator = iter(sequence)

    monkeypatch.setattr(
        "app.modules.preprocessor.ball_tracker.cv2.VideoCapture",
        lambda _: _FakeCapture(total_frames=13),
    )
    monkeypatch.setattr(tracker, "_infer", lambda frame: next(iterator))

    detections = tracker.track(
        Path("video.mp4"),
        release_frame_idx=10,
        fps=30.0,
        batter_mode=BatterMode.PRESENT,
        batter_roi=roi,
    )

    assert len(detections) == 3
    assert tracker._last_roi_entry_frame_idx == 11
    assert detections[-1].x == pytest.approx(140.0)
    assert detections[-1].y == pytest.approx(140.0)


def test_track_returns_empty_list_when_confidence_never_exceeds_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _make_tracker(monkeypatch)
    sequence = [(None, None, 0.0), (None, None, 0.0)] + [
        (100.0, 400.0, BALL_CONF_RAW_THRESHOLD - 0.05) for _ in range(5)
    ]
    iterator = iter(sequence)

    monkeypatch.setattr(
        "app.modules.preprocessor.ball_tracker.cv2.VideoCapture",
        lambda _: _FakeCapture(total_frames=7),
    )
    monkeypatch.setattr(tracker, "_infer", lambda frame: next(iterator))

    detections = tracker.track(
        Path("video.mp4"),
        release_frame_idx=2,
        fps=30.0,
        batter_mode=BatterMode.NONE,
        batter_roi=None,
    )

    assert detections == []


@pytest.mark.asyncio
async def test_run_raises_when_raw_path_is_too_short(monkeypatch: pytest.MonkeyPatch) -> None:
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
            annotated_frame=np.zeros((STANDARDIZED_HEIGHT, STANDARDIZED_WIDTH, 3), dtype=np.uint8),
        )

    class _FakeBatterDetector:
        def detect(self, video_path: Path, calibration_data: CalibrationData):
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
        "app.modules.preprocessor.service.cv2.VideoCapture",
        lambda _: _FakeVideoCaptureForFps(),
    )

    with pytest.raises(PreprocessingError, match="Ball path too short: 2 detections"):
        await preprocessor.run(Path("input.mp4"), calibration)
