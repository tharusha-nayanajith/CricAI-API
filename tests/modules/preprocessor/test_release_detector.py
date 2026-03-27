from pathlib import Path

import numpy as np
import pytest

from app.exceptions import PreprocessingError
from app.modules.preprocessor.models import BallDetection, ReleasePoint
from app.modules.preprocessor.release_detector import ReleaseDetector
from app.modules.preprocessor.service import VideoPreprocessor
from tests.conftest import CalibrationDataFactory


class _FakeOnnxInput:
    def __init__(self, name: str, shape: list[int]) -> None:
        self.name = name
        self.shape = shape


class _FakeOnnxOutput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeOnnxSession:
    def __init__(self, confidence: float) -> None:
        self._confidence = confidence

    def get_inputs(self) -> list[_FakeOnnxInput]:
        return [_FakeOnnxInput("features", [1, 6])]

    def get_outputs(self) -> list[_FakeOnnxOutput]:
        return [_FakeOnnxOutput("score")]

    def run(self, output_names: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        _ = output_names
        _ = inputs
        return [np.array([[self._confidence]], dtype=np.float32)]


class _FakeInterpreter:
    def __init__(self) -> None:
        self._heatmaps = np.zeros((1, 9, 9, 18), dtype=np.float32)
        self._offsets = np.zeros((1, 9, 9, 36), dtype=np.float32)
        # Make right wrist / elbow / shoulder confidently detectable.
        self._heatmaps[0, 4, 4, 10] = 8.0
        self._heatmaps[0, 4, 4, 8] = 8.0
        self._heatmaps[0, 4, 4, 6] = 8.0

    def allocate_tensors(self) -> None:
        return None

    def get_input_details(self) -> list[dict[str, object]]:
        return [{"index": 1, "shape": np.array([1, 257, 257, 3]), "dtype": np.float32}]

    def get_output_details(self) -> list[dict[str, int]]:
        return [{"index": 2}, {"index": 3}]

    def set_tensor(self, index: int, value: np.ndarray) -> None:
        _ = index
        _ = value

    def invoke(self) -> None:
        return None

    def get_tensor(self, index: int) -> np.ndarray:
        if index == 2:
            return self._heatmaps
        return self._offsets


def _create_detector(monkeypatch: pytest.MonkeyPatch, confidence: float) -> ReleaseDetector:
    detector = ReleaseDetector(Path("release.onnx"), Path("pose.tflite"))
    monkeypatch.setattr(
        detector,
        "_create_onnx_session",
        lambda _: _FakeOnnxSession(confidence),
    )
    monkeypatch.setattr(
        detector,
        "_create_tflite_interpreter",
        lambda _: _FakeInterpreter(),
    )
    detector.load_models()
    return detector


def test_process_frame_returns_none_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = _create_detector(monkeypatch, confidence=0.3)
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)

    result = detector.process_frame(frame, frame_idx=10, fps=30.0)

    assert result is None


def test_process_frame_returns_release_point_above_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _create_detector(monkeypatch, confidence=0.95)
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)

    result = detector.process_frame(frame, frame_idx=15, fps=30.0)

    assert isinstance(result, ReleasePoint)
    assert result.frame_idx == 15
    assert result.timestamp_s == pytest.approx(0.5)


def test_reset_clears_state_for_next_delivery(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = _create_detector(monkeypatch, confidence=0.95)
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)

    first = detector.process_frame(frame, frame_idx=1, fps=25.0)
    locked = detector.process_frame(frame, frame_idx=2, fps=25.0)
    detector.reset()
    second = detector.process_frame(frame, frame_idx=3, fps=25.0)

    assert first is not None
    assert locked is None
    assert second is not None
    assert second.frame_idx == 3


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._index = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == 5:
            return 30.0
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        return None


class _AlwaysNoneDetector:
    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int, fps: float) -> None:
        _ = frame_bgr
        _ = frame_idx
        _ = fps
        return None

    def reset(self) -> None:
        return None


@pytest.mark.asyncio
async def test_run_raises_when_no_release_frame_found(monkeypatch: pytest.MonkeyPatch) -> None:
    preprocessor = VideoPreprocessor()
    frames = [np.zeros((1280, 720, 3), dtype=np.uint8) for _ in range(3)]
    calibration = CalibrationDataFactory()

    async def _fake_standardize(video_path: Path) -> Path:
        return video_path

    class _FakeBatterDetector:
        def detect(self, video_path: Path, calibration_data) -> tuple:
            _ = video_path
            _ = calibration_data
            return "none", None

    class _FakeBallTracker:
        def track(self, *args, **kwargs):
            _ = args
            _ = kwargs
            return [
                BallDetection(frame_idx=1, timestamp_s=0.1, x=10.0, y=20.0, confidence=0.9),
                BallDetection(frame_idx=2, timestamp_s=0.2, x=15.0, y=25.0, confidence=0.8),
                BallDetection(frame_idx=3, timestamp_s=0.3, x=20.0, y=30.0, confidence=0.7),
            ]

    class _FakeVideoCaptureForFps(_FakeCapture):
        def get(self, prop: int) -> float:
            if prop == 5:
                return 30.0
            return 0.0

    monkeypatch.setattr(preprocessor, "standardize_video", _fake_standardize)
    monkeypatch.setattr(
        "app.modules.preprocessor.service.get_batter_detector",
        lambda: _FakeBatterDetector(),
    )
    monkeypatch.setattr(
        "app.modules.preprocessor.service.get_ball_tracker",
        lambda: _FakeBallTracker(),
    )
    monkeypatch.setattr(
        "app.modules.preprocessor.service.get_release_detector",
        lambda: _AlwaysNoneDetector(),
    )
    monkeypatch.setattr(
        "app.modules.preprocessor.service.cv2.VideoCapture",
        lambda _: _FakeVideoCaptureForFps(frames),
    )

    with pytest.raises(PreprocessingError, match="No release frame detected"):
        await preprocessor.run(Path("input.mp4"), calibration)
