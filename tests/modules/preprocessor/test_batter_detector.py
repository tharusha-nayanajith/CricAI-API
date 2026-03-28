from pathlib import Path

import cv2
import numpy as np
import pytest

from app.exceptions import PreprocessingError
from app.modules.preprocessor.batter_detector import BatterDetector
from app.modules.preprocessor.constants import STANDARDIZED_HEIGHT, STANDARDIZED_WIDTH
from app.modules.preprocessor.models import BatterMode, BatterROI
from tests.conftest import CalibrationDataFactory, KeypointFactory


class _FakeInterpreter:
    def __init__(self, output: np.ndarray) -> None:
        self._output = output

    def get_input_details(self) -> list[dict[str, object]]:
        return [{"index": 1, "shape": np.array([1, 257, 257, 3]), "dtype": np.float32}]

    def get_output_details(self) -> list[dict[str, int]]:
        return [{"index": 2}]

    def set_tensor(self, index: int, value: np.ndarray) -> None:
        _ = index
        _ = value

    def invoke(self) -> None:
        return None

    def get_tensor(self, index: int) -> np.ndarray:
        _ = index
        return self._output


class _FakeCapture:
    def __init__(self, total_frames: int) -> None:
        self._total_frames = total_frames
        self._current_frame = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._total_frames)
        return 0.0

    def set(self, prop: int, value: int) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._current_frame = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        frame = np.zeros((STANDARDIZED_HEIGHT, STANDARDIZED_WIDTH, 3), dtype=np.uint8)
        return True, frame

    def release(self) -> None:
        return None


def _make_calibration(points: list[tuple[float, float, int]]):
    calibration = CalibrationDataFactory()
    keypoints = [
        KeypointFactory(x=x, y=y, channel_index=channel_index)
        for x, y, channel_index in points
    ]
    return calibration.model_copy(
        update={
            "image_size": (STANDARDIZED_WIDTH, STANDARDIZED_HEIGHT),
            "keypoints": keypoints,
        }
    )


def test_derive_roi_returns_expected_box_dimensions() -> None:
    detector = BatterDetector(_FakeInterpreter(np.zeros((1, 1, 17, 3), dtype=np.float32)))
    calibration = _make_calibration(
        [(619.0, 200.0, 0), (599.0, 210.0, 1), (609.0, 205.0, 2)],
    )

    roi = detector.derive_roi(calibration)

    assert roi.width == 80
    assert roi.height == 120
    assert roi.x == 70
    assert roi.y == 145


def test_derive_roi_raises_with_fewer_than_two_stump_keypoints() -> None:
    detector = BatterDetector(_FakeInterpreter(np.zeros((1, 1, 17, 3), dtype=np.float32)))
    calibration = _make_calibration([(619.0, 200.0, 0)])

    with pytest.raises(PreprocessingError, match="At least 2 stump keypoints"):
        detector.derive_roi(calibration)


def test_derive_roi_applies_minimum_floor() -> None:
    detector = BatterDetector(_FakeInterpreter(np.zeros((1, 1, 17, 3), dtype=np.float32)))
    calibration = _make_calibration([(669.0, 60.0, 0), (669.0, 60.0, 1)])

    roi = detector.derive_roi(calibration)

    assert roi.width == 80
    assert roi.height == 120


def test_derive_roi_falls_back_to_reprojected_keypoints_when_detected_points_are_far(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = BatterDetector(_FakeInterpreter(np.zeros((1, 1, 17, 3), dtype=np.float32)))
    calibration = _make_calibration(
        [(300.0, 500.0, 0), (320.0, 510.0, 1), (310.0, 505.0, 2)],
    ).model_copy(update={"detected_channels": 6})
    projected = [
        KeypointFactory(x=619.0, y=200.0, channel_index=0),
        KeypointFactory(x=599.0, y=210.0, channel_index=1),
        KeypointFactory(x=609.0, y=205.0, channel_index=2),
    ]

    monkeypatch.setattr(detector, "_project_striker_keypoints", lambda _: projected)

    roi = detector.derive_roi(calibration)

    assert roi.width == 80
    assert roi.height == 120
    assert roi.x == 70
    assert roi.y == 145


def test_derive_roi_uses_detected_keypoints_when_they_match_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = BatterDetector(_FakeInterpreter(np.zeros((1, 1, 17, 3), dtype=np.float32)))
    calibration = _make_calibration(
        [(619.0, 200.0, 0), (599.0, 210.0, 1), (609.0, 205.0, 2)],
    ).model_copy(update={"detected_channels": 6})
    projected = [
        KeypointFactory(x=617.0, y=201.0, channel_index=0),
        KeypointFactory(x=597.0, y=212.0, channel_index=1),
        KeypointFactory(x=608.0, y=206.0, channel_index=2),
    ]

    monkeypatch.setattr(detector, "_project_striker_keypoints", lambda _: projected)

    roi = detector.derive_roi(calibration)

    assert roi.width == 80
    assert roi.height == 120
    assert roi.x == 70
    assert roi.y == 145


def test_derive_roi_ignores_projection_fallback_when_fov_is_invalid() -> None:
    detector = BatterDetector(_FakeInterpreter(np.zeros((1, 1, 17, 3), dtype=np.float32)))
    calibration = _make_calibration(
        [(619.0, 200.0, 0), (599.0, 210.0, 1), (609.0, 205.0, 2)],
    ).model_copy(update={"fov": 0.0, "detected_channels": 6})

    roi = detector.derive_roi(calibration)

    assert roi.width == 80
    assert roi.height == 120
    assert roi.x == 70
    assert roi.y == 145


def test_roi_keypoints_in_frame_space_unflips_x_coordinates() -> None:
    detector = BatterDetector(_FakeInterpreter(np.zeros((1, 1, 17, 3), dtype=np.float32)))
    calibration = _make_calibration(
        [(619.0, 200.0, 0), (599.0, 210.0, 1), (609.0, 205.0, 2)],
    )

    keypoints = detector._roi_keypoints_in_frame_space(calibration)

    assert [round(keypoint.x, 1) for keypoint in keypoints] == [100.0, 120.0, 110.0]


def test_sample_frame_indices_are_evenly_spaced_within_window() -> None:
    detector = BatterDetector(_FakeInterpreter(np.zeros((1, 1, 17, 3), dtype=np.float32)))

    indices = detector._sample_frame_indices(100)

    assert len(indices) == detector.SAMPLE_COUNT
    assert all(15 <= idx <= 50 for idx in indices)
    gaps = [indices[idx + 1] - indices[idx] for idx in range(len(indices) - 1)]
    assert max(gaps) - min(gaps) <= 1


def test_person_in_roi_returns_true_when_score_exceeds_threshold() -> None:
    output = np.zeros((1, 1, 17, 3), dtype=np.float32)
    output[0, 0, 3, 2] = 0.8
    detector = BatterDetector(_FakeInterpreter(output))

    result = detector._person_in_roi(
        np.zeros((STANDARDIZED_HEIGHT, STANDARDIZED_WIDTH, 3), dtype=np.uint8),
        BatterROI(x=10, y=10, width=100, height=120),
    )

    assert result is True


def test_person_in_roi_returns_false_when_all_scores_are_low() -> None:
    output = np.zeros((1, 1, 17, 3), dtype=np.float32)
    output[0, 0, :, 2] = 0.2
    detector = BatterDetector(_FakeInterpreter(output))

    result = detector._person_in_roi(
        np.zeros((STANDARDIZED_HEIGHT, STANDARDIZED_WIDTH, 3), dtype=np.uint8),
        BatterROI(x=10, y=10, width=100, height=120),
    )

    assert result is False


def test_detect_returns_present_when_majority_vote_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = BatterDetector(_FakeInterpreter(np.zeros((1, 1, 17, 3), dtype=np.float32)))
    calibration = _make_calibration([(100.0, 200.0, 0), (120.0, 210.0, 1)])
    outcomes = iter([True, True, True, False, False])

    monkeypatch.setattr(
        "app.modules.preprocessor.batter_detector.cv2.VideoCapture",
        lambda _: _FakeCapture(100),
    )
    monkeypatch.setattr(detector, "_person_in_roi", lambda frame, roi: next(outcomes))

    mode, roi = detector.detect(Path("video.mp4"), calibration)

    assert mode is BatterMode.PRESENT
    assert roi is not None


def test_detect_returns_none_when_majority_vote_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = BatterDetector(_FakeInterpreter(np.zeros((1, 1, 17, 3), dtype=np.float32)))
    calibration = _make_calibration([(100.0, 200.0, 0), (120.0, 210.0, 1)])
    outcomes = iter([False, False, True, False, False])

    monkeypatch.setattr(
        "app.modules.preprocessor.batter_detector.cv2.VideoCapture",
        lambda _: _FakeCapture(100),
    )
    monkeypatch.setattr(detector, "_person_in_roi", lambda frame, roi: next(outcomes))

    mode, roi = detector.detect(Path("video.mp4"), calibration)

    assert mode is BatterMode.NONE
    assert roi is None
