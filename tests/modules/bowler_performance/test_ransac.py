import numpy as np
import pytest

from app.modules.bowler_performance.ransac import (
    MAX_FRAME_GAP,
    MIN_INLIERS,
    BallPathCleaner,
    RANSACResult,
)
from app.modules.preprocessor.models import BallDetection


def _detection(
    frame_idx: int,
    timestamp_s: float,
    x: float,
    y: float,
    confidence: float = 0.9,
) -> BallDetection:
    return BallDetection(
        frame_idx=frame_idx,
        timestamp_s=timestamp_s,
        x=x,
        y=y,
        confidence=confidence,
    )


def test_clean_returns_none_when_raw_path_has_too_few_detections() -> None:
    cleaner = BallPathCleaner()
    raw_path = [
        _detection(idx, idx / 30.0, 10.0 + idx, 20.0 + idx)
        for idx in range(MIN_INLIERS - 1)
    ]

    result = cleaner.clean(raw_path, fps=30.0)

    assert result is None


def test_clean_returns_ransac_result_for_noisy_parabolic_path() -> None:
    cleaner = BallPathCleaner()
    rng = np.random.default_rng(0)
    raw_path: list[BallDetection] = []

    for frame_idx in range(20):
        timestamp_s = frame_idx / 30.0
        x = 100.0 + 12.0 * timestamp_s + rng.normal(0.0, 1.0)
        y = 220.0 + 260.0 * timestamp_s - 180.0 * (timestamp_s**2) + rng.normal(0.0, 1.0)
        raw_path.append(_detection(frame_idx, timestamp_s, x, y))

    for frame_idx in range(20, 25):
        raw_path.append(
            _detection(
                frame_idx,
                frame_idx / 30.0,
                float(rng.uniform(0.0, 720.0)),
                float(rng.uniform(0.0, 1280.0)),
                confidence=0.2,
            )
        )

    rng.shuffle(raw_path)
    result = cleaner.clean(raw_path, fps=30.0)

    assert isinstance(result, RANSACResult)
    assert len(result.selected_track) >= 18
    assert len(result.inliers) >= 15
    assert {detection.frame_idx for detection in result.inliers}.issubset(
        {detection.frame_idx for detection in result.selected_track}
    )


def test_detect_bounce_identifies_turning_point_in_y() -> None:
    cleaner = BallPathCleaner()
    inliers = [
        _detection(0, 0.0, 0.0, 100.0),
        _detection(1, 1 / 30.0, 0.0, 120.0),
        _detection(2, 2 / 30.0, 0.0, 140.0),
        _detection(3, 3 / 30.0, 0.0, 160.0),
        _detection(4, 4 / 30.0, 0.0, 145.0),
        _detection(5, 5 / 30.0, 0.0, 130.0),
        _detection(6, 6 / 30.0, 0.0, 115.0),
    ]

    bounce_frame, bounce_t = cleaner._detect_bounce(inliers, fps=30.0)

    assert bounce_frame == 3
    assert bounce_t == inliers[3].timestamp_s


def test_clean_keeps_post_bounce_inliers_with_piecewise_fit() -> None:
    cleaner = BallPathCleaner()
    raw_path: list[BallDetection] = []

    for frame_idx in range(12):
        timestamp_s = frame_idx / 30.0
        if frame_idx <= 5:
            x_val = 150.0 + 12.0 * frame_idx
            y_val = 120.0 + 34.0 * frame_idx
        else:
            x_val = 150.0 + 12.0 * frame_idx
            y_val = 120.0 + 34.0 * 5 - 22.0 * (frame_idx - 5)
        raw_path.append(
            _detection(frame_idx, timestamp_s, x_val, y_val, confidence=0.9)
        )

    result = cleaner.clean(raw_path, fps=30.0)

    assert result is not None
    assert result.bounce_frame == 5
    assert max(detection.frame_idx for detection in result.inliers) >= 9


def test_clean_returns_none_for_empty_path() -> None:
    cleaner = BallPathCleaner()

    result = cleaner.clean([], fps=30.0)

    assert result is None


def test_group_candidates_by_frame_sorts_candidates_by_confidence() -> None:
    cleaner = BallPathCleaner()
    raw_path = [
        _detection(5, 5 / 30.0, 110.0, 210.0, confidence=0.55),
        _detection(5, 5 / 30.0, 112.0, 212.0, confidence=0.92),
        _detection(6, 6 / 30.0, 120.0, 220.0, confidence=0.75),
    ]

    frame_groups = cleaner._group_candidates_by_frame(raw_path)

    assert len(frame_groups) == 2
    assert [detection.confidence for detection in frame_groups[0]] == [0.92, 0.55]


def test_split_frame_groups_breaks_on_large_frame_gap() -> None:
    cleaner = BallPathCleaner()
    frame_groups = cleaner._group_candidates_by_frame(
        [
            _detection(0, 0.0, 100.0, 200.0),
            _detection(1, 1 / 30.0, 103.0, 210.0),
            _detection(2, 2 / 30.0, 106.0, 220.0),
            _detection(MAX_FRAME_GAP + 3, (MAX_FRAME_GAP + 3) / 30.0, 110.0, 230.0),
            _detection(MAX_FRAME_GAP + 4, (MAX_FRAME_GAP + 4) / 30.0, 114.0, 240.0),
        ]
    )

    segments = cleaner._split_frame_groups(frame_groups)

    assert [[group[0].frame_idx for group in segment] for segment in segments] == [
        [0, 1, 2],
        [MAX_FRAME_GAP + 3, MAX_FRAME_GAP + 4],
    ]


def test_select_track_skips_short_junk_prefix_and_keeps_longer_delivery() -> None:
    cleaner = BallPathCleaner()
    raw_path = [
        _detection(24, 24 / 30.0, 510.0, 444.0, confidence=0.99),
        _detection(25, 25 / 30.0, 506.0, 434.0, confidence=0.97),
        _detection(26, 26 / 30.0, 508.0, 434.0, confidence=0.98),
        _detection(27, 27 / 30.0, 506.0, 420.0, confidence=0.95),
        _detection(28, 28 / 30.0, 254.0, 150.0, confidence=0.82),
        _detection(29, 29 / 30.0, 268.0, 176.0, confidence=0.84),
        _detection(30, 30 / 30.0, 284.0, 200.0, confidence=0.87),
        _detection(31, 31 / 30.0, 292.0, 222.0, confidence=0.88),
        _detection(32, 32 / 30.0, 298.0, 242.0, confidence=0.89),
        _detection(33, 33 / 30.0, 306.0, 258.0, confidence=0.90),
    ]

    selected = cleaner._select_track(raw_path)

    assert [detection.frame_idx for detection in selected] == [28, 29, 30, 31, 32, 33]


def test_select_track_prefers_early_delivery_over_late_reappearance() -> None:
    cleaner = BallPathCleaner()
    raw_path = [
        *[
            _detection(
                frame_idx,
                frame_idx / 30.0,
                250.0 + 10.0 * (frame_idx - 20),
                150.0 + 18.0 * (frame_idx - 20),
                confidence=0.82,
            )
            for frame_idx in range(20, 28)
        ],
        *[
            _detection(
                frame_idx,
                frame_idx / 30.0,
                430.0 + 4.0 * (frame_idx - 40),
                640.0 + 3.0 * (frame_idx - 40),
                confidence=0.95,
            )
            for frame_idx in range(40, 52)
        ],
    ]

    selected = cleaner._select_track(raw_path)

    assert [detection.frame_idx for detection in selected] == list(range(20, 28))


def test_select_track_prefers_smooth_candidate_path_over_jumpy_noise() -> None:
    cleaner = BallPathCleaner()
    raw_path: list[BallDetection] = []

    for frame_idx in range(8):
        timestamp_s = frame_idx / 30.0
        raw_path.append(
            _detection(
                frame_idx,
                timestamp_s,
                120.0 + 11.0 * frame_idx,
                180.0 + 20.0 * frame_idx,
                confidence=0.62,
            )
        )
        raw_path.append(
            _detection(
                frame_idx,
                timestamp_s,
                40.0 if frame_idx % 2 == 0 else 680.0,
                80.0 if frame_idx % 2 == 0 else 1180.0,
                confidence=0.97,
            )
        )

    selected = cleaner._select_track(raw_path)

    assert [detection.x for detection in selected] == pytest.approx(
        [120.0 + 11.0 * frame_idx for frame_idx in range(8)]
    )
