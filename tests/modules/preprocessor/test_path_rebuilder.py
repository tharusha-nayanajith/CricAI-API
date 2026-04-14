from app.modules.preprocessor.models import BallDetection, FrameBallDetections
from app.modules.preprocessor.path_rebuilder import DeliveryPathRebuilder


def _detection(frame_idx: int, x: float, y: float, confidence: float = 0.9) -> BallDetection:
    return BallDetection(
        frame_idx=frame_idx,
        timestamp_s=frame_idx / 30.0,
        x=x,
        y=y,
        confidence=confidence,
    )


def test_rebuilder_trims_unsupported_tail_after_roi_entry() -> None:
    rebuilder = DeliveryPathRebuilder()
    raw_candidates = [
        _detection(1, 101.0, 110.0),
        _detection(2, 104.0, 126.0),
        _detection(3, 109.0, 144.0),
        _detection(4, 116.0, 164.0),
        _detection(5, 125.0, 186.0),
        _detection(6, 136.0, 210.0),
        _detection(7, 149.0, 236.0),
        _detection(8, 164.0, 264.0),
        _detection(9, 420.0, 760.0),
        _detection(10, 500.0, 820.0),
        _detection(11, 560.0, 900.0),
    ]
    grouped_candidates = [
        FrameBallDetections(
            frame_idx=detection.frame_idx,
            timestamp_s=detection.timestamp_s,
            detections=[detection],
        )
        for detection in raw_candidates
    ]

    rebuilt = rebuilder.rebuild(
        raw_candidates,
        fps=30.0,
        grouped_candidates=grouped_candidates,
        roi_entry_frame_idx=7,
    )

    assert rebuilt is not None
    assert rebuilt[0].frame_idx == 1
    assert rebuilt[-1].frame_idx == 8
    assert all(detection.x < 250.0 for detection in rebuilt)


def test_rebuilder_prefers_consistent_branch_over_far_distractor() -> None:
    rebuilder = DeliveryPathRebuilder()
    consistent = [
        _detection(1, 100.0, 110.0, 0.88),
        _detection(2, 104.0, 126.0, 0.88),
        _detection(3, 109.0, 144.0, 0.88),
        _detection(4, 116.0, 164.0, 0.88),
        _detection(5, 125.0, 186.0, 0.88),
        _detection(6, 136.0, 178.0, 0.88),
        _detection(7, 149.0, 164.0, 0.88),
        _detection(8, 164.0, 152.0, 0.88),
    ]
    distractors = [
        _detection(4, 420.0, 300.0, 0.95),
        _detection(5, 470.0, 330.0, 0.95),
        _detection(6, 520.0, 360.0, 0.95),
    ]
    raw_candidates = consistent + distractors
    grouped_candidates = []
    for frame_idx in range(1, 9):
        detections = [d for d in raw_candidates if d.frame_idx == frame_idx]
        grouped_candidates.append(
            FrameBallDetections(
                frame_idx=frame_idx,
                timestamp_s=frame_idx / 30.0,
                detections=detections,
            )
        )

    rebuilt = rebuilder.rebuild(
        raw_candidates,
        fps=30.0,
        grouped_candidates=grouped_candidates,
        roi_entry_frame_idx=None,
    )

    assert rebuilt is not None
    assert rebuilt[-1].frame_idx == 8
    assert max(detection.x for detection in rebuilt) < 200.0
