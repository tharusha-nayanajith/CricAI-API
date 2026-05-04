import numpy as np

from app.modules.bowler_performance.release_visuals import (
    LANDMARK_LEFT_ANKLE,
    LANDMARK_LEFT_ELBOW,
    LANDMARK_LEFT_HIP,
    LANDMARK_LEFT_KNEE,
    LANDMARK_LEFT_SHOULDER,
    LANDMARK_LEFT_WRIST,
    LANDMARK_NOSE,
    LANDMARK_RIGHT_ANKLE,
    LANDMARK_RIGHT_ELBOW,
    LANDMARK_RIGHT_HIP,
    LANDMARK_RIGHT_KNEE,
    LANDMARK_RIGHT_SHOULDER,
    LANDMARK_RIGHT_WRIST,
    build_release_visual_analysis,
)
from app.modules.preprocessor.models import ReleasePoint


def test_build_release_visual_analysis_returns_expected_overlays(monkeypatch) -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    release_point = ReleasePoint(
        frame_idx=10,
        timestamp_s=0.33,
        hand_position=(830.0, 140.0),
        confidence=0.92,
        annotated_frame=frame.copy(),
        raw_frame=frame.copy(),
    )

    class FakeLandmark:
        def __init__(self, id: int, x: float, y: float, visibility: float = 0.99) -> None:
            self.id = id
            self.x = x
            self.y = y
            self.visibility = visibility

    monkeypatch.setattr(
        "app.modules.bowler_performance.release_visuals._extract_pose_landmarks",
        lambda _frame: [
            FakeLandmark(LANDMARK_NOSE, 700.0, 160.0),
            FakeLandmark(LANDMARK_LEFT_SHOULDER, 610.0, 250.0),
            FakeLandmark(LANDMARK_RIGHT_SHOULDER, 760.0, 250.0),
            FakeLandmark(LANDMARK_LEFT_ELBOW, 660.0, 205.0),
            FakeLandmark(LANDMARK_RIGHT_ELBOW, 795.0, 185.0),
            FakeLandmark(LANDMARK_LEFT_WRIST, 690.0, 165.0),
            FakeLandmark(LANDMARK_RIGHT_WRIST, 820.0, 145.0),
            FakeLandmark(LANDMARK_LEFT_HIP, 650.0, 410.0),
            FakeLandmark(LANDMARK_RIGHT_HIP, 760.0, 412.0),
            FakeLandmark(LANDMARK_LEFT_KNEE, 640.0, 535.0),
            FakeLandmark(LANDMARK_RIGHT_KNEE, 770.0, 535.0),
            FakeLandmark(LANDMARK_LEFT_ANKLE, 635.0, 650.0),
            FakeLandmark(LANDMARK_RIGHT_ANKLE, 780.0, 655.0),
        ],
    )

    analysis, overlay_image = build_release_visual_analysis(frame, release_point)

    assert analysis is not None
    assert overlay_image is not None
    assert analysis.release_frame_width == 1280
    assert analysis.release_frame_height == 720
    assert analysis.bowling_arm == "right"
    assert len(analysis.overlays) == 4
    overlay_ids = {overlay.overlay_id for overlay in analysis.overlays}
    assert overlay_ids == {
        "release_arm_line",
        "body_lean_line",
        "front_leg_angle",
        "release_to_hip_alignment",
    }
    body_lean_overlay = next(
        overlay for overlay in analysis.overlays if overlay.overlay_id == "body_lean_line"
    )
    assert body_lean_overlay.observation
    assert body_lean_overlay.reason
    assert body_lean_overlay.recommendation
    assert any("hip" in note.lower() or "release" in note.lower() for note in analysis.summary_notes)


def test_build_release_visual_analysis_returns_none_when_pose_missing(monkeypatch) -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "app.modules.bowler_performance.release_visuals._extract_pose_landmarks",
        lambda _frame: [],
    )

    analysis, overlay_image = build_release_visual_analysis(frame, release_point=None)

    assert analysis is None
    assert overlay_image is None
