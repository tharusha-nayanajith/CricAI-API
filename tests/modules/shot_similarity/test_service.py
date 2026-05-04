import numpy as np
import pytest

import app.modules.shot_similarity.service as shot_similarity_service
from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts
from app.modules.preprocessor.models import BatterROI
from app.modules.shot_similarity.models import PoseKeypoint, ShotReference, ShotSimilarityResult


def _artifacts(with_contact: bool = True) -> VideoArtifacts:
    return VideoArtifacts(
        release_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        ball_path=[],
        bat_contact_frame=(np.ones((8, 8, 3), dtype=np.uint8) if with_contact else None),
        release_point=None,
    )


@pytest.mark.asyncio
async def test_run_returns_best_match(monkeypatch: pytest.MonkeyPatch) -> None:
    service = shot_similarity_service.ShotSimilarityService()
    user_array = np.arange(99, dtype=np.float32)
    golden_keypoints = [
        PoseKeypoint(x=float(idx), y=float(idx + 1), z=float(idx + 2), visibility=0.9)
        for idx in range(33)
    ]
    monkeypatch.setattr(
        shot_similarity_service,
        "_extract_pose_landmarks_2d",
        lambda frame, selected_landmarks: [],
    )
    monkeypatch.setattr(
        shot_similarity_service,
        "_extract_keypoints_from_frame",
        lambda frame, selected_landmarks: user_array,
    )
    monkeypatch.setattr(
        shot_similarity_service,
        "_array_to_keypoints",
        lambda keypoints_array: golden_keypoints,
    )
    monkeypatch.setattr(
        shot_similarity_service,
        "_load_reference_library",
        lambda: {
            "Virat Kohli": {
                "cover_drive": ShotReference(keypoints=golden_keypoints),
            }
        },
    )

    result = await service.run(_artifacts(), video_url="sample.mp4")

    assert isinstance(result, ShotSimilarityResult)
    assert result.matched_player == "Virat Kohli"
    assert result.shot_type == "drive"
    assert result.keypoints_detected == 33 * shot_similarity_service.TARGET_FRAME_COUNT
    assert result.compared_frame == "30_frame_pose_sequence"


@pytest.mark.asyncio
async def test_run_filters_by_classified_shot_type(monkeypatch: pytest.MonkeyPatch) -> None:
    service = shot_similarity_service.ShotSimilarityService()
    user_array = np.arange(99, dtype=np.float32)
    drive_keypoints = [
        PoseKeypoint(x=float(idx), y=float(idx + 1), z=float(idx + 2), visibility=0.9)
        for idx in range(33)
    ]
    cut_keypoints = [
        PoseKeypoint(x=float(idx + 5), y=float(idx + 6), z=float(idx + 7), visibility=0.9)
        for idx in range(33)
    ]
    monkeypatch.setattr(
        shot_similarity_service,
        "_extract_pose_landmarks_2d",
        lambda frame, selected_landmarks: [],
    )
    monkeypatch.setattr(
        shot_similarity_service,
        "_extract_keypoints_from_frame",
        lambda frame, selected_landmarks: user_array,
    )
    monkeypatch.setattr(
        shot_similarity_service,
        "_array_to_keypoints",
        lambda keypoints_array: drive_keypoints,
    )
    monkeypatch.setattr(
        shot_similarity_service,
        "_load_reference_library",
        lambda: {
            "Virat Kohli": {
                "cover_drive": ShotReference(keypoints=drive_keypoints),
                "cut_shot": ShotReference(keypoints=cut_keypoints),
            }
        },
    )

    result = await service.run(
        _artifacts(),
        video_url="sample.mp4",
        classified_shot_type="cut",
    )

    assert result.matched_player == "Virat Kohli"
    assert result.shot_type == "cut"


def test_extract_pose_frame_keypoints_maps_roi_crop_to_full_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    roi = BatterROI(x=50, y=20, width=100, height=60)

    class Landmark:
        x = 10.0
        y = 15.0
        z = 0.2
        visibility = 0.8

    monkeypatch.setattr(
        shot_similarity_service,
        "_extract_pose_landmarks_2d",
        lambda frame, selected_landmarks: [Landmark() for _ in selected_landmarks],
    )
    monkeypatch.setattr(
        shot_similarity_service,
        "_extract_keypoints_from_frame",
        lambda frame, selected_landmarks: None,
    )

    keypoints = shot_similarity_service._extract_pose_frame_keypoints(frame, roi)

    assert len(keypoints) == shot_similarity_service.LANDMARK_COUNT
    assert keypoints[0].x == pytest.approx(0.3)
    assert keypoints[0].y == pytest.approx(0.35)
    assert keypoints[0].visibility == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_run_rejects_low_quality_pose_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = shot_similarity_service.ShotSimilarityService()
    reference_keypoints = [
        PoseKeypoint(x=float(idx), y=float(idx + 1), z=float(idx + 2), visibility=0.9)
        for idx in range(33)
    ]
    monkeypatch.setattr(
        shot_similarity_service,
        "_load_reference_library",
        lambda: {
            "Virat Kohli": {
                "cover_drive": ShotReference(keypoints=reference_keypoints),
            }
        },
    )
    monkeypatch.setattr(
        shot_similarity_service,
        "_extract_pose_landmarks_2d",
        lambda frame, selected_landmarks: [],
    )
    monkeypatch.setattr(
        shot_similarity_service,
        "_extract_keypoints_from_frame",
        lambda frame, selected_landmarks: None,
    )

    with pytest.raises(FeatureError, match="Pose detection quality is too low"):
        await service.run(_artifacts(), video_url="sample.mp4")


@pytest.mark.asyncio
async def test_run_requires_video_or_bat_contact_frame() -> None:
    service = shot_similarity_service.ShotSimilarityService()

    with pytest.raises(FeatureError, match="standardized batting video or a batter contact frame"):
        await service.run(_artifacts(with_contact=False))
