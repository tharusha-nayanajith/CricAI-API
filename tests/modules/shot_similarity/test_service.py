import numpy as np
import pytest

import app.modules.shot_similarity.service as shot_similarity_service
from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts
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
    assert result.keypoints_detected == 33
    assert result.compared_frame == "bat_contact_frame"


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


@pytest.mark.asyncio
async def test_run_requires_bat_contact_frame() -> None:
    service = shot_similarity_service.ShotSimilarityService()

    with pytest.raises(FeatureError, match="Shot similarity requires a batter contact frame"):
        await service.run(_artifacts(with_contact=False))
