from pathlib import Path

import numpy as np
import pytest

import app.modules.shot_classifier.service as shot_classifier_service
from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts
from app.modules.preprocessor.models import BatContactResult, ContactMethod
from app.modules.shot_classifier.models import ShotClassifierResult


def _artifacts(
    roi_entry_frame_idx: int | None = 15,
    with_bat_contact: bool = False,
) -> VideoArtifacts:
    return VideoArtifacts(
        release_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        ball_path=[],
        bat_contact_frame=None,
        release_point=None,
        batter_roi_entry_frame_idx=roi_entry_frame_idx,
        bat_contact=(
            BatContactResult(
                contact_frame_idx=40,
                timestamp_s=40 / 30,
                annotated_frame=np.zeros((8, 8, 3), dtype=np.uint8),
                detection_score=0.8,
                method=ContactMethod.BALL_VELOCITY,
            )
            if with_bat_contact
            else None
        ),
    )


@pytest.mark.asyncio
async def test_run_returns_prediction_from_roi_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    service = shot_classifier_service.ShotClassifierService()

    class _FakeModel:
        def predict(self, batch, verbose=0):
            _ = verbose
            assert batch.shape == (1, 30, 224, 224, 3)
            predictions = np.zeros((1, 10), dtype=np.float32)
            predictions[0, 6] = 0.91
            predictions[0, 0] = 0.09
            return predictions

    monkeypatch.setattr(shot_classifier_service, "_load_model", lambda: _FakeModel())
    monkeypatch.setattr(
        shot_classifier_service,
        "_read_clip_frames",
        lambda video_path, start_frame_idx, frame_count, frame_size: np.zeros(
            (frame_count, frame_size[0], frame_size[1], 3),
            dtype=np.uint8,
        ),
    )

    result = await service.run(_artifacts(), Path("input.mp4"), video_url="input.mp4")

    assert isinstance(result, ShotClassifierResult)
    assert result.predicted_shot == "pull"
    assert result.confidence == pytest.approx(0.91)
    assert result.frame_start_index == 15
    assert result.frame_end_index == 44
    assert result.trigger_source == "batter_roi_entry"


@pytest.mark.asyncio
async def test_run_falls_back_to_bat_contact_when_roi_entry_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = shot_classifier_service.ShotClassifierService()

    class _FakeModel:
        def predict(self, batch, verbose=0):
            _ = batch, verbose
            predictions = np.zeros((1, 10), dtype=np.float32)
            predictions[0, 8] = 0.77
            return predictions

    monkeypatch.setattr(shot_classifier_service, "_load_model", lambda: _FakeModel())
    monkeypatch.setattr(
        shot_classifier_service,
        "_read_clip_frames",
        lambda video_path, start_frame_idx, frame_count, frame_size: np.zeros(
            (frame_count, frame_size[0], frame_size[1], 3),
            dtype=np.uint8,
        ),
    )

    result = await service.run(
        _artifacts(roi_entry_frame_idx=None, with_bat_contact=True),
        Path("input.mp4"),
    )

    assert result.predicted_shot == "straight"
    assert result.frame_start_index == 11
    assert result.trigger_source == "bat_contact_fallback"


@pytest.mark.asyncio
async def test_run_requires_roi_entry_or_bat_contact() -> None:
    service = shot_classifier_service.ShotClassifierService()

    with pytest.raises(
        FeatureError,
        match="Shot classifier requires a batter ROI entry frame or bat-contact fallback frame",
    ):
        await service.run(_artifacts(roi_entry_frame_idx=None), Path("input.mp4"))
