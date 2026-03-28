import numpy as np
import pytest

import app.modules.action_legality.service as action_service
from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts
from app.modules.action_legality.models import (
    ActionLegalityMetadata,
    ActionLegalityResult,
    ActionLegalityScaler,
)
from app.modules.action_legality.service import ActionLegalityService
from app.modules.preprocessor.models import ReleasePoint


def _artifacts(with_raw_frame: bool = True) -> VideoArtifacts:
    raw_frame = np.full((8, 8, 3), 50, dtype=np.uint8)
    annotated_frame = np.full((8, 8, 3), 200, dtype=np.uint8)
    release_point = ReleasePoint(
        frame_idx=12,
        timestamp_s=0.4,
        hand_position=(300.0, 260.0),
        confidence=0.93,
        annotated_frame=annotated_frame,
        raw_frame=(raw_frame if with_raw_frame else None),
    )
    return VideoArtifacts(
        release_frame=annotated_frame,
        ball_path=[],
        bat_contact_frame=None,
        release_point=release_point,
    )


class _FakeModel:
    def __init__(self, prediction: float) -> None:
        self._prediction = prediction

    def predict(self, values: np.ndarray, verbose: int = 0) -> np.ndarray:
        _ = values, verbose
        return np.asarray([[self._prediction]], dtype=np.float32)


@pytest.mark.asyncio
async def test_run_returns_action_legality_result(monkeypatch: pytest.MonkeyPatch) -> None:
    service = ActionLegalityService()
    artifacts = _artifacts()

    monkeypatch.setattr(
        action_service,
        "_load_metadata",
        lambda: ActionLegalityMetadata(feature_dim=2, select_landmarks=[11, 12]),
    )
    monkeypatch.setattr(
        action_service,
        "_load_scaler",
        lambda: ActionLegalityScaler(mean=[0.0, 0.0], scale=[1.0, 1.0]),
    )
    monkeypatch.setattr(action_service, "_load_model", lambda: _FakeModel(0.82))
    monkeypatch.setattr(
        action_service,
        "_extract_keypoints_from_frame",
        lambda frame, selected_landmarks: np.asarray([0.5, 0.4], dtype=np.float32),
    )
    monkeypatch.setattr(
        action_service,
        "_normalize_keypoints_by_torso",
        lambda features, selected_landmarks: features,
    )

    result = await service.run(artifacts, video_url="sample.mp4")

    assert isinstance(result, ActionLegalityResult)
    assert result.verdict == "illegal"
    assert result.illegal_probability == pytest.approx(0.82)
    assert result.legal_probability == pytest.approx(0.18)
    assert result.confidence == pytest.approx(0.82)
    assert result.release_frame_index == 12
    assert result.release_timestamp_s == pytest.approx(0.4)
    assert result.release_confidence == pytest.approx(0.93)
    assert result.video_url == "sample.mp4"
    assert result.used_annotated_release_frame is False


@pytest.mark.asyncio
async def test_run_uses_annotated_frame_when_raw_release_frame_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ActionLegalityService()
    artifacts = _artifacts(with_raw_frame=False)

    monkeypatch.setattr(
        action_service,
        "_load_metadata",
        lambda: ActionLegalityMetadata(feature_dim=2, select_landmarks=[11]),
    )
    monkeypatch.setattr(
        action_service,
        "_load_scaler",
        lambda: ActionLegalityScaler(mean=[0.0, 0.0], scale=[1.0, 1.0]),
    )
    monkeypatch.setattr(action_service, "_load_model", lambda: _FakeModel(0.2))
    monkeypatch.setattr(
        action_service,
        "_extract_keypoints_from_frame",
        lambda frame, selected_landmarks: np.asarray([frame[0, 0, 0], 0.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        action_service,
        "_normalize_keypoints_by_torso",
        lambda features, selected_landmarks: features,
    )

    result = await service.run(artifacts)

    assert result.verdict == "legal"
    assert result.used_annotated_release_frame is True
    assert result.normalized_keypoints[0] == pytest.approx(200.0)


@pytest.mark.asyncio
async def test_run_raises_feature_error_when_pose_landmarks_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ActionLegalityService()

    monkeypatch.setattr(
        action_service,
        "_load_metadata",
        lambda: ActionLegalityMetadata(feature_dim=2, select_landmarks=[11, 12]),
    )
    monkeypatch.setattr(
        action_service,
        "_load_scaler",
        lambda: ActionLegalityScaler(mean=[0.0, 0.0], scale=[1.0, 1.0]),
    )
    monkeypatch.setattr(action_service, "_load_model", lambda: _FakeModel(0.4))
    monkeypatch.setattr(
        action_service,
        "_extract_keypoints_from_frame",
        lambda frame, selected_landmarks: None,
    )

    with pytest.raises(FeatureError, match="Pose landmarks were not detected"):
        await service.run(_artifacts())
