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
            predictions = np.zeros((1, len(shot_classifier_service.SHOT_CLASS_LABELS)), dtype=np.float32)
            predictions[0, shot_classifier_service.SHOT_CLASS_LABELS.index("pull")] = 0.91
            predictions[0, shot_classifier_service.SHOT_CLASS_LABELS.index("cut")] = 0.09
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
    monkeypatch.setattr(
        shot_classifier_service,
        "_extract_features",
        lambda frames_normalized: np.zeros((shot_classifier_service.FEATURE_DIM,), dtype=np.float32),
    )
    monkeypatch.setattr(
        shot_classifier_service,
        "_run_mistake_analysis",
        lambda reference_shot, features: {"mistakes": []},
    )
    monkeypatch.setattr(
        shot_classifier_service,
        "_build_technique_map",
        lambda technique_frame_bgr, handedness: (
            {
                "torso": 88.0,
                "front_elbow": 77.0,
                "back_elbow": 66.0,
                "back_knee": 55.0,
                "shoulders": 44.0,
            },
            [{"body_part": "Torso", "score": 88.0, "metric": "spine_alignment"}],
        ),
    )

    result = await service.run(_artifacts(), Path("input.mp4"), video_url="input.mp4")

    assert isinstance(result, ShotClassifierResult)
    assert result.predicted_shot == "pull"
    assert result.confidence == pytest.approx(0.91)
    assert result.frame_start_index == 15
    assert result.frame_end_index == 44
    assert result.trigger_source == "batter_roi_entry"
    assert result.technique_map == {
        "torso": 88.0,
        "front_elbow": 77.0,
        "back_elbow": 66.0,
        "back_knee": 55.0,
        "shoulders": 44.0,
    }
    assert result.visual_feedback == {
        "mistakes": [],
        "technique_map": result.technique_map,
        "technique_details": [{"body_part": "Torso", "score": 88.0, "metric": "spine_alignment"}],
    }


@pytest.mark.asyncio
async def test_run_falls_back_to_bat_contact_when_roi_entry_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = shot_classifier_service.ShotClassifierService()

    class _FakeModel:
        def predict(self, batch, verbose=0):
            _ = batch, verbose
            predictions = np.zeros((1, len(shot_classifier_service.SHOT_CLASS_LABELS)), dtype=np.float32)
            predictions[0, shot_classifier_service.SHOT_CLASS_LABELS.index("drive")] = 0.77
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
    monkeypatch.setattr(
        shot_classifier_service,
        "_extract_features",
        lambda frames_normalized: np.zeros((shot_classifier_service.FEATURE_DIM,), dtype=np.float32),
    )
    monkeypatch.setattr(
        shot_classifier_service,
        "_run_mistake_analysis",
        lambda reference_shot, features: {"mistakes": []},
    )
    monkeypatch.setattr(
        shot_classifier_service,
        "_build_technique_map",
        lambda technique_frame_bgr, handedness: ({}, []),
    )

    result = await service.run(
        _artifacts(roi_entry_frame_idx=None, with_bat_contact=True),
        Path("input.mp4"),
    )

    assert result.predicted_shot == "drive"
    assert result.frame_start_index == 11
    assert result.trigger_source == "bat_contact_fallback"


@pytest.mark.asyncio
async def test_run_requires_roi_entry_or_bat_contact() -> None:
    service = shot_classifier_service.ShotClassifierService()

    with pytest.raises(
        FeatureError,
        match="Shot classifier requires a batter ROI entry frame, bat-contact fallback frame, or ball-path fallback frame",
    ):
        await service.run(_artifacts(roi_entry_frame_idx=None), Path("input.mp4"))


@pytest.mark.asyncio
async def test_run_uses_intended_shot_for_analysis_when_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = shot_classifier_service.ShotClassifierService()
    recorded_reference_shots: list[str] = []

    class _FakeModel:
        def predict(self, batch, verbose=0):
            _ = batch, verbose
            predictions = np.zeros((1, len(shot_classifier_service.SHOT_CLASS_LABELS)), dtype=np.float32)
            predictions[0, shot_classifier_service.SHOT_CLASS_LABELS.index("pull")] = 0.91
            predictions[0, shot_classifier_service.SHOT_CLASS_LABELS.index("drive")] = 0.22
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
    monkeypatch.setattr(
        shot_classifier_service,
        "_extract_features",
        lambda frames_normalized: np.zeros((shot_classifier_service.FEATURE_DIM,), dtype=np.float32),
    )

    def _fake_mistake_analysis(reference_shot: str, features: np.ndarray) -> dict[str, object]:
        _ = features
        recorded_reference_shots.append(reference_shot)
        return {"mistakes": [{"severity": "warning"}]}

    monkeypatch.setattr(shot_classifier_service, "_run_mistake_analysis", _fake_mistake_analysis)
    monkeypatch.setattr(
        shot_classifier_service,
        "_build_technique_map",
        lambda technique_frame_bgr, handedness: (
            {
                "torso": 90.0,
                "front_elbow": 80.0,
                "back_elbow": 70.0,
                "back_knee": 60.0,
                "shoulders": 50.0,
            },
            [],
        ),
    )

    result = await service.run(
        _artifacts(),
        Path("input.mp4"),
        video_url="input.mp4",
        intended_shot="cover drive",
    )

    assert recorded_reference_shots == ["drive"]
    assert result.predicted_shot == "pull"
    assert result.intended_shot == "drive"
    assert result.intent_match is False
    assert result.intended_shot_score == pytest.approx(0.22)
    assert result.mistake_analysis_basis == "intended_shot"
    assert result.mistake_analysis_reference_shot == "drive"


def test_build_technique_map_uses_local_pose_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    landmarks = {
        11: (0.40, 0.30, 0.0, 0.99),
        12: (0.58, 0.32, 0.0, 0.99),
        13: (0.35, 0.45, 0.0, 0.99),
        14: (0.68, 0.40, 0.0, 0.99),
        15: (0.30, 0.58, 0.0, 0.99),
        16: (0.78, 0.55, 0.0, 0.99),
        23: (0.44, 0.62, 0.0, 0.99),
        24: (0.57, 0.64, 0.0, 0.99),
        25: (0.48, 0.80, 0.0, 0.99),
        26: (0.60, 0.82, 0.0, 0.99),
        27: (0.47, 0.97, 0.0, 0.99),
        28: (0.63, 0.96, 0.0, 0.99),
    }
    monkeypatch.setattr(shot_classifier_service, "_extract_pose_landmarks", lambda frame_bgr: landmarks)

    technique_map, details = shot_classifier_service._build_technique_map(
        technique_frame_bgr=np.zeros((32, 32, 3), dtype=np.uint8),
        handedness=None,
    )

    assert set(technique_map) == {"torso", "front_elbow", "back_elbow", "back_knee", "shoulders"}
    assert len(details) == 5
    assert all(0.0 <= value <= 100.0 for value in technique_map.values())



def test_resolve_model_path_uses_named_asset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    model_path = assets_dir / "model_weights.h5"
    model_path.write_bytes(b"weights")

    monkeypatch.delenv("SHOT_CLASSIFIER_MODEL_PATH", raising=False)
    monkeypatch.setattr(shot_classifier_service, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(shot_classifier_service, "MODEL_PATH", model_path)
    monkeypatch.setattr(
        shot_classifier_service,
        "EXTERNAL_MODEL_PATH",
        tmp_path / "external_model_weights.h5",
    )

    assert shot_classifier_service._resolve_model_path() == model_path



def test_resolve_model_path_falls_back_to_any_h5_asset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    fallback_model = assets_dir / "custom_weights.h5"
    fallback_model.write_bytes(b"weights")

    monkeypatch.delenv("SHOT_CLASSIFIER_MODEL_PATH", raising=False)
    monkeypatch.setattr(shot_classifier_service, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(shot_classifier_service, "MODEL_PATH", assets_dir / "model_weights.h5")
    monkeypatch.setattr(
        shot_classifier_service,
        "EXTERNAL_MODEL_PATH",
        tmp_path / "external_model_weights.h5",
    )

    assert shot_classifier_service._resolve_model_path() == fallback_model
