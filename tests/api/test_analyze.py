import json
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pytest

import app.api.analyze as analyze_module
from app.api.deps import require_entitlement
from app.main import app
from app.modules.action_legality.models import ActionLegalityResult
from app.modules.preprocessor.models import ReleasePoint
from app.modules.shot_classifier.models import ShotClassifierResult
from app.modules.shot_similarity.models import ShotSimilarityResult
from app.modules.users.models import UserProfile
from app.storage.results import get_job_status, initialize_job_status
from tests.conftest import CalibrationDataFactory


@pytest.mark.asyncio
async def test_analyze_with_valid_payload_returns_job_id(
    test_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = CalibrationDataFactory()
    current_user = UserProfile(
        id=uuid4(),
        email="authorized@example.com",
        full_name="Authorized User",
        created_at=datetime.now(UTC),
        is_active=True,
        revenuecat_customer_id=None,
        entitlement_status="active",
        entitlement_expires_at=None,
        current_tier="coach",
        clips_used_this_month=0,
        quota_reset_at=datetime.now(UTC),
    )

    async def fake_require_entitlement() -> UserProfile:
        return current_user

    async def fake_enforce_clip_quota(session, user_id):
        _ = session, user_id
        return current_user

    app.dependency_overrides[require_entitlement] = fake_require_entitlement
    monkeypatch.setattr(analyze_module._user_service, "enforce_clip_quota", fake_enforce_clip_quota)
    response = await test_client.post(
        "/analyze",
        files={"video": ("sample.mp4", b"00", "video/mp4")},
        data={
            "calibration": calibration.model_dump_json(),
            "features": "bowler_performance,action_legality,shot_classifier,shot_similarity",
        },
    )

    assert response.status_code == 200
    assert "job_id" in response.json()


@pytest.mark.asyncio
async def test_analyze_normalizes_intended_shot_before_queueing(
    test_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = CalibrationDataFactory()
    current_user = _authorized_user()
    queued_jobs: list[dict[str, object]] = []

    async def fake_require_entitlement() -> UserProfile:
        return current_user

    async def fake_enforce_clip_quota(session, user_id):
        _ = session, user_id
        return current_user

    def fake_copy_upload_to_temp(upload):
        return f"/tmp/{upload.filename}"

    def fake_delay(
        job_id,
        selected_features,
        source_video_path,
        filename,
        calibration_payload,
        intended_shot=None,
    ):
        queued_jobs.append(
            {
                "job_id": job_id,
                "selected_features": selected_features,
                "source_video_path": source_video_path,
                "filename": filename,
                "calibration_payload": calibration_payload,
                "intended_shot": intended_shot,
            }
        )

    app.dependency_overrides[require_entitlement] = fake_require_entitlement
    monkeypatch.setattr(analyze_module._user_service, "enforce_clip_quota", fake_enforce_clip_quota)
    monkeypatch.setattr(analyze_module, "_copy_upload_to_temp", fake_copy_upload_to_temp)
    monkeypatch.setattr(analyze_module.process_video_job, "delay", fake_delay)

    response = await test_client.post(
        "/analyze",
        files={"video": ("sample.mp4", b"00", "video/mp4")},
        data={
            "calibration": calibration.model_dump_json(),
            "features": "shot_classifier",
            "intended_shot": "cover drive",
        },
    )

    assert response.status_code == 200
    assert queued_jobs[0]["intended_shot"] == "drive"


@pytest.mark.asyncio
async def test_analyze_with_malformed_calibration_returns_422(
    test_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_user = UserProfile(
        id=uuid4(),
        email="authorized@example.com",
        full_name="Authorized User",
        created_at=datetime.now(UTC),
        is_active=True,
        revenuecat_customer_id=None,
        entitlement_status="active",
        entitlement_expires_at=None,
        current_tier="coach",
        clips_used_this_month=0,
        quota_reset_at=datetime.now(UTC),
    )

    async def fake_require_entitlement() -> UserProfile:
        return current_user

    async def fake_enforce_clip_quota(session, user_id):
        _ = session, user_id
        return current_user

    app.dependency_overrides[require_entitlement] = fake_require_entitlement
    monkeypatch.setattr(analyze_module._user_service, "enforce_clip_quota", fake_enforce_clip_quota)
    response = await test_client.post(
        "/analyze",
        files={"video": ("sample.mp4", b"00", "video/mp4")},
        data={"calibration": json.dumps({"invalid": "payload"})},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_process_job_runs_action_legality_without_ball_tracking(
    fake_redis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = fake_redis
    calibration = CalibrationDataFactory()
    job_id = "action-only-job"
    recorded_require_ball_path: list[bool] = []

    async def fake_preprocessor_run(video_path, calibration_data, require_ball_path=True):
        _ = video_path, calibration_data
        recorded_require_ball_path.append(require_ball_path)
        annotated = np.zeros((4, 4, 3), dtype=np.uint8)
        raw = np.ones((4, 4, 3), dtype=np.uint8)
        return analyze_module.VideoArtifacts(
            release_frame=annotated,
            ball_path=[],
            bat_contact_frame=None,
            release_point=ReleasePoint(
                frame_idx=7,
                timestamp_s=0.23,
                hand_position=(10.0, 20.0),
                confidence=0.91,
                annotated_frame=annotated,
                raw_frame=raw,
            ),
        )

    async def fake_action_legality_run(artifacts, video_url=None):
        _ = artifacts, video_url
        return ActionLegalityResult(
            verdict="legal",
            illegal_probability=0.11,
            legal_probability=0.89,
            confidence=0.89,
            release_frame_index=7,
            release_timestamp_s=0.23,
            release_confidence=0.91,
            selected_landmarks=[11, 13, 15, 12, 14, 16, 23, 25, 27],
            normalized_keypoints=[0.0] * 27,
            video_url="upload.mp4",
            used_annotated_release_frame=False,
        )

    monkeypatch.setattr(analyze_module._preprocessor, "run", fake_preprocessor_run)

    monkeypatch.setattr(
        analyze_module._action_legality_service,
        "run",
        fake_action_legality_run,
    )

    await initialize_job_status(job_id)
    await analyze_module.process_job(
        job_id=job_id,
        selected_features=["action_legality"],
        video_bytes=b"123",
        filename="upload.mp4",
        calibration=calibration,
    )

    job_status = await get_job_status(job_id)

    assert recorded_require_ball_path == [False]
    assert job_status.action_legality.status == "done"
    assert job_status.action_legality.result is not None
    assert job_status.action_legality.result["verdict"] == "legal"
    assert job_status.bowler_performance.status == "pending"


@pytest.mark.asyncio
async def test_process_job_runs_shot_similarity_with_preprocessed_contact_frame(
    fake_redis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = fake_redis
    calibration = CalibrationDataFactory()
    job_id = "shot-similarity-job"
    recorded_require_ball_path: list[bool] = []

    async def fake_preprocessor_run(video_path, calibration_data, require_ball_path=True):
        _ = video_path, calibration_data
        recorded_require_ball_path.append(require_ball_path)
        annotated = np.zeros((4, 4, 3), dtype=np.uint8)
        return analyze_module.VideoArtifacts(
            release_frame=annotated,
            ball_path=[],
            bat_contact_frame=np.ones((4, 4, 3), dtype=np.uint8),
            release_point=ReleasePoint(
                frame_idx=7,
                timestamp_s=0.23,
                hand_position=(10.0, 20.0),
                confidence=0.91,
                annotated_frame=annotated,
                raw_frame=annotated,
            ),
        )

    async def fake_shot_similarity_run(artifacts, video_url=None):
        _ = artifacts, video_url
        return ShotSimilarityResult(
            similarity_percentage=92.5,
            matched_player="Virat Kohli",
            shot_type="cover_drive",
            keypoints_detected=33,
            confidence=91.0,
            feedback=["Open up your left shoulder."],
            compared_frame="bat_contact_frame",
            video_url="upload.mp4",
        )

    monkeypatch.setattr(analyze_module._preprocessor, "run", fake_preprocessor_run)

    monkeypatch.setattr(
        analyze_module._shot_similarity_service,
        "run",
        fake_shot_similarity_run,
    )

    await initialize_job_status(job_id)
    await analyze_module.process_job(
        job_id=job_id,
        selected_features=["shot_similarity"],
        video_bytes=b"123",
        filename="upload.mp4",
        calibration=calibration,
    )

    job_status = await get_job_status(job_id)

    assert recorded_require_ball_path == [True]
    assert job_status.shot_similarity.status == "done"
    assert job_status.shot_similarity.result is not None
    assert job_status.shot_similarity.result["matched_player"] == "Virat Kohli"


@pytest.mark.asyncio
async def test_process_job_runs_shot_classifier_with_roi_entry_frame(
    fake_redis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = fake_redis
    calibration = CalibrationDataFactory()
    job_id = "shot-classifier-job"
    recorded_require_ball_path: list[bool] = []

    async def fake_preprocessor_run(video_path, calibration_data, require_ball_path=True):
        _ = video_path, calibration_data
        recorded_require_ball_path.append(require_ball_path)
        annotated = np.zeros((4, 4, 3), dtype=np.uint8)
        return analyze_module.VideoArtifacts(
            release_frame=annotated,
            ball_path=[],
            bat_contact_frame=None,
            release_point=ReleasePoint(
                frame_idx=7,
                timestamp_s=0.23,
                hand_position=(10.0, 20.0),
                confidence=0.91,
                annotated_frame=annotated,
                raw_frame=annotated,
            ),
            batter_roi_entry_frame_idx=18,
        )

    async def fake_shot_classifier_run(artifacts, video_path, video_url=None):
        _ = artifacts, video_path, video_url
        return ShotClassifierResult(
            predicted_shot="cover",
            confidence=0.93,
            probabilities={"cover": 0.93},
            frames_used=30,
            frame_start_index=18,
            frame_end_index=47,
            roi_entry_frame_index=18,
            trigger_source="batter_roi_entry",
            video_url="upload.mp4",
        )

    monkeypatch.setattr(analyze_module._preprocessor, "run", fake_preprocessor_run)

    monkeypatch.setattr(
        analyze_module._shot_classifier_service,
        "run",
        fake_shot_classifier_run,
    )

    await initialize_job_status(job_id)
    await analyze_module.process_job(
        job_id=job_id,
        selected_features=["shot_classifier"],
        video_bytes=b"123",
        filename="upload.mp4",
        calibration=calibration,
    )

    job_status = await get_job_status(job_id)

    assert recorded_require_ball_path == [True]
    assert job_status.shot_classifier.status == "done"
    assert job_status.shot_classifier.result is not None
    assert job_status.shot_classifier.result["predicted_shot"] == "cover"









def _authorized_user() -> UserProfile:
    return UserProfile(
        id=uuid4(),
        email="authorized@example.com",
        full_name="Authorized User",
        created_at=datetime.now(UTC),
        is_active=True,
        revenuecat_customer_id=None,
        entitlement_status="active",
        entitlement_expires_at=None,
        current_tier="coach",
        clips_used_this_month=0,
        quota_reset_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_analyze_session_with_shared_calibration_reuses_single_payload(
    test_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = CalibrationDataFactory()
    current_user = _authorized_user()
    queued_jobs: list[dict[str, object]] = []

    async def fake_require_entitlement() -> UserProfile:
        return current_user

    async def fake_enforce_clip_quota(session, user_id):
        _ = session, user_id
        return current_user

    def fake_copy_upload_to_temp(upload):
        return f"/tmp/{upload.filename}"

    def fake_delay(job_id, selected_features, source_video_path, filename, calibration_payload, intended_shot=None):
        queued_jobs.append(
            {
                "job_id": job_id,
                "selected_features": selected_features,
                "source_video_path": source_video_path,
                "filename": filename,
                "calibration_payload": calibration_payload,
                "intended_shot": intended_shot,
            }
        )

    app.dependency_overrides[require_entitlement] = fake_require_entitlement
    monkeypatch.setattr(analyze_module._user_service, "enforce_clip_quota", fake_enforce_clip_quota)
    monkeypatch.setattr(analyze_module, "_copy_upload_to_temp", fake_copy_upload_to_temp)
    monkeypatch.setattr(analyze_module.process_video_job, "delay", fake_delay)

    response = await test_client.post(
        "/analyze/session",
        files=[
            ("videos", ("delivery1.mp4", b"01", "video/mp4")),
            ("videos", ("delivery2.mp4", b"02", "video/mp4")),
        ],
        data={
            "calibration": calibration.model_dump_json(),
            "features": "bowler_performance",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "session_id" in payload
    assert len(payload["delivery_ids"]) == 2
    assert len(queued_jobs) == 2
    assert queued_jobs[0]["calibration_payload"] == queued_jobs[1]["calibration_payload"]
    assert queued_jobs[0]["selected_features"] == ["bowler_performance"]
    assert queued_jobs[1]["selected_features"] == ["bowler_performance"]


@pytest.mark.asyncio
async def test_analyze_session_with_per_delivery_calibrations_uses_upload_order(
    test_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration_one = CalibrationDataFactory(position=(1.0, 2.0, 3.0), yaw=1.0)
    calibration_two = CalibrationDataFactory(position=(9.0, 8.0, 7.0), yaw=9.0)
    current_user = _authorized_user()
    queued_jobs: list[dict[str, object]] = []

    async def fake_require_entitlement() -> UserProfile:
        return current_user

    async def fake_enforce_clip_quota(session, user_id):
        _ = session, user_id
        return current_user

    def fake_copy_upload_to_temp(upload):
        return f"/tmp/{upload.filename}"

    def fake_delay(job_id, selected_features, source_video_path, filename, calibration_payload, intended_shot=None):
        queued_jobs.append(
            {
                "filename": filename,
                "calibration_payload": calibration_payload,
                "intended_shot": intended_shot,
            }
        )

    app.dependency_overrides[require_entitlement] = fake_require_entitlement
    monkeypatch.setattr(analyze_module._user_service, "enforce_clip_quota", fake_enforce_clip_quota)
    monkeypatch.setattr(analyze_module, "_copy_upload_to_temp", fake_copy_upload_to_temp)
    monkeypatch.setattr(analyze_module.process_video_job, "delay", fake_delay)

    response = await test_client.post(
        "/analyze/session",
        files=[
            ("videos", ("delivery1.mp4", b"01", "video/mp4")),
            ("videos", ("delivery2.mp4", b"02", "video/mp4")),
        ],
        data=[
            ("calibration", calibration_one.model_dump_json()),
            ("calibration", calibration_two.model_dump_json()),
            ("features", "bowler_performance"),
        ],
    )

    assert response.status_code == 200
    assert len(queued_jobs) == 2
    assert queued_jobs[0]["filename"] == "delivery1.mp4"
    assert queued_jobs[1]["filename"] == "delivery2.mp4"
    assert queued_jobs[0]["calibration_payload"]["position"] == [1.0, 2.0, 3.0]
    assert queued_jobs[1]["calibration_payload"]["position"] == [9.0, 8.0, 7.0]


@pytest.mark.asyncio
async def test_analyze_session_rejects_invalid_calibration_count(
    test_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = CalibrationDataFactory()
    current_user = _authorized_user()

    async def fake_require_entitlement() -> UserProfile:
        return current_user

    async def fake_enforce_clip_quota(session, user_id):
        _ = session, user_id
        return current_user

    app.dependency_overrides[require_entitlement] = fake_require_entitlement
    monkeypatch.setattr(analyze_module._user_service, "enforce_clip_quota", fake_enforce_clip_quota)

    response = await test_client.post(
        "/analyze/session",
        files=[
            ("videos", ("delivery1.mp4", b"01", "video/mp4")),
            ("videos", ("delivery2.mp4", b"02", "video/mp4")),
            ("videos", ("delivery3.mp4", b"03", "video/mp4")),
        ],
        data=[
            ("calibration", calibration.model_dump_json()),
            ("calibration", calibration.model_dump_json()),
        ],
    )

    assert response.status_code == 422
    assert "Session calibration count" in response.text
