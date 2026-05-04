from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.api.deps import get_current_user
from app.main import app
from app.modules.users.models import UserProfile
from app.storage.database import get_sessionmaker
from app.storage.history import add_session_delivery, create_analysis_job, create_analysis_session
from app.storage.results import initialize_job_status, store_result
from app.models.job import FeatureResult


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
async def test_history_jobs_returns_feature_previews(test_client) -> None:
    current_user = _authorized_user()

    async def fake_get_current_user() -> UserProfile:
        return current_user

    app.dependency_overrides[get_current_user] = fake_get_current_user

    async with get_sessionmaker()() as session:
        await create_analysis_job(
            session,
            job_id="job-history-1",
            user_id=current_user.id,
            filename="delivery.mp4",
            requested_features=["bowler_performance", "action_legality"],
        )
        await session.commit()

    await initialize_job_status("job-history-1", ["bowler_performance", "action_legality"])
    await store_result(
        "job-history-1",
        "bowler_performance",
        FeatureResult(
            status="done",
            result={
                "speed_kmh": 128.4,
                "lengthClass": "good_length",
                "thumbnailImageUrl": "/results/job-history-1/artifacts/bowler_performance/thumbnail.jpg",
            },
            error=None,
        ),
    )
    await store_result(
        "job-history-1",
        "action_legality",
        FeatureResult(
            status="done",
            result={"verdict": "legal", "confidence": 0.88},
            error=None,
        ),
    )

    response = await test_client.get("/history/jobs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["job_id"] == "job-history-1"
    assert payload["items"][0]["requested_features"] == [
        "bowler_performance",
        "action_legality",
    ]
    assert payload["items"][0]["thumbnail_image_url"] == (
        "/results/job-history-1/artifacts/bowler_performance/thumbnail.jpg"
    )
    assert payload["items"][0]["feature_results"]["bowler_performance"]["preview"]["speed_kmh"] == 128.4
    assert payload["items"][0]["feature_results"]["action_legality"]["preview"]["verdict"] == "legal"


@pytest.mark.asyncio
async def test_history_sessions_returns_delivery_counts(test_client) -> None:
    current_user = _authorized_user()

    async def fake_get_current_user() -> UserProfile:
        return current_user

    app.dependency_overrides[get_current_user] = fake_get_current_user

    async with get_sessionmaker()() as session:
        await create_analysis_session(session, session_id="session-history-1", user_id=current_user.id)
        await create_analysis_job(
            session,
            job_id="session-job-1",
            user_id=current_user.id,
            filename="ball-1.mp4",
            requested_features=["bowler_performance"],
            session_id="session-history-1",
        )
        await add_session_delivery(
            session,
            session_id="session-history-1",
            job_id="session-job-1",
            sequence_no=0,
            filename="ball-1.mp4",
        )
        await session.commit()

    await initialize_job_status("session-job-1", ["bowler_performance"])
    await store_result(
        "session-job-1",
        "bowler_performance",
        FeatureResult(
            status="done",
            result={
                "speed_kmh": 131.2,
                "thumbnailImageUrl": "/results/session-job-1/artifacts/bowler_performance/thumbnail.jpg",
            },
            error=None,
        ),
    )

    response = await test_client.get("/history/sessions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["session_id"] == "session-history-1"
    assert payload["items"][0]["delivery_count"] == 1
    assert payload["items"][0]["avg_speed_kmh"] == 131.2
    assert payload["items"][0]["thumbnail_image_url"] == (
        "/results/session-job-1/artifacts/bowler_performance/thumbnail.jpg"
    )
