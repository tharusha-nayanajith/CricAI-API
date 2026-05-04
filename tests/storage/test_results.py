from uuid import uuid4

import pytest

from app.models.job import FeatureResult
from app.storage.database import get_sessionmaker
from app.storage.history import create_analysis_job
from app.storage.results import (
    get_job_status,
    initialize_job_status,
    set_feature_status,
    store_result,
)


@pytest.mark.asyncio
async def test_store_result_persists_feature_result(fake_redis) -> None:
    await initialize_job_status("job-1")
    feature_result = FeatureResult(status="done", result={"speed": 120}, error=None)

    await store_result("job-1", "bowler_performance", feature_result)
    job_status = await get_job_status("job-1")

    assert job_status.bowler_performance == feature_result
    assert job_status.overall_status == "processing"


@pytest.mark.asyncio
async def test_get_job_status_returns_initialized_status(fake_redis) -> None:
    expected = await initialize_job_status("job-2")

    actual = await get_job_status("job-2")

    assert actual == expected


@pytest.mark.asyncio
async def test_set_feature_status_updates_existing_job(fake_redis) -> None:
    await initialize_job_status("job-3")

    await set_feature_status("job-3", "action_legality", "processing")
    job_status = await get_job_status("job-3")

    assert job_status.action_legality.status == "processing"
    assert job_status.overall_status == "processing"


@pytest.mark.asyncio
async def test_requested_feature_subset_can_complete_job(fake_redis) -> None:
    await initialize_job_status("job-4", ["bowler_performance"])
    await store_result(
        "job-4",
        "bowler_performance",
        FeatureResult(status="done", result={"speed": 120}, error=None),
    )

    job_status = await get_job_status("job-4")

    assert job_status.requested_features == ["bowler_performance"]
    assert job_status.overall_status == "done"


@pytest.mark.asyncio
async def test_get_job_status_falls_back_to_database_when_redis_is_empty(fake_redis) -> None:
    async with get_sessionmaker()() as session:
        await create_analysis_job(
            session,
            job_id="job-db",
            user_id=uuid4(),
            filename="sample.mp4",
            requested_features=["bowler_performance"],
        )
        await session.commit()

    await initialize_job_status("job-db", ["bowler_performance"])
    await store_result(
        "job-db",
        "bowler_performance",
        FeatureResult(status="done", result={"speed": 120}, error=None),
    )
    fake_redis._values.clear()

    job_status = await get_job_status("job-db")

    assert job_status.requested_features == ["bowler_performance"]
    assert job_status.bowler_performance.result == {"speed": 120}
    assert job_status.overall_status == "done"
