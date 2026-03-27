import pytest

from app.models.job import FeatureResult
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
