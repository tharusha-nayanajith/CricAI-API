import pytest

from app.models.job import FeatureResult
from app.storage.results import initialize_job_status, store_result


@pytest.mark.asyncio
async def test_results_unknown_job_returns_404(test_client) -> None:
    response = await test_client.get("/results/unknown-job")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_results_known_job_returns_job_status_shape(test_client) -> None:
    await initialize_job_status("known-job")
    await store_result(
        "known-job",
        "bowler_performance",
        FeatureResult(status="done", result={}, error=None),
    )

    response = await test_client.get("/results/known-job")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "known-job",
        "overall_status": "processing",
        "bowler_performance": {"status": "done", "result": {}, "error": None},
        "action_legality": {"status": "pending", "result": None, "error": None},
        "shot_classifier": {"status": "pending", "result": None, "error": None},
        "shot_similarity": {"status": "pending", "result": None, "error": None},
    }
