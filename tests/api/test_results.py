import pytest

from app.models.job import FeatureResult
from app.modules.bowler_performance.models import (
    BowlerCoachingFeedback,
    BowlerPerformanceResult,
)
from app.storage.results import get_job_status, initialize_job_status, store_result


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
        "requested_features": [
            "bowler_performance",
            "action_legality",
            "shot_classifier",
            "shot_similarity",
        ],
        "bowler_performance": {"status": "done", "result": {}, "error": None},
        "action_legality": {"status": "pending", "result": None, "error": None},
        "shot_classifier": {"status": "pending", "result": None, "error": None},
        "shot_similarity": {"status": "pending", "result": None, "error": None},
    }


@pytest.mark.asyncio
async def test_generate_bowler_delivery_coaching_updates_cached_result(
    test_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await initialize_job_status("known-job")
    bowler_result = BowlerPerformanceResult(
        speed_kmh=132.4,
        swing_metres=0.18,
        bounce_point=None,
        length_class=None,
        confidence=0.91,
        inlier_count=8,
        raw_speed_ms=36.78,
        trajectory_reliable=True,
        trajectory_warning=None,
    )
    await store_result(
        "known-job",
        "bowler_performance",
        FeatureResult(
            status="done",
            result=bowler_result.model_dump(by_alias=True),
            error=None,
        ),
    )

    expected_feedback = BowlerCoachingFeedback(
        analysis_scope="single_delivery",
        sample_size=1,
        summary="Single delivery review ready.",
        strengths=["Good pace baseline."],
        improvements=["Tighten target line."],
        next_steps=["Repeat the same release shape."],
    )
    monkeypatch.setattr(
        "app.api.results.generate_single_delivery_coaching",
        lambda result: expected_feedback,
    )

    response = await test_client.post("/results/known-job/bowler-coaching")

    assert response.status_code == 200
    assert response.json()["analysisScope"] == "single_delivery"

    refreshed = await get_job_status("known-job")
    assert refreshed.bowler_performance.result["coachingFeedback"]["summary"] == expected_feedback.summary
