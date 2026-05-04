import pytest

from app.modules.bowler_performance.models import (
    BowlerCoachingFeedback,
    BowlerPerformanceResult,
)
from app.models.job import FeatureResult
from app.models.session import SessionDeliveryRef
from app.storage.results import initialize_job_status, store_result
from app.storage.sessions import store_session


@pytest.mark.asyncio
async def test_generate_session_bowler_coaching_caches_summary(
    test_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "session-1"
    delivery_ids = ["delivery-1", "delivery-2"]
    await store_session(
        session_id,
        [
            SessionDeliveryRef(delivery_id=delivery_ids[0], filename="a.mp4"),
            SessionDeliveryRef(delivery_id=delivery_ids[1], filename="b.mp4"),
        ],
    )

    for idx, delivery_id in enumerate(delivery_ids, start=1):
        await initialize_job_status(delivery_id)
        bowler_result = BowlerPerformanceResult(
            speed_kmh=128.0 + idx,
            swing_metres=0.1 * idx,
            bounce_point=None,
            length_class=None,
            confidence=0.9,
            inlier_count=8,
            raw_speed_ms=35.5 + idx,
            trajectory_reliable=True,
            trajectory_warning=None,
        )
        await store_result(
            delivery_id,
            "bowler_performance",
            FeatureResult(
                status="done",
                result=bowler_result.model_dump(by_alias=True),
                error=None,
            ),
        )

    expected_feedback = BowlerCoachingFeedback(
        analysis_scope="multi_delivery",
        sample_size=2,
        summary="Spell review ready.",
        strengths=["Pace held up across the sample."],
        improvements=["Sharpen line consistency."],
        next_steps=["Compare this spell against the next one."],
    )
    monkeypatch.setattr(
        "app.api.sessions.generate_multi_delivery_coaching",
        lambda deliveries: expected_feedback,
    )

    response = await test_client.post(f"/sessions/{session_id}/bowler-coaching")

    assert response.status_code == 200
    assert response.json()["analysisScope"] == "multi_delivery"

    session_result_response = await test_client.get(f"/sessions/{session_id}/results")

    assert session_result_response.status_code == 200
    assert (
        session_result_response.json()["summary"]["coaching_feedback"]["summary"]
        == expected_feedback.summary
    )
