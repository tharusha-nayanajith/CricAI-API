from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.session import SessionResult
from app.modules.bowler_performance.coaching import generate_multi_delivery_coaching
from app.modules.bowler_performance.models import BowlerCoachingFeedback, BowlerPerformanceResult
from app.storage.sessions import (
    get_session_cached_bowler_coaching,
    get_session_delivery_refs,
    get_session_result,
    store_session_cached_bowler_coaching,
)
from app.storage.results import get_job_status

router = APIRouter(tags=["sessions"])


class BowlerSessionCoachingRequest(BaseModel):
    force_refresh: bool = False


@router.get("/sessions/{session_id}/results", response_model=SessionResult)
async def session_results(session_id: str) -> SessionResult:
    try:
        return await get_session_result(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc


@router.post(
    "/sessions/{session_id}/bowler-coaching",
    response_model=BowlerCoachingFeedback,
)
async def session_bowler_coaching(
    session_id: str,
    request: BowlerSessionCoachingRequest | None = None,
) -> BowlerCoachingFeedback:
    try:
        if not (request and request.force_refresh):
            cached = await get_session_cached_bowler_coaching(session_id)
            if cached is not None:
                return cached

        delivery_refs = await get_session_delivery_refs(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc

    bowler_results: list[BowlerPerformanceResult] = []
    has_incomplete_delivery = False
    for delivery_ref in delivery_refs:
        try:
            job_status = await get_job_status(delivery_ref.delivery_id)
        except KeyError:
            has_incomplete_delivery = True
            continue

        feature = job_status.bowler_performance
        if feature.status in {"pending", "processing"}:
            has_incomplete_delivery = True
            continue
        if feature.status == "done" and feature.result is not None:
            bowler_results.append(BowlerPerformanceResult.model_validate(feature.result))

    if has_incomplete_delivery:
        raise HTTPException(
            status_code=409,
            detail="Session coaching is available only after all deliveries finish processing.",
        )
    if len(bowler_results) < 2:
        raise HTTPException(
            status_code=409,
            detail="At least two completed bowler deliveries are required for session coaching.",
        )

    coaching_feedback = generate_multi_delivery_coaching(bowler_results)
    await store_session_cached_bowler_coaching(session_id, coaching_feedback)
    return coaching_feedback
