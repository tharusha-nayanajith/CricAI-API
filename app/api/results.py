import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.config import get_redis
from app.models.job import FeatureResult, JobStatus
from app.modules.bowler_performance.coaching import generate_single_delivery_coaching
from app.modules.bowler_performance.models import BowlerCoachingFeedback, BowlerPerformanceResult
from app.storage.artifacts import get_artifact_path
from app.storage.results import get_job_status, store_result

router = APIRouter(tags=["results"])
TERMINAL_STATUSES = {"done", "partial", "failed"}


class BowlerCoachingRequest(BaseModel):
    force_refresh: bool = False


@router.get("/results/{job_id}", response_model=JobStatus)
async def results(job_id: str) -> JobStatus:
    try:
        return await get_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


@router.post(
    "/results/{job_id}/bowler-coaching",
    response_model=BowlerCoachingFeedback,
)
async def generate_bowler_delivery_coaching(
    job_id: str,
    request: BowlerCoachingRequest | None = None,
) -> BowlerCoachingFeedback:
    try:
        job_status = await get_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc

    bowler_feature = job_status.bowler_performance
    if bowler_feature.status != "done" or bowler_feature.result is None:
        raise HTTPException(
            status_code=409,
            detail="Bowler performance result must be completed before coaching can be generated.",
        )

    bowler_result = BowlerPerformanceResult.model_validate(bowler_feature.result)
    if bowler_result.coaching_feedback is not None and not (request and request.force_refresh):
        return bowler_result.coaching_feedback

    coaching_feedback = generate_single_delivery_coaching(bowler_result)
    updated_result = bowler_result.model_copy(
        update={
            "coaching_feedback": coaching_feedback,
            "flutter_payload": [
                entry.model_copy(update={"coaching_feedback": coaching_feedback})
                for entry in bowler_result.flutter_payload
            ],
        }
    )
    await store_result(
        job_id,
        "bowler_performance",
        FeatureResult(
            status=bowler_feature.status,
            result=updated_result.model_dump(by_alias=True),
            error=bowler_feature.error,
        ),
    )
    return coaching_feedback


@router.get("/results/{job_id}/artifacts/{feature_name}/{artifact_name}")
async def result_artifact(job_id: str, feature_name: str, artifact_name: str) -> FileResponse:
    try:
        _ = await get_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc

    try:
        artifact_path = get_artifact_path(job_id, feature_name, artifact_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid artifact path.") from exc
    if not artifact_path.exists() or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(artifact_path)


@router.get("/results/{job_id}/events")
async def result_events(job_id: str, request: Request) -> StreamingResponse:
    redis = get_redis()
    channel = f"results_events:{job_id}"

    try:
        initial_status = await get_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc

    async def event_stream():
        pubsub = redis.pubsub()
        await pubsub.subscribe(channel)
        try:
            initial_payload = initial_status.model_dump_json()
            yield f"event: status\ndata: {initial_payload}\n\n"
            if initial_status.overall_status in TERMINAL_STATUSES:
                return

            while True:
                if await request.is_disconnected():
                    return

                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=15.0,
                )
                if message is None:
                    yield ": keep-alive\n\n"
                    await asyncio.sleep(0)
                    continue

                payload = message["data"]
                yield f"event: status\ndata: {payload}\n\n"

                job_status = JobStatus.model_validate_json(payload)
                if job_status.overall_status in TERMINAL_STATUSES:
                    return
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
