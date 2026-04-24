import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from app.config import get_redis
from app.models.job import JobStatus
from app.storage.artifacts import get_artifact_path
from app.storage.results import get_job_status

router = APIRouter(tags=["results"])
TERMINAL_STATUSES = {"done", "partial", "failed"}


@router.get("/results/{job_id}", response_model=JobStatus)
async def results(job_id: str) -> JobStatus:
    try:
        return await get_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


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
