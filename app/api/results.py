from fastapi import APIRouter, HTTPException

from app.models.job import JobStatus
from app.storage.results import get_job_status

router = APIRouter(tags=["results"])


@router.get("/results/{job_id}", response_model=JobStatus)
async def results(job_id: str) -> JobStatus:
    try:
        return await get_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
