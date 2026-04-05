from fastapi import APIRouter, HTTPException

from app.modules.presentation.models import PresentationBundle
from app.modules.presentation.service import PresentationService
from app.storage.results import get_job_status
from app.storage.video import get_playback_video_url

router = APIRouter(tags=["presentation"])
_presentation_service = PresentationService()


@router.get("/presentation/{job_id}", response_model=PresentationBundle)
async def presentation(job_id: str) -> PresentationBundle:
    try:
        job_status = await get_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    playback_video_url = await get_playback_video_url(job_id)
    return _presentation_service.build_bundle(job_status, playback_video_url=playback_video_url)
