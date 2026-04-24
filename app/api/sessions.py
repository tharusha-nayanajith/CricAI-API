from fastapi import APIRouter, HTTPException

from app.models.session import SessionResult
from app.storage.sessions import get_session_result

router = APIRouter(tags=["sessions"])


@router.get("/sessions/{session_id}/results", response_model=SessionResult)
async def session_results(session_id: str) -> SessionResult:
    try:
        return await get_session_result(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc
