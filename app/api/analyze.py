from __future__ import annotations

import asyncio
import json
import shutil
from json import JSONDecodeError
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_entitlement
from app.exceptions import AuthenticationError, AuthorizationError
from app.models.calibration import CalibrationData
from app.modules.users.models import UserProfile
from app.modules.users.service import UserService
from app.storage.calibration import store_calibration
from app.storage.database import get_db_session
from app.storage.results import initialize_job_status
from app.tasks import process_video_job

router = APIRouter(tags=["analyze"])
VIDEO_FILE = File(...)
CALIBRATION_FORM = Form(...)
FEATURES_FORM = Form("bowler_performance,action_legality,shot_classifier,shot_similarity")

ALL_FEATURES = {
    "bowler_performance",
    "action_legality",
    "shot_classifier",
    "shot_similarity",
}
_user_service = UserService()


def _copy_upload_to_temp(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "upload.mp4").suffix or ".mp4"
    with NamedTemporaryFile(prefix="crickai_upload_", suffix=suffix, delete=False) as temp_file:
        upload.file.seek(0)
        shutil.copyfileobj(upload.file, temp_file)
        return temp_file.name


def _delete_file(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


@router.post("/analyze")
async def analyze_video(
    current_user: Annotated[UserProfile, Depends(require_entitlement)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    *,
    video: UploadFile = VIDEO_FILE,
    calibration: str = CALIBRATION_FORM,
    features: str = FEATURES_FORM,
) -> dict[str, str]:
    try:
        calibration_payload = json.loads(calibration)
        calibration_data = CalibrationData.model_validate(calibration_payload)
    except JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Calibration must be valid JSON.") from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    selected_features = [feature.strip() for feature in features.split(",") if feature.strip()]
    if not selected_features:
        selected_features = sorted(ALL_FEATURES)

    invalid_features = sorted(set(selected_features) - ALL_FEATURES)
    if invalid_features:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported features requested: {', '.join(invalid_features)}",
        )

    try:
        await _user_service.enforce_clip_quota(session, current_user.id)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    loop = asyncio.get_running_loop()
    source_video_path = await loop.run_in_executor(None, _copy_upload_to_temp, video)

    job_id = str(uuid4())
    await store_calibration(job_id, calibration_data)
    await initialize_job_status(job_id)
    try:
        process_video_job.delay(
            job_id,
            selected_features,
            source_video_path,
            video.filename or "upload.mp4",
            calibration_payload,
        )
    except Exception as exc:
        await loop.run_in_executor(None, _delete_file, source_video_path)
        raise HTTPException(status_code=503, detail="Background worker is unavailable.") from exc
    return {"job_id": job_id}
