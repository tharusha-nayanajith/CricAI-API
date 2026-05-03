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
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_entitlement
from app.exceptions import AuthenticationError, AuthorizationError
from app.models.calibration import CalibrationData
from app.models.session import SessionCreateResponse, SessionDeliveryRef
from app.modules.shot_classifier.models import SHOT_CLASS_LABELS, normalize_shot_label
from app.modules.users.models import UserProfile
from app.modules.users.service import UserService
from app.storage.calibration import store_calibration
from app.storage.database import get_db_session
from app.storage.results import initialize_job_status
from app.storage.sessions import store_session

router = APIRouter(tags=["analyze"])
VIDEO_FILE = File(...)
VIDEOS_FILE = File(...)
CALIBRATION_FORM = Form(...)
SESSION_CALIBRATION_FORM = Form(...)
FEATURES_FORM = Form("bowler_performance,action_legality,shot_classifier,shot_similarity")
INTENDED_SHOT_FORM = Form(None)
SESSION_FEATURES = ["bowler_performance"]

ALL_FEATURES = {
    "bowler_performance",
    "action_legality",
    "shot_classifier",
    "shot_similarity",
}
_user_service = UserService()


def _dispatch_process_video_job(
    job_id: str,
    selected_features: list[str],
    source_video_path: str,
    filename: str,
    calibration_payload: dict[str, object],
    intended_shot: str | None = None,
) -> None:
    # Import lazily so auth-only API startup does not require the full ML stack.
    from app.tasks import process_video_job

    process_video_job.delay(
        job_id,
        selected_features,
        source_video_path,
        filename,
        calibration_payload,
        intended_shot,
    )


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


def _parse_calibration(calibration: str) -> tuple[dict[str, object], CalibrationData]:
    try:
        calibration_payload = json.loads(calibration)
        calibration_data = CalibrationData.model_validate(calibration_payload)
    except JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Calibration must be valid JSON.") from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    return calibration_payload, calibration_data


def _parse_session_calibrations(
    calibrations: list[str],
    clip_count: int,
) -> list[tuple[dict[str, object], CalibrationData]]:
    if not calibrations:
        raise HTTPException(status_code=422, detail="At least one calibration is required.")

    parsed = [_parse_calibration(calibration) for calibration in calibrations]
    if len(parsed) == 1:
        return parsed * clip_count
    if len(parsed) != clip_count:
        raise HTTPException(
            status_code=422,
            detail=(
                "Session calibration count must be either 1 shared calibration or exactly one "
                "calibration per uploaded video."
            ),
        )
    return parsed


def _parse_features(features: str) -> list[str]:
    selected_features = [feature.strip() for feature in features.split(",") if feature.strip()]
    if not selected_features:
        selected_features = sorted(ALL_FEATURES)

    invalid_features = sorted(set(selected_features) - ALL_FEATURES)
    if invalid_features:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported features requested: {', '.join(invalid_features)}",
        )
    return selected_features


def _parse_intended_shot(intended_shot: str | None) -> str | None:
    normalized_intended_shot = normalize_shot_label(intended_shot)
    if normalized_intended_shot is None:
        return None
    if normalized_intended_shot not in SHOT_CLASS_LABELS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported intended_shot '{intended_shot}'. "
                f"Expected one of: {', '.join(SHOT_CLASS_LABELS)}"
            ),
        )
    return normalized_intended_shot


async def _enforce_quota(session: AsyncSession, user_id: str, clip_count: int) -> None:
    try:
        for _ in range(max(1, clip_count)):
            await _user_service.enforce_clip_quota(session, user_id)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/analyze")
async def analyze_video(
    current_user: Annotated[UserProfile, Depends(require_entitlement)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    *,
    video: UploadFile = VIDEO_FILE,
    calibration: str = CALIBRATION_FORM,
    features: str = FEATURES_FORM,
    intended_shot: str | None = INTENDED_SHOT_FORM,
) -> dict[str, str]:
    calibration_payload, calibration_data = _parse_calibration(calibration)
    selected_features = _parse_features(features)
    normalized_intended_shot = _parse_intended_shot(intended_shot)
    await _enforce_quota(session, current_user.id, 1)

    loop = asyncio.get_running_loop()
    source_video_path = await loop.run_in_executor(None, _copy_upload_to_temp, video)

    job_id = str(uuid4())
    try:
        await store_calibration(job_id, calibration_data)
        await initialize_job_status(job_id)
        _dispatch_process_video_job(
            job_id,
            selected_features,
            source_video_path,
            video.filename or "upload.mp4",
            calibration_payload,
            normalized_intended_shot,
        )
    except RedisError as exc:
        await loop.run_in_executor(None, _delete_file, source_video_path)
        raise HTTPException(status_code=503, detail="Redis is unavailable.") from exc
    except Exception as exc:
        await loop.run_in_executor(None, _delete_file, source_video_path)
        raise HTTPException(status_code=503, detail="Background worker is unavailable.") from exc
    return {"job_id": job_id}


@router.post("/analyze/session", response_model=SessionCreateResponse)
async def analyze_session(
    current_user: Annotated[UserProfile, Depends(require_entitlement)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    *,
    videos: list[UploadFile] = VIDEOS_FILE,
    calibration: list[str] = SESSION_CALIBRATION_FORM,
    features: str = FEATURES_FORM,
) -> SessionCreateResponse:
    if not videos:
        raise HTTPException(status_code=422, detail="At least one video is required.")

    session_calibrations = _parse_session_calibrations(calibration, len(videos))
    _ = features
    selected_features = SESSION_FEATURES
    await _enforce_quota(session, current_user.id, len(videos))

    loop = asyncio.get_running_loop()
    source_video_paths = await asyncio.gather(
        *(loop.run_in_executor(None, _copy_upload_to_temp, video) for video in videos)
    )

    session_id = str(uuid4())
    delivery_refs: list[SessionDeliveryRef] = []
    try:
        for video, source_video_path, calibration_entry in zip(
            videos,
            source_video_paths,
            session_calibrations,
            strict=True,
        ):
            calibration_payload, calibration_data = calibration_entry
            job_id = str(uuid4())
            filename = video.filename or "upload.mp4"
            await store_calibration(job_id, calibration_data)
            await initialize_job_status(job_id)
            _dispatch_process_video_job(
                job_id,
                selected_features,
                source_video_path,
                filename,
                calibration_payload,
            )
            delivery_refs.append(SessionDeliveryRef(delivery_id=job_id, filename=filename))
        await store_session(session_id, delivery_refs)
    except RedisError as exc:
        await asyncio.gather(
            *(loop.run_in_executor(None, _delete_file, path) for path in source_video_paths),
            return_exceptions=True,
        )
        raise HTTPException(status_code=503, detail="Redis is unavailable.") from exc
    except Exception as exc:
        await asyncio.gather(
            *(loop.run_in_executor(None, _delete_file, path) for path in source_video_paths),
            return_exceptions=True,
        )
        raise HTTPException(status_code=503, detail="Background worker is unavailable.") from exc
    return SessionCreateResponse(
        session_id=session_id,
        delivery_ids=[item.delivery_id for item in delivery_refs],
    )
