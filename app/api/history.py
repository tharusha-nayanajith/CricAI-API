from __future__ import annotations

from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.models.history import (
    FeatureHistoryPreview,
    JobHistoryItem,
    JobHistoryResponse,
    SessionHistoryItem,
    SessionHistoryResponse,
)
from app.models.job import FEATURE_NAMES, FeatureName
from app.modules.users.models import UserProfile
from app.storage.database import get_db_session
from app.storage.history import (
    list_analysis_jobs_for_user,
    list_analysis_sessions_for_user,
    summarize_session_jobs,
)

router = APIRouter(tags=["history"])


def _feature_preview(feature_name: str, payload: dict[str, Any]) -> dict[str, object] | None:
    if feature_name == "bowler_performance":
        preview: dict[str, object] = {}
        for key in ("speed_kmh", "lengthClass", "length_class", "thumbnailImageUrl"):
            value = payload.get(key)
            if value is not None:
                preview[key] = value
        wicket_risk = payload.get("wicketRisk")
        if isinstance(wicket_risk, dict):
            percentage = wicket_risk.get("percentage")
            if percentage is not None:
                preview["wicketRiskPercentage"] = percentage
        return preview or None
    if feature_name == "action_legality":
        preview = {}
        for key in ("verdict", "confidence", "overlay_image_url", "release_frame_image_url"):
            value = payload.get(key)
            if value is not None:
                preview[key] = value
        return preview or None
    if feature_name == "shot_classifier":
        preview = {}
        for key in ("predicted_shot", "confidence", "thumbnail_url"):
            value = payload.get(key)
            if value is not None:
                preview[key] = value
        return preview or None
    if feature_name == "shot_similarity":
        preview = {}
        for key in ("similarity_score", "top_match_label", "top_match_score"):
            value = payload.get(key)
            if value is not None:
                preview[key] = value
        return preview or None
    return None


@router.get("/history/jobs", response_model=JobHistoryResponse)
async def list_job_history(
    current_user: Annotated[UserProfile, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    *,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> JobHistoryResponse:
    jobs, total = await list_analysis_jobs_for_user(
        session,
        user_id=current_user.id,
        limit=limit,
        offset=offset,
    )

    items: list[JobHistoryItem] = []
    for job in jobs:
        feature_results: dict[FeatureName, FeatureHistoryPreview] = {}
        thumbnail_image_url: str | None = None
        for feature_name in FEATURE_NAMES:
            feature_record = next(
                (item for item in job.feature_results if item.feature_name == feature_name),
                None,
            )
            if feature_record is None:
                continue
            preview = (
                _feature_preview(feature_name, feature_record.result_json)
                if isinstance(feature_record.result_json, dict)
                else None
            )
            if thumbnail_image_url is None and isinstance(preview, dict):
                candidate = preview.get("thumbnailImageUrl")
                if isinstance(candidate, str) and candidate:
                    thumbnail_image_url = candidate
            feature_results[cast(FeatureName, feature_name)] = FeatureHistoryPreview(
                status=feature_record.status,
                error=feature_record.error,
                preview=preview,
            )
        items.append(
            JobHistoryItem(
                job_id=job.id,
                session_id=job.session_id,
                filename=job.filename,
                overall_status=job.overall_status,
                requested_features=[
                    cast(FeatureName, feature_name)
                    for feature_name in (job.requested_features or list(FEATURE_NAMES))
                    if feature_name in FEATURE_NAMES
                ],
                created_at=job.created_at,
                updated_at=job.updated_at,
                thumbnail_image_url=thumbnail_image_url,
                feature_results=feature_results,
            )
        )

    return JobHistoryResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/history/sessions", response_model=SessionHistoryResponse)
async def list_session_history(
    current_user: Annotated[UserProfile, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    *,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> SessionHistoryResponse:
    sessions, total = await list_analysis_sessions_for_user(
        session,
        user_id=current_user.id,
        limit=limit,
        offset=offset,
    )

    items: list[SessionHistoryItem] = []
    for session_record in sessions:
        jobs = [
            delivery.job
            for delivery in sorted(session_record.deliveries, key=lambda item: item.sequence_no)
            if delivery.job is not None
        ]
        avg_speed_kmh, max_speed_kmh, thumbnail_image_url = summarize_session_jobs(jobs)
        items.append(
            SessionHistoryItem(
                session_id=session_record.id,
                overall_status=session_record.overall_status,
                delivery_count=len(session_record.deliveries),
                created_at=session_record.created_at,
                updated_at=session_record.updated_at,
                avg_speed_kmh=avg_speed_kmh,
                max_speed_kmh=max_speed_kmh,
                thumbnail_image_url=thumbnail_image_url,
            )
        )

    return SessionHistoryResponse(items=items, total=total, limit=limit, offset=offset)
