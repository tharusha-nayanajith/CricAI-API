from __future__ import annotations

from statistics import mean

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.models.job import FEATURE_NAMES
from app.storage.database import (
    AnalysisFeatureResultRecord,
    AnalysisJobRecord,
    AnalysisSessionRecord,
    SessionDeliveryRecord,
)


async def create_analysis_job(
    session,
    *,
    job_id: str,
    user_id,
    filename: str,
    requested_features: list[str],
    session_id: str | None = None,
) -> AnalysisJobRecord:
    record = AnalysisJobRecord(
        id=job_id,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        requested_features=requested_features,
        overall_status="pending",
    )
    session.add(record)
    for feature_name in FEATURE_NAMES:
        session.add(
            AnalysisFeatureResultRecord(
                job_id=job_id,
                feature_name=feature_name,
                status="pending",
            )
        )
    await session.flush()
    return record


async def list_analysis_jobs_for_user(
    session,
    *,
    user_id,
    limit: int,
    offset: int,
) -> tuple[list[AnalysisJobRecord], int]:
    total_stmt = select(func.count()).select_from(AnalysisJobRecord).where(
        AnalysisJobRecord.user_id == user_id
    )
    total = int((await session.execute(total_stmt)).scalar_one())
    stmt = (
        select(AnalysisJobRecord)
        .where(AnalysisJobRecord.user_id == user_id)
        .options(selectinload(AnalysisJobRecord.feature_results))
        .order_by(AnalysisJobRecord.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    jobs = (await session.scalars(stmt)).all()
    return jobs, total


async def list_analysis_sessions_for_user(
    session,
    *,
    user_id,
    limit: int,
    offset: int,
) -> tuple[list[AnalysisSessionRecord], int]:
    total_stmt = select(func.count()).select_from(AnalysisSessionRecord).where(
        AnalysisSessionRecord.user_id == user_id
    )
    total = int((await session.execute(total_stmt)).scalar_one())
    stmt = (
        select(AnalysisSessionRecord)
        .where(AnalysisSessionRecord.user_id == user_id)
        .options(
            selectinload(AnalysisSessionRecord.deliveries)
            .selectinload(SessionDeliveryRecord.job)
            .selectinload(AnalysisJobRecord.feature_results)
        )
        .order_by(AnalysisSessionRecord.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    sessions = (await session.scalars(stmt)).all()
    return sessions, total


def summarize_session_jobs(jobs: list[AnalysisJobRecord]) -> tuple[float | None, float | None, str | None]:
    speeds: list[float] = []
    thumbnail_image_url: str | None = None

    for job in jobs:
        bowler_feature = next(
            (item for item in job.feature_results if item.feature_name == "bowler_performance"),
            None,
        )
        if bowler_feature is None or not isinstance(bowler_feature.result_json, dict):
            continue
        speed = bowler_feature.result_json.get("speed_kmh")
        if isinstance(speed, (int, float)):
            speeds.append(float(speed))
        if thumbnail_image_url is None:
            thumbnail = bowler_feature.result_json.get("thumbnailImageUrl")
            if isinstance(thumbnail, str) and thumbnail:
                thumbnail_image_url = thumbnail

    avg_speed = round(mean(speeds), 2) if speeds else None
    max_speed = round(max(speeds), 2) if speeds else None
    return avg_speed, max_speed, thumbnail_image_url


async def create_analysis_session(
    session,
    *,
    session_id: str,
    user_id,
) -> AnalysisSessionRecord:
    record = AnalysisSessionRecord(
        id=session_id,
        user_id=user_id,
        overall_status="pending",
    )
    session.add(record)
    await session.flush()
    return record


async def add_session_delivery(
    session,
    *,
    session_id: str,
    job_id: str,
    sequence_no: int,
    filename: str,
) -> SessionDeliveryRecord:
    record = SessionDeliveryRecord(
        session_id=session_id,
        job_id=job_id,
        sequence_no=sequence_no,
        filename=filename,
    )
    session.add(record)
    await session.flush()
    return record
