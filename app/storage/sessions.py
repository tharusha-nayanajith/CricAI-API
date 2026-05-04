import json
from statistics import mean
from typing import Literal

from app.config import get_redis
from app.models.job import JobStatus
from app.models.session import (
    BowlerSessionDelivery,
    SessionDeliveryRef,
    SessionProgress,
    SessionResult,
    SessionSummary,
)
from app.modules.bowler_performance.models import BowlerCoachingFeedback
from app.storage.database import AnalysisSessionRecord, SessionDeliveryRecord, get_sessionmaker
from app.storage.results import get_job_status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

SESSIONS_TTL_SECONDS = 3600
SessionOverallStatus = Literal["pending", "processing", "done", "partial", "failed"]


def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


async def store_session(session_id: str, deliveries: list[SessionDeliveryRef]) -> None:
    redis = get_redis()
    payload = {
        "session_id": session_id,
        "deliveries": [item.model_dump() for item in deliveries],
        "bowler_coaching_feedback": None,
    }
    await redis.set(_session_key(session_id), json.dumps(payload), ex=SESSIONS_TTL_SECONDS)


async def _get_session_payload(session_id: str) -> dict[str, object]:
    redis = get_redis()
    value = await redis.get(_session_key(session_id))
    if value is None:
        raise KeyError(session_id)
    return json.loads(value)


async def get_session_delivery_refs(session_id: str) -> list[SessionDeliveryRef]:
    try:
        payload = await _get_session_payload(session_id)
    except KeyError:
        async with get_sessionmaker()() as session:
            stmt = (
                select(SessionDeliveryRecord)
                .where(SessionDeliveryRecord.session_id == session_id)
                .order_by(SessionDeliveryRecord.sequence_no.asc())
            )
            deliveries = (await session.scalars(stmt)).all()
        if not deliveries:
            raise
        return [
            SessionDeliveryRef(delivery_id=item.job_id, filename=item.filename) for item in deliveries
        ]
    deliveries = payload.get("deliveries", [])
    return [SessionDeliveryRef.model_validate(item) for item in deliveries]


async def get_session_cached_bowler_coaching(
    session_id: str,
) -> BowlerCoachingFeedback | None:
    try:
        payload = await _get_session_payload(session_id)
        coaching_feedback = payload.get("bowler_coaching_feedback")
    except KeyError:
        async with get_sessionmaker()() as session:
            record = await session.get(AnalysisSessionRecord, session_id)
        if record is None or record.bowler_coaching_feedback is None:
            return None
        coaching_feedback = record.bowler_coaching_feedback
    if coaching_feedback is None:
        return None
    return BowlerCoachingFeedback.model_validate(coaching_feedback)


async def store_session_cached_bowler_coaching(
    session_id: str,
    coaching_feedback: BowlerCoachingFeedback,
) -> None:
    redis = get_redis()
    try:
        payload = await _get_session_payload(session_id)
        payload["bowler_coaching_feedback"] = coaching_feedback.model_dump(by_alias=True)
        await redis.set(_session_key(session_id), json.dumps(payload), ex=SESSIONS_TTL_SECONDS)
    except KeyError:
        pass

    async with get_sessionmaker()() as session:
        record = await session.get(AnalysisSessionRecord, session_id)
        if record is None:
            return
        record.bowler_coaching_feedback = coaching_feedback.model_dump(by_alias=True)
        await session.commit()


async def get_session_result(session_id: str) -> SessionResult:
    try:
        payload = await _get_session_payload(session_id)
        deliveries = payload.get("deliveries", [])
        delivery_refs = [SessionDeliveryRef.model_validate(item) for item in deliveries]
        cached_coaching = payload.get("bowler_coaching_feedback")
    except KeyError:
        async with get_sessionmaker()() as session:
            stmt = (
                select(AnalysisSessionRecord)
                .where(AnalysisSessionRecord.id == session_id)
                .options(selectinload(AnalysisSessionRecord.deliveries))
            )
            record = await session.scalar(stmt)
        if record is None:
            raise
        delivery_refs = [
            SessionDeliveryRef(delivery_id=item.job_id, filename=item.filename)
            for item in sorted(record.deliveries, key=lambda item: item.sequence_no)
        ]
        cached_coaching = record.bowler_coaching_feedback
    jobs_by_id: dict[str, JobStatus] = {}
    for item in delivery_refs:
        try:
            jobs_by_id[item.delivery_id] = await get_job_status(item.delivery_id)
        except KeyError:
            continue

    progress = _build_progress(list(jobs_by_id.values()), len(delivery_refs))
    return SessionResult(
        session_id=session_id,
        overall_status=_compute_session_overall_status(progress),
        progress=progress,
        deliveries=[
            _build_bowler_delivery(item, jobs_by_id.get(item.delivery_id))
            for item in delivery_refs
        ],
        summary=_build_summary(
            list(jobs_by_id.values()),
            BowlerCoachingFeedback.model_validate(cached_coaching)
            if cached_coaching is not None
            else None,
        ),
    )


def _build_bowler_delivery(
    delivery_ref: SessionDeliveryRef,
    job: JobStatus | None,
) -> BowlerSessionDelivery:
    if job is None:
        from app.models.job import FeatureResult

        return BowlerSessionDelivery(
            delivery_id=delivery_ref.delivery_id,
            filename=delivery_ref.filename,
            overall_status="pending",
            bowler_performance=FeatureResult(status="pending", result=None, error=None),
        )
    return BowlerSessionDelivery(
        delivery_id=delivery_ref.delivery_id,
        filename=delivery_ref.filename,
        overall_status=job.overall_status,
        bowler_performance=job.bowler_performance,
    )


def _build_progress(deliveries: list[JobStatus], total: int) -> SessionProgress:
    pending = sum(1 for job in deliveries if job.overall_status == "pending")
    processing = sum(1 for job in deliveries if job.overall_status == "processing")
    completed = sum(1 for job in deliveries if job.overall_status == "done")
    partial = sum(1 for job in deliveries if job.overall_status == "partial")
    failed = sum(1 for job in deliveries if job.overall_status == "failed")
    missing = max(0, total - len(deliveries))
    pending += missing
    return SessionProgress(
        total=total,
        pending=pending,
        processing=processing,
        completed=completed,
        failed=failed,
        partial=partial,
    )


def _compute_session_overall_status(progress: SessionProgress) -> SessionOverallStatus:
    if progress.total == 0:
        return "pending"
    terminal_count = progress.completed + progress.partial + progress.failed
    if progress.completed == progress.total:
        return "done"
    if progress.failed == progress.total:
        return "failed"
    if terminal_count == progress.total and (progress.partial > 0 or progress.failed > 0):
        return "partial"
    if progress.processing > 0 or terminal_count > 0:
        return "processing"
    return "pending"


def _build_summary(
    deliveries: list[JobStatus],
    coaching_feedback: BowlerCoachingFeedback | None = None,
) -> SessionSummary:
    speeds: list[float] = []
    wicket_risks: list[float] = []
    length_breakdown: dict[str, int] = {}

    for job in deliveries:
        bowler = job.bowler_performance.result or {}
        speed = bowler.get("speed_kmh")
        if isinstance(speed, (int, float)):
            speeds.append(float(speed))

        wicket_risk = bowler.get("wicketRisk")
        if isinstance(wicket_risk, dict):
            percentage = wicket_risk.get("percentage")
            if isinstance(percentage, (int, float)):
                wicket_risks.append(float(percentage))

        length_class = bowler.get("length_class") or bowler.get("lengthClass")
        if isinstance(length_class, str) and length_class:
            length_breakdown[length_class] = length_breakdown.get(length_class, 0) + 1

    return SessionSummary(
        avg_speed_kmh=(round(mean(speeds), 2) if speeds else None),
        max_speed_kmh=(round(max(speeds), 2) if speeds else None),
        avg_wicket_risk_percentage=(round(mean(wicket_risks), 2) if wicket_risks else None),
        length_breakdown=length_breakdown,
        coaching_feedback=coaching_feedback,
    )
