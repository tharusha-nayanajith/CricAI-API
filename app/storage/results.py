from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, cast

from app.config import get_redis
from app.models.job import FEATURE_NAMES, FeatureResult, JobStatus
from app.storage.database import (
    AnalysisJobRecord,
    AnalysisSessionRecord,
    SessionDeliveryRecord,
    get_sessionmaker,
)
from sqlalchemy import select
from sqlalchemy.orm import selectinload

RESULTS_TTL_SECONDS = 3600
FeatureName = Literal[
    "bowler_performance",
    "action_legality",
    "shot_classifier",
    "shot_similarity",
]
FeatureStatus = Literal["pending", "processing", "done", "failed"]
OverallStatus = Literal["pending", "processing", "done", "partial", "failed"]


def _default_feature_result() -> FeatureResult:
    return FeatureResult(status="pending", result=None, error=None)


def _build_default_job_status(job_id: str) -> JobStatus:
    return JobStatus(
        job_id=job_id,
        overall_status="pending",
        requested_features=list(FEATURE_NAMES),
        bowler_performance=_default_feature_result(),
        action_legality=_default_feature_result(),
        shot_classifier=_default_feature_result(),
        shot_similarity=_default_feature_result(),
    )


def _results_key(job_id: str) -> str:
    return f"results:{job_id}"


def _results_events_channel(job_id: str) -> str:
    return f"results_events:{job_id}"


def _compute_overall_status(job_status: JobStatus) -> OverallStatus:
    requested_features = job_status.requested_features or list(FEATURE_NAMES)
    statuses = [getattr(job_status, name).status for name in requested_features]
    if not statuses:
        return "pending"
    if all(status == "done" for status in statuses):
        return "done"
    if all(status == "failed" for status in statuses):
        return "failed"
    if any(status == "failed" for status in statuses) and any(
        status == "done" for status in statuses
    ):
        return "partial"
    if any(status in {"processing", "done", "failed"} for status in statuses):
        return "processing"
    return "pending"


def _compute_session_overall_status_from_jobs(job_statuses: list[str]) -> OverallStatus:
    if not job_statuses:
        return "pending"
    if all(status == "done" for status in job_statuses):
        return "done"
    if all(status == "failed" for status in job_statuses):
        return "failed"
    if all(status in {"done", "partial", "failed"} for status in job_statuses) and any(
        status in {"partial", "failed"} for status in job_statuses
    ):
        return "partial"
    if any(status in {"processing", "done", "partial", "failed"} for status in job_statuses):
        return "processing"
    return "pending"


async def _save_job_status(job_status: JobStatus) -> None:
    redis = get_redis()
    payload = job_status.model_dump_json()
    await redis.set(_results_key(job_status.job_id), payload, ex=RESULTS_TTL_SECONDS)
    await redis.publish(_results_events_channel(job_status.job_id), payload)


async def initialize_job_status(
    job_id: str,
    requested_features: list[str] | None = None,
) -> JobStatus:
    job_status = _build_default_job_status(job_id)
    if requested_features is not None:
        job_status.requested_features = [
            cast(FeatureName, feature_name)
            for feature_name in requested_features
            if feature_name in FEATURE_NAMES
        ]
        job_status.overall_status = _compute_overall_status(job_status)
    await _save_job_status(job_status)
    return job_status


def _build_job_status_from_record(record: AnalysisJobRecord) -> JobStatus:
    feature_results = {
        item.feature_name: FeatureResult(
            status=cast(FeatureStatus, item.status),
            result=item.result_json,
            error=item.error,
        )
        for item in record.feature_results
    }
    job_status = JobStatus(
        job_id=record.id,
        overall_status=cast(OverallStatus, record.overall_status),
        requested_features=[
            cast(FeatureName, feature_name)
            for feature_name in (record.requested_features or list(FEATURE_NAMES))
            if feature_name in FEATURE_NAMES
        ],
        bowler_performance=feature_results.get("bowler_performance", _default_feature_result()),
        action_legality=feature_results.get("action_legality", _default_feature_result()),
        shot_classifier=feature_results.get("shot_classifier", _default_feature_result()),
        shot_similarity=feature_results.get("shot_similarity", _default_feature_result()),
    )
    job_status.overall_status = _compute_overall_status(job_status)
    return job_status


async def _load_job_record(job_id: str) -> AnalysisJobRecord | None:
    async with get_sessionmaker()() as session:
        stmt = (
            select(AnalysisJobRecord)
            .where(AnalysisJobRecord.id == job_id)
            .options(selectinload(AnalysisJobRecord.feature_results))
        )
        return await session.scalar(stmt)


async def _persist_feature_result(job_id: str, module_name: str, result: FeatureResult) -> None:
    async with get_sessionmaker()() as session:
        stmt = (
            select(AnalysisJobRecord)
            .where(AnalysisJobRecord.id == job_id)
            .options(selectinload(AnalysisJobRecord.feature_results))
        )
        job_record = await session.scalar(stmt)
        if job_record is None:
            return

        now = datetime.now(UTC)
        feature_record = next(
            (item for item in job_record.feature_results if item.feature_name == module_name),
            None,
        )
        if feature_record is None:
            return
        feature_record.status = result.status
        feature_record.result_json = result.result
        feature_record.error = result.error
        if result.status == "processing" and feature_record.started_at is None:
            feature_record.started_at = now
        if result.status in {"done", "failed"}:
            if feature_record.started_at is None:
                feature_record.started_at = now
            feature_record.completed_at = now
        job_status = _build_job_status_from_record(job_record)
        setattr(job_status, module_name, result)
        job_status.overall_status = _compute_overall_status(job_status)
        job_record.overall_status = job_status.overall_status
        if job_record.session_id is not None:
            session_stmt = (
                select(AnalysisSessionRecord)
                .where(AnalysisSessionRecord.id == job_record.session_id)
                .options(
                    selectinload(AnalysisSessionRecord.deliveries).selectinload(
                        SessionDeliveryRecord.job
                    )
                )
            )
            session_record = await session.scalar(session_stmt)
            if session_record is not None:
                child_statuses = [
                    delivery.job.overall_status
                    for delivery in session_record.deliveries
                    if delivery.job is not None
                ]
                session_record.overall_status = _compute_session_overall_status_from_jobs(
                    child_statuses
                )
        await session.commit()


async def _persist_feature_status(job_id: str, module_name: str, status: FeatureStatus) -> None:
    async with get_sessionmaker()() as session:
        stmt = (
            select(AnalysisJobRecord)
            .where(AnalysisJobRecord.id == job_id)
            .options(selectinload(AnalysisJobRecord.feature_results))
        )
        job_record = await session.scalar(stmt)
        if job_record is None:
            return

        now = datetime.now(UTC)
        feature_record = next(
            (item for item in job_record.feature_results if item.feature_name == module_name),
            None,
        )
        if feature_record is None:
            return
        feature_record.status = status
        if status == "processing" and feature_record.started_at is None:
            feature_record.started_at = now
        if status in {"done", "failed"}:
            feature_record.completed_at = now
        job_status = _build_job_status_from_record(job_record)
        updated_feature = getattr(job_status, module_name).model_copy(update={"status": status})
        setattr(job_status, module_name, updated_feature)
        job_status.overall_status = _compute_overall_status(job_status)
        job_record.overall_status = job_status.overall_status
        if job_record.session_id is not None:
            session_stmt = (
                select(AnalysisSessionRecord)
                .where(AnalysisSessionRecord.id == job_record.session_id)
                .options(
                    selectinload(AnalysisSessionRecord.deliveries).selectinload(
                        SessionDeliveryRecord.job
                    )
                )
            )
            session_record = await session.scalar(session_stmt)
            if session_record is not None:
                child_statuses = [
                    delivery.job.overall_status
                    for delivery in session_record.deliveries
                    if delivery.job is not None
                ]
                session_record.overall_status = _compute_session_overall_status_from_jobs(
                    child_statuses
                )
        await session.commit()


async def store_result(job_id: str, module_name: str, result: FeatureResult) -> None:
    if module_name not in FEATURE_NAMES:
        raise ValueError(f"Unsupported module name: {module_name}")

    redis = get_redis()
    current_value = await redis.get(_results_key(job_id))
    if current_value is not None:
        job_status = JobStatus.model_validate_json(current_value)
    else:
        record = await _load_job_record(job_id)
        job_status = (
            _build_job_status_from_record(record)
            if record is not None
            else _build_default_job_status(job_id)
        )
    setattr(job_status, module_name, result)
    job_status.overall_status = _compute_overall_status(job_status)
    await _save_job_status(job_status)
    await _persist_feature_result(job_id, module_name, result)


async def get_job_status(job_id: str) -> JobStatus:
    redis = get_redis()
    value = await redis.get(_results_key(job_id))
    if value is None:
        record = await _load_job_record(job_id)
        if record is None:
            raise KeyError(job_id)
        job_status = _build_job_status_from_record(record)
        await _save_job_status(job_status)
        return job_status
    job_status = JobStatus.model_validate_json(value)
    if not job_status.requested_features:
        record = await _load_job_record(job_id)
        if record is not None:
            job_status.requested_features = [
                cast(FeatureName, feature_name)
                for feature_name in (record.requested_features or list(FEATURE_NAMES))
                if feature_name in FEATURE_NAMES
            ]
            job_status.overall_status = _compute_overall_status(job_status)
    return job_status


async def set_feature_status(job_id: str, module_name: str, status: str) -> None:
    if module_name not in FEATURE_NAMES:
        raise ValueError(f"Unsupported module name: {module_name}")
    if status not in {"pending", "processing", "done", "failed"}:
        raise ValueError(f"Unsupported feature status: {status}")

    redis = get_redis()
    current_value = await redis.get(_results_key(job_id))
    if current_value is not None:
        job_status = JobStatus.model_validate_json(current_value)
    else:
        record = await _load_job_record(job_id)
        job_status = (
            _build_job_status_from_record(record)
            if record is not None
            else _build_default_job_status(job_id)
        )
    current_feature = getattr(job_status, module_name)
    updated_feature = current_feature.model_copy(update={"status": cast(FeatureStatus, status)})
    setattr(job_status, module_name, updated_feature)
    job_status.overall_status = _compute_overall_status(job_status)
    await _save_job_status(job_status)
    await _persist_feature_status(job_id, module_name, cast(FeatureStatus, status))
