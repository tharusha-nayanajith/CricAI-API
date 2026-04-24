from typing import Literal, cast

from app.config import get_redis
from app.models.job import FeatureResult, JobStatus

RESULTS_TTL_SECONDS = 3600
FEATURE_NAMES = (
    "bowler_performance",
    "action_legality",
    "shot_classifier",
    "shot_similarity",
)
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
    statuses = [getattr(job_status, name).status for name in FEATURE_NAMES]
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


async def _save_job_status(job_status: JobStatus) -> None:
    redis = get_redis()
    payload = job_status.model_dump_json()
    await redis.set(_results_key(job_status.job_id), payload, ex=RESULTS_TTL_SECONDS)
    await redis.publish(_results_events_channel(job_status.job_id), payload)


async def initialize_job_status(job_id: str) -> JobStatus:
    job_status = _build_default_job_status(job_id)
    await _save_job_status(job_status)
    return job_status


async def store_result(job_id: str, module_name: str, result: FeatureResult) -> None:
    if module_name not in FEATURE_NAMES:
        raise ValueError(f"Unsupported module name: {module_name}")

    redis = get_redis()
    current_value = await redis.get(_results_key(job_id))
    job_status = (
        JobStatus.model_validate_json(current_value)
        if current_value is not None
        else _build_default_job_status(job_id)
    )
    setattr(job_status, module_name, result)
    job_status.overall_status = _compute_overall_status(job_status)
    await _save_job_status(job_status)


async def get_job_status(job_id: str) -> JobStatus:
    redis = get_redis()
    value = await redis.get(_results_key(job_id))
    if value is None:
        raise KeyError(job_id)
    return JobStatus.model_validate_json(value)


async def set_feature_status(job_id: str, module_name: str, status: str) -> None:
    if module_name not in FEATURE_NAMES:
        raise ValueError(f"Unsupported module name: {module_name}")
    if status not in {"pending", "processing", "done", "failed"}:
        raise ValueError(f"Unsupported feature status: {status}")

    redis = get_redis()
    current_value = await redis.get(_results_key(job_id))
    job_status = (
        JobStatus.model_validate_json(current_value)
        if current_value is not None
        else _build_default_job_status(job_id)
    )
    current_feature = getattr(job_status, module_name)
    updated_feature = current_feature.model_copy(update={"status": cast(FeatureStatus, status)})
    setattr(job_status, module_name, updated_feature)
    job_status.overall_status = _compute_overall_status(job_status)
    await _save_job_status(job_status)
