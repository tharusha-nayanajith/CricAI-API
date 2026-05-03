from __future__ import annotations

import asyncio
from celery.signals import worker_process_init
from loguru import logger

from app.celery_app import celery_app

_shot_classifier_warmup = None

try:
    from app.modules.shot_classifier.service import warmup_shot_classifier as _shot_classifier_warmup
except ModuleNotFoundError as exc:
    logger.warning("Shot classifier warmup disabled because an optional dependency is missing: {}", exc)


@worker_process_init.connect
def _warm_models_on_worker_start(**_: object) -> None:
    if _shot_classifier_warmup is None:
        logger.info("Worker process init: skipping shot classifier warmup")
        return
    logger.info("Worker process init: warming shot classifier")
    _shot_classifier_warmup()


@celery_app.task(name="app.tasks.process_video_job")
def process_video_job(
    job_id: str,
    selected_features: list[str],
    source_video_path: str,
    filename: str,
    calibration_payload: dict[str, object],
    intended_shot: str | None = None,
) -> None:
    # Import lazily so the worker can boot even when optional ML deps are absent.
    from app.jobs import process_job

    logger.info("Starting Celery job {}", job_id)
    asyncio.run(
        process_job(
            job_id=job_id,
            selected_features=selected_features,
            source_video_path=source_video_path,
            filename=filename,
            calibration_payload=calibration_payload,
            intended_shot=intended_shot,
        )
    )
