from __future__ import annotations

import asyncio
from celery.signals import worker_process_init
from loguru import logger

from app.celery_app import celery_app
from app.jobs import process_job
from app.modules.shot_classifier.service import warmup_shot_classifier


@worker_process_init.connect
def _warm_models_on_worker_start(**_: object) -> None:
    logger.info("Worker process init: warming shot classifier")
    warmup_shot_classifier()


@celery_app.task(name="app.tasks.process_video_job")
def process_video_job(
    job_id: str,
    selected_features: list[str],
    source_video_path: str,
    filename: str,
    calibration_payload: dict[str, object],
) -> None:
    logger.info("Starting Celery job {}", job_id)
    asyncio.run(
        process_job(
            job_id=job_id,
            selected_features=selected_features,
            source_video_path=source_video_path,
            filename=filename,
            calibration_payload=calibration_payload,
        )
    )
