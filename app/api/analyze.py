import json
from json import JSONDecodeError
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from loguru import logger
from pydantic import ValidationError

from app.models.calibration import CalibrationData
from app.storage.calibration import store_calibration
from app.storage.results import initialize_job_status

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


async def process_job(job_id: str, selected_features: list[str]) -> None:
    logger.info("Queued job {} with features {}", job_id, selected_features)


@router.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
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

    await video.read()

    job_id = str(uuid4())
    await store_calibration(job_id, calibration_data)
    await initialize_job_status(job_id)
    background_tasks.add_task(process_job, job_id, selected_features)
    return {"job_id": job_id}
