import asyncio
import json
import shutil
import tempfile
from functools import partial
from json import JSONDecodeError
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from loguru import logger
from pydantic import ValidationError

from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts
from app.models.calibration import CalibrationData
from app.models.job import FeatureResult
from app.modules.bowler_performance.service import BowlerPerformanceAnalyzer
from app.modules.preprocessor.models import BallDetection
from app.modules.preprocessor.service import VideoPreprocessor
from app.storage.calibration import store_calibration
from app.storage.results import initialize_job_status, set_feature_status, store_result

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
_preprocessor = VideoPreprocessor()
_bowler_analyzer = BowlerPerformanceAnalyzer()


def _write_video_bytes(video_path: Path, video_bytes: bytes) -> None:
    video_path.write_bytes(video_bytes)


def _cleanup_job_dir(job_dir: Path) -> None:
    shutil.rmtree(job_dir, ignore_errors=True)


def _derive_fps(ball_path: list[BallDetection]) -> float:
    for previous, current in zip(ball_path, ball_path[1:], strict=False):
        dt = current.timestamp_s - previous.timestamp_s
        frame_gap = current.frame_idx - previous.frame_idx
        if dt > 0.0 and frame_gap > 0:
            return frame_gap / dt
    return 30.0


async def run_bowler_performance(
    job_id: str,
    artifacts: VideoArtifacts,
    calibration: CalibrationData,
    fps: float,
) -> None:
    await set_feature_status(job_id, "bowler_performance", "processing")
    try:
        result = await _bowler_analyzer.run(artifacts, calibration, fps)
        await store_result(
            job_id,
            "bowler_performance",
            FeatureResult(status="done", result=result.model_dump()),
        )
    except FeatureError as exc:
        logger.error("bowler_performance failed for {}: {}", job_id, exc)
        await store_result(
            job_id,
            "bowler_performance",
            FeatureResult(status="failed", error=str(exc)),
        )
    except Exception as exc:
        logger.exception("Unexpected bowler_performance failure for {}", job_id)
        await store_result(
            job_id,
            "bowler_performance",
            FeatureResult(status="failed", error=str(exc)),
        )


async def process_job(
    job_id: str,
    selected_features: list[str],
    video_bytes: bytes,
    filename: str,
    calibration: CalibrationData,
) -> None:
    logger.info("Queued job {} with features {}", job_id, selected_features)
    if "bowler_performance" not in selected_features:
        logger.info("No implemented background features selected for {}", job_id)
        return

    job_dir = Path(tempfile.mkdtemp(prefix=f"crickai_{job_id}_"))
    safe_name = Path(filename or "upload.mp4").name or "upload.mp4"
    video_path = job_dir / safe_name
    loop = asyncio.get_running_loop()

    try:
        await loop.run_in_executor(None, partial(_write_video_bytes, video_path, video_bytes))
        artifacts = await _preprocessor.run(video_path, calibration)
    except Exception as exc:
        logger.error("Preprocessor failed for {}: {}", job_id, exc)
        await store_result(
            job_id,
            "bowler_performance",
            FeatureResult(status="failed", error=f"Preprocessor failed: {exc}"),
        )
    else:
        fps = _derive_fps(artifacts.ball_path)
        await run_bowler_performance(job_id, artifacts, calibration, fps)
    finally:
        await loop.run_in_executor(None, partial(_cleanup_job_dir, job_dir))


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

    video_bytes = await video.read()

    job_id = str(uuid4())
    await store_calibration(job_id, calibration_data)
    await initialize_job_status(job_id)
    background_tasks.add_task(
        process_job,
        job_id,
        selected_features,
        video_bytes,
        video.filename or "upload.mp4",
        calibration_data,
    )
    return {"job_id": job_id}
