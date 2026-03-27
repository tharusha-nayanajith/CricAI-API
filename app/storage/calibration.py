from app.config import get_redis
from app.models.calibration import CalibrationData

CALIBRATION_TTL_SECONDS = 3600


async def store_calibration(job_id: str, data: CalibrationData) -> None:
    redis = get_redis()
    await redis.set(f"calib:{job_id}", data.model_dump_json(), ex=CALIBRATION_TTL_SECONDS)


async def get_calibration(job_id: str) -> CalibrationData | None:
    redis = get_redis()
    value = await redis.get(f"calib:{job_id}")
    if value is None:
        return None
    return CalibrationData.model_validate_json(value)
