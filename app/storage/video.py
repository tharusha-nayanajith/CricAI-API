from pathlib import Path

from app.config import get_redis

PLAYBACK_VIDEO_TTL_SECONDS = 3600


async def upload_playback_video(job_id: str, video_path: Path) -> str:
    redis = get_redis()
    value = str(video_path)
    await redis.set(f"playback:{job_id}", value, ex=PLAYBACK_VIDEO_TTL_SECONDS)
    return value


async def get_playback_video_url(job_id: str) -> str | None:
    redis = get_redis()
    value = await redis.get(f"playback:{job_id}")
    if value is None:
        return None
    return str(value)
