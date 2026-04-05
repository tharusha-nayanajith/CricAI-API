from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from redis.asyncio import Redis


class Settings(BaseSettings):
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/crickai",
        validation_alias="DATABASE_URL",
    )
    s3_bucket: str = Field(default="crickai-results", validation_alias="S3_BUCKET")
    aws_region: str = Field(default="us-east-1", validation_alias="AWS_REGION")
    s3_playback_prefix: str = Field(
        default="deliveries",
        validation_alias="S3_PLAYBACK_PREFIX",
    )
    s3_presign_ttl_seconds: int = Field(
        default=3600,
        validation_alias="S3_PRESIGN_TTL_SECONDS",
    )
    jwt_secret: str = Field(default="change-me", validation_alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", validation_alias="JWT_ALGORITHM")
    revenuecat_webhook_secret: str = Field(
        default="change-me",
        validation_alias="REVENUECAT_WEBHOOK_SECRET",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_redis() -> Redis:
    settings = get_settings()
    return Redis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)
