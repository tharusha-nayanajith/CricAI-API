from collections.abc import AsyncIterator
from pathlib import Path

import factory
import pytest
from httpx import ASGITransport, AsyncClient

import app.api.deps as deps_module
import app.modules.users.service as user_service_module
import app.storage.database as database_module
from app.config import Settings
from app.main import app
from app.models.calibration import CalibrationData, Keypoint
from app.models.job import FeatureResult, JobStatus
from app.storage.database import dispose_database, init_database


class FakeRedis:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._values.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        _ = ex
        self._values[key] = value
        return True

    async def delete(self, key: str) -> int:
        if key in self._values:
            del self._values[key]
            return 1
        return 0


class KeypointFactory(factory.Factory):
    class Meta:
        model = Keypoint

    x = 10.0
    y = 20.0
    score = 0.95
    channel_index = 1


class CalibrationDataFactory(factory.Factory):
    class Meta:
        model = CalibrationData

    image_size = (1920, 1080)
    fov = 90.0
    yaw = 2.5
    position = (1.0, 2.0, 3.0)
    principal_point = (960.0, 540.0)
    rotation = (0.1, 0.2, 0.3)
    score = 0.88
    detected_channels = 2
    total_detections = 5
    keypoints = factory.List([factory.SubFactory(KeypointFactory)])


class FeatureResultFactory(factory.Factory):
    class Meta:
        model = FeatureResult

    status = "pending"
    result = None
    error = None


class JobStatusFactory(factory.Factory):
    class Meta:
        model = JobStatus

    job_id = "known-job"
    overall_status = "pending"
    bowler_performance = factory.SubFactory(FeatureResultFactory)
    action_legality = factory.SubFactory(FeatureResultFactory)
    shot_classifier = factory.SubFactory(FeatureResultFactory)
    shot_similarity = factory.SubFactory(FeatureResultFactory)


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    fake = FakeRedis()
    monkeypatch.setattr("app.storage.calibration.get_redis", lambda: fake)
    monkeypatch.setattr("app.storage.results.get_redis", lambda: fake)
    monkeypatch.setattr("app.storage.video.get_redis", lambda: fake)
    monkeypatch.setattr("app.storage.sessions.get_redis", lambda: fake)
    monkeypatch.setattr("app.modules.users.service.get_redis", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
async def test_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> AsyncIterator[Settings]:
    settings = Settings.model_construct(
        redis_url="redis://localhost:6379/0",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        s3_bucket="test-bucket",
        aws_region="us-east-1",
        jwt_secret="test-secret",
        jwt_algorithm="HS256",
        revenuecat_webhook_secret="revenuecat-secret",
    )
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr("app.storage.database.get_settings", lambda: settings)
    monkeypatch.setattr("app.modules.users.service.get_settings", lambda: settings)
    database_module.get_engine.cache_clear()
    database_module.get_sessionmaker.cache_clear()
    await init_database()
    yield settings
    app.dependency_overrides.clear()
    await dispose_database()


@pytest.fixture
async def test_client(fake_redis: FakeRedis, test_settings: Settings) -> AsyncIterator[AsyncClient]:
    _ = fake_redis, test_settings, deps_module, user_service_module
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
