from collections.abc import AsyncIterator

import factory
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.calibration import CalibrationData, Keypoint
from app.models.job import FeatureResult, JobStatus


class FakeRedis:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._values.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        _ = ex
        self._values[key] = value
        return True


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
    return fake


@pytest.fixture
async def test_client(fake_redis: FakeRedis) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
