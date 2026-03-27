import pytest

from app.storage.calibration import get_calibration, store_calibration
from tests.conftest import CalibrationDataFactory


@pytest.mark.asyncio
async def test_store_then_get_calibration_round_trips(fake_redis) -> None:
    calibration = CalibrationDataFactory()

    await store_calibration("job-1", calibration)
    restored = await get_calibration("job-1")

    assert restored == calibration


@pytest.mark.asyncio
async def test_get_missing_calibration_returns_none(fake_redis) -> None:
    result = await get_calibration("missing-job")

    assert result is None
