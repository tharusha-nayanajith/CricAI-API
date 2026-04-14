import pytest

from app.models.calibration import Keypoint
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


def test_calibration_best_per_channel_keeps_highest_score_keypoint() -> None:
    calibration = CalibrationDataFactory().model_copy(
        update={
            "detected_channels": 2,
            "total_detections": 4,
            "keypoints": [
                Keypoint(x=10.0, y=20.0, score=0.4, channel_index=0),
                Keypoint(x=99.0, y=88.0, score=0.1, channel_index=0),
                Keypoint(x=30.0, y=40.0, score=0.8, channel_index=2),
                Keypoint(x=35.0, y=45.0, score=0.9, channel_index=2),
            ],
        }
    )

    sanitized = calibration.best_per_channel()

    assert len(sanitized.keypoints) == 2
    assert sanitized.detected_channels == 2
    assert sanitized.total_detections == 2
    assert [keypoint.channel_index for keypoint in sanitized.keypoints] == [0, 2]
    assert sanitized.keypoints[0].x == 10.0
    assert sanitized.keypoints[1].x == 35.0
