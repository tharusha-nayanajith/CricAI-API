import json

import pytest

from tests.conftest import CalibrationDataFactory


@pytest.mark.asyncio
async def test_analyze_with_valid_payload_returns_job_id(test_client) -> None:
    calibration = CalibrationDataFactory()
    response = await test_client.post(
        "/analyze",
        files={"video": ("sample.mp4", b"00", "video/mp4")},
        data={
            "calibration": calibration.model_dump_json(),
            "features": "bowler_performance,action_legality,shot_classifier,shot_similarity",
        },
    )

    assert response.status_code == 200
    assert "job_id" in response.json()


@pytest.mark.asyncio
async def test_analyze_with_malformed_calibration_returns_422(test_client) -> None:
    response = await test_client.post(
        "/analyze",
        files={"video": ("sample.mp4", b"00", "video/mp4")},
        data={"calibration": json.dumps({"invalid": "payload"})},
    )

    assert response.status_code == 422
