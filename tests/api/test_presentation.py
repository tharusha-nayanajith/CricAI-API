import pytest

import app.api.presentation as presentation_module
from app.models.job import FeatureResult
from app.storage.results import initialize_job_status, store_result


@pytest.mark.asyncio
async def test_presentation_unknown_job_returns_404(test_client) -> None:
    response = await test_client.get("/presentation/unknown-job")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_presentation_returns_flutter_view_bundle(
    test_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_playback_video_url(job_id: str) -> str | None:
        assert job_id == "known-job"
        return "https://cdn.example.com/deliveries/known-job/playback.mp4"

    monkeypatch.setattr(presentation_module, "get_playback_video_url", fake_get_playback_video_url)
    await initialize_job_status("known-job")
    await store_result(
        "known-job",
        "bowler_performance",
        FeatureResult(
            status="done",
            result={
                "videoURL": "delivery.mp4",
                "speed_kmh": 132.4,
                "swing_metres": 0.41,
                "bounce_point": {"x_metres": -0.12, "z_metres": 6.1},
                "length_class": "good_length",
                "confidence": 0.93,
                "inlier_count": 14,
                "raw_speed_ms": 36.78,
                "trajectoryReliable": True,
                "trajectoryWarning": None,
                "deliveryFeatures": {
                    "releaseFrameIdx": 12,
                    "releaseTimestampS": 0.4,
                    "bounceFrameIdx": 24,
                    "bounceTimestampS": 0.8,
                    "contactFrameIdx": 36,
                    "contactTimestampS": 1.2,
                },
                "wicketRisk": {"riskBand": "medium", "percentage": 47.5},
                "ballTrack": {"trajectoryMode": "anchor_fitted"},
                "cameraCalibration": {"fovy": 42.0},
                "flutterPayload": [{"ballTrack": {"trajectoryMode": "anchor_fitted"}}],
            },
        ),
    )
    await store_result(
        "known-job",
        "action_legality",
        FeatureResult(
            status="done",
            result={
                "verdict": "legal",
                "release_frame_index": 12,
                "release_timestamp_s": 0.4,
                "video_url": "delivery.mp4",
            },
        ),
    )
    await store_result(
        "known-job",
        "shot_classifier",
        FeatureResult(
            status="done",
            result={
                "predicted_shot": "cover",
                "confidence": 0.91,
                "frame_start_index": 30,
                "frame_end_index": 59,
                "video_url": "delivery.mp4",
            },
        ),
    )
    response = await test_client.get("/presentation/known-job")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "known-job",
        "overall_status": "processing",
        "original_video_url": "https://cdn.example.com/deliveries/known-job/playback.mp4",
        "playback_video_url": "https://cdn.example.com/deliveries/known-job/playback.mp4",
        "available_views": [
            "three_d_view",
            "bowler_performance",
            "action_legality",
            "shot_classifier",
        ],
        "markers": [
            {"id": "release", "label": "Release", "frame_idx": 12.0, "timestamp_s": 0.4},
            {"id": "bounce", "label": "Bounce", "frame_idx": 24.0, "timestamp_s": 0.8},
            {"id": "contact", "label": "Contact", "frame_idx": 36.0, "timestamp_s": 1.2},
            {
                "id": "shot_window_start",
                "label": "Shot Window Start",
                "frame_idx": 30.0,
                "timestamp_s": None,
            },
            {
                "id": "shot_window_end",
                "label": "Shot Window End",
                "frame_idx": 59.0,
                "timestamp_s": None,
            },
        ],
        "views": [
            {
                "id": "three_d_view",
                "label": "3D View",
                "status": "ready",
                "button_enabled": True,
                "button_order": 1,
                "payload": {
                    "videoUrl": "https://cdn.example.com/deliveries/known-job/playback.mp4",
                    "speedKmh": 132.4,
                    "swingMetres": 0.41,
                    "bouncePoint": {"x_metres": -0.12, "z_metres": 6.1},
                    "lengthClass": "good_length",
                    "confidence": 0.93,
                    "inlierCount": 14,
                    "rawSpeedMs": 36.78,
                    "trajectoryReliable": True,
                    "trajectoryWarning": None,
                    "ballTrack": {"trajectoryMode": "anchor_fitted"},
                    "cameraCalibration": {"fovy": 42.0},
                    "deliveryFeatures": {
                        "releaseFrameIdx": 12,
                        "releaseTimestampS": 0.4,
                        "bounceFrameIdx": 24,
                        "bounceTimestampS": 0.8,
                        "contactFrameIdx": 36,
                        "contactTimestampS": 1.2,
                    },
                    "wicketRisk": {"riskBand": "medium", "percentage": 47.5},
                    "flutterPayload": [{"ballTrack": {"trajectoryMode": "anchor_fitted"}}],
                    "playbackVideoUrl": "https://cdn.example.com/deliveries/known-job/playback.mp4",
                },
                "error": None,
            },
            {
                "id": "bowler_performance",
                "label": "Bowler",
                "status": "ready",
                "button_enabled": True,
                "button_order": 2,
                "payload": {
                    "videoURL": "https://cdn.example.com/deliveries/known-job/playback.mp4",
                    "speed_kmh": 132.4,
                    "swing_metres": 0.41,
                    "bounce_point": {"x_metres": -0.12, "z_metres": 6.1},
                    "length_class": "good_length",
                    "confidence": 0.93,
                    "inlier_count": 14,
                    "raw_speed_ms": 36.78,
                    "trajectoryReliable": True,
                    "trajectoryWarning": None,
                    "deliveryFeatures": {
                        "releaseFrameIdx": 12,
                        "releaseTimestampS": 0.4,
                        "bounceFrameIdx": 24,
                        "bounceTimestampS": 0.8,
                        "contactFrameIdx": 36,
                        "contactTimestampS": 1.2,
                    },
                    "wicketRisk": {"riskBand": "medium", "percentage": 47.5},
                    "ballTrack": {"trajectoryMode": "anchor_fitted"},
                    "cameraCalibration": {"fovy": 42.0},
                    "flutterPayload": [{"ballTrack": {"trajectoryMode": "anchor_fitted"}}],
                },
                "error": None,
            },
            {
                "id": "action_legality",
                "label": "Legality",
                "status": "ready",
                "button_enabled": True,
                "button_order": 3,
                "payload": {
                    "verdict": "legal",
                    "release_frame_index": 12,
                    "release_timestamp_s": 0.4,
                    "video_url": "https://cdn.example.com/deliveries/known-job/playback.mp4",
                },
                "error": None,
            },
            {
                "id": "shot_classifier",
                "label": "Shot Type",
                "status": "ready",
                "button_enabled": True,
                "button_order": 4,
                "payload": {
                    "predicted_shot": "cover",
                    "confidence": 0.91,
                    "frame_start_index": 30,
                    "frame_end_index": 59,
                    "video_url": "https://cdn.example.com/deliveries/known-job/playback.mp4",
                },
                "error": None,
            },
            {
                "id": "shot_similarity",
                "label": "Shot Match",
                "status": "pending",
                "button_enabled": False,
                "button_order": 5,
                "payload": None,
                "error": None,
            },
        ],
    }
