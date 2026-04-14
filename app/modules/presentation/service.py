from __future__ import annotations

from typing import Any

from app.models.job import FeatureResult, JobStatus

from .models import PresentationBundle, PresentationMarker, PresentationView


class PresentationService:
    def build_bundle(
        self,
        job_status: JobStatus,
        *,
        playback_video_url: str | None = None,
    ) -> PresentationBundle:
        bowler_result = _with_playback_url(
            _feature_payload(job_status.bowler_performance),
            playback_video_url,
        )
        action_result = _with_playback_url(
            _feature_payload(job_status.action_legality),
            playback_video_url,
        )
        shot_classifier_result = _with_playback_url(
            _feature_payload(job_status.shot_classifier),
            playback_video_url,
        )
        shot_similarity_result = _with_playback_url(
            _feature_payload(job_status.shot_similarity),
            playback_video_url,
        )

        views = [
            _build_three_d_view(job_status.bowler_performance, bowler_result, playback_video_url),
            _build_feature_view(
                feature_id="bowler_performance",
                label="Bowler",
                button_order=2,
                feature=job_status.bowler_performance,
                payload=bowler_result,
            ),
            _build_feature_view(
                feature_id="action_legality",
                label="Legality",
                button_order=3,
                feature=job_status.action_legality,
                payload=action_result,
            ),
            _build_feature_view(
                feature_id="shot_classifier",
                label="Shot Type",
                button_order=4,
                feature=job_status.shot_classifier,
                payload=shot_classifier_result,
            ),
            _build_feature_view(
                feature_id="shot_similarity",
                label="Shot Match",
                button_order=5,
                feature=job_status.shot_similarity,
                payload=shot_similarity_result,
            ),
        ]
        resolved_playback_url = playback_video_url or _pick_first_video_url(
            bowler_result,
            action_result,
            shot_classifier_result,
            shot_similarity_result,
        )
        return PresentationBundle(
            job_id=job_status.job_id,
            overall_status=job_status.overall_status,
            original_video_url=resolved_playback_url,
            playback_video_url=resolved_playback_url,
            available_views=[view.id for view in views if view.button_enabled],
            markers=_build_markers(
                bowler_result,
                action_result,
                shot_classifier_result,
            ),
            views=views,
        )


def _build_three_d_view(
    feature: FeatureResult,
    payload: dict[str, Any] | None,
    playback_video_url: str | None,
) -> PresentationView:
    if feature.status == "failed":
        return PresentationView(
            id="three_d_view",
            label="3D View",
            status="failed",
            button_enabled=False,
            button_order=1,
            error=feature.error,
        )
    if feature.status != "done":
        return PresentationView(
            id="three_d_view",
            label="3D View",
            status="pending",
            button_enabled=False,
            button_order=1,
        )
    if payload is None:
        return PresentationView(
            id="three_d_view",
            label="3D View",
            status="unavailable",
            button_enabled=False,
            button_order=1,
            error="Bowler performance result payload is missing.",
        )

    ball_track = payload.get("ballTrack")
    camera_calibration = payload.get("cameraCalibration")
    if ball_track is None or camera_calibration is None:
        return PresentationView(
            id="three_d_view",
            label="3D View",
            status="unavailable",
            button_enabled=False,
            button_order=1,
            error="3D reconstruction payload is not available for this delivery.",
        )

    return PresentationView(
        id="three_d_view",
        label="3D View",
        status="ready",
        button_enabled=True,
        button_order=1,
        payload={
            "videoUrl": payload.get("videoURL"),
            "speedKmh": payload.get("speed_kmh"),
            "swingMetres": payload.get("swing_metres"),
            "bouncePoint": payload.get("bounce_point"),
            "lengthClass": payload.get("length_class"),
            "confidence": payload.get("confidence"),
            "inlierCount": payload.get("inlier_count"),
            "rawSpeedMs": payload.get("raw_speed_ms"),
            "trajectoryReliable": payload.get("trajectoryReliable"),
            "trajectoryWarning": payload.get("trajectoryWarning"),
            "ballTrack": ball_track,
            "cameraCalibration": camera_calibration,
            "deliveryFeatures": payload.get("deliveryFeatures"),
            "wicketRisk": payload.get("wicketRisk"),
            "flutterPayload": payload.get("flutterPayload"),
            "playbackVideoUrl": playback_video_url,
        },
    )


def _build_feature_view(
    *,
    feature_id: str,
    label: str,
    button_order: int,
    feature: FeatureResult,
    payload: dict[str, Any] | None,
) -> PresentationView:
    status_map = {
        "pending": ("pending", False),
        "processing": ("pending", False),
        "done": ("ready", True),
        "failed": ("failed", False),
    }
    view_status, button_enabled = status_map[feature.status]
    return PresentationView(
        id=feature_id,
        label=label,
        status=view_status,
        button_enabled=button_enabled,
        button_order=button_order,
        payload=payload if feature.status == "done" else None,
        error=feature.error,
    )


def _feature_payload(feature: FeatureResult) -> dict[str, Any] | None:
    if feature.result is None:
        return None
    return dict(feature.result)


def _pick_first_video_url(*payloads: dict[str, Any] | None) -> str | None:
    for payload in payloads:
        if payload is None:
            continue
        video_url = payload.get("videoURL") or payload.get("video_url")
        if isinstance(video_url, str) and video_url and not video_url.startswith("s3://"):
            return video_url
    return None


def _with_playback_url(
    payload: dict[str, Any] | None,
    playback_video_url: str | None,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    hydrated = dict(payload)
    if playback_video_url is None:
        return hydrated
    if "videoURL" in hydrated or "video_url" not in hydrated:
        hydrated["videoURL"] = playback_video_url
    if "video_url" in hydrated or "videoURL" not in hydrated:
        hydrated["video_url"] = playback_video_url
    return hydrated


def _build_markers(
    bowler_result: dict[str, Any] | None,
    action_result: dict[str, Any] | None,
    shot_classifier_result: dict[str, Any] | None,
) -> list[PresentationMarker]:
    markers = [
        _build_release_marker(bowler_result, action_result),
        _build_bounce_marker(bowler_result),
        _build_contact_marker(bowler_result),
        _build_shot_window_start_marker(shot_classifier_result),
        _build_shot_window_end_marker(shot_classifier_result),
    ]
    return [marker for marker in markers if marker is not None]


def _build_release_marker(
    bowler_result: dict[str, Any] | None,
    action_result: dict[str, Any] | None,
) -> PresentationMarker | None:
    delivery_features = _delivery_features(bowler_result)
    release_frame_idx = _number_value(delivery_features, "releaseFrameIdx")
    release_timestamp_s = _number_value(delivery_features, "releaseTimestampS")
    if release_frame_idx is None:
        release_frame_idx = _number_value(action_result, "release_frame_index")
    if release_timestamp_s is None:
        release_timestamp_s = _number_value(action_result, "release_timestamp_s")
    if release_frame_idx is None and release_timestamp_s is None:
        return None
    return PresentationMarker(
        id="release",
        label="Release",
        frame_idx=release_frame_idx,
        timestamp_s=release_timestamp_s,
    )


def _build_bounce_marker(bowler_result: dict[str, Any] | None) -> PresentationMarker | None:
    delivery_features = _delivery_features(bowler_result)
    bounce_frame_idx = _number_value(delivery_features, "bounceFrameIdx")
    bounce_timestamp_s = _number_value(delivery_features, "bounceTimestampS")
    if bounce_frame_idx is None and bounce_timestamp_s is None:
        return None
    return PresentationMarker(
        id="bounce",
        label="Bounce",
        frame_idx=bounce_frame_idx,
        timestamp_s=bounce_timestamp_s,
    )


def _build_contact_marker(bowler_result: dict[str, Any] | None) -> PresentationMarker | None:
    delivery_features = _delivery_features(bowler_result)
    contact_frame_idx = _number_value(delivery_features, "contactFrameIdx")
    contact_timestamp_s = _number_value(delivery_features, "contactTimestampS")
    if contact_frame_idx is None and contact_timestamp_s is None:
        return None
    return PresentationMarker(
        id="contact",
        label="Contact",
        frame_idx=contact_frame_idx,
        timestamp_s=contact_timestamp_s,
    )


def _build_shot_window_start_marker(
    shot_classifier_result: dict[str, Any] | None,
) -> PresentationMarker | None:
    frame_idx = _number_value(shot_classifier_result, "frame_start_index")
    if frame_idx is None:
        return None
    return PresentationMarker(
        id="shot_window_start",
        label="Shot Window Start",
        frame_idx=frame_idx,
    )


def _build_shot_window_end_marker(
    shot_classifier_result: dict[str, Any] | None,
) -> PresentationMarker | None:
    frame_idx = _number_value(shot_classifier_result, "frame_end_index")
    if frame_idx is None:
        return None
    return PresentationMarker(
        id="shot_window_end",
        label="Shot Window End",
        frame_idx=frame_idx,
    )


def _delivery_features(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    value = payload.get("deliveryFeatures")
    if isinstance(value, dict):
        return value
    return None


def _number_value(payload: dict[str, Any] | None, key: str) -> float | None:
    if payload is None:
        return None
    value = payload.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None
