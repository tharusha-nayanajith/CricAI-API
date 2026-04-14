from __future__ import annotations

import asyncio
from functools import partial

import numpy as np
from loguru import logger

from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts
from app.models.calibration import CalibrationData
from app.modules.bowler_performance.camera import (
    ReconstructionSanity,
    assess_world_points,
    build_extrinsic_matrix,
    build_intrinsic_matrix,
    filter_world_point_outliers,
    pixels_to_world_points,
)
from app.modules.bowler_performance.metrics import build_result
from app.modules.bowler_performance.models import (
    BallTrackPayload,
    BouncePoint,
    BowlerPerformanceResult,
    CameraCalibrationPayload,
    DeliveryFeatures,
    FlutterPayloadEntry,
    TrajectoryPoint3D,
    TrajectoryPointPitch,
)
from app.modules.bowler_performance.pitch_coordinates import (
    BATTING_STUMP_Z_METRES,
    BOWLING_STUMP_Z_METRES,
    STUMP_HALF_WIDTH_METRES,
    build_pitch_frame,
    world_points_to_pitch_points,
    world_to_pitch,
)
from app.modules.bowler_performance.ransac import BallPathCleaner
from app.modules.bowler_performance.trajectory import AnchorTrajectory, build_anchor_trajectory
from app.modules.bowler_performance.wicket_risk import predict_wicket_risk
from app.modules.preprocessor.models import BallDetection


class BowlerPerformanceAnalyzer:
    def __init__(self) -> None:
        self._cleaner = BallPathCleaner()

    async def run(
        self,
        artifacts: VideoArtifacts,
        calibration: CalibrationData,
        fps: float,
        video_url: str | None = None,
    ) -> BowlerPerformanceResult:
        logger.info(
            "Starting bowler_performance analysis detections={} release_shape={} "
            "bat_contact_available={}",
            len(artifacts.ball_path),
            artifacts.release_frame.shape,
            artifacts.bat_contact_frame is not None,
        )
        loop = asyncio.get_running_loop()
        calibration = calibration.best_per_channel()

        try:
            logger.info("Cleaning ball path with RANSAC")
            ransac_result = await loop.run_in_executor(
                None,
                partial(self._cleaner.clean, artifacts.ball_path, fps),
            )
            if ransac_result is None:
                raise FeatureError("RANSAC found too few inliers in ball path")

            logger.info("Building camera matrices from calibration")
            intrinsic = await loop.run_in_executor(
                None,
                partial(build_intrinsic_matrix, calibration),
            )
            extrinsic = await loop.run_in_executor(
                None,
                partial(build_extrinsic_matrix, calibration),
            )

            logger.info(
                "Selected trajectory detections={} RANSAC inliers={}",
                len(ransac_result.selected_track),
                len(ransac_result.inliers),
            )
            logger.info(
                "Unprojecting {} inlier detections to world space",
                len(ransac_result.inliers),
            )
            world_points = await loop.run_in_executor(
                None,
                partial(
                    pixels_to_world_points,
                    ransac_result.inliers,
                    intrinsic,
                    extrinsic,
                    fps,
                ),
            )
            filtered_world_result = await loop.run_in_executor(
                None,
                partial(filter_world_point_outliers, world_points),
            )
            world_points = filtered_world_result.points
            logger.info(
                "Filtered world points raw={} kept={} removed_frames={}",
                len(ransac_result.inliers),
                len(world_points),
                filtered_world_result.removed_frame_indices,
            )
            if len(world_points) < 3:
                raise FeatureError("Too few valid 3D world points after outlier filtering")
            reconstruction_sanity = await loop.run_in_executor(
                None,
                partial(assess_world_points, world_points),
            )
            logger.info(
                "Sanity detail all_on_ground={} depth_range={} step_jump={} "
                "y_abs_max={} z_min={} z_max={} z_span={} max_step={}",
                reconstruction_sanity.all_points_on_ground,
                reconstruction_sanity.implausible_depth_range,
                reconstruction_sanity.implausible_step_jump,
                reconstruction_sanity.world_y_abs_max,
                reconstruction_sanity.world_z_min,
                reconstruction_sanity.world_z_max,
                reconstruction_sanity.world_z_span,
                reconstruction_sanity.max_step_distance_m,
            )

            logger.info("Converting world points to standardized pitch coordinates")
            pitch_frame = await loop.run_in_executor(
                None,
                partial(build_pitch_frame, calibration, intrinsic, extrinsic),
            )

            pitch_points = await loop.run_in_executor(
                None,
                partial(world_points_to_pitch_points, world_points, pitch_frame),
            )

            anchor_trajectory = None
            if artifacts.release_point is not None:
                anchor_trajectory = await loop.run_in_executor(
                    None,
                    partial(
                        build_anchor_trajectory,
                        artifacts,
                        ransac_result.inliers,
                        ransac_result.bounce_frame,
                        intrinsic,
                        extrinsic,
                        pitch_frame,
                    ),
                )
            canonical_bounce_point = _canonical_bounce_point_from_trajectory(
                anchor_trajectory,
                ransac_result.bounce_frame,
            )

            logger.info("Computing bowler performance metrics")
            result = await loop.run_in_executor(
                None,
                partial(
                    build_result,
                    world_points,
                    pitch_points,
                    ransac_result.inliers,
                    ransac_result.bounce_frame,
                    (
                        float(artifacts.release_point.timestamp_s)
                        if artifacts.release_point is not None
                        else None
                    ),
                    reconstruction_sanity.trajectory_reliable,
                    _trajectory_warning(reconstruction_sanity),
                    canonical_bounce_point,
                ),
            )
            _log_bounce_neighborhood_debug(
                world_points,
                pitch_points,
                ransac_result.bounce_frame,
                calibration,
            )
            camera_calibration = await loop.run_in_executor(
                None,
                partial(
                    _build_camera_calibration_payload,
                    calibration,
                    intrinsic,
                    extrinsic,
                ),
            )
            ball_track = None
            if artifacts.release_point is not None:
                ball_track = await loop.run_in_executor(
                    None,
                    partial(
                        _build_ball_track_payload,
                        artifacts,
                        pitch_frame,
                        ransac_result.bounce_frame,
                        ransac_result.inliers,
                        intrinsic,
                        extrinsic,
                        result.speed_kmh,
                        result.swing_metres,
                        (
                            float(result.bounce_point.x_metres)
                            if result.bounce_point is not None
                            else None
                        ),
                        anchor_trajectory,
                    ),
                )
            _log_stadium_basis_debug(
                pitch_frame,
                result,
                ball_track,
                ransac_result.bounce_frame,
            )
            effective_ball_track = ball_track
            delivery_features = _build_delivery_features(
                artifacts,
                fps,
                ransac_result.selected_track,
                ransac_result.inliers,
                ransac_result.bounce_frame,
                result,
                effective_ball_track,
            )
            wicket_risk = (
                await loop.run_in_executor(
                    None,
                    partial(predict_wicket_risk, delivery_features),
                )
                if delivery_features is not None
                else None
            )
            result = result.model_copy(
                update={
                    "delivery_features": delivery_features,
                    "wicket_risk": wicket_risk,
                    "video_url": video_url,
                    "ball_track": effective_ball_track,
                    "camera_calibration": camera_calibration,
                    "flutter_payload": [
                        FlutterPayloadEntry(
                            video_url=video_url,
                            delivery_features=delivery_features,
                            wicket_risk=wicket_risk,
                            ball_track=effective_ball_track,
                            camera_calibration=camera_calibration,
                        )
                    ],
                }
            )
        except FeatureError:
            raise
        except Exception as exc:
            raise FeatureError("Bowler performance analysis failed unexpectedly") from exc

        logger.info(
            "Completed bowler_performance analysis speed_kmh={} swing_metres={} "
            "length_class={} bounce_point={} wicket_risk_band={} wicket_risk_pct={} "
            "wicket_risk_model={} inlier_count={} trajectory_reliable={}",
            (
                f"{result.speed_kmh:.2f}"
                if result.speed_kmh is not None
                else "unavailable"
            ),
            (
                f"{result.swing_metres:.3f}"
                if result.swing_metres is not None
                else "unavailable"
            ),
            (
                result.length_class.value
                if result.length_class is not None
                else "unavailable"
            ),
            (
                {
                    "x": round(float(result.bounce_point.x_metres), 3),
                    "z": round(float(result.bounce_point.z_metres), 3),
                }
                if result.bounce_point is not None
                else None
            ),
            (
                result.wicket_risk.risk_band.value
                if result.wicket_risk is not None
                else "unavailable"
            ),
            (
                f"{result.wicket_risk.percentage:.1f}"
                if result.wicket_risk is not None
                else "unavailable"
            ),
            (
                f"{result.wicket_risk.model_name}:{result.wicket_risk.model_version}"
                if result.wicket_risk is not None
                else "unavailable"
            ),
            result.inlier_count,
            result.trajectory_reliable,
        )
        return result


def _log_bounce_neighborhood_debug(
    world_points: list[tuple[BallDetection, np.ndarray]],
    pitch_points: list[tuple[BallDetection, np.ndarray]],
    bounce_frame: int | None,
    calibration: CalibrationData,
) -> None:
    batting_stump_bases = {
        keypoint.channel_index: [float(keypoint.x), float(keypoint.y)]
        for keypoint in calibration.keypoints
        if keypoint.channel_index in (0, 2, 4)
    }
    if bounce_frame is None or not world_points or not pitch_points:
        logger.info(
            "Bounce neighborhood debug bounce_frame={} batting_stump_bases={} samples={}",
            bounce_frame,
            batting_stump_bases,
            [],
        )
        return

    pitch_by_frame = {
        detection.frame_idx: pitch_point
        for detection, pitch_point in pitch_points
    }
    samples: list[dict[str, object]] = []
    for detection, world_point in world_points:
        if abs(detection.frame_idx - bounce_frame) > 2:
            continue
        pitch_point = pitch_by_frame.get(detection.frame_idx)
        samples.append(
            {
                "frame_idx": int(detection.frame_idx),
                "pixel_x": float(detection.x),
                "pixel_y": float(detection.y),
                "world_x": float(world_point[0]),
                "world_y": float(world_point[1]),
                "world_z": float(world_point[2]),
                "pitch_x": float(pitch_point[0]) if pitch_point is not None else None,
                "pitch_z": float(pitch_point[2]) if pitch_point is not None else None,
            }
        )

    samples.sort(key=lambda item: (abs(item["frame_idx"] - bounce_frame), item["frame_idx"]))
    logger.info(
        "Bounce neighborhood debug bounce_frame={} batting_stump_bases={} samples={}",
        bounce_frame,
        batting_stump_bases,
        samples,
    )


def _trajectory_warning(sanity: ReconstructionSanity) -> str | None:
    if sanity.trajectory_reliable:
        return None

    reasons: list[str] = []
    if sanity.all_points_on_ground:
        reasons.append("world points collapsed onto the ground plane")
    if sanity.implausible_depth_range:
        reasons.append("depth range is implausibly large")
    if sanity.implausible_step_jump:
        reasons.append("frame-to-frame world jumps are implausibly large")
    if not reasons:
        reasons.append("world reconstruction failed sanity checks")
    return (
        "3D speed and swing metrics are unavailable because "
        + "; ".join(reasons)
        + ". The stadium path uses anchor-fitted visualization points instead."
    )


def _build_ball_track_payload(
    artifacts: VideoArtifacts,
    pitch_frame,
    bounce_frame: int | None,
    detections: list,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    speed_kmh: float | None,
    swing_metres: float | None,
    canonical_bounce_pitch_x: float | None,
    trajectory: AnchorTrajectory | None = None,
) -> BallTrackPayload | None:
    if trajectory is None:
        trajectory = build_anchor_trajectory(
            artifacts,
            detections,
            bounce_frame,
            intrinsic,
            extrinsic,
            pitch_frame,
        )
    if trajectory is None:
        return None

    frame_values = trajectory.frame_values
    world_xyz = trajectory.world_points
    pitch_xyz = trajectory.pitch_points.copy()

    bounce_index = (
        int(np.argmin(np.abs(frame_values - bounce_frame)))
        if bounce_frame is not None
        else None
    )
    if bounce_index is not None and canonical_bounce_pitch_x is not None:
        pitch_xyz[:, 0] = _stabilize_lateral_path(
            pitch_xyz[:, 0],
            bounce_index,
            canonical_bounce_pitch_x,
        )

    stadium_xyz = world_xyz.copy()
    stadium_xyz[:, 0] = pitch_xyz[:, 0]

    release_pitch = pitch_xyz[0]
    bounce_pitch = pitch_xyz[bounce_index] if bounce_index is not None else None
    target_pitch = pitch_xyz[-1]
    pitch_x_axis = getattr(pitch_frame, "x_axis_world", None)
    logger.info(
        "Ball-track lateral debug x_axis_world={} release_world={} "
        "bounce_world={} target_world={} release_pitch={} bounce_pitch={} "
        "target_pitch={} first5_3d_x={} first5_pitch_x={}",
        (
            [float(value) for value in pitch_x_axis.tolist()]
            if pitch_x_axis is not None
            else None
        ),
        [float(value) for value in trajectory.release_anchor.tolist()],
        [float(value) for value in trajectory.bounce_anchor.tolist()],
        [float(value) for value in trajectory.target_anchor.tolist()],
        [float(value) for value in release_pitch.tolist()],
        [float(value) for value in bounce_pitch.tolist()] if bounce_pitch is not None else None,
        [float(value) for value in target_pitch.tolist()],
        [float(value) for value in stadium_xyz[:5, 0].tolist()],
        [float(value) for value in pitch_xyz[:5, 0].tolist()],
    )

    def fit_axis(values: np.ndarray) -> list[float]:
        degree = 2 if len(frame_values) >= 3 else 1
        coeffs = np.polyfit(frame_values, values, deg=degree)
        if degree == 1:
            coeffs = np.asarray([0.0, coeffs[0], coeffs[1]], dtype=np.float64)
        return [float(value) for value in coeffs]

    def deviation(values: np.ndarray, coeffs: list[float]) -> float:
        fitted = np.polyval(np.asarray(coeffs, dtype=np.float64), frame_values)
        return float(np.sqrt(np.mean((values - fitted) ** 2)))

    coeff_x = fit_axis(stadium_xyz[:, 0])
    coeff_y = fit_axis(stadium_xyz[:, 1])
    coeff_z = fit_axis(stadium_xyz[:, 2])
    deviations = [
        deviation(stadium_xyz[:, 0], coeff_x),
        deviation(stadium_xyz[:, 1], coeff_y),
        deviation(stadium_xyz[:, 2], coeff_z),
    ]
    return BallTrackPayload(
        pitch_x=(
            float(pitch_xyz[bounce_index, 0]) if bounce_index is not None else None
        ),
        pitch_y=(
            float(pitch_xyz[bounce_index, 2]) if bounce_index is not None else None
        ),
        min_frame_idx=float(frame_values.min()),
        max_frame_idx=float(frame_values.max()),
        parameter_x_array=coeff_x,
        parameter_y_array=coeff_y,
        parameter_z_array=coeff_z,
        deviation_array=deviations,
        speed=(float(speed_kmh) if speed_kmh is not None else None),
        spin=None,
        swing=(float(swing_metres) if swing_metres is not None else None),
        trajectory_mode="anchor_fitted",
        trajectory_points_3d=[
            {
                "frame_idx": float(frame_value),
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
            }
            for frame_value, point in zip(frame_values, stadium_xyz, strict=True)
        ],
        trajectory_points_pitch=[
            {
                "frame_idx": float(frame_value),
                "pitch_x": float(point[0]),
                "pitch_z": float(point[2]),
            }
            for frame_value, point in zip(frame_values, pitch_xyz, strict=True)
        ],
    )


def _stabilize_lateral_path(
    lateral_values: np.ndarray,
    bounce_index: int,
    canonical_bounce_pitch_x: float,
) -> np.ndarray:
    stabilized = np.asarray(lateral_values, dtype=np.float64).copy()
    if stabilized.size == 0:
        return stabilized

    stabilized[bounce_index] = float(canonical_bounce_pitch_x)
    stabilized[: bounce_index + 1] = _quadratic_lateral_segment(
        stabilized[: bounce_index + 1],
        float(stabilized[0]),
        float(canonical_bounce_pitch_x),
    )
    stabilized[bounce_index:] = _quadratic_lateral_segment(
        stabilized[bounce_index:],
        float(canonical_bounce_pitch_x),
        float(stabilized[-1]),
    )
    stabilized[bounce_index] = float(canonical_bounce_pitch_x)
    return stabilized


def _quadratic_lateral_segment(
    observed_values: np.ndarray,
    start_value: float,
    end_value: float,
) -> np.ndarray:
    value_count = len(observed_values)
    if value_count <= 2:
        segment = np.asarray(observed_values, dtype=np.float64).copy()
        segment[0] = start_value
        segment[-1] = end_value
        return segment

    t_values = np.linspace(0.0, 1.0, value_count, dtype=np.float64)
    coeff = 2.0 * (1.0 - t_values) * t_values
    base = ((1.0 - t_values) ** 2) * start_value + (t_values**2) * end_value
    valid_mask = coeff > 1e-6
    if not np.any(valid_mask):
        segment = np.linspace(start_value, end_value, value_count, dtype=np.float64)
        segment[0] = start_value
        segment[-1] = end_value
        return segment

    observed = np.asarray(observed_values, dtype=np.float64)
    control_value = float(
        np.mean(
            (observed[valid_mask] - base[valid_mask])
            / coeff[valid_mask]
        )
    )
    segment = (
        ((1.0 - t_values) ** 2) * start_value
        + coeff * control_value
        + (t_values**2) * end_value
    )
    segment[0] = start_value
    segment[-1] = end_value
    return segment.astype(np.float64)


def _build_camera_calibration_payload(
    calibration: CalibrationData,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> CameraCalibrationPayload:
    rotation_matrix = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    focal = float(intrinsic[0, 0])
    cx, cy = calibration.principal_point
    intrinsic_payload = [
        focal,
        0.0,
        0.0,
        0.0,
        focal,
        0.0,
        float(cx),
        float(cy),
        1.0,
    ]
    return CameraCalibrationPayload(
        principal_point_array=[float(cx), float(cy)],
        score=float(calibration.score),
        translation_array=[float(value) for value in translation.tolist()],
        dimensions=[int(value) for value in calibration.image_size],
        distortion=0.0,
        instrinsic_matrix_array=intrinsic_payload,
        position_array=[float(value) for value in calibration.position],
        rotation_euler_array=[float(value) for value in calibration.rotation],
        focal=focal,
        rotation_matrix_array=[float(value) for value in rotation_matrix.reshape(-1).tolist()],
        fovy=float(calibration.fov),
    )


def _canonical_bounce_point_from_trajectory(
    trajectory: AnchorTrajectory | None,
    bounce_frame: int | None,
) -> BouncePoint | None:
    if trajectory is None or bounce_frame is None or trajectory.pitch_points.size == 0:
        return None

    bounce_index = int(np.argmin(np.abs(trajectory.frame_values - float(bounce_frame))))
    bounce_pitch = trajectory.pitch_points[bounce_index]
    return BouncePoint(
        x_metres=float(bounce_pitch[0]),
        z_metres=float(bounce_pitch[2]),
    )


def _log_stadium_basis_debug(
    pitch_frame,
    result: BowlerPerformanceResult,
    ball_track: BallTrackPayload | None,
    bounce_frame: int | None,
) -> None:
    if not hasattr(pitch_frame, "batting_origin_world") or not hasattr(
        pitch_frame,
        "x_axis_world",
    ):
        logger.info(
            "Stadium basis debug bounce_result={} ball_track_pitch_x={} "
            "bounce_track_pitch={} bounce_track_3d={} batting_stumps_pitch={} "
            "bowling_stumps_pitch={}",
            (
                {
                    "x": float(result.bounce_point.x_metres),
                    "z": float(result.bounce_point.z_metres),
                }
                if result.bounce_point is not None
                else None
            ),
            ball_track.pitch_x if ball_track is not None else None,
            None,
            None,
            None,
            None,
        )
        return

    batting_stumps_world = [
        [-STUMP_HALF_WIDTH_METRES, 0.0, BATTING_STUMP_Z_METRES],
        [0.0, 0.0, BATTING_STUMP_Z_METRES],
        [STUMP_HALF_WIDTH_METRES, 0.0, BATTING_STUMP_Z_METRES],
    ]
    bowling_stumps_world = [
        [-STUMP_HALF_WIDTH_METRES, 0.0, BOWLING_STUMP_Z_METRES],
        [0.0, 0.0, BOWLING_STUMP_Z_METRES],
        [STUMP_HALF_WIDTH_METRES, 0.0, BOWLING_STUMP_Z_METRES],
    ]
    batting_stumps_pitch = [
        world_to_pitch(np.asarray(point, dtype=np.float64), pitch_frame).tolist()
        for point in batting_stumps_world
    ]
    bowling_stumps_pitch = [
        world_to_pitch(np.asarray(point, dtype=np.float64), pitch_frame).tolist()
        for point in bowling_stumps_world
    ]
    bounce_track_pitch = None
    bounce_track_3d = None
    if ball_track is not None and bounce_frame is not None:
        if ball_track.trajectory_points_pitch:
            bounce_track_pitch = min(
                ball_track.trajectory_points_pitch,
                key=lambda point: abs(point.frame_idx - float(bounce_frame)),
            )
        if ball_track.trajectory_points_3d:
            bounce_track_3d = min(
                ball_track.trajectory_points_3d,
                key=lambda point: abs(point.frame_idx - float(bounce_frame)),
            )

    logger.info(
        "Stadium basis debug bounce_result={} ball_track_pitch_x={} "
        "bounce_track_pitch={} bounce_track_3d={} batting_stumps_pitch={} "
        "bowling_stumps_pitch={}",
        (
            {
                "x": float(result.bounce_point.x_metres),
                "z": float(result.bounce_point.z_metres),
            }
            if result.bounce_point is not None
            else None
        ),
        ball_track.pitch_x if ball_track is not None else None,
        (
            {
                "frame_idx": float(bounce_track_pitch.frame_idx),
                "pitch_x": float(bounce_track_pitch.pitch_x),
                "pitch_z": float(bounce_track_pitch.pitch_z),
            }
            if bounce_track_pitch is not None
            else None
        ),
        (
            {
                "frame_idx": float(bounce_track_3d.frame_idx),
                "x": float(bounce_track_3d.x),
                "y": float(bounce_track_3d.y),
                "z": float(bounce_track_3d.z),
            }
            if bounce_track_3d is not None
            else None
        ),
        batting_stumps_pitch,
        bowling_stumps_pitch,
    )



def _build_delivery_features(
    artifacts: VideoArtifacts,
    fps: float,
    selected_track: list[BallDetection],
    inliers: list[BallDetection],
    bounce_frame: int | None,
    result: BowlerPerformanceResult,
    ball_track: BallTrackPayload | None,
) -> DeliveryFeatures | None:
    if artifacts.release_point is None:
        return None

    release_frame_idx = float(artifacts.release_point.frame_idx)
    release_timestamp_s = float(artifacts.release_point.timestamp_s)
    bounce_frame_idx = float(bounce_frame) if bounce_frame is not None else None
    bounce_timestamp_s = _timestamp_for_frame(selected_track, bounce_frame)
    contact_frame_idx = (
        float(artifacts.bat_contact.contact_frame_idx)
        if artifacts.bat_contact is not None
        else (
            float(selected_track[-1].frame_idx)
            if selected_track
            else None
        )
    )
    contact_timestamp_s = (
        float(artifacts.bat_contact.timestamp_s)
        if artifacts.bat_contact is not None
        else (
            float(selected_track[-1].timestamp_s)
            if selected_track
            else None
        )
    )

    release_to_bounce_ms = _delta_ms(release_timestamp_s, bounce_timestamp_s)
    bounce_to_contact_ms = _delta_ms(bounce_timestamp_s, contact_timestamp_s)
    release_to_contact_ms = _delta_ms(release_timestamp_s, contact_timestamp_s)

    pre_bounce_detection_count, post_bounce_detection_count = _split_counts(
        inliers,
        bounce_frame,
    )
    release_point_3d = _nearest_trajectory_point(
        ball_track.trajectory_points_3d if ball_track is not None else [],
        release_frame_idx,
    )
    contact_point_3d = _nearest_trajectory_point(
        ball_track.trajectory_points_3d if ball_track is not None else [],
        contact_frame_idx,
    )
    release_point_pitch = _nearest_pitch_point(
        ball_track.trajectory_points_pitch if ball_track is not None else [],
        release_frame_idx,
    )
    contact_point_pitch = _nearest_pitch_point(
        ball_track.trajectory_points_pitch if ball_track is not None else [],
        contact_frame_idx,
    )
    bounce_pitch_x = (
        float(result.bounce_point.x_metres)
        if result.bounce_point is not None
        else None
    )
    bounce_pitch_y = (
        float(result.bounce_point.z_metres)
        if result.bounce_point is not None
        else None
    )
    release_pitch_x = release_point_pitch.pitch_x if release_point_pitch is not None else None
    contact_pitch_x = contact_point_pitch.pitch_x if contact_point_pitch is not None else None
    pre_bounce_lateral_delta = (
        bounce_pitch_x - release_pitch_x
        if bounce_pitch_x is not None and release_pitch_x is not None
        else None
    )
    post_bounce_lateral_delta = (
        contact_pitch_x - bounce_pitch_x
        if contact_pitch_x is not None and bounce_pitch_x is not None
        else None
    )

    return DeliveryFeatures(
        batter_mode=(
            artifacts.batter_mode.value if artifacts.batter_mode is not None else "none"
        ),
        has_bat_contact=str(artifacts.bat_contact is not None),
        contact_method=(
            artifacts.bat_contact.method.value if artifacts.bat_contact is not None else "none"
        ),
        trajectory_reliable=str(result.trajectory_reliable),
        length_class=(
            result.length_class.value if result.length_class is not None else None
        ),
        line_bucket=_line_bucket_from_x(bounce_pitch_x),
        pace_bucket=_pace_bucket_from_release_to_bounce_ms(release_to_bounce_ms),
        fps=float(fps),
        release_frame_idx=release_frame_idx,
        bounce_frame_idx=bounce_frame_idx,
        contact_frame_idx=contact_frame_idx,
        release_timestamp_s=release_timestamp_s,
        bounce_timestamp_s=bounce_timestamp_s,
        contact_timestamp_s=contact_timestamp_s,
        release_to_bounce_ms=release_to_bounce_ms,
        bounce_to_contact_ms=bounce_to_contact_ms,
        release_to_contact_ms=release_to_contact_ms,
        pre_bounce_detection_count=pre_bounce_detection_count,
        post_bounce_detection_count=post_bounce_detection_count,
        selected_track_detection_count=len(selected_track),
        inlier_count=len(inliers),
        bounce_pitch_x=bounce_pitch_x,
        bounce_pitch_y=bounce_pitch_y,
        tracking_confidence=float(result.confidence),
        release_confidence=float(artifacts.release_point.confidence),
        contact_score=(
            float(artifacts.bat_contact.detection_score)
            if artifacts.bat_contact is not None
            and artifacts.bat_contact.detection_score is not None
            else None
        ),
        release_height_m=release_point_3d.y if release_point_3d is not None else None,
        contact_height_m=contact_point_3d.y if contact_point_3d is not None else None,
        release_pitch_x=release_pitch_x,
        contact_pitch_x=contact_pitch_x,
        pre_bounce_lateral_delta=pre_bounce_lateral_delta,
        post_bounce_lateral_delta=post_bounce_lateral_delta,
        approach_to_stumps=contact_pitch_x,
    )


def _timestamp_for_frame(
    detections: list[BallDetection],
    frame_idx: int | None,
) -> float | None:
    if frame_idx is None or not detections:
        return None
    detection = min(detections, key=lambda item: abs(item.frame_idx - frame_idx))
    return float(detection.timestamp_s)


def _delta_ms(start_s: float | None, end_s: float | None) -> float | None:
    if start_s is None or end_s is None:
        return None
    return float((end_s - start_s) * 1000.0)


def _split_counts(
    detections: list[BallDetection],
    bounce_frame: int | None,
) -> tuple[int, int]:
    if bounce_frame is None:
        return len(detections), 0
    pre = sum(1 for detection in detections if detection.frame_idx <= bounce_frame)
    post = max(0, len(detections) - pre)
    return pre, post


def _line_bucket_from_x(bounce_pitch_x: float | None) -> str | None:
    if bounce_pitch_x is None:
        return None
    if bounce_pitch_x <= -0.65:
        return "wide_outside_off"
    if bounce_pitch_x <= -0.22:
        return "outside_off"
    if bounce_pitch_x <= 0.08:
        return "off_stump"
    if bounce_pitch_x <= 0.24:
        return "middle"
    if bounce_pitch_x <= 0.5:
        return "leg_stump"
    return "wide_leg"


def _pace_bucket_from_release_to_bounce_ms(release_to_bounce_ms: float | None) -> str | None:
    if release_to_bounce_ms is None:
        return None
    if release_to_bounce_ms < 650.0:
        return "fast"
    if release_to_bounce_ms > 900.0:
        return "slow"
    return "medium"


def _nearest_trajectory_point(
    points: list[TrajectoryPoint3D],
    frame_idx: float | None,
) -> TrajectoryPoint3D | None:
    if not points:
        return None
    if frame_idx is None:
        return points[-1]
    return min(points, key=lambda point: abs(point.frame_idx - frame_idx))


def _nearest_pitch_point(
    points: list[TrajectoryPointPitch],
    frame_idx: float | None,
) -> TrajectoryPointPitch | None:
    if not points:
        return None
    if frame_idx is None:
        return points[-1]
    return min(points, key=lambda point: abs(point.frame_idx - frame_idx))

