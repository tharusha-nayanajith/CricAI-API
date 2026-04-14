from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, pi, sin, tan

import cv2
import numpy as np
from loguru import logger

from app.models.calibration import CalibrationData
from app.modules.preprocessor.models import BallDetection

STUMP_HALF_WIDTH_METRES = 0.0954
WORLD_X_ABS_LIMIT_METRES = 5.0
WORLD_Y_MIN_METRES = -0.1
WORLD_Y_MAX_METRES = 4.0
WORLD_Z_ABS_LIMIT_METRES = 40.0
WORLD_STEP_MIN_THRESHOLD_METRES = 5.0
WORLD_SEGMENT_MIN_THRESHOLD_METRES = 8.0
WORLD_SANITY_Z_SPAN_THRESHOLD_METRES = 30.0
WORLD_SANITY_Z_ABS_THRESHOLD_METRES = 40.0
WORLD_SANITY_STEP_THRESHOLD_METRES = 6.5
STUMP_HEIGHT_METRES = 0.711
BATTING_STUMP_Z_METRES = -10.059
BOWLING_STUMP_Z_METRES = 10.059

STUMP_WORLD_BY_CHANNEL: dict[int, tuple[float, float, float]] = {
    0: (-STUMP_HALF_WIDTH_METRES, 0.0, BATTING_STUMP_Z_METRES),
    1: (-STUMP_HALF_WIDTH_METRES, STUMP_HEIGHT_METRES, BATTING_STUMP_Z_METRES),
    2: (0.0, 0.0, BATTING_STUMP_Z_METRES),
    3: (0.0, STUMP_HEIGHT_METRES, BATTING_STUMP_Z_METRES),
    4: (STUMP_HALF_WIDTH_METRES, 0.0, BATTING_STUMP_Z_METRES),
    5: (STUMP_HALF_WIDTH_METRES, STUMP_HEIGHT_METRES, BATTING_STUMP_Z_METRES),
    6: (-STUMP_HALF_WIDTH_METRES, 0.0, BOWLING_STUMP_Z_METRES),
    7: (-STUMP_HALF_WIDTH_METRES, STUMP_HEIGHT_METRES, BOWLING_STUMP_Z_METRES),
    8: (0.0, 0.0, BOWLING_STUMP_Z_METRES),
    9: (0.0, STUMP_HEIGHT_METRES, BOWLING_STUMP_Z_METRES),
    10: (STUMP_HALF_WIDTH_METRES, 0.0, BOWLING_STUMP_Z_METRES),
    11: (STUMP_HALF_WIDTH_METRES, STUMP_HEIGHT_METRES, BOWLING_STUMP_Z_METRES),
}


@dataclass(slots=True)
class ReconstructionSanity:
    point_count: int
    world_y_abs_max: float | None
    world_z_min: float | None
    world_z_max: float | None
    world_z_span: float | None
    max_step_distance_m: float | None
    median_step_distance_m: float | None
    all_points_on_ground: bool
    implausible_depth_range: bool
    implausible_step_jump: bool
    trajectory_reliable: bool


@dataclass(slots=True)
class WorldPointFilterResult:
    points: list[tuple[BallDetection, np.ndarray]]
    removed_frame_indices: list[int]


@dataclass(slots=True)
class RefinedCameraPose:
    extrinsic: np.ndarray
    position: np.ndarray
    rotation_euler: np.ndarray
    reprojection_error_px: float | None
    correspondence_count: int
    refined: bool


def build_intrinsic_matrix(calibration: CalibrationData) -> np.ndarray:
    image_height = float(calibration.image_size[1])
    fov_radians = calibration.fov * pi / 180.0
    focal_length_px = image_height / (tan(fov_radians / 2.0) * 2.0)
    cx, cy = calibration.principal_point
    return np.array(
        [
            [focal_length_px, 0.0, float(cx)],
            [0.0, focal_length_px, float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def build_extrinsic_matrix(calibration: CalibrationData) -> np.ndarray:
    rotation_euler = np.array(calibration.rotation, dtype=np.float64)
    rotation_matrix = _euler_to_rotation(rotation_euler)
    camera_position = np.array(calibration.position, dtype=np.float64)
    translation = -rotation_matrix @ camera_position
    return np.hstack([rotation_matrix, translation.reshape(3, 1)])


def refine_extrinsic_matrix(
    calibration: CalibrationData,
    intrinsic: np.ndarray,
    initial_extrinsic: np.ndarray,
) -> RefinedCameraPose:
    correspondences = _stump_correspondences(calibration)
    initial_position, initial_rotation_euler = decompose_extrinsic_matrix(initial_extrinsic)
    if len(correspondences) < 4:
        return RefinedCameraPose(
            extrinsic=initial_extrinsic,
            position=initial_position,
            rotation_euler=initial_rotation_euler,
            reprojection_error_px=None,
            correspondence_count=len(correspondences),
            refined=False,
        )

    object_points = np.asarray(
        [world_point for _keypoint, world_point in correspondences],
        dtype=np.float64,
    )
    image_points = np.asarray(
        [[keypoint.x, keypoint.y] for keypoint, _world_point in correspondences],
        dtype=np.float64,
    )
    rotation_matrix = initial_extrinsic[:, :3]
    translation = initial_extrinsic[:, 3].reshape(3, 1)
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
    distortion = np.zeros((4, 1), dtype=np.float64)

    try:
        solved, rotation_vector, translation = cv2.solvePnP(
            object_points,
            image_points,
            intrinsic,
            distortion,
            rvec=rotation_vector,
            tvec=translation,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        solved = False

    if not solved:
        return RefinedCameraPose(
            extrinsic=initial_extrinsic,
            position=initial_position,
            rotation_euler=initial_rotation_euler,
            reprojection_error_px=None,
            correspondence_count=len(correspondences),
            refined=False,
        )

    refined_rotation, _ = cv2.Rodrigues(rotation_vector)
    refined_translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
    refined_extrinsic = np.hstack([refined_rotation, refined_translation])
    refined_position = (-refined_rotation.T @ refined_translation).reshape(3)
    refined_rotation_euler = _rotation_to_euler(refined_rotation)
    reprojection_error = _reprojection_error_px(
        object_points,
        image_points,
        intrinsic,
        refined_rotation,
        refined_translation,
    )
    logger.info(
        "Camera pose refinement correspondences={} reprojection_error_px={} "
        "position={} rotation_euler={}",
        len(correspondences),
        reprojection_error,
        [float(value) for value in refined_position.tolist()],
        [float(value) for value in refined_rotation_euler.tolist()],
    )
    return RefinedCameraPose(
        extrinsic=refined_extrinsic,
        position=refined_position,
        rotation_euler=refined_rotation_euler,
        reprojection_error_px=reprojection_error,
        correspondence_count=len(correspondences),
        refined=True,
    )


def decompose_extrinsic_matrix(extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation_matrix = np.asarray(extrinsic[:, :3], dtype=np.float64)
    translation = np.asarray(extrinsic[:, 3], dtype=np.float64).reshape(3, 1)
    position = (-rotation_matrix.T @ translation).reshape(3)
    rotation_euler = _rotation_to_euler(rotation_matrix)
    return position, rotation_euler


def _euler_to_rotation(rotation_euler: np.ndarray) -> np.ndarray:
    rx, ry, rz = rotation_euler[0], rotation_euler[1], rotation_euler[2]
    cos_z = cos(rz)
    sin_z = sin(rz)
    cos_y = cos(ry)
    sin_y = sin(ry)
    cos_x = cos(rx)
    sin_x = sin(rx)

    row0 = [cos_z * cos_y, sin_z * cos_y, -sin_y]
    row1 = [
        (-cos_z) * sin_y * sin_x + sin_z * cos_x,
        (-cos_z) * cos_x - (sin_z * sin_y) * sin_x,
        (-cos_y) * sin_x,
    ]
    row2 = [
        (-sin_z) * sin_x - (cos_z * sin_y) * cos_x,
        (-sin_z) * sin_y * cos_x + cos_z * sin_x,
        (-cos_y) * cos_x,
    ]
    return np.array([row0, row1, row2], dtype=np.float64)


def _rotation_to_euler(rotation_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation_matrix, dtype=np.float64)
    sin_y = -float(matrix[0, 2])
    sin_y = min(1.0, max(-1.0, sin_y))
    ry = asin(sin_y)
    cos_y = cos(ry)
    if abs(cos_y) > 1e-9:
        rx = atan2(-float(matrix[1, 2]), -float(matrix[2, 2]))
        rz = atan2(float(matrix[0, 1]), float(matrix[0, 0]))
    else:
        rx = 0.0
        rz = atan2(float(matrix[1, 0]), -float(matrix[1, 1]))
    return np.array([rx, ry, rz], dtype=np.float64)


def _stump_correspondences(
    calibration: CalibrationData,
) -> list[tuple[object, tuple[float, float, float]]]:
    correspondences: list[tuple[object, tuple[float, float, float]]] = []
    for keypoint in calibration.keypoints:
        world_point = STUMP_WORLD_BY_CHANNEL.get(keypoint.channel_index)
        if world_point is None:
            continue
        correspondences.append((keypoint, world_point))
    return correspondences


def _reprojection_error_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    intrinsic: np.ndarray,
    rotation_matrix: np.ndarray,
    translation: np.ndarray,
) -> float:
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation,
        intrinsic,
        np.zeros((4, 1), dtype=np.float64),
    )
    residual = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def project_world_points_to_image(
    world_points: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> np.ndarray:
    points = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
    rotation_matrix = np.asarray(extrinsic[:, :3], dtype=np.float64)
    translation = np.asarray(extrinsic[:, 3], dtype=np.float64).reshape(3, 1)
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
    projected, _ = cv2.projectPoints(
        points,
        rotation_vector,
        translation,
        intrinsic,
        np.zeros((4, 1), dtype=np.float64),
    )
    return projected.reshape(-1, 2).astype(np.float64)


def unproject_to_ground(
    pixel_x: float,
    pixel_y: float,
    K: np.ndarray,
    RT: np.ndarray,
) -> np.ndarray | None:
    return unproject_to_height(pixel_x, pixel_y, K, RT, world_y=0.0)



def unproject_to_height(
    pixel_x: float,
    pixel_y: float,
    K: np.ndarray,
    RT: np.ndarray,
    world_y: float = 0.0,
) -> np.ndarray | None:
    image_point = np.array([pixel_x, pixel_y, 1.0], dtype=np.float64)
    ray_cam = np.linalg.inv(K) @ image_point

    rotation = RT[:, :3]
    translation = RT[:, 3]
    camera_center = -rotation.T @ translation
    ray_world = rotation.T @ ray_cam
    norm = float(np.linalg.norm(ray_world))
    if norm < 1e-9:
        return None

    ray_world /= norm
    y_component = float(ray_world[1])
    if abs(y_component) < 1e-9:
        return None

    ray_scale = (float(world_y) - float(camera_center[1])) / y_component
    if ray_scale < 0.0:
        return None

    return camera_center + ray_scale * ray_world


def unproject_to_plane_z(
    pixel_x: float,
    pixel_y: float,
    K: np.ndarray,
    RT: np.ndarray,
    target_z: float,
) -> np.ndarray | None:
    image_point = np.array([pixel_x, pixel_y, 1.0], dtype=np.float64)
    ray_cam = np.linalg.inv(K) @ image_point

    rotation = RT[:, :3]
    translation = RT[:, 3]
    camera_center = -rotation.T @ translation
    ray_world = rotation.T @ ray_cam
    norm = float(np.linalg.norm(ray_world))
    if norm < 1e-9:
        return None

    ray_world /= norm
    z_component = float(ray_world[2])
    if abs(z_component) < 1e-9:
        return None

    ray_scale = (float(target_z) - float(camera_center[2])) / z_component
    if ray_scale < 0.0:
        return None

    return camera_center + ray_scale * ray_world


def pixels_to_world_points(
    detections: list[BallDetection],
    K: np.ndarray,
    RT: np.ndarray,
    fps: float = 30.0,
    iterations: int = 3,
) -> list[tuple[BallDetection, np.ndarray]]:
    if not detections:
        return []

    world_points: list[tuple[BallDetection, np.ndarray]] = []
    for detection in detections:
        world_point = unproject_to_ground(detection.x, detection.y, K, RT)
        if world_point is None:
            continue
        world_points.append((detection, world_point))

    if len(world_points) < 4:
        return world_points

    safe_fps = fps if fps > 0.0 else 30.0
    gravity = 9.81

    for _ in range(max(0, iterations)):
        frames = np.asarray(
            [detection.frame_idx for detection, _world_point in world_points],
            dtype=np.float64,
        )
        heights = np.asarray(
            [world_point[1] for _detection, world_point in world_points],
            dtype=np.float64,
        )
        time_values = frames / safe_fps
        corrected_heights = heights + (0.5 * gravity * (time_values**2))
        velocity_y, initial_y = np.polyfit(time_values, corrected_heights, 1)

        refined_points: list[tuple[BallDetection, np.ndarray]] = []
        for detection in detections:
            time_value = float(detection.frame_idx) / safe_fps
            estimated_y = float(
                (velocity_y * time_value)
                + initial_y
                - (0.5 * gravity * (time_value**2))
            )
            estimated_y = max(estimated_y, -0.05)
            world_point = unproject_to_height(
                detection.x,
                detection.y,
                K,
                RT,
                world_y=estimated_y,
            )
            if world_point is None:
                continue
            refined_points.append((detection, world_point))

        if refined_points:
            world_points = refined_points

    return world_points


def filter_world_point_outliers(
    world_points: list[tuple[BallDetection, np.ndarray]],
) -> WorldPointFilterResult:
    if not world_points:
        return WorldPointFilterResult(points=[], removed_frame_indices=[])

    bounded_points: list[tuple[BallDetection, np.ndarray]] = []
    removed_frames: set[int] = set()
    for detection, point in world_points:
        xyz = np.asarray(point, dtype=np.float64)
        if not np.all(np.isfinite(xyz)):
            removed_frames.add(int(detection.frame_idx))
            continue
        if (
            abs(float(xyz[0])) > WORLD_X_ABS_LIMIT_METRES
            or float(xyz[1]) < WORLD_Y_MIN_METRES
            or float(xyz[1]) > WORLD_Y_MAX_METRES
            or abs(float(xyz[2])) > WORLD_Z_ABS_LIMIT_METRES
        ):
            removed_frames.add(int(detection.frame_idx))
            continue
        bounded_points.append((detection, xyz))

    if len(bounded_points) <= 2:
        return WorldPointFilterResult(
            points=bounded_points,
            removed_frame_indices=sorted(removed_frames),
        )

    kept_points = bounded_points
    while len(kept_points) >= 3:
        point_values = np.asarray([point for _detection, point in kept_points], dtype=np.float64)
        step_distances = np.linalg.norm(np.diff(point_values, axis=0), axis=1)
        if step_distances.size == 0:
            break
        median_step = float(np.median(step_distances))
        threshold = max(WORLD_STEP_MIN_THRESHOLD_METRES, median_step * 4.0)
        spike_index: int | None = None
        for index in range(1, len(kept_points) - 1):
            prev_step = float(np.linalg.norm(point_values[index] - point_values[index - 1]))
            next_step = float(np.linalg.norm(point_values[index + 1] - point_values[index]))
            bridge_step = float(np.linalg.norm(point_values[index + 1] - point_values[index - 1]))
            if prev_step > threshold and next_step > threshold and bridge_step <= threshold:
                spike_index = index
                break
        if spike_index is None:
            break
        removed_frames.add(int(kept_points[spike_index][0].frame_idx))
        kept_points = kept_points[:spike_index] + kept_points[spike_index + 1 :]

    if len(kept_points) >= 2:
        point_values = np.asarray([point for _detection, point in kept_points], dtype=np.float64)
        step_distances = np.linalg.norm(np.diff(point_values, axis=0), axis=1)
        if step_distances.size > 0:
            median_step = float(np.median(step_distances))
            threshold = max(WORLD_SEGMENT_MIN_THRESHOLD_METRES, median_step * 4.0)
            segment_ranges: list[tuple[int, int]] = []
            start_index = 0
            for index, step_distance in enumerate(step_distances, start=1):
                if float(step_distance) > threshold:
                    segment_ranges.append((start_index, index))
                    start_index = index
            segment_ranges.append((start_index, len(kept_points)))
            best_start, best_end = max(
                segment_ranges,
                key=lambda item: (item[1] - item[0], -item[0]),
            )
            if best_start != 0 or best_end != len(kept_points):
                for detection, _point in kept_points[:best_start] + kept_points[best_end:]:
                    removed_frames.add(int(detection.frame_idx))
                kept_points = kept_points[best_start:best_end]

    return WorldPointFilterResult(
        points=kept_points,
        removed_frame_indices=sorted(removed_frames),
    )


def assess_world_points(
    world_points: list[tuple[BallDetection, np.ndarray]],
) -> ReconstructionSanity:
    if not world_points:
        return ReconstructionSanity(
            point_count=0,
            world_y_abs_max=None,
            world_z_min=None,
            world_z_max=None,
            world_z_span=None,
            max_step_distance_m=None,
            median_step_distance_m=None,
            all_points_on_ground=False,
            implausible_depth_range=False,
            implausible_step_jump=False,
            trajectory_reliable=False,
        )

    world_xyz = np.asarray([point for _detection, point in world_points], dtype=np.float64)
    if len(world_xyz) >= 2:
        deltas = np.diff(world_xyz, axis=0)
        step_distances = np.linalg.norm(deltas, axis=1)
        max_step_distance = float(np.max(step_distances))
        median_step_distance = float(np.median(step_distances))
    else:
        max_step_distance = None
        median_step_distance = None

    world_y_abs_max = float(np.max(np.abs(world_xyz[:, 1])))
    world_z_values = world_xyz[:, 2]
    world_z_min = float(np.min(world_z_values))
    world_z_max = float(np.max(world_z_values))
    world_z_span = world_z_max - world_z_min
    world_z_p5 = float(np.percentile(world_z_values, 5))
    world_z_p95 = float(np.percentile(world_z_values, 95))
    world_z_span_robust = world_z_p95 - world_z_p5

    all_points_on_ground = world_y_abs_max < 1e-6
    implausible_depth_range = (
        world_z_span_robust > WORLD_SANITY_Z_SPAN_THRESHOLD_METRES
        or abs(world_z_p5) > WORLD_SANITY_Z_ABS_THRESHOLD_METRES
    )
    implausible_step_jump = (
        max_step_distance is not None
        and max_step_distance > WORLD_SANITY_STEP_THRESHOLD_METRES
    )
    trajectory_reliable = not (
        all_points_on_ground or implausible_depth_range or implausible_step_jump
    )

    return ReconstructionSanity(
        point_count=len(world_points),
        world_y_abs_max=world_y_abs_max,
        world_z_min=world_z_min,
        world_z_max=world_z_max,
        world_z_span=world_z_span,
        max_step_distance_m=max_step_distance,
        median_step_distance_m=median_step_distance,
        all_points_on_ground=all_points_on_ground,
        implausible_depth_range=implausible_depth_range,
        implausible_step_jump=implausible_step_jump,
        trajectory_reliable=trajectory_reliable,
    )
