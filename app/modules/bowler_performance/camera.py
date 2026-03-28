from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, tan

import numpy as np

from app.models.calibration import CalibrationData
from app.modules.preprocessor.models import BallDetection


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


def unproject_to_ground(
    pixel_x: float,
    pixel_y: float,
    K: np.ndarray,
    RT: np.ndarray,
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
    if abs(float(ray_world[1])) < 1e-9:
        return None

    ray_scale = -float(camera_center[1]) / float(ray_world[1])
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
) -> list[tuple[BallDetection, np.ndarray]]:
    world_points: list[tuple[BallDetection, np.ndarray]] = []
    for detection in detections:
        world_point = unproject_to_ground(detection.x, detection.y, K, RT)
        if world_point is None:
            continue
        world_points.append((detection, world_point))
    return world_points


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
    world_z_min = float(np.min(world_xyz[:, 2]))
    world_z_max = float(np.max(world_xyz[:, 2]))
    world_z_span = world_z_max - world_z_min

    all_points_on_ground = world_y_abs_max < 1e-6
    implausible_depth_range = world_z_span > 50.0 or abs(world_z_min) > 50.0
    implausible_step_jump = (
        max_step_distance is not None and max_step_distance > 10.0
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
