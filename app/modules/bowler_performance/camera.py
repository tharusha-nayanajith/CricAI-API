from __future__ import annotations

from math import cos, pi, sin, tan

import numpy as np

from app.models.calibration import CalibrationData
from app.modules.preprocessor.models import BallDetection


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
