from __future__ import annotations

from math import cos, pi, sin, tan

import numpy as np

from app.models.calibration import CalibrationData
from app.modules.preprocessor.models import BallDetection


def build_intrinsic_matrix(calibration: CalibrationData) -> np.ndarray:
    image_width = float(calibration.image_size[0])
    fov_radians = calibration.fov * pi / 180.0
    focal_length_px = (image_width / 2.0) / tan(fov_radians / 2.0)
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
    rotation_vector = np.array(calibration.rotation, dtype=np.float64)
    rotation_matrix = _rodrigues(rotation_vector)
    camera_position = np.array(calibration.position, dtype=np.float64)
    translation = -rotation_matrix @ camera_position
    return np.hstack([rotation_matrix, translation.reshape(3, 1)])


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-9:
        return np.eye(3, dtype=np.float64)

    k = rvec / theta
    k_skew = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ],
        dtype=np.float64,
    )
    identity = np.eye(3, dtype=np.float64)
    return (
        identity * cos(theta)
        + (1.0 - cos(theta)) * np.outer(k, k)
        + k_skew * sin(theta)
    )


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
