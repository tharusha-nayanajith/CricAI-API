from pydantic import BaseModel


class Keypoint(BaseModel):
    x: float
    y: float
    score: float
    channel_index: int


class CalibrationData(BaseModel):
    image_size: tuple[int, int]
    fov: float
    yaw: float
    position: tuple[float, float, float]
    principal_point: tuple[float, float]
    rotation: tuple[float, float, float]
    score: float
    detected_channels: int
    total_detections: int
    keypoints: list[Keypoint]
