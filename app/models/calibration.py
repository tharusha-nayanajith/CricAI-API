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

    def best_per_channel(self) -> "CalibrationData":
        best_by_channel: dict[int, Keypoint] = {}
        for keypoint in self.keypoints:
            previous = best_by_channel.get(keypoint.channel_index)
            if previous is None or keypoint.score > previous.score:
                best_by_channel[keypoint.channel_index] = keypoint

        sanitized_keypoints = [
            best_by_channel[channel_index]
            for channel_index in sorted(best_by_channel)
        ]
        return self.model_copy(
            update={
                "keypoints": sanitized_keypoints,
                "detected_channels": len(sanitized_keypoints),
                "total_detections": len(sanitized_keypoints),
            }
        )
