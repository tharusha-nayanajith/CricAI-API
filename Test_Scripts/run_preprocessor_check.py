import asyncio
import json
import re
from pathlib import Path

import cv2

from app.models.calibration import CalibrationData, Keypoint
from app.modules.preprocessor.service import VideoPreprocessor, get_batter_detector

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "preprocessor_outputs"


def parse_calibration_file(calibration_path: Path) -> CalibrationData:
    text = calibration_path.read_text(encoding="utf-8")

    def extract_float(label: str) -> float:
        match = re.search(rf"{re.escape(label)}:\s*([-\d.]+)", text)
        if match is None:
            raise ValueError(f"Missing calibration field: {label}")
        return float(match.group(1))

    def extract_int(label: str) -> int:
        match = re.search(rf"{re.escape(label)}:\s*(\d+)", text)
        if match is None:
            raise ValueError(f"Missing calibration field: {label}")
        return int(match.group(1))

    def extract_list(label: str) -> list[float]:
        match = re.search(rf"{re.escape(label)}:\s*\[([^\]]+)\]", text)
        if match is None:
            raise ValueError(f"Missing calibration field: {label}")
        return [float(item.strip()) for item in match.group(1).split(",")]

    keypoints: list[Keypoint] = []
    for channel_index, x_val, y_val, score in re.findall(
        r"Ch\s+(\d+):\s+x=([-\d.]+),\s+y=([-\d.]+),\s+score=([-\d.]+)",
        text,
    ):
        keypoints.append(
            Keypoint(
                channel_index=int(channel_index),
                x=float(x_val),
                y=float(y_val),
                score=float(score),
            )
        )

    image_size = tuple(int(value) for value in extract_list("Image Size"))
    position = tuple(extract_list("Position"))
    principal_point = tuple(extract_list("Principal Point"))
    rotation = tuple(extract_list("Rotation"))

    return CalibrationData(
        image_size=image_size,
        fov=extract_float("FOV"),
        yaw=extract_float("Yaw"),
        position=position,
        principal_point=principal_point,
        rotation=rotation,
        score=extract_float("Score"),
        detected_channels=extract_int("Detected Channels"),
        total_detections=extract_int("Total Detections"),
        keypoints=keypoints,
    )


async def process_video(video_path: Path, calibration_path: Path) -> dict[str, object]:
    preprocessor = VideoPreprocessor()
    calibration = parse_calibration_file(calibration_path)
    standardized_path = await preprocessor.standardize_video(video_path)
    batter_detector = get_batter_detector()
    batter_mode, batter_roi = batter_detector.detect(standardized_path, calibration)
    artifacts = await preprocessor.run(standardized_path, calibration)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    release_frame_path = OUTPUT_DIR / f"{video_path.stem}_release.jpg"
    cv2.imwrite(str(release_frame_path), artifacts.release_frame)

    return {
        "video": video_path.name,
        "calibration": calibration_path.name,
        "standardized_video": standardized_path.name,
        "batter_mode": batter_mode.value,
        "batter_roi": (
            {
                "x": batter_roi.x,
                "y": batter_roi.y,
                "width": batter_roi.width,
                "height": batter_roi.height,
            }
            if batter_roi is not None
            else None
        ),
        "release_frame": release_frame_path.name,
        "release_frame_shape": list(artifacts.release_frame.shape),
    }


async def main() -> None:
    pairs = [
        ("batter_video_01.mp4", "batter_video_01_calibiration_output.txt"),
        ("batter_video_02.mp4", "batter_video_02_calibiration_output.txt"),
        ("bowler_video_01.mp4", "bowler_video_01_calibiration_output.txt"),
    ]

    results: list[dict[str, object]] = []
    for video_name, calibration_name in pairs:
        video_path = BASE_DIR / video_name
        calibration_path = BASE_DIR / calibration_name
        print(f"Processing {video_name} ...")
        result = await process_video(video_path, calibration_path)
        results.append(result)
        print(
            f"  batter_mode={result['batter_mode']} "
            f"release_frame={result['release_frame']}"
        )

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
