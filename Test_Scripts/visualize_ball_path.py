import argparse
import asyncio
import json
from pathlib import Path

import cv2

from app.models.artifacts import VideoArtifacts
from app.modules.preprocessor.models import BallDetection
from app.modules.preprocessor.service import VideoPreprocessor
from Test_Scripts.run_preprocessor_check import OUTPUT_DIR, parse_calibration_file

BASE_DIR = Path(__file__).resolve().parent


def draw_ball_path(artifacts: VideoArtifacts) -> cv2.typing.MatLike:
    canvas = artifacts.release_frame.copy()
    path = artifacts.ball_path

    for idx, detection in enumerate(path):
        point = (int(detection.x), int(detection.y))
        color = (0, 255, 0) if idx < len(path) - 1 else (0, 255, 255)
        radius = 4 if idx < len(path) - 1 else 7
        cv2.circle(canvas, point, radius, color, -1)

        if idx > 0:
            previous = path[idx - 1]
            previous_point = (int(previous.x), int(previous.y))
            cv2.line(canvas, previous_point, point, (255, 220, 0), 2)

    cv2.putText(
        canvas,
        f"Ball detections: {len(path)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def serialize_ball_path(ball_path: list[BallDetection]) -> list[dict[str, float | int]]:
    return [
        {
            "frame_idx": detection.frame_idx,
            "timestamp_s": detection.timestamp_s,
            "x": detection.x,
            "y": detection.y,
            "confidence": detection.confidence,
        }
        for detection in ball_path
    ]


async def process_pair(video_path: Path, calibration_path: Path) -> None:
    calibration = parse_calibration_file(calibration_path)
    preprocessor = VideoPreprocessor()
    artifacts = await preprocessor.run(video_path, calibration)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path = OUTPUT_DIR / f"{video_path.stem}_ball_path.jpg"
    json_path = OUTPUT_DIR / f"{video_path.stem}_ball_path.json"

    overlay = draw_ball_path(artifacts)
    cv2.imwrite(str(image_path), overlay)
    json_path.write_text(
        json.dumps(serialize_ball_path(artifacts.ball_path), indent=2),
        encoding="utf-8",
    )

    print(
        f"{video_path.name}: saved {image_path.name} and {json_path.name} "
        f"with {len(artifacts.ball_path)} detections"
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize raw detected ball path.")
    parser.add_argument("--video", help="Video filename inside Test_Scripts")
    parser.add_argument("--calibration", help="Calibration txt filename inside Test_Scripts")
    args = parser.parse_args()

    if args.video and args.calibration:
        pairs = [(args.video, args.calibration)]
    else:
        pairs = [
            ("batter_video_01.mp4", "batter_video_01_calibiration_output.txt"),
            ("batter_video_02.mp4", "batter_video_02_calibiration_output.txt"),
            ("bowler_video_01.mp4", "bowler_video_01_calibiration_output.txt"),
        ]

    for video_name, calibration_name in pairs:
        await process_pair(BASE_DIR / video_name, BASE_DIR / calibration_name)


if __name__ == "__main__":
    asyncio.run(main())
