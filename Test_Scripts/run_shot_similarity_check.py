from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from app.models.calibration import CalibrationData, Keypoint
from app.modules.preprocessor.service import VideoPreprocessor
from app.modules.shot_similarity.service import ShotSimilarityService

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "shot_similarity_outputs"
DEFAULT_FOV_OVERRIDE = 90.0
VIDEO_PAIRS = [
    ("batter_video_01.mp4", "batter_video_01_calibiration_output.txt"),
    ("batter_video_02.mp4", "batter_video_02_calibiration_output.txt"),
]


def _latest_calibration_block(text: str) -> str:
    marker = "=== Calibration Output ==="
    if marker not in text:
        return text
    blocks = [block.strip() for block in text.split(marker) if block.strip()]
    if not blocks:
        return text
    return f"{marker}\n{blocks[-1]}"


def parse_calibration_file(
    calibration_path: Path,
    fallback_fov: float | None = DEFAULT_FOV_OVERRIDE,
) -> CalibrationData:
    text = _latest_calibration_block(calibration_path.read_text(encoding="utf-8"))

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

    keypoints_by_channel: dict[int, Keypoint] = {}
    for channel_index, x_val, y_val, score in re.findall(
        r"Ch\s+(\d+):\s+x=([-\d.]+),\s+y=([-\d.]+),\s+score=([-\d.]+)",
        text,
    ):
        channel = int(channel_index)
        keypoints_by_channel[channel] = Keypoint(
            channel_index=channel,
            x=float(x_val),
            y=float(y_val),
            score=float(score),
        )
    keypoints = [keypoints_by_channel[channel] for channel in sorted(keypoints_by_channel)]

    image_size = tuple(int(value) for value in extract_list("Image Size"))
    position = tuple(extract_list("Position"))
    principal_point = tuple(extract_list("Principal Point"))
    rotation = tuple(extract_list("Rotation"))
    parsed_fov = extract_float("FOV")
    effective_fov = parsed_fov
    if effective_fov <= 0.0 and fallback_fov is not None:
        effective_fov = fallback_fov
        print(
            f"[warn] {calibration_path.name} has FOV={parsed_fov}. "
            f"Using fallback FOV={effective_fov} for local testing."
        )

    return CalibrationData(
        image_size=image_size,
        fov=effective_fov,
        yaw=extract_float("Yaw"),
        position=position,
        principal_point=principal_point,
        rotation=rotation,
        score=extract_float("Score"),
        detected_channels=extract_int("Detected Channels"),
        total_detections=extract_int("Total Detections"),
        keypoints=keypoints,
    )


async def process_pair(
    video_path: Path,
    calibration_path: Path,
    fallback_fov: float | None,
) -> dict[str, object]:
    calibration = parse_calibration_file(calibration_path, fallback_fov=fallback_fov)
    preprocessor = VideoPreprocessor()
    shot_similarity = ShotSimilarityService()

    artifacts = await preprocessor.run(video_path, calibration, require_ball_path=True)
    result = await shot_similarity.run(artifacts, video_url=video_path.name)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": video_path.name,
        "calibration": calibration_path.name,
        "shot_similarity": result.model_dump(),
    }
    output_path = OUTPUT_DIR / f"{video_path.stem}_shot_similarity.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run shot similarity on sample batter videos.")
    parser.add_argument(
        "--video",
        help="Optional video filename from Test_Scripts to run only one pair.",
    )
    parser.add_argument(
        "--fallback-fov",
        type=float,
        default=DEFAULT_FOV_OVERRIDE,
        help="Fallback FOV used when calibration text has FOV <= 0. Default: 90.",
    )
    args = parser.parse_args()

    selected_pairs = VIDEO_PAIRS
    if args.video:
        selected_pairs = [pair for pair in VIDEO_PAIRS if pair[0] == args.video]
        if not selected_pairs:
            raise ValueError(f"No configured calibration pair found for video: {args.video}")

    summaries: list[dict[str, object]] = []
    for video_name, calibration_name in selected_pairs:
        video_path = BASE_DIR / video_name
        calibration_path = BASE_DIR / calibration_name
        print(f"Processing {video_name} with {calibration_name} ...")
        payload = await process_pair(video_path, calibration_path, args.fallback_fov)
        summaries.append(payload)
        print(json.dumps(payload, indent=2))

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
