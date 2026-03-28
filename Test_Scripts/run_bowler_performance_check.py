from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.models.calibration import CalibrationData, Keypoint
from app.modules.bowler_performance.camera import (
    assess_world_points,
    build_extrinsic_matrix,
    build_intrinsic_matrix,
    pixels_to_world_points,
)
from app.modules.bowler_performance.service import BowlerPerformanceAnalyzer
from app.modules.bowler_performance.pitch_coordinates import (
    build_pitch_frame,
    world_points_to_pitch_points,
)
from app.modules.bowler_performance.ransac import BallPathCleaner
from app.modules.preprocessor.models import BallDetection
from app.modules.preprocessor.service import VideoPreprocessor

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "bowler_performance_outputs"
DEFAULT_FOV_OVERRIDE = 90.0
VIDEO_PAIRS = [
    ("bowler_video_01.mp4", "bowler_video_01_calibiration_output.txt"),
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


def read_video_props(video_path: Path) -> tuple[float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
    finally:
        cap.release()
    if fps <= 0.0:
        raise RuntimeError(f"Video FPS could not be read from: {video_path}")
    return fps, width, height


def serialize_ball_detection(detection: BallDetection) -> dict[str, float | int]:
    return {
        "frame_idx": detection.frame_idx,
        "timestamp_s": detection.timestamp_s,
        "x": detection.x,
        "y": detection.y,
        "confidence": detection.confidence,
    }


def serialize_world_point(
    detection: BallDetection,
    world_point: np.ndarray,
) -> dict[str, float | int]:
    return {
        **serialize_ball_detection(detection),
        "world_x": float(world_point[0]),
        "world_y": float(world_point[1]),
        "world_z": float(world_point[2]),
    }


def serialize_pitch_point(
    detection: BallDetection,
    pitch_point: np.ndarray,
) -> dict[str, float | int]:
    return {
        **serialize_ball_detection(detection),
        "pitch_x": float(pitch_point[0]),
        "pitch_y": float(pitch_point[1]),
        "pitch_z": float(pitch_point[2]),
    }


def build_reconstruction_sanity(
    world_points: list[tuple[BallDetection, np.ndarray]],
) -> dict[str, float | int | bool | None]:
    sanity = assess_world_points(world_points)
    return {
        "point_count": sanity.point_count,
        "world_y_abs_max": sanity.world_y_abs_max,
        "world_z_min": sanity.world_z_min,
        "world_z_max": sanity.world_z_max,
        "world_z_span": sanity.world_z_span,
        "max_step_distance_m": sanity.max_step_distance_m,
        "median_step_distance_m": sanity.median_step_distance_m,
        "all_points_on_ground": sanity.all_points_on_ground,
        "implausible_depth_range": sanity.implausible_depth_range,
        "implausible_step_jump": sanity.implausible_step_jump,
        "trajectory_reliable": sanity.trajectory_reliable,
    }


def _scale_point(
    x_val: float,
    y_val: float,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    scale_x = target_width / source_width if source_width else 1.0
    scale_y = target_height / source_height if source_height else 1.0
    return int(round(x_val * scale_x)), int(round(y_val * scale_y))


def render_overlay_video(
    video_path: Path,
    output_path: Path,
    raw_path: list[BallDetection],
    selected_track: list[BallDetection],
    inliers: list[BallDetection],
    bounce_frame: int | None,
    result_payload: dict[str, Any],
    standardized_size: tuple[int, int],
) -> None:
    def _fmt_metric(value: float | None, format_spec: str) -> str:
        if value is None:
            return "unavailable"
        return format(value, format_spec)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    raw_by_frame: dict[int, list[BallDetection]] = {}
    for detection in raw_path:
        raw_by_frame.setdefault(detection.frame_idx, []).append(detection)
    selected_by_frame = {detection.frame_idx: detection for detection in selected_track}
    inlier_by_frame = {detection.frame_idx: detection for detection in inliers}
    bounce_detection = None
    bounce_candidates = selected_track or inliers or raw_path
    if bounce_frame is not None and bounce_candidates:
        bounce_detection = min(
            bounce_candidates,
            key=lambda item: abs(item.frame_idx - bounce_frame),
        )

    source_width, source_height = standardized_size
    frame_idx = 0
    drawn_raw_points: list[tuple[int, int]] = []
    drawn_selected: list[tuple[int, int]] = []
    drawn_inliers: list[tuple[int, int]] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            for detection in raw_by_frame.get(frame_idx, []):
                drawn_raw_points.append(
                    _scale_point(
                        detection.x,
                        detection.y,
                        source_width,
                        source_height,
                        width,
                        height,
                    )
                )

            selected_detection = selected_by_frame.get(frame_idx)
            if selected_detection is not None:
                drawn_selected.append(
                    _scale_point(
                        selected_detection.x,
                        selected_detection.y,
                        source_width,
                        source_height,
                        width,
                        height,
                    )
                )

            inlier_detection = inlier_by_frame.get(frame_idx)
            if inlier_detection is not None:
                drawn_inliers.append(
                    _scale_point(
                        inlier_detection.x,
                        inlier_detection.y,
                        source_width,
                        source_height,
                        width,
                        height,
                    )
                )

            for start, end in zip(drawn_selected, drawn_selected[1:], strict=False):
                cv2.line(frame, start, end, (255, 255, 0), 2)
            for start, end in zip(drawn_inliers, drawn_inliers[1:], strict=False):
                cv2.line(frame, start, end, (0, 255, 0), 3)

            for point in drawn_raw_points:
                cv2.circle(frame, point, 4, (90, 90, 255), -1)
            for point in drawn_selected:
                cv2.circle(frame, point, 5, (255, 255, 0), -1)
            for point in drawn_inliers:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)

            if bounce_detection is not None:
                bounce_point = _scale_point(
                    bounce_detection.x,
                    bounce_detection.y,
                    source_width,
                    source_height,
                    width,
                    height,
                )
                cv2.circle(frame, bounce_point, 10, (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    "bounce",
                    (bounce_point[0] + 10, bounce_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            metrics_lines = [
                f"frame={frame_idx}",
                "pink=raw cyan=selected green=inliers yellow=bounce",
                "raw_detections="
                f"{len(raw_path)} selected={len(selected_track)} "
                f"inliers={result_payload['inlier_count']}",
                f"speed_kmh={_fmt_metric(result_payload['speed_kmh'], '.2f')}",
                f"raw_speed_ms={_fmt_metric(result_payload['raw_speed_ms'], '.3f')}",
                f"swing_m={_fmt_metric(result_payload['swing_metres'], '.3f')}",
                f"length={result_payload['length_class']}",
            ]
            bounce_point_payload = result_payload.get("bounce_point")
            if bounce_point_payload is not None:
                metrics_lines.append(
                    "bounce_xz=("
                    f"{bounce_point_payload['x_metres']:.3f}, "
                    f"{bounce_point_payload['z_metres']:.3f})"
                )
            trajectory_warning = result_payload.get("trajectory_warning")
            if trajectory_warning:
                metrics_lines.append("trajectory=unavailable")

            for idx, line in enumerate(metrics_lines):
                cv2.putText(
                    frame,
                    line,
                    (12, 28 + idx * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()


async def process_pair(
    video_path: Path,
    calibration_path: Path,
    fallback_fov: float | None,
) -> dict[str, object]:
    calibration = parse_calibration_file(calibration_path, fallback_fov=fallback_fov)
    preprocessor = VideoPreprocessor()
    cleaner = BallPathCleaner()
    analyzer = BowlerPerformanceAnalyzer()
    loop = asyncio.get_running_loop()

    standardized_path = await preprocessor.standardize_video(video_path)
    standardized_fps, standardized_width, standardized_height = read_video_props(standardized_path)
    artifacts = await preprocessor.run(standardized_path, calibration)

    ransac_result = await loop.run_in_executor(
        None,
        partial(cleaner.clean, artifacts.ball_path, standardized_fps),
    )
    if ransac_result is None:
        raise RuntimeError("RANSAC did not find enough inliers to build diagnostics.")

    intrinsic = await loop.run_in_executor(None, partial(build_intrinsic_matrix, calibration))
    extrinsic = await loop.run_in_executor(None, partial(build_extrinsic_matrix, calibration))
    world_points = await loop.run_in_executor(
        None,
        partial(pixels_to_world_points, ransac_result.inliers, intrinsic, extrinsic),
    )
    reconstruction_sanity = await loop.run_in_executor(
        None,
        partial(assess_world_points, world_points),
    )
    pitch_frame = await loop.run_in_executor(
        None,
        partial(build_pitch_frame, calibration, intrinsic, extrinsic),
    )
    pitch_points = await loop.run_in_executor(
        None,
        partial(world_points_to_pitch_points, world_points, pitch_frame),
    )
    result = await analyzer.run(
        artifacts,
        calibration,
        standardized_fps,
        video_url=video_path.name,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overlay_path = OUTPUT_DIR / f"{video_path.stem}_bowler_overlay.mp4"
    diagnostics_path = OUTPUT_DIR / f"{video_path.stem}_bowler_diagnostics.json"
    try:
        await loop.run_in_executor(
            None,
            partial(
                render_overlay_video,
                video_path,
                overlay_path,
                artifacts.ball_path,
                ransac_result.selected_track,
                ransac_result.inliers,
                ransac_result.bounce_frame,
                result.model_dump(),
                (standardized_width, standardized_height),
            ),
        )
    except OSError as exc:
        raise RuntimeError(
            "Failed to write overlay video. Free some disk space and rerun."
        ) from exc

    diagnostics = {
        "video": video_path.name,
        "calibration": calibration_path.name,
        "effective_fov": calibration.fov,
        "standardized_video": standardized_path.name,
        "standardized_fps": standardized_fps,
        "ball_path_detections": len(artifacts.ball_path),
        "has_bat_contact_frame": artifacts.bat_contact_frame is not None,
        "overlay_video": overlay_path.name,
        "raw_ball_path": [serialize_ball_detection(detection) for detection in artifacts.ball_path],
        "ransac": {
            "selected_count": len(ransac_result.selected_track),
            "inlier_count": len(ransac_result.inliers),
            "bounce_frame": ransac_result.bounce_frame,
            "bounce_t": ransac_result.bounce_t,
            "para_x": {
                "a": ransac_result.para_x.a,
                "b": ransac_result.para_x.b,
                "c": ransac_result.para_x.c,
            },
            "para_y": {
                "a": ransac_result.para_y.a,
                "b": ransac_result.para_y.b,
                "c": ransac_result.para_y.c,
            },
            "selected_track": [
                serialize_ball_detection(detection)
                for detection in ransac_result.selected_track
            ],
            "inliers": [serialize_ball_detection(detection) for detection in ransac_result.inliers],
        },
        "world_points": [
            serialize_world_point(detection, world_point)
            for detection, world_point in world_points
        ],
        "reconstruction_sanity": build_reconstruction_sanity(world_points),
        "pitch_frame": {
            "batting_origin_world": pitch_frame.batting_origin_world.tolist(),
            "x_axis_world": pitch_frame.x_axis_world.tolist(),
            "z_axis_world": pitch_frame.z_axis_world.tolist(),
            "scale": pitch_frame.scale,
            "measured_pitch_length": pitch_frame.measured_pitch_length,
            "measured_batting_center_world": (
                pitch_frame.measured_batting_center_world.tolist()
                if pitch_frame.measured_batting_center_world is not None
                else None
            ),
            "measured_bowling_center_world": (
                pitch_frame.measured_bowling_center_world.tolist()
                if pitch_frame.measured_bowling_center_world is not None
                else None
            ),
            "length_reliable": pitch_frame.length_reliable,
            "final_bounce_z_before_classification": (
                result.bounce_point.z_metres if result.bounce_point is not None else None
            ),
        },
        "pitch_points": [
            serialize_pitch_point(detection, pitch_point)
            for detection, pitch_point in pitch_points
        ],
        "bowler_performance": result.model_dump(),
        "flutter_payload": result.model_dump(by_alias=True),
    }
    try:
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            "Failed to write diagnostics JSON. Free some disk space and rerun."
        ) from exc
    return diagnostics


def get_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run bowler performance on real sample videos.")
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

    free_bytes = get_free_bytes(OUTPUT_DIR if OUTPUT_DIR.exists() else BASE_DIR)
    print(f"Output free space: {free_bytes / (1024 * 1024):.1f} MB")

    summaries: list[dict[str, object]] = []
    for video_name, calibration_name in selected_pairs:
        video_path = BASE_DIR / video_name
        calibration_path = BASE_DIR / calibration_name
        print(f"Processing {video_name} with {calibration_name} ...")
        diagnostics = await process_pair(video_path, calibration_path, args.fallback_fov)
        summaries.append(
            {
                "video": diagnostics["video"],
                "overlay_video": diagnostics["overlay_video"],
                "ball_path_detections": diagnostics["ball_path_detections"],
                "selected_track_detections": diagnostics["ransac"]["selected_count"],
                "ransac_inliers": diagnostics["ransac"]["inlier_count"],
                "reconstruction_sanity": diagnostics["reconstruction_sanity"],
                "bowler_performance": diagnostics["bowler_performance"],
            }
        )
        print(json.dumps(summaries[-1], indent=2))

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
