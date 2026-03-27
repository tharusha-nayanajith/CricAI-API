import argparse
import asyncio
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import cv2
from loguru import logger

from app.exceptions import PreprocessingError
from app.modules.preprocessor.constants import BAT_CONTACT_AUDIO_WINDOW_FRAMES
from app.modules.preprocessor.models import (
    BallDetection,
    BatterMode,
    BatterROI,
    DeliveryContext,
)
from app.modules.preprocessor.service import (
    VideoPreprocessor,
    get_ball_tracker,
    get_bat_contact_detector,
    get_batter_detector,
)
from Test_Scripts.run_preprocessor_check import OUTPUT_DIR, parse_calibration_file

BASE_DIR = Path(__file__).resolve().parent


@dataclass(slots=True)
class ContactVisualization:
    frame_idx: int
    timestamp_s: float
    label: str
    score: float | None = None
    exact: bool = False


@dataclass(slots=True)
class SearchVisualization:
    anchor_label: str | None
    anchor_frame_idx: int | None
    search_start_frame: int | None
    search_end_frame: int | None
    window_message: str | None


def _nearest_detection(
    ball_path: list[BallDetection],
    frame_idx: int,
) -> BallDetection | None:
    if not ball_path:
        return None
    return min(ball_path, key=lambda detection: abs(detection.frame_idx - frame_idx))


def _annotate_frame(
    frame: Any,
    frame_idx: int,
    batter_roi: BatterROI | None,
    ball_path: list[BallDetection],
    contact_frame_idx: int | None,
    contact_method: str | None,
    contact_score: float | None,
    exact_contact: bool,
    search_visualization: SearchVisualization | None,
) -> None:
    if batter_roi is not None:
        top_left = (batter_roi.x, batter_roi.y)
        bottom_right = (batter_roi.x + batter_roi.width, batter_roi.y + batter_roi.height)
        cv2.rectangle(frame, top_left, bottom_right, (255, 180, 0), 2)
        cv2.putText(
            frame,
            "Batter ROI",
            (batter_roi.x, max(30, batter_roi.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )

    visible_detections = [
        detection for detection in ball_path if detection.frame_idx <= frame_idx
    ]
    for idx, detection in enumerate(visible_detections):
        point = (int(detection.x), int(detection.y))
        color = (0, 255, 0) if idx < len(visible_detections) - 1 else (0, 255, 255)
        radius = 4 if idx < len(visible_detections) - 1 else 6
        cv2.circle(frame, point, radius, color, -1)
        if idx > 0:
            previous = visible_detections[idx - 1]
            previous_point = (int(previous.x), int(previous.y))
            cv2.line(frame, previous_point, point, (255, 220, 0), 2)

    cv2.putText(
        frame,
        f"Frame: {frame_idx}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if search_visualization is not None:
        if (
            search_visualization.search_start_frame is not None
            and search_visualization.search_end_frame is not None
            and search_visualization.search_start_frame
            <= frame_idx
            <= search_visualization.search_end_frame
        ):
            height, width = frame.shape[:2]
            cv2.rectangle(frame, (8, 8), (width - 8, height - 8), (0, 200, 255), 3)
            cv2.putText(
                frame,
                "AUDIO SEARCH WINDOW",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

        if (
            search_visualization.anchor_frame_idx is not None
            and frame_idx == search_visualization.anchor_frame_idx
            and search_visualization.anchor_label is not None
        ):
            cv2.putText(
                frame,
                f"ANCHOR: {search_visualization.anchor_label}",
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 200, 0),
                2,
                cv2.LINE_AA,
            )

        if search_visualization.window_message is not None:
            cv2.putText(
                frame,
                search_visualization.window_message,
                (20, 255),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

    if contact_frame_idx is None:
        cv2.putText(
            frame,
            "Bat contact not detected",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        return

    cv2.putText(
        frame,
        f"Contact frame: {contact_frame_idx}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if contact_method is not None and contact_score is not None:
        cv2.putText(
            frame,
            f"Method: {contact_method} | Score: {contact_score:.3f}",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if contact_method is not None and contact_score is None:
        cv2.putText(
            frame,
            f"Method: {contact_method}",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if frame_idx != contact_frame_idx:
        return

    _height, width = frame.shape[:2]
    overlay = frame.copy()
    banner_color = (0, 0, 180) if exact_contact else (0, 120, 220)
    cv2.rectangle(overlay, (0, 0), (width, 140), banner_color, -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0.0, frame)
    cv2.putText(
        frame,
        "BAT CONTACT" if exact_contact else "APPROX BAT CONTACT",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            f"Frame {contact_frame_idx} | {contact_method} | score={contact_score:.3f}"
            if contact_score is not None
            else f"Frame {contact_frame_idx} | {contact_method}"
        ),
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    nearest = _nearest_detection(ball_path, frame_idx)
    if nearest is not None:
        point = (int(nearest.x), int(nearest.y))
        cv2.circle(frame, point, 18, (0, 0, 255), 3)
        cv2.line(frame, (point[0] - 24, point[1]), (point[0] + 24, point[1]), (0, 0, 255), 2)
        cv2.line(frame, (point[0], point[1] - 24), (point[0], point[1] + 24), (0, 0, 255), 2)


async def _collect_preprocessor_context(
    video_path: Path,
    calibration_path: Path,
) -> tuple[
    Path,
    float,
    BatterMode,
    BatterROI | None,
    list[BallDetection],
    ContactVisualization | None,
    SearchVisualization | None,
]:
    calibration = parse_calibration_file(calibration_path)
    preprocessor = VideoPreprocessor()
    loop = asyncio.get_event_loop()

    standardized_path = await preprocessor.standardize_video(video_path)
    batter_detector = get_batter_detector()
    batter_mode, batter_roi = await loop.run_in_executor(
        None,
        partial(batter_detector.detect, standardized_path, calibration),
    )
    fps = await loop.run_in_executor(
        None,
        partial(VideoPreprocessor._read_fps, standardized_path),
    )
    ctx = DeliveryContext(
        standardized_video_path=standardized_path,
        batter_mode=batter_mode,
        batter_roi=batter_roi,
        fps=fps,
    )
    release_point = await preprocessor._detect_release(ctx)
    ball_tracker = get_ball_tracker()
    ball_path = await loop.run_in_executor(
        None,
        partial(
            ball_tracker.track,
            standardized_path,
            release_point.frame_idx,
            fps,
            batter_mode,
            batter_roi,
        ),
    )
    if len(ball_path) < 3:
        raise PreprocessingError(f"Ball path too short: {len(ball_path)} detections")

    contact_visualization = None
    search_visualization = None
    if batter_mode is BatterMode.PRESENT and batter_roi is not None:
        bat_contact_detector = get_bat_contact_detector()
        search_visualization = _build_search_visualization(
            standardized_path,
            fps,
            batter_roi,
            ball_path,
        )
        try:
            bat_contact = await loop.run_in_executor(
                None,
                partial(
                    bat_contact_detector.detect,
                    standardized_path,
                    fps,
                    batter_roi,
                    ball_path,
                ),
            )
        except Exception as exc:
            logger.warning("Bat contact detection failed during visualization: {}", exc)
            bat_contact = None

        if bat_contact is not None:
            contact_visualization = ContactVisualization(
                frame_idx=bat_contact.contact_frame_idx,
                timestamp_s=bat_contact.timestamp_s,
                label=bat_contact.method.value,
                score=bat_contact.detection_score,
                exact=True,
            )
        else:
            approx_estimate = _estimate_contact_frame(
                standardized_path,
                fps,
                batter_roi,
                ball_path,
            )
            if approx_estimate is not None:
                contact_visualization = ContactVisualization(
                    frame_idx=approx_estimate[0],
                    timestamp_s=approx_estimate[0] / fps if fps > 0 else 0.0,
                    label=approx_estimate[1],
                    score=None,
                    exact=False,
                )

    return (
        standardized_path,
        fps,
        batter_mode,
        batter_roi,
        ball_path,
        contact_visualization,
        search_visualization,
    )


def _build_search_visualization(
    video_path: Path,
    fps: float,
    batter_roi: BatterROI,
    ball_path: list[BallDetection],
) -> SearchVisualization:
    bat_contact_detector = get_bat_contact_detector()
    _ = batter_roi
    audio_result = bat_contact_detector._detect_impact_frame(video_path, fps)
    anchor_frame_idx = None if audio_result is None else int(audio_result["impact_frame"])
    search_start_frame = (
        None
        if anchor_frame_idx is None
        else max(0, anchor_frame_idx - BAT_CONTACT_AUDIO_WINDOW_FRAMES)
    )
    search_end_frame = (
        None
        if anchor_frame_idx is None
        else anchor_frame_idx + BAT_CONTACT_AUDIO_WINDOW_FRAMES
    )
    anchor_label = "audio_spike" if anchor_frame_idx is not None else None
    return SearchVisualization(
        anchor_label=anchor_label,
        anchor_frame_idx=anchor_frame_idx,
        search_start_frame=search_start_frame,
        search_end_frame=search_end_frame,
        window_message=(
            None
            if anchor_frame_idx is not None
            else "Audio impact detection unavailable"
        ),
    )


def _estimate_contact_frame(
    video_path: Path,
    fps: float,
    batter_roi: BatterROI,
    ball_path: list[BallDetection],
) -> tuple[int, str] | None:
    bat_contact_detector = get_bat_contact_detector()
    _ = batter_roi
    audio_result = bat_contact_detector._detect_impact_frame(video_path, fps)
    if audio_result is None:
        return None
    refined_frame_idx, method, _score = bat_contact_detector._refine_impact_frame(
        ball_path,
        int(audio_result["impact_frame"]),
    )
    return refined_frame_idx, method.value


def _write_visualization_video(
    video_path: Path,
    output_path: Path,
    fps: float,
    batter_roi: BatterROI | None,
    ball_path: list[BallDetection],
    bat_contact: ContactVisualization | None,
    search_visualization: SearchVisualization | None,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PreprocessingError(f"Unable to open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    effective_fps = fps if fps > 0 else float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        effective_fps,
        (width, height),
    )

    contact_frame_idx = None if bat_contact is None else bat_contact.frame_idx
    contact_method = None if bat_contact is None else bat_contact.label
    contact_score = None if bat_contact is None else bat_contact.score
    exact_contact = False if bat_contact is None else bat_contact.exact

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _annotate_frame(
                frame,
                frame_idx,
                batter_roi,
                ball_path,
                contact_frame_idx,
                contact_method,
                contact_score,
                exact_contact,
                search_visualization,
            )
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()


async def process_pair(video_path: Path, calibration_path: Path) -> None:
    (
        standardized_path,
        fps,
        batter_mode,
        batter_roi,
        ball_path,
        bat_contact,
        search_visualization,
    ) = await _collect_preprocessor_context(video_path, calibration_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{video_path.stem}_bat_contact_visualization.mp4"
    _write_visualization_video(
        standardized_path,
        output_path,
        fps,
        batter_roi,
        ball_path,
        bat_contact,
        search_visualization,
    )

    if bat_contact is None:
        print(
            f"{video_path.name}: saved {output_path.name} "
            f"(batter_mode={batter_mode.value}, no bat contact detected)"
        )
        return

    if not bat_contact.exact:
        print(
            f"{video_path.name}: saved {output_path.name} "
            f"with approximate contact frame {bat_contact.frame_idx} via {bat_contact.label}"
        )
        return

    print(
        (
            f"{video_path.name}: saved {output_path.name} "
            f"with contact frame {bat_contact.frame_idx} "
            f"via {bat_contact.label} score={bat_contact.score:.3f}"
        )
        if bat_contact.score is not None
        else (
            f"{video_path.name}: saved {output_path.name} "
            f"with contact frame {bat_contact.frame_idx} "
            f"via {bat_contact.label}"
        )
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a full video visualization with the detected bat contact frame marked.",
    )
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
