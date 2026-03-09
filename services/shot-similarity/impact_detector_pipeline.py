"""
impact_detector_pipeline.py
----------------------------
Orchestrates the full cricket bat-ball impact detection pipeline.

Pipeline steps:
    1. Extract audio from video and detect audio spike (impact_audio_detector)
    2. Convert spike time → estimated impact frame
    3. Extract ±5 frames around that estimate (frame_extractor)
    4. Detect cricket ball using YOLOv8 in those frames (ball_detector)
    5. Refine impact frame using ball velocity analysis (impact_frame_refiner)

Returns:
    {"impact_frame": int, "impact_time": float}

Design decisions:
    - YOLO model is loaded once and passed through the pipeline (avoids reload overhead)
    - Window size is configurable (default ±5 frames) for performance tuning
    - All expensive operations are deferred: audio extraction, YOLO load, frame seek
"""

import cv2
from typing import Optional

from impact_audio_detector import detect_impact_frame
from frame_extractor import extract_frames_around_impact, get_video_metadata
from impact_frame_refiner import refine_impact_frame
from ball_detector import load_model


def run_impact_detection_pipeline(
    video_path: str,
    frame_window: int = 5,
    yolo_model_path: str = "yolov8n.pt",
    preloaded_model=None,
) -> dict:
    """
    Run the complete bat-ball impact detection pipeline on a cricket video.

    Args:
        video_path:       Path to the cricket shot video file.
        frame_window:     Number of frames to examine before/after audio spike.
                          Smaller = faster; larger = more robust to audio timing error.
                          Default is 5 (total 11-frame window).
        yolo_model_path:  Path or name of YOLOv8 weights to use.
                          "yolov8n.pt" (nano) is recommended for mobile/API use.
        preloaded_model:  If you already have a YOLO model loaded, pass it here
                          to avoid redundant loading across multiple calls.

    Returns:
        dict:
            "impact_frame"   (int)   – Final impact frame index (0-based).
            "impact_time"    (float) – Impact time in seconds (derived from frame + FPS).
            "method"         (str)   – Detection method used: "ball_velocity" or "audio_fallback".
            "ball_detections" (int)  – Number of frames where ball was detected.
            "audio_estimate" (int)   – Initial audio-based estimate (before refinement).

    Raises:
        FileNotFoundError: If the video file does not exist or cannot be opened.
        ValueError:        If the video has no audio track or is unreadable.
    """
    # -----------------------------------------------------------------------
    # Step 1 & 2: Audio spike detection + frame conversion
    # -----------------------------------------------------------------------
    audio_result = detect_impact_frame(video_path)
    audio_impact_frame: int = audio_result["impact_frame"]
    audio_impact_time: float = audio_result["impact_time"]

    # -----------------------------------------------------------------------
    # Step 3: Extract frames around estimated impact
    # -----------------------------------------------------------------------
    frames = extract_frames_around_impact(
        video_path=video_path,
        impact_frame=audio_impact_frame,
        window=frame_window,
    )

    # -----------------------------------------------------------------------
    # Step 4 & 5: Ball detection + velocity-based refinement
    # -----------------------------------------------------------------------
    model = preloaded_model if preloaded_model is not None else load_model(yolo_model_path)

    refinement = refine_impact_frame(
        frames=frames,
        audio_impact_frame=audio_impact_frame,
        model=model,
    )

    final_impact_frame: int = refinement["impact_frame"]

    # -----------------------------------------------------------------------
    # Convert the final frame back to a precise timestamp using video FPS
    # -----------------------------------------------------------------------
    metadata = get_video_metadata(video_path)
    fps = metadata["fps"]
    final_impact_time = round(final_impact_frame / fps, 4) if fps > 0 else audio_impact_time

    return {
        "impact_frame": final_impact_frame,
        "impact_time": final_impact_time,
        "method": refinement["method"],
        "ball_detections": refinement["ball_detections"],
        "audio_estimate": audio_impact_frame,
    }
