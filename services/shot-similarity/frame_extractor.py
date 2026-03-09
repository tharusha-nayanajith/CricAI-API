"""
frame_extractor.py
------------------
Extract a small window of frames around an estimated impact frame from a video.

Why extract only a window?
    Processing the entire video with YOLO would be slow and memory-intensive.
    Since audio detection already gives us a good estimate of the impact time,
    we only need to inspect a ±N frame buffer around that estimate.

Usage:
    frames = extract_frames_around_impact(
        video_path="cricket_shot.mp4",
        impact_frame=42,
        window=5,        # frames before and after
    )
    # frames → [(37, np.ndarray), (38, np.ndarray), ..., (47, np.ndarray)]
"""

import cv2
import numpy as np
from typing import List, Tuple


def extract_frames_around_impact(
    video_path: str,
    impact_frame: int,
    window: int = 5,
) -> List[Tuple[int, np.ndarray]]:
    """
    Extract video frames within [impact_frame - window, impact_frame + window].

    The range is clamped to valid frame indices (0 … total_frames - 1) so
    the caller does not need to worry about boundary conditions.

    Args:
        video_path:   Path to the video file.
        impact_frame: Estimated impact frame index (0-based).
        window:       Number of frames to extract before and after impact_frame.

    Returns:
        List of (frame_index, BGR_numpy_array) tuples, in chronological order.

    Raises:
        FileNotFoundError: If the video file cannot be opened.
        ValueError:        If no frames could be extracted from the range.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Cannot open video file: '{video_path}'. "
            "Ensure the path is correct and the codec is supported."
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video '{video_path}' reports 0 frames.")

    # Clamp to valid range
    start_frame = max(0, impact_frame - window)
    end_frame = min(total_frames - 1, impact_frame + window)

    frames: List[Tuple[int, np.ndarray]] = []

    # Seek directly to start_frame (much faster than decoding from the beginning)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            # Decoding failed – try to continue with remaining frames
            break
        frames.append((frame_idx, frame))

    cap.release()

    if not frames:
        raise ValueError(
            f"No frames could be extracted from video '{video_path}' "
            f"in range [{start_frame}, {end_frame}]."
        )

    return frames


def get_video_metadata(video_path: str) -> dict:
    """
    Return basic metadata (fps, total frames, width, height) for a video.

    Useful for converting between frame indices and timestamps.

    Args:
        video_path: Path to the video file.

    Returns:
        dict with keys "fps", "total_frames", "width", "height", "duration_seconds".

    Raises:
        FileNotFoundError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: '{video_path}'.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": round(total_frames / fps, 4) if fps > 0 else 0.0,
    }
