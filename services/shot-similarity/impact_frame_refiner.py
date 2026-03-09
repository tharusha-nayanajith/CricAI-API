"""
impact_frame_refiner.py
-----------------------
Refines the audio-estimated impact frame using YOLO ball tracking.

Logic:
    1. Run ball_detector on each frame in the window.
    2. Collect ball centre positions across frames.
    3. Compute Euclidean velocity between successive frames.
    4. The frame with the largest velocity change (delta-v) is the true impact:
       at impact, the ball reverses or sharply changes direction.
    5. Falls back to the audio-estimated frame if < 2 ball detections exist.

This approach is robust to:
    - Brief occlusion of the ball (we skip non-detections and interpolate)
    - Minor audio spike mis-timing (audio ± a few frames)
"""

import numpy as np
from typing import List, Tuple, Optional
import cv2

from ball_detector import detect_ball, load_model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_velocity(positions: List[Tuple[int, Tuple[int, int]]]) -> List[Tuple[int, float]]:
    """
    Compute scalar velocity between successive detected ball positions.

    Args:
        positions: List of (frame_index, (cx, cy)) for frames where ball was found.

    Returns:
        List of (frame_index, velocity) where velocity is the Euclidean distance
        between the ball centres in consecutive detections.
        The frame_index is the *later* of the two frames in each pair.
    """
    velocities = []
    for i in range(1, len(positions)):
        f_prev, (x_prev, y_prev) = positions[i - 1]
        f_curr, (x_curr, y_curr) = positions[i]

        # Normalise by the number of frames between detections
        # (handles sparse detections gracefully)
        frame_gap = max(1, f_curr - f_prev)
        dist = np.sqrt((x_curr - x_prev) ** 2 + (y_curr - y_prev) ** 2)
        velocity = dist / frame_gap
        velocities.append((f_curr, velocity))

    return velocities


def _find_max_velocity_change(velocities: List[Tuple[int, float]]) -> Optional[int]:
    """
    Find the frame with the largest change in ball velocity (delta-v).

    The impact frame is characterised by a sudden drop or spike in speed
    as the ball reverses direction after hitting the bat.

    Args:
        velocities: List of (frame_index, speed) pairs.

    Returns:
        Frame index of maximum velocity change, or None if fewer than 2 entries.
    """
    if len(velocities) < 2:
        return None

    max_delta = -1.0
    best_frame = None

    for i in range(1, len(velocities)):
        f_curr, v_curr = velocities[i]
        _, v_prev = velocities[i - 1]
        delta = abs(v_curr - v_prev)

        if delta > max_delta:
            max_delta = delta
            best_frame = f_curr

    return best_frame


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refine_impact_frame(
    frames: List[Tuple[int, np.ndarray]],
    audio_impact_frame: int,
    model=None,
) -> dict:
    """
    Refine the estimated impact frame using ball velocity analysis.

    Args:
        frames:              List of (frame_index, BGR_ndarray) tuples from frame_extractor.
        audio_impact_frame:  Fall-back frame index from audio spike detection.
        model:               Pre-loaded YOLO model (optional; loaded on first use if None).

    Returns:
        dict with keys:
            "impact_frame"  (int)  – Refined impact frame index.
            "method"        (str)  – "ball_velocity" or "audio_fallback".
            "ball_detections" (int) – Number of frames where ball was detected.
    """
    if model is None:
        model = load_model()

    # Step 1: Detect ball in every frame
    detected_positions: List[Tuple[int, Tuple[int, int]]] = []

    for frame_idx, frame in frames:
        result = detect_ball(frame, model=model)
        if result is not None:
            detected_positions.append((frame_idx, (result["x"], result["y"])))

    num_detections = len(detected_positions)

    # Step 2: Need at least 2 detections to compute velocity
    if num_detections < 2:
        return {
            "impact_frame": audio_impact_frame,
            "method": "audio_fallback",
            "ball_detections": num_detections,
        }

    # Step 3: Compute velocity between successive detections
    velocities = _compute_velocity(detected_positions)

    # Step 4: Find the frame with the largest velocity change
    best_frame = _find_max_velocity_change(velocities)

    if best_frame is None:
        return {
            "impact_frame": audio_impact_frame,
            "method": "audio_fallback",
            "ball_detections": num_detections,
        }

    return {
        "impact_frame": best_frame,
        "method": "ball_velocity",
        "ball_detections": num_detections,
    }
