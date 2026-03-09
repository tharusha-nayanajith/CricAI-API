"""
ball_detector.py
----------------
Detects the cricket ball in video frames using YOLOv8.

Optimised for:
    - Low memory usage (nano model, yolov8n.pt)
    - Fast inference suitable for mobile-app backend usage
    - Processes only specific frames rather than full video

Usage:
    model = load_model()
    result = detect_ball(frame, model)
    # result → {"x": 120, "y": 240, "confidence": 0.91}  or  None
"""

import numpy as np
import cv2
from typing import Optional

# Lazy import so the module is importable even if ultralytics is not yet installed
try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

# When using a CUSTOM trained cricket ball model:
#   - Class 0 = cricket ball (single-class dataset from Roboflow)
# When using the default COCO yolov8n.pt:
#   - Class 32 = sports ball (less accurate for cricket)
_CRICKET_BALL_CLASS_ID = 0   # Custom model class
_COCO_SPORTS_BALL_CLASS_ID = 32  # COCO fallback class

# Set to your custom trained weights after fine-tuning:
#   e.g. "runs/detect/train/weights/best.pt"
# Leave as "yolov8n.pt" to use the COCO pretrained model (less accurate)
_DEFAULT_MODEL_PATH = "yolov8n.pt"

# Minimum confidence threshold for a valid ball detection
_MIN_CONFIDENCE = 0.35

_model_instance: Optional[object] = None
_model_path_used: str = ""



def load_model(model_path: str = _DEFAULT_MODEL_PATH) -> object:
    """
    Load (or return cached) YOLOv8 model.

    For best cricket ball detection, pass your custom-trained weights:
        load_model("runs/detect/train/weights/best.pt")

    Falls back to COCO pretrained yolov8n.pt if no custom model is available.

    Args:
        model_path: Path or name of the YOLO model weights.

    Returns:
        Loaded YOLO model object.

    Raises:
        ImportError: If the `ultralytics` package is not installed.
    """
    global _model_instance, _model_path_used

    if not _ULTRALYTICS_AVAILABLE:
        raise ImportError(
            "The 'ultralytics' package is required for ball detection. "
            "Install it with: pip install ultralytics"
        )

    # Reload if a different model path is requested
    if _model_instance is None or _model_path_used != model_path:
        _model_instance = YOLO(model_path)
        _model_path_used = model_path

    return _model_instance



# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_ball(
    frame: np.ndarray,
    model=None,
    confidence_threshold: float = _MIN_CONFIDENCE,
) -> Optional[dict]:
    """
    Detect the cricket ball in a single video frame.

    Runs YOLOv8 inference and returns the highest-confidence bounding box
    that matches the "sports ball" COCO class.

    Args:
        frame:                BGR NumPy array (as returned by OpenCV).
        model:                Pre-loaded YOLO model. If None, load_model() is called.
        confidence_threshold: Minimum confidence to accept a detection.

    Returns:
        dict with keys:
            "x"          (int)   – Ball centre X coordinate (pixels)
            "y"          (int)   – Ball centre Y coordinate (pixels)
            "confidence" (float) – Detection confidence  [0.0 – 1.0]
            "bbox"       (dict)  – {"x1", "y1", "x2", "y2"} bounding box corners
        Returns None if no ball is detected above the threshold.

    Raises:
        ImportError: If the `ultralytics` package is not installed.
        ValueError:  If the frame is None or empty.
    """
    if frame is None or frame.size == 0:
        raise ValueError("detect_ball() received an empty or None frame.")

    if model is None:
        model = load_model()

    # Resize frame for faster inference while preserving aspect ratio
    # Target: 640 px on the longest side (YOLO's native training resolution)
    frame_for_inference = _resize_for_inference(frame, target_size=640)

    # Auto-select class filter based on which model is loaded:
    #   - Custom cricket model → class 0 (cricket ball), no filter needed (single class)
    #   - COCO pretrained model → class 32 (sports ball)
    is_custom_model = _model_path_used not in ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt")
    class_filter = None if is_custom_model else [_COCO_SPORTS_BALL_CLASS_ID]

    results = model(
        frame_for_inference,
        classes=class_filter,
        verbose=False,
    )

    best_detection = None
    best_confidence = 0.0

    # Scale factor to map back to original frame coordinates
    scale_x = frame.shape[1] / frame_for_inference.shape[1]
    scale_y = frame.shape[0] / frame_for_inference.shape[0]

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            confidence = float(box.conf[0])

            if confidence < confidence_threshold:
                continue

            if confidence > best_confidence:
                best_confidence = confidence

                # Bounding box in xyxy format (inference frame)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Scale back to original frame coordinates
                x1_orig = int(x1 * scale_x)
                y1_orig = int(y1 * scale_y)
                x2_orig = int(x2 * scale_x)
                y2_orig = int(y2 * scale_y)

                # Ball centre
                cx = int((x1_orig + x2_orig) / 2)
                cy = int((y1_orig + y2_orig) / 2)

                best_detection = {
                    "x": cx,
                    "y": cy,
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": x1_orig,
                        "y1": y1_orig,
                        "x2": x2_orig,
                        "y2": y2_orig,
                    },
                }

    return best_detection


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resize_for_inference(frame: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    Resize a frame so its longest side equals `target_size`, preserving aspect ratio.

    This drastically speeds up YOLO inference without affecting detection quality
    for objects of reasonable size.

    Args:
        frame:       Input BGR frame.
        target_size: Pixel length of the longest side after resize.

    Returns:
        Resized BGR frame.
    """
    h, w = frame.shape[:2]
    if max(h, w) <= target_size:
        return frame

    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
