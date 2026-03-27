from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger

from app.exceptions import PreprocessingError
from app.modules.preprocessor.constants import (
    BALL_CONF_RAW_THRESHOLD,
    BALL_EARLY_STOP_CONF,
    BALL_EARLY_STOP_MIN_FRAME,
    BALL_EARLY_STOP_Y,
    MAX_BALL_TRACK_FRAMES,
    STANDARDIZED_HEIGHT,
    STANDARDIZED_WIDTH,
)
from app.modules.preprocessor.models import BallDetection, BatterMode, BatterROI


class BallTracker:
    def __init__(self, model_path: Path):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.output_name = self.session.get_outputs()[0].name
        self._frame_buffer: list[np.ndarray] = []
        self._last_roi_entry_frame_idx: int | None = None
        logger.info("Ball tracker provider: {}", self.session.get_providers()[0])

    def reset(self) -> None:
        self._frame_buffer = []
        self._last_roi_entry_frame_idx = None

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        if (
            frame_bgr.shape[0] != STANDARDIZED_HEIGHT
            or frame_bgr.shape[1] != STANDARDIZED_WIDTH
        ):
            frame_bgr = cv2.resize(frame_bgr, (STANDARDIZED_WIDTH, STANDARDIZED_HEIGHT))

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = frame_rgb.astype(np.float32) * (1.0 / 127.5) - 1.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def _infer(self, frame_bgr: np.ndarray) -> tuple[float | None, float | None, float]:
        processed = self._preprocess(frame_bgr)
        self._frame_buffer.append(processed)
        if len(self._frame_buffer) > 3:
            self._frame_buffer.pop(0)
        if len(self._frame_buffer) < 3:
            return None, None, 0.0

        outputs = self.session.run(
            [self.output_name],
            {
                "input_image1": self._frame_buffer[0],
                "input_image2": self._frame_buffer[1],
                "input_image3": self._frame_buffer[2],
            },
        )
        scores = outputs[0][0][0]
        _, max_val, _, max_loc = cv2.minMaxLoc(scores)
        peak_x = max_loc[0] * (STANDARDIZED_WIDTH / scores.shape[1])
        peak_y = max_loc[1] * (STANDARDIZED_HEIGHT / scores.shape[0])
        return peak_x, peak_y, float(max_val)

    def track(
        self,
        video_path: Path,
        release_frame_idx: int,
        fps: float,
        batter_mode: BatterMode,
        batter_roi: BatterROI | None,
    ) -> list[BallDetection]:
        self.reset()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise PreprocessingError(f"Unable to open video file: {video_path}")

        detections: list[BallDetection] = []
        termination_reason = "end_of_video"
        start_frame_idx = max(0, release_frame_idx - 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

        try:
            for _current_frame_idx in range(start_frame_idx, release_frame_idx):
                ret, frame = cap.read()
                if not ret:
                    termination_reason = "warmup_read_failed"
                    break
                self._infer(frame)

            if termination_reason == "warmup_read_failed":
                return detections

            for current_frame_idx in range(
                release_frame_idx,
                release_frame_idx + MAX_BALL_TRACK_FRAMES,
            ):
                ret, frame = cap.read()
                if not ret:
                    break

                x, y, conf = self._infer(frame)
                if x is None or y is None:
                    continue

                if conf >= BALL_CONF_RAW_THRESHOLD:
                    detections.append(
                        BallDetection(
                            frame_idx=current_frame_idx,
                            timestamp_s=current_frame_idx / fps if fps > 0 else 0.0,
                            x=x,
                            y=y,
                            confidence=conf,
                        )
                    )

                if batter_mode is BatterMode.PRESENT and batter_roi is not None:
                    if (
                        self._last_roi_entry_frame_idx is None
                        and self._ball_in_roi(x, y, batter_roi)
                    ):
                        self._last_roi_entry_frame_idx = current_frame_idx
                        logger.info(
                            "Ball entered batter ROI at frame {}",
                            current_frame_idx,
                        )
                else:
                    tracked = current_frame_idx - release_frame_idx
                    if (
                        tracked > BALL_EARLY_STOP_MIN_FRAME
                        and conf < BALL_EARLY_STOP_CONF
                        and y < BALL_EARLY_STOP_Y
                    ):
                        termination_reason = "early_stop_low_confidence"
                        break
        finally:
            cap.release()

        logger.info(
            "Ball tracker detections={} termination_reason={} roi_entry_frame={}",
            len(detections),
            termination_reason,
            self._last_roi_entry_frame_idx,
        )
        return detections

    def _ball_in_roi(
        self,
        x: float,
        y: float,
        roi: BatterROI,
    ) -> bool:
        return roi.x <= x <= roi.x + roi.width and roi.y <= y <= roi.y + roi.height
