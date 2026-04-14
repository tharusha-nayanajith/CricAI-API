from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from loguru import logger

from app.exceptions import PreprocessingError
from app.modules.preprocessor.constants import (
    BALL_CONF_RAW_THRESHOLD,
    BALL_DYNAMIC_SEARCH_BASE_RADIUS_PX,
    BALL_DYNAMIC_SEARCH_MAX_RADIUS_PX,
    BALL_DYNAMIC_SEARCH_MAX_RECOVERY_RADIUS_PX,
    BALL_DYNAMIC_SEARCH_MIN_HISTORY,
    BALL_DYNAMIC_SEARCH_RADIUS_SCALE,
    BALL_EARLY_STOP_CONF,
    BALL_EARLY_STOP_MIN_FRAME,
    BALL_EARLY_STOP_Y,
    BALL_MAX_CANDIDATES_PER_FRAME,
    BALL_PEAK_NMS_RADIUS,
    MAX_BALL_TRACK_FRAMES,
    STANDARDIZED_HEIGHT,
    STANDARDIZED_WIDTH,
)
from app.modules.preprocessor.models import (
    BallDetection,
    BatterMode,
    BatterROI,
    FrameBallDetections,
)
from app.modules.preprocessor.path_rebuilder import DeliveryPathRebuilder

try:
    import onnxruntime as ort
except ImportError:
    ort = SimpleNamespace(InferenceSession=None, SessionOptions=lambda: SimpleNamespace())


class BallTracker:
    def __init__(self, model_path: Path):
        if ort.InferenceSession is None:
            raise PreprocessingError("onnxruntime is required for ball tracking.")
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
        self._last_frame_candidates: list[FrameBallDetections] = []
        self._accepted_track_history: list[BallDetection] = []
        self._rebuilder = DeliveryPathRebuilder()
        logger.info("Ball tracker provider: {}", self.session.get_providers()[0])

    @property
    def last_roi_entry_frame_idx(self) -> int | None:
        return self._last_roi_entry_frame_idx

    @property
    def last_frame_candidates(self) -> list[FrameBallDetections]:
        return [
            FrameBallDetections(
                frame_idx=frame.frame_idx,
                timestamp_s=frame.timestamp_s,
                detections=list(frame.detections),
            )
            for frame in self._last_frame_candidates
        ]

    def reset(self) -> None:
        self._frame_buffer = []
        self._last_roi_entry_frame_idx = None
        self._last_frame_candidates = []
        self._accepted_track_history = []

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

    def _infer_candidates(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float]]:
        processed = self._preprocess(frame_bgr)
        self._frame_buffer.append(processed)
        if len(self._frame_buffer) > 3:
            self._frame_buffer.pop(0)
        if len(self._frame_buffer) < 3:
            return []

        outputs = self.session.run(
            [self.output_name],
            {
                "input_image1": self._frame_buffer[0],
                "input_image2": self._frame_buffer[1],
                "input_image3": self._frame_buffer[2],
            },
        )
        scores = outputs[0][0][0]
        return self._extract_candidates(scores)

    def _infer(self, frame_bgr: np.ndarray) -> tuple[float | None, float | None, float]:
        candidates = self._infer_candidates(frame_bgr)
        if not candidates:
            return None, None, 0.0
        return candidates[0]

    def _filter_candidates_by_tracking_window(
        self,
        candidates: list[tuple[float, float, float]],
    ) -> list[tuple[float, float, float]]:
        if (
            len(candidates) <= 1
            or len(self._accepted_track_history) < BALL_DYNAMIC_SEARCH_MIN_HISTORY
        ):
            return candidates

        expected_x, expected_y, radius_px = self._tracking_expectation()
        in_window = [
            candidate
            for candidate in candidates
            if np.hypot(candidate[0] - expected_x, candidate[1] - expected_y) <= radius_px
        ]
        if in_window:
            return in_window

        nearest = min(
            candidates,
            key=lambda candidate: np.hypot(
                candidate[0] - expected_x,
                candidate[1] - expected_y,
            ),
        )
        nearest_distance = float(np.hypot(nearest[0] - expected_x, nearest[1] - expected_y))
        if nearest_distance <= BALL_DYNAMIC_SEARCH_MAX_RECOVERY_RADIUS_PX:
            return [nearest]
        return []

    def _tracking_expectation(self) -> tuple[float, float, float]:
        last = self._accepted_track_history[-1]
        previous = self._accepted_track_history[-2]
        vx = float(last.x - previous.x)
        vy = float(last.y - previous.y)
        expected_x = float(last.x + vx)
        expected_y = float(last.y + vy)
        speed_px = float(np.hypot(vx, vy))
        radius_px = min(
            BALL_DYNAMIC_SEARCH_MAX_RADIUS_PX,
            BALL_DYNAMIC_SEARCH_BASE_RADIUS_PX + speed_px * BALL_DYNAMIC_SEARCH_RADIUS_SCALE,
        )
        return expected_x, expected_y, radius_px

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

        raw_detections: list[BallDetection] = []
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
                return raw_detections

            for current_frame_idx in range(
                release_frame_idx,
                release_frame_idx + MAX_BALL_TRACK_FRAMES,
            ):
                ret, frame = cap.read()
                if not ret:
                    break

                candidates = self._infer_candidates(frame)
                candidates = self._filter_candidates_by_tracking_window(candidates)
                frame_detections: list[BallDetection] = []
                best_x: float | None = None
                best_y: float | None = None
                best_conf = 0.0

                for x, y, conf in candidates:
                    if best_x is None:
                        best_x = x
                        best_y = y
                        best_conf = conf

                    if conf < BALL_CONF_RAW_THRESHOLD:
                        continue

                    detection = BallDetection(
                        frame_idx=current_frame_idx,
                        timestamp_s=current_frame_idx / fps if fps > 0 else 0.0,
                        x=x,
                        y=y,
                        confidence=conf,
                    )
                    raw_detections.append(detection)
                    frame_detections.append(detection)

                if frame_detections:
                    self._last_frame_candidates.append(
                        FrameBallDetections(
                            frame_idx=current_frame_idx,
                            timestamp_s=(current_frame_idx / fps if fps > 0 else 0.0),
                            detections=list(frame_detections),
                        )
                    )
                    best_detection = max(
                        frame_detections,
                        key=lambda detection: detection.confidence,
                    )
                    self._accepted_track_history.append(best_detection)
                    if len(self._accepted_track_history) > 6:
                        self._accepted_track_history.pop(0)

                if best_x is None or best_y is None:
                    continue

                if batter_mode is BatterMode.PRESENT and batter_roi is not None:
                    if self._last_roi_entry_frame_idx is None and any(
                        self._ball_in_roi(detection.x, detection.y, batter_roi)
                        for detection in frame_detections
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
                        and best_conf < BALL_EARLY_STOP_CONF
                        and best_y < BALL_EARLY_STOP_Y
                    ):
                        termination_reason = "early_stop_low_confidence"
                        break
        finally:
            cap.release()

        rebuilt_path = self._rebuilder.rebuild(
            raw_detections,
            fps,
            grouped_candidates=self._last_frame_candidates,
            roi_entry_frame_idx=self._last_roi_entry_frame_idx,
        )
        final_path = rebuilt_path if rebuilt_path is not None else raw_detections

        logger.info(
            "Ball tracker raw_detections={} rebuilt_detections={} "
            "candidate_frames={} termination_reason={} roi_entry_frame={}",
            len(raw_detections),
            len(final_path),
            len(self._last_frame_candidates),
            termination_reason,
            self._last_roi_entry_frame_idx,
        )
        return final_path

    def _extract_candidates(self, scores: np.ndarray) -> list[tuple[float, float, float]]:
        working_scores = np.array(scores, copy=True)
        candidates: list[tuple[float, float, float]] = []
        scale_x = STANDARDIZED_WIDTH / scores.shape[1]
        scale_y = STANDARDIZED_HEIGHT / scores.shape[0]
        radius_x = max(1, int(round(BALL_PEAK_NMS_RADIUS / scale_x)))
        radius_y = max(1, int(round(BALL_PEAK_NMS_RADIUS / scale_y)))

        for _ in range(BALL_MAX_CANDIDATES_PER_FRAME):
            _, max_val, _, max_loc = cv2.minMaxLoc(working_scores)
            if max_val <= 0.0:
                break

            peak_x = max_loc[0] * scale_x
            peak_y = max_loc[1] * scale_y
            candidates.append((peak_x, peak_y, float(max_val)))

            x_min = max(0, max_loc[0] - radius_x)
            x_max = min(working_scores.shape[1], max_loc[0] + radius_x + 1)
            y_min = max(0, max_loc[1] - radius_y)
            y_max = min(working_scores.shape[0], max_loc[1] + radius_y + 1)
            working_scores[y_min:y_max, x_min:x_max] = 0.0

        return candidates

    def _ball_in_roi(
        self,
        x: float,
        y: float,
        roi: BatterROI,
    ) -> bool:
        return roi.x <= x <= roi.x + roi.width and roi.y <= y <= roi.y + roi.height
