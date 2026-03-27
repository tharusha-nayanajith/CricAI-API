from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from app.exceptions import PreprocessingError
from app.models.calibration import CalibrationData
from app.modules.preprocessor.constants import STANDARDIZED_HEIGHT, STANDARDIZED_WIDTH
from app.modules.preprocessor.models import BatterMode, BatterROI


class BatterDetector:
    KEYPOINT_SCORE_THRESHOLD = 0.3
    SAMPLE_COUNT = 5
    SAMPLE_WINDOW_START = 0.15
    SAMPLE_WINDOW_END = 0.50
    VOTE_THRESHOLD = 3

    def __init__(self, posenet_interpreter) -> None:
        self._interpreter = posenet_interpreter

    def derive_roi(self, calibration: CalibrationData) -> BatterROI:
        stump_kps = [kp for kp in calibration.keypoints if kp.channel_index in {0, 1, 2, 3, 4, 5}]
        if len(stump_kps) < 2:
            raise PreprocessingError(
                "At least 2 stump keypoints from channels 0-5 are required to derive batter ROI."
            )

        stump_min_x = min(kp.x for kp in stump_kps)
        stump_max_x = max(kp.x for kp in stump_kps)
        stump_min_y = min(kp.y for kp in stump_kps)
        stump_max_y = max(kp.y for kp in stump_kps)

        stump_width = stump_max_x - stump_min_x
        stump_height = stump_max_y - stump_min_y
        centroid_x = sum(kp.x for kp in stump_kps) / len(stump_kps)
        centroid_y = sum(kp.y for kp in stump_kps) / len(stump_kps)

        box_w = int(max(stump_width * 3, 80))
        box_h = int(max(stump_height * 3, 120))

        x = max(0, int(centroid_x - box_w / 2))
        y = max(0, int(centroid_y - box_h / 2))
        width = max(0, min(box_w, STANDARDIZED_WIDTH - x))
        height = max(0, min(box_h, STANDARDIZED_HEIGHT - y))

        roi = BatterROI(x=x, y=y, width=width, height=height)
        logger.info(
            "Derived batter ROI x={} y={} width={} height={}",
            roi.x,
            roi.y,
            roi.width,
            roi.height,
        )
        return roi

    def detect(
        self,
        video_path: Path,
        calibration: CalibrationData,
    ) -> tuple[BatterMode, BatterROI | None]:
        roi = self.derive_roi(calibration)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise PreprocessingError(f"Unable to open video file: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_indices = self._sample_frame_indices(total_frames)
            votes = 0
            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                if self._person_in_roi(frame, roi):
                    votes += 1
        finally:
            cap.release()

        logger.info("Batter detector votes: {}/{}", votes, self.SAMPLE_COUNT)
        if votes >= self.VOTE_THRESHOLD:
            logger.info("Batter mode: {}", BatterMode.PRESENT.value)
            return BatterMode.PRESENT, roi

        logger.info("Batter mode: {}", BatterMode.NONE.value)
        return BatterMode.NONE, None

    def _sample_frame_indices(self, total_frames: int) -> list[int]:
        if total_frames <= 0:
            return [0] * self.SAMPLE_COUNT

        start = int(total_frames * self.SAMPLE_WINDOW_START)
        end = int(total_frames * self.SAMPLE_WINDOW_END)
        if end < start:
            end = start

        if self.SAMPLE_COUNT == 1:
            return [start]

        return [
            int(round(start + (end - start) * idx / (self.SAMPLE_COUNT - 1)))
            for idx in range(self.SAMPLE_COUNT)
        ]

    def _person_in_roi(
        self,
        frame_bgr: np.ndarray,
        roi: BatterROI,
    ) -> bool:
        if roi.width <= 0 or roi.height <= 0:
            return False

        crop = frame_bgr[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()
        if not input_details or not output_details:
            raise PreprocessingError("Pose interpreter metadata is unavailable.")

        _, input_h, input_w, _ = input_details[0]["shape"]
        input_dtype = input_details[0]["dtype"]
        resized = cv2.resize(crop, (input_w, input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        if input_dtype == np.uint8:
            model_input = rgb[np.newaxis].astype(np.uint8)
        else:
            model_input = (rgb.astype(np.float32) / 127.5 - 1.0)[np.newaxis]

        self._interpreter.set_tensor(int(input_details[0]["index"]), model_input)
        self._interpreter.invoke()
        output = np.asarray(self._interpreter.get_tensor(int(output_details[0]["index"])))
        return self._has_person(output)

    def _has_person(self, output: np.ndarray) -> bool:
        if output.ndim == 4 and output.shape[1] == 1 and output.shape[-1] == 3:
            scores = output[0][0][:, 2]
            return bool(np.any(scores > self.KEYPOINT_SCORE_THRESHOLD))

        if output.ndim == 4 and output.shape[-1] >= 17:
            heatmap_scores = 1.0 / (1.0 + np.exp(-output[0]))
            max_scores = np.max(heatmap_scores[..., :17], axis=(0, 1))
            return bool(np.any(max_scores > self.KEYPOINT_SCORE_THRESHOLD))

        return False
