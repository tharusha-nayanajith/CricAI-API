from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from app.exceptions import PreprocessingError
from app.modules.preprocessor.constants import RELEASE_CONFIDENCE_THRESHOLD
from app.modules.preprocessor.models import ReleasePoint

RELEASE_INPUT_SIZE = 512
RELEASE_SCALE = 1.0 / 127.5
RELEASE_BIAS = -1.0

POSENET_SCALE = 1.0 / 127.5
POSENET_BIAS = -1.0
POSENET_STRIDE = 16

KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10


@dataclass(slots=True)
class _Keypoint:
    x: float
    y: float
    confidence: float

    def valid(self, min_conf: float = 0.3) -> bool:
        return self.confidence >= min_conf

    def xy(self) -> tuple[int, int]:
        return int(self.x), int(self.y)


class _ReleaseClassifier:
    def __init__(self, session: Any, threshold: float) -> None:
        self._session = session
        self._threshold = threshold
        self._input_names = [i.name for i in session.get_inputs()]
        self._output_name = session.get_outputs()[0].name
        self._buffer: list[np.ndarray] = []
        self.release_frame: int | None = None
        self._triggered = False

    def push(self, frame_bgr: np.ndarray, frame_idx: int) -> float | None:
        self._buffer.append(self._preprocess(frame_bgr))
        if len(self._buffer) > 3:
            self._buffer.pop(0)
        if len(self._buffer) < min(3, len(self._input_names)):
            return None

        input_feed: dict[str, np.ndarray] = {}
        for idx, name in enumerate(self._input_names):
            input_feed[name] = self._buffer[idx]

        out = self._session.run([self._output_name], input_feed)
        prob = float(np.asarray(out[0]).reshape(-1)[0])
        if not self._triggered and prob >= self._threshold:
            self._triggered = True
            self.release_frame = frame_idx
        return prob

    def reset(self) -> None:
        self._buffer = []
        self.release_frame = None
        self._triggered = False

    @staticmethod
    def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (RELEASE_INPUT_SIZE, RELEASE_INPUT_SIZE))
        norm = resized.astype(np.float32) * RELEASE_SCALE + RELEASE_BIAS
        return np.transpose(norm, (2, 0, 1))[np.newaxis]


class _PoseNetDecoder:
    def __init__(self, interpreter: Any) -> None:
        self._interpreter = interpreter
        self._interpreter.allocate_tensors()
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()
        if not input_details or len(output_details) < 2:
            raise PreprocessingError("Pose model has invalid IO metadata.")

        self._input_idx = int(input_details[0]["index"])
        self._input_dtype = input_details[0]["dtype"]
        _, self._input_h, self._input_w, _ = input_details[0]["shape"]
        self._out_heatmap_idx = int(output_details[0]["index"])
        self._out_offsets_idx = int(output_details[1]["index"])

    def get_bowling_arm(
        self, frame_bgr: np.ndarray
    ) -> tuple[str, _Keypoint, _Keypoint, _Keypoint]:
        h, w = frame_bgr.shape[:2]
        heatmaps, offsets = self._infer(frame_bgr)

        left_wrist = self._decode_kp(heatmaps, offsets, KP_LEFT_WRIST, w, h)
        right_wrist = self._decode_kp(heatmaps, offsets, KP_RIGHT_WRIST, w, h)
        if right_wrist.confidence >= left_wrist.confidence:
            arm = "right"
            wrist = right_wrist
            elbow = self._decode_kp(heatmaps, offsets, KP_RIGHT_ELBOW, w, h)
            shoulder = self._decode_kp(heatmaps, offsets, KP_RIGHT_SHOULDER, w, h)
        else:
            arm = "left"
            wrist = left_wrist
            elbow = self._decode_kp(heatmaps, offsets, KP_LEFT_ELBOW, w, h)
            shoulder = self._decode_kp(heatmaps, offsets, KP_LEFT_SHOULDER, w, h)
        return arm, wrist, elbow, shoulder

    def _infer(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._input_w, self._input_h))
        if self._input_dtype == np.uint8:
            inp = resized[np.newaxis].astype(np.uint8)
        else:
            inp = (resized.astype(np.float32) * POSENET_SCALE + POSENET_BIAS)[np.newaxis]

        self._interpreter.set_tensor(self._input_idx, inp)
        self._interpreter.invoke()
        heatmaps = np.squeeze(self._interpreter.get_tensor(self._out_heatmap_idx))
        offsets = np.squeeze(self._interpreter.get_tensor(self._out_offsets_idx))
        return heatmaps, offsets

    def _decode_kp(
        self,
        heatmaps: np.ndarray,
        offsets: np.ndarray,
        kp_idx: int,
        orig_w: int,
        orig_h: int,
    ) -> _Keypoint:
        grid = int(heatmaps.shape[0])
        nkp = int(heatmaps.shape[2])
        hm = heatmaps[:, :, kp_idx]
        flat = int(np.argmax(hm))
        row, col = divmod(flat, grid)
        confidence = float(self._sigmoid(float(hm[row, col])))

        oy = float(offsets[row, col, kp_idx])
        ox = float(offsets[row, col, kp_idx + nkp])
        y_in = float(np.clip(row * POSENET_STRIDE + oy, 0, self._input_h - 1))
        x_in = float(np.clip(col * POSENET_STRIDE + ox, 0, self._input_w - 1))
        return _Keypoint(
            x=x_in * orig_w / self._input_w,
            y=y_in * orig_h / self._input_h,
            confidence=confidence,
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        return float(1.0 / (1.0 + np.exp(-value)))


class ReleaseDetector:
    def __init__(self, release_model_path: Path, posenet_model_path: Path):
        self.release_model_path = release_model_path
        self.posenet_model_path = posenet_model_path
        self.release_session: Any | None = None
        self.posenet_interpreter: Any | None = None
        self.classifier: _ReleaseClassifier | None = None
        self.pose_decoder: _PoseNetDecoder | None = None

    def load_models(self) -> None:
        logger.info("Loading release detector models")
        self.release_session = self._create_onnx_session(self.release_model_path)
        self.posenet_interpreter = self._create_tflite_interpreter(self.posenet_model_path)
        self.classifier = _ReleaseClassifier(
            self.release_session,
            threshold=RELEASE_CONFIDENCE_THRESHOLD,
        )
        self.pose_decoder = _PoseNetDecoder(self.posenet_interpreter)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        fps: float,
    ) -> ReleasePoint | None:
        self._ensure_loaded()
        assert self.classifier is not None
        assert self.pose_decoder is not None

        prob = self.classifier.push(frame_bgr, frame_idx)
        if prob is None or self.classifier.release_frame != frame_idx:
            return None

        _, wrist, elbow, shoulder = self.pose_decoder.get_bowling_arm(frame_bgr)
        hand_position = (wrist.x, wrist.y)
        annotated = self._draw_release(frame_bgr, wrist, elbow, shoulder, prob, frame_idx)
        timestamp = frame_idx / fps if fps > 0 else 0.0
        return ReleasePoint(
            frame_idx=frame_idx,
            timestamp_s=timestamp,
            hand_position=hand_position,
            confidence=float(prob),
            annotated_frame=annotated,
        )

    def reset(self) -> None:
        if self.classifier is not None:
            self.classifier.reset()

    def _ensure_loaded(self) -> None:
        if self.classifier is None or self.pose_decoder is None:
            raise PreprocessingError("ReleaseDetector models are not loaded.")

    @staticmethod
    def _create_onnx_session(model_path: Path) -> Any:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise PreprocessingError("onnxruntime is required for release detection.") from exc

        return ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    @staticmethod
    def _create_tflite_interpreter(model_path: Path) -> Any:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise PreprocessingError(
                    "A TFLite interpreter is required for pose detection."
                ) from exc
            Interpreter = tf.lite.Interpreter
        return Interpreter(model_path=str(model_path))

    @staticmethod
    def _draw_release(
        frame_bgr: np.ndarray,
        wrist: _Keypoint,
        elbow: _Keypoint,
        shoulder: _Keypoint,
        prob: float,
        frame_idx: int,
    ) -> np.ndarray:
        out = frame_bgr.copy()
        if shoulder.valid() and elbow.valid():
            cv2.line(out, shoulder.xy(), elbow.xy(), (255, 200, 0), 2)
        if elbow.valid() and wrist.valid():
            cv2.line(out, elbow.xy(), wrist.xy(), (0, 200, 255), 3)
        if shoulder.valid():
            cv2.circle(out, shoulder.xy(), 6, (255, 200, 0), -1)
        if elbow.valid():
            cv2.circle(out, elbow.xy(), 7, (0, 200, 255), -1)
        if wrist.valid():
            wrist_xy = wrist.xy()
            cv2.circle(out, wrist_xy, 10, (0, 0, 255), -1)
            cv2.circle(out, wrist_xy, 20, (0, 0, 255), 2)
        cv2.putText(
            out,
            f"frame={frame_idx} prob={prob:.3f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return out
