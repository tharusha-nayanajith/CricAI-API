from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from scipy.signal import butter, filtfilt, find_peaks

from app.exceptions import PreprocessingError
from app.modules.preprocessor.constants import (
    AUDIO_HIGHPASS_CUTOFF_HZ,
    AUDIO_MIN_PEAK_DISTANCE_S,
    AUDIO_MIN_PROMINENCE,
    AUDIO_SMOOTHING_WINDOW,
    AUDIO_TARGET_SR,
    BAT_CONTACT_AUDIO_WINDOW_FRAMES,
)
from app.modules.preprocessor.models import (
    BallDetection,
    BatContactResult,
    BatterROI,
    ContactMethod,
)

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip


class BatContactDetector:
    def detect(
        self,
        video_path: Path,
        fps: float,
        batter_roi: BatterROI,
        ball_path: list[BallDetection],
    ) -> BatContactResult | None:
        _ = batter_roi

        audio_result = self._detect_impact_frame(video_path, fps)
        if audio_result is None:
            logger.warning("Audio impact detection failed for {}", video_path)
            return None

        contact_frame_idx, method, detection_score = self._refine_impact_frame(
            ball_path,
            audio_result["impact_frame"],
        )

        raw_frame = self._read_frame(video_path, contact_frame_idx)
        logger.info(
            "Bat contact detected at frame {} via {}",
            contact_frame_idx,
            method.value,
        )
        return BatContactResult(
            contact_frame_idx=contact_frame_idx,
            timestamp_s=contact_frame_idx / fps,
            annotated_frame=raw_frame,
            detection_score=detection_score,
            method=method,
        )

    def _detect_impact_frame(
        self,
        video_path: Path,
        fps: float,
    ) -> dict[str, int | float] | None:
        try:
            audio, sr = self._extract_audio_array(video_path)
            filtered_audio = self._high_pass_filter(audio, sr)
            envelope = self._smooth_signal(filtered_audio)
            _peak_idx, impact_time = self._find_impact_peak(envelope, sr)
        except Exception as exc:
            logger.warning("Audio impact detection failed for {}: {}", video_path, exc)
            return None

        effective_fps = fps if fps > 0 else self._read_video_fps(video_path)
        if effective_fps <= 0:
            effective_fps = 30.0

        impact_frame = int(impact_time * effective_fps)
        logger.info(
            "Audio impact estimate for {}: frame={} time={:.4f}s",
            video_path,
            impact_frame,
            impact_time,
        )
        return {
            "impact_frame": impact_frame,
            "impact_time": round(impact_time, 4),
        }

    def _extract_audio_array(
        self,
        video_path: Path,
        target_sr: int = AUDIO_TARGET_SR,
    ) -> tuple[np.ndarray, int]:
        clip = VideoFileClip(str(video_path))
        try:
            if clip.audio is None:
                raise ValueError(f"Video '{video_path}' has no audio track.")

            audio_array = clip.audio.to_soundarray(fps=target_sr)
        finally:
            clip.close()

        if audio_array.ndim == 2:
            audio_array = audio_array.mean(axis=1)

        return audio_array.astype(np.float32), target_sr

    def _high_pass_filter(
        self,
        signal: np.ndarray,
        sr: int,
        cutoff_hz: float = AUDIO_HIGHPASS_CUTOFF_HZ,
    ) -> np.ndarray:
        nyquist = sr / 2.0
        normalized_cutoff = cutoff_hz / nyquist
        b, a = butter(5, normalized_cutoff, btype="high", analog=False)
        return filtfilt(b, a, signal)

    def _smooth_signal(
        self,
        signal: np.ndarray,
        window_size: int = AUDIO_SMOOTHING_WINDOW,
    ) -> np.ndarray:
        envelope = np.abs(signal)
        window = min(window_size, max(1, envelope.shape[0]))
        kernel = np.ones(window, dtype=np.float32) / window
        return np.convolve(envelope, kernel, mode="same")

    def _find_impact_peak(
        self,
        envelope: np.ndarray,
        sr: int,
    ) -> tuple[int, float]:
        min_prominence = AUDIO_MIN_PROMINENCE * float(envelope.max())
        min_distance = int(AUDIO_MIN_PEAK_DISTANCE_S * sr)
        peaks, _properties = find_peaks(
            envelope,
            prominence=min_prominence,
            distance=min_distance,
        )

        if len(peaks) == 0:
            peak_idx = int(np.argmax(envelope))
        else:
            peak_idx = int(peaks[np.argmax(envelope[peaks])])

        return peak_idx, peak_idx / sr

    def _refine_impact_frame(
        self,
        ball_path: list[BallDetection],
        audio_impact_frame: int,
    ) -> tuple[int, ContactMethod, float | None]:
        window_start = audio_impact_frame - BAT_CONTACT_AUDIO_WINDOW_FRAMES
        window_end = audio_impact_frame + BAT_CONTACT_AUDIO_WINDOW_FRAMES
        candidate_positions = [
            detection
            for detection in ball_path
            if window_start <= detection.frame_idx <= window_end
        ]

        if len(candidate_positions) < 2:
            logger.info(
                "Bat contact falling back to audio estimate: insufficient ball detections ({})",
                len(candidate_positions),
            )
            return audio_impact_frame, ContactMethod.AUDIO_FALLBACK, None

        velocities = self._compute_velocity(candidate_positions)
        best_frame, max_delta = self._find_max_velocity_change(velocities)
        if best_frame is None:
            logger.info("Bat contact falling back to audio estimate: no velocity delta found")
            return audio_impact_frame, ContactMethod.AUDIO_FALLBACK, None

        return best_frame, ContactMethod.BALL_VELOCITY, max_delta

    def _compute_velocity(
        self,
        positions: list[BallDetection],
    ) -> list[tuple[int, float]]:
        velocities: list[tuple[int, float]] = []
        for idx in range(1, len(positions)):
            previous = positions[idx - 1]
            current = positions[idx]
            frame_gap = max(1, current.frame_idx - previous.frame_idx)
            distance = float(
                np.sqrt((current.x - previous.x) ** 2 + (current.y - previous.y) ** 2)
            )
            velocities.append((current.frame_idx, distance / frame_gap))
        return velocities

    def _find_max_velocity_change(
        self,
        velocities: list[tuple[int, float]],
    ) -> tuple[int | None, float | None]:
        if len(velocities) < 2:
            return None, None

        max_delta = -1.0
        best_frame: int | None = None
        for idx in range(1, len(velocities)):
            frame_idx, current_velocity = velocities[idx]
            _previous_frame_idx, previous_velocity = velocities[idx - 1]
            delta = abs(current_velocity - previous_velocity)
            if delta > max_delta:
                max_delta = delta
                best_frame = frame_idx

        if best_frame is None:
            return None, None
        return best_frame, max_delta

    def _read_frame(self, video_path: Path, frame_idx: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise PreprocessingError(f"Unable to open video file: {video_path}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                raise PreprocessingError(
                    f"Unable to read frame {frame_idx} from video file: {video_path}"
                )
            return frame
        finally:
            cap.release()

    def _read_video_fps(self, video_path: Path) -> float:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise PreprocessingError(f"Unable to open video file: {video_path}")
        try:
            return float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            cap.release()
