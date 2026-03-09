"""
impact_audio_detector.py
------------------------
Detects the bat-ball impact frame in a cricket video using audio spike analysis.

Pipeline:
1. Extract audio from the video (moviepy).
2. Apply a high-pass filter to remove background noise (scipy).
3. Smooth the signal with a rolling window.
4. Detect the strongest amplitude peak (scipy.signal.find_peaks).
5. Convert spike timestamp → video frame index using FPS.

Returns:
    dict: {"impact_frame": int, "impact_time": float}
"""

import numpy as np
import librosa
import cv2
from scipy.signal import butter, filtfilt, find_peaks
try:
    from moviepy import VideoFileClip          # moviepy v2.x
except ImportError:
    from moviepy.editor import VideoFileClip  # moviepy v1.x fallback


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_audio_array(video_path: str, target_sr: int = 22050):
    """
    Extract mono audio from the video file as a NumPy array.

    Args:
        video_path: Absolute or relative path to the video file.
        target_sr:  Target sample rate for resampling.

    Returns:
        Tuple[np.ndarray, int]: (audio_samples, sample_rate)

    Raises:
        ValueError: If the video has no audio track.
    """
    clip = VideoFileClip(video_path)

    if clip.audio is None:
        clip.close()
        raise ValueError(f"Video '{video_path}' has no audio track.")

    # Write a temporary WAV to a BytesIO-like pipeline via librosa
    # moviepy gives us an audio array at the clip's fps
    audio_fps = target_sr
    audio_array = clip.audio.to_soundarray(fps=audio_fps)
    clip.close()

    # Convert stereo → mono
    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=1)

    audio_array = audio_array.astype(np.float32)
    return audio_array, audio_fps


def _high_pass_filter(signal: np.ndarray, sr: int, cutoff_hz: float = 1000.0) -> np.ndarray:
    """
    Apply a Butterworth high-pass filter to remove low-frequency background noise.

    A bat-ball impact produces a sharp transient (predominantly > 1 kHz),
    while crowd noise and ambient sounds are concentrated at lower frequencies.

    Args:
        signal:     1-D audio signal.
        sr:         Sample rate in Hz.
        cutoff_hz:  High-pass cutoff frequency in Hz.

    Returns:
        Filtered signal as a NumPy array.
    """
    nyquist = sr / 2.0
    normalized_cutoff = cutoff_hz / nyquist
    # 5th-order Butterworth for a steep roll-off
    b, a = butter(5, normalized_cutoff, btype="high", analog=False)
    return filtfilt(b, a, signal)


def _smooth_signal(signal: np.ndarray, window_size: int = 512) -> np.ndarray:
    """
    Smooth the absolute amplitude envelope with a uniform moving average.

    Args:
        signal:      1-D audio signal (ideally already filtered).
        window_size: Number of samples in the moving-average window.

    Returns:
        Smoothed amplitude envelope.
    """
    envelope = np.abs(signal)
    kernel = np.ones(window_size) / window_size
    return np.convolve(envelope, kernel, mode="same")


def _find_impact_peak(envelope: np.ndarray, sr: int) -> tuple:
    """
    Locate the single strongest amplitude peak in the envelope.

    Strategy:
      - Run scipy find_peaks with a minimum prominence threshold so that
        minor fluctuations are ignored.
      - Among all detected peaks, return the one with the highest amplitude.
      - Fall back to argmax if no peaks are found.

    Args:
        envelope: Smoothed amplitude envelope.
        sr:       Sample rate.

    Returns:
        Tuple[int, float]: (peak_sample_index, peak_time_seconds)
    """
    # Minimum prominence = 20% of the global max to filter trivial bumps
    min_prominence = 0.20 * envelope.max()
    # Minimum distance between peaks: 0.1 seconds
    min_distance = int(0.1 * sr)

    peaks, properties = find_peaks(
        envelope,
        prominence=min_prominence,
        distance=min_distance,
    )

    if len(peaks) == 0:
        # Fallback: just use the highest point
        peak_idx = int(np.argmax(envelope))
    else:
        # Pick the peak with the largest amplitude value
        peak_amplitudes = envelope[peaks]
        peak_idx = int(peaks[np.argmax(peak_amplitudes)])

    peak_time = peak_idx / sr
    return peak_idx, peak_time


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_impact_frame(video_path: str, smoothing_window: int = 512) -> dict:
    """
    Detect the cricket bat-ball impact frame from a video using audio analysis.

    Args:
        video_path:      Path to the cricket shot video file.
        smoothing_window: Moving-average window size (samples) for envelope smoothing.

    Returns:
        dict with keys:
            "impact_frame" (int)  – estimated frame index (0-based).
            "impact_time"  (float) – time in seconds of the audio spike.

    Raises:
        ValueError: If the video has no audio track.
        FileNotFoundError: If the video file does not exist.
    """
    # 1. Extract audio
    audio, sr = _extract_audio_array(video_path)

    # 2. High-pass filter: keep transient (impact) frequencies, drop rumble/crowd
    filtered_audio = _high_pass_filter(audio, sr, cutoff_hz=1000.0)

    # 3. Smooth the amplitude envelope
    envelope = _smooth_signal(filtered_audio, window_size=smoothing_window)

    # 4. Detect the strongest peak
    _, impact_time = _find_impact_peak(envelope, sr)

    # 5. Convert spike time → video frame index
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        fps = 30.0  # sensible default

    impact_frame = int(impact_time * fps)

    return {
        "impact_frame": impact_frame,
        "impact_time": round(impact_time, 4),
    }
