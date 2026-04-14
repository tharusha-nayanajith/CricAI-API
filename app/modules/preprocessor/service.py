import asyncio
import os
import shutil
import subprocess
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from app.exceptions import PreprocessingError
from app.models.artifacts import VideoArtifacts
from app.models.calibration import CalibrationData
from app.modules.preprocessor.ball_tracker import BallTracker
from app.modules.preprocessor.bat_contact_detector import BatContactDetector
from app.modules.preprocessor.batter_detector import BatterDetector
from app.modules.preprocessor.constants import (
    BALL_DETECTION_MODEL_PATH,
    POSENET_MODEL_PATH,
    RELEASE_MODEL_PATH,
    STANDARDIZED_HEIGHT,
    STANDARDIZED_WIDTH,
)
from app.modules.preprocessor.models import BatterMode, DeliveryContext, ReleasePoint
from app.modules.preprocessor.release_detector import ReleaseDetector

_release_detector: ReleaseDetector | None = None
_batter_detector: BatterDetector | None = None
_ball_tracker: BallTracker | None = None
_bat_contact_detector: BatContactDetector | None = None


def get_release_detector() -> ReleaseDetector:
    global _release_detector
    if _release_detector is None:
        _release_detector = ReleaseDetector(RELEASE_MODEL_PATH, POSENET_MODEL_PATH)
        _release_detector.load_models()
    return _release_detector


def get_batter_detector() -> BatterDetector:
    global _batter_detector
    if _batter_detector is None:
        release_detector = get_release_detector()
        if release_detector.posenet_interpreter is None:
            raise PreprocessingError("Release detector PoseNet interpreter is not available.")
        _batter_detector = BatterDetector(release_detector.posenet_interpreter)
    return _batter_detector


def get_ball_tracker() -> BallTracker:
    global _ball_tracker
    if _ball_tracker is None:
        _ball_tracker = BallTracker(BALL_DETECTION_MODEL_PATH)
    return _ball_tracker


def get_bat_contact_detector() -> BatContactDetector:
    global _bat_contact_detector
    if _bat_contact_detector is None:
        _bat_contact_detector = BatContactDetector()
    return _bat_contact_detector


class VideoPreprocessor:
    def __init__(self) -> None:
        self._release_frame: np.ndarray | None = None
        self._release_point: ReleasePoint | None = None

    async def standardize_video(self, video_path: Path) -> Path:
        logger.info("Starting video standardization for {}", video_path)
        loop = asyncio.get_event_loop()
        width, height = await loop.run_in_executor(None, partial(self._read_dimensions, video_path))

        if width == STANDARDIZED_WIDTH and height == STANDARDIZED_HEIGHT:
            logger.info("Video already standardized: {}", video_path)
            return video_path

        ffmpeg_binary = self._resolve_ffmpeg_binary()
        standardized_path = video_path.with_name(
            f"{video_path.stem}_standardized{video_path.suffix}"
        )
        cmd = [
            ffmpeg_binary,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"scale={STANDARDIZED_WIDTH}:{STANDARDIZED_HEIGHT}",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            str(standardized_path),
        ]
        result = await loop.run_in_executor(None, partial(self._run_ffmpeg, cmd))
        if result.returncode != 0:
            raise PreprocessingError(
                f"ffmpeg failed while standardizing {video_path}: {result.stderr.strip()}"
            )

        logger.info("Completed video standardization: {}", standardized_path)
        return standardized_path

    async def run(
        self,
        video_path: Path,
        calibration: CalibrationData,
        require_ball_path: bool = True,
    ) -> VideoArtifacts:
        self._release_frame = None
        self._release_point = None
        std_path = await self.standardize_video(video_path)
        batter_detector = get_batter_detector()
        loop = asyncio.get_event_loop()
        batter_mode, batter_roi = await loop.run_in_executor(
            None,
            partial(batter_detector.detect, std_path, calibration),
        )

        ctx = DeliveryContext(
            standardized_video_path=std_path,
            batter_mode=batter_mode,
            batter_roi=batter_roi,
            fps=await loop.run_in_executor(None, partial(self._read_fps, std_path)),
        )
        ctx.release_point = await self._detect_release(ctx)
        if not require_ball_path:
            self._release_frame = ctx.release_point.annotated_frame
            self._release_point = ctx.release_point
            return VideoArtifacts(
                release_frame=ctx.release_point.annotated_frame,
                ball_path=[],
                bat_contact_frame=None,
                standardized_video_path=std_path,
                release_point=ctx.release_point,
                batter_mode=ctx.batter_mode,
                bat_contact=None,
                ball_candidates_by_frame=[],
            )

        ball_tracker = get_ball_tracker()
        raw_path = await loop.run_in_executor(
            None,
            partial(
                ball_tracker.track,
                std_path,
                ctx.release_point.frame_idx,
                ctx.fps,
                ctx.batter_mode,
                ctx.batter_roi,
            ),
        )
        ctx.batter_roi_entry_frame_idx = getattr(ball_tracker, "last_roi_entry_frame_idx", None)
        ball_candidates_by_frame = getattr(ball_tracker, "last_frame_candidates", [])
        if len(raw_path) < 3:
            raise PreprocessingError(f"Ball path too short: {len(raw_path)} detections")
        if ctx.batter_mode == BatterMode.PRESENT:
            if ctx.batter_roi is None:
                raise PreprocessingError("Batter ROI is required when batter mode is present")
            bat_contact_detector = get_bat_contact_detector()
            ctx.bat_contact = await loop.run_in_executor(
                None,
                partial(
                    bat_contact_detector.detect,
                    std_path,
                    ctx.fps,
                    ctx.batter_roi,
                    raw_path,
                ),
            )
        else:
            ctx.bat_contact = None
        self._release_frame = ctx.release_point.annotated_frame
        self._release_point = ctx.release_point

        return VideoArtifacts(
            release_frame=ctx.release_point.annotated_frame,
            ball_path=raw_path,
            bat_contact_frame=(
                ctx.bat_contact.annotated_frame if ctx.bat_contact is not None else None
            ),
            standardized_video_path=std_path,
            release_point=ctx.release_point,
            batter_mode=ctx.batter_mode,
            batter_roi_entry_frame_idx=ctx.batter_roi_entry_frame_idx,
            bat_contact=ctx.bat_contact,
            ball_candidates_by_frame=ball_candidates_by_frame,
        )

    async def _detect_release(self, ctx: DeliveryContext) -> ReleasePoint:
        release_detector = get_release_detector()
        loop = asyncio.get_event_loop()
        cap = cv2.VideoCapture(str(ctx.standardized_video_path))
        if not cap.isOpened():
            raise PreprocessingError(f"Unable to open video file: {ctx.standardized_video_path}")

        frame_idx = 0
        try:
            while True:
                ret, frame = await loop.run_in_executor(None, cap.read)
                if not ret:
                    break

                release_point = await loop.run_in_executor(
                    None,
                    partial(release_detector.process_frame, frame, frame_idx, ctx.fps),
                )
                if release_point is not None:
                    release_detector.reset()
                    return release_point
                frame_idx += 1
        finally:
            cap.release()

        raise PreprocessingError(
            f"No release frame detected in standardized video: {ctx.standardized_video_path}"
        )

    @staticmethod
    def _read_dimensions(video_path: Path) -> tuple[int, int]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise PreprocessingError(f"Unable to open video file: {video_path}")
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        finally:
            cap.release()

    @staticmethod
    def _read_fps(video_path: Path) -> float:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise PreprocessingError(f"Unable to open video file: {video_path}")
        try:
            return float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            cap.release()

    @staticmethod
    def _run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    @staticmethod
    def _resolve_ffmpeg_binary() -> str:
        env_path = os.getenv("FFMPEG_PATH")
        if env_path:
            env_binary = Path(env_path)
            if env_binary.exists():
                return str(env_binary)

        which_binary = shutil.which("ffmpeg")
        if which_binary:
            return which_binary

        repo_root = Path(__file__).resolve().parents[3]
        project_binary = repo_root / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
        if project_binary.exists():
            return str(project_binary)

        winget_binary = (
            Path.home()
            / "AppData"
            / "Local"
            / "Microsoft"
            / "WinGet"
            / "Packages"
            / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
            / "ffmpeg-8.1-full_build"
            / "bin"
            / "ffmpeg.exe"
        )
        if winget_binary.exists():
            return str(winget_binary)

        raise PreprocessingError(
            "ffmpeg binary was not found. Set FFMPEG_PATH or place ffmpeg "
            "at tools/ffmpeg/bin/ffmpeg.exe."
        )


class PreprocessorService(VideoPreprocessor):
    """Backward-compatible alias for preprocessor service wiring."""
