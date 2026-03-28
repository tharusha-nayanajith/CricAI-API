from __future__ import annotations

import asyncio
from functools import partial

from loguru import logger

from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts
from app.models.calibration import CalibrationData
from app.modules.bowler_performance.camera import (
    build_extrinsic_matrix,
    build_intrinsic_matrix,
    pixels_to_world_points,
)
from app.modules.bowler_performance.metrics import build_result
from app.modules.bowler_performance.models import BowlerPerformanceResult
from app.modules.bowler_performance.pitch_coordinates import (
    build_pitch_frame,
    world_points_to_pitch_points,
)
from app.modules.bowler_performance.ransac import BallPathCleaner


class BowlerPerformanceAnalyzer:
    def __init__(self) -> None:
        self._cleaner = BallPathCleaner()

    async def run(
        self,
        artifacts: VideoArtifacts,
        calibration: CalibrationData,
        fps: float,
    ) -> BowlerPerformanceResult:
        logger.info(
            "Starting bowler_performance analysis detections={} release_shape={} "
            "bat_contact_available={}",
            len(artifacts.ball_path),
            artifacts.release_frame.shape,
            artifacts.bat_contact_frame is not None,
        )
        loop = asyncio.get_running_loop()

        try:
            logger.info("Cleaning ball path with RANSAC")
            ransac_result = await loop.run_in_executor(
                None,
                partial(self._cleaner.clean, artifacts.ball_path, fps),
            )
            if ransac_result is None:
                raise FeatureError("RANSAC found too few inliers in ball path")

            logger.info("Building camera matrices from calibration")
            intrinsic = await loop.run_in_executor(
                None,
                partial(build_intrinsic_matrix, calibration),
            )
            extrinsic = await loop.run_in_executor(
                None,
                partial(build_extrinsic_matrix, calibration),
            )

            logger.info(
                "Selected trajectory detections={} RANSAC inliers={}",
                len(ransac_result.selected_track),
                len(ransac_result.inliers),
            )
            logger.info(
                "Unprojecting {} inlier detections to world space",
                len(ransac_result.inliers),
            )
            world_points = await loop.run_in_executor(
                None,
                partial(pixels_to_world_points, ransac_result.inliers, intrinsic, extrinsic),
            )
            if len(world_points) < 3:
                raise FeatureError("Too few valid 3D world points after unprojection")

            logger.info("Converting world points to standardized pitch coordinates")
            pitch_frame = await loop.run_in_executor(
                None,
                partial(build_pitch_frame, calibration, intrinsic, extrinsic),
            )
            if pitch_frame is None:
                raise FeatureError("Unable to derive pitch coordinates from calibration")

            pitch_points = await loop.run_in_executor(
                None,
                partial(world_points_to_pitch_points, world_points, pitch_frame),
            )

            logger.info("Computing bowler performance metrics")
            result = await loop.run_in_executor(
                None,
                partial(
                    build_result,
                    world_points,
                    pitch_points,
                    ransac_result.inliers,
                    ransac_result.bounce_frame,
                ),
            )
        except FeatureError:
            raise
        except Exception as exc:
            raise FeatureError("Bowler performance analysis failed unexpectedly") from exc

        logger.info(
            "Completed bowler_performance analysis speed_kmh={:.2f} swing_metres={:.3f} "
            "inlier_count={}",
            result.speed_kmh,
            result.swing_metres,
            result.inlier_count,
        )
        return result
