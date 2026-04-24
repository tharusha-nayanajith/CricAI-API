from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from loguru import logger

from app.exceptions import FeatureError
from app.models.artifacts import VideoArtifacts
from app.models.calibration import CalibrationData
from app.models.job import FeatureResult
from app.modules.shot_classifier.models import ShotClassifierResult
from app.modules.bowler_performance.service import BowlerPerformanceAnalyzer
from app.modules.preprocessor.models import BallDetection
from app.modules.preprocessor.service import VideoPreprocessor
from app.storage.results import set_feature_status, store_result

_preprocessor = VideoPreprocessor()
_bowler_analyzer = BowlerPerformanceAnalyzer()


class _LazyServiceProxy:
    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._instance: Any | None = None

    def _get_instance(self) -> Any:
        if self._instance is None:
            self._instance = self._factory()
        return self._instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_instance(), name)


def _build_action_legality_service() -> Any:
    from app.modules.action_legality.service import ActionLegalityService

    return ActionLegalityService()


def _build_shot_classifier_service() -> Any:
    from app.modules.shot_classifier.service import ShotClassifierService

    return ShotClassifierService()


def _build_shot_similarity_service() -> Any:
    from app.modules.shot_similarity.service import ShotSimilarityService

    return ShotSimilarityService()


_action_legality_service = _LazyServiceProxy(_build_action_legality_service)
_shot_classifier_service = _LazyServiceProxy(_build_shot_classifier_service)
_shot_similarity_service = _LazyServiceProxy(_build_shot_similarity_service)


def _cleanup_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Failed to delete temporary upload {}", path)


def _derive_fps(ball_path: list[BallDetection]) -> float:
    for previous, current in zip(ball_path, ball_path[1:], strict=False):
        dt = current.timestamp_s - previous.timestamp_s
        frame_gap = current.frame_idx - previous.frame_idx
        if dt > 0.0 and frame_gap > 0:
            return frame_gap / dt
    return 30.0


async def run_bowler_performance(
    job_id: str,
    artifacts: VideoArtifacts,
    calibration: CalibrationData,
    fps: float,
    video_url: str,
) -> None:
    await set_feature_status(job_id, "bowler_performance", "processing")
    try:
        result = await _bowler_analyzer.run(artifacts, calibration, fps, video_url=video_url)
        await store_result(
            job_id,
            "bowler_performance",
            FeatureResult(status="done", result=result.model_dump(by_alias=True)),
        )
    except FeatureError as exc:
        logger.error("bowler_performance failed for {}: {}", job_id, exc)
        await store_result(
            job_id,
            "bowler_performance",
            FeatureResult(status="failed", error=str(exc)),
        )
    except Exception as exc:
        logger.exception("Unexpected bowler_performance failure for {}", job_id)
        await store_result(
            job_id,
            "bowler_performance",
            FeatureResult(status="failed", error=str(exc)),
        )


async def run_action_legality(
    job_id: str,
    artifacts: VideoArtifacts,
    video_url: str,
) -> None:
    await set_feature_status(job_id, "action_legality", "processing")
    try:
        result = await _action_legality_service.run(artifacts, video_url=video_url, job_id=job_id)
        await store_result(
            job_id,
            "action_legality",
            FeatureResult(status="done", result=result.model_dump()),
        )
    except FeatureError as exc:
        logger.error("action_legality failed for {}: {}", job_id, exc)
        await store_result(
            job_id,
            "action_legality",
            FeatureResult(status="failed", error=str(exc)),
        )
    except Exception as exc:
        logger.exception("Unexpected action_legality failure for {}", job_id)
        await store_result(
            job_id,
            "action_legality",
            FeatureResult(status="failed", error=str(exc)),
        )


async def run_shot_similarity(
    job_id: str,
    artifacts: VideoArtifacts,
    video_url: str,
    classified_shot_type: str | None,
) -> None:
    await set_feature_status(job_id, "shot_similarity", "processing")
    try:
        result = await _shot_similarity_service.run(
            artifacts,
            video_url=video_url,
            classified_shot_type=classified_shot_type,
        )
        await store_result(
            job_id,
            "shot_similarity",
            FeatureResult(status="done", result=result.model_dump()),
        )
    except FeatureError as exc:
        logger.error("shot_similarity failed for {}: {}", job_id, exc)
        await store_result(
            job_id,
            "shot_similarity",
            FeatureResult(status="failed", error=str(exc)),
        )
    except Exception as exc:
        logger.exception("Unexpected shot_similarity failure for {}", job_id)
        await store_result(
            job_id,
            "shot_similarity",
            FeatureResult(status="failed", error=str(exc)),
        )


async def _compute_shot_classifier_result(
    artifacts: VideoArtifacts,
    video_path: Path,
    video_url: str,
) -> ShotClassifierResult:
    return await _shot_classifier_service.run(
        artifacts,
        video_path=video_path,
        video_url=video_url,
    )


async def run_shot_classifier(
    job_id: str,
    artifacts: VideoArtifacts,
    video_path: Path,
    video_url: str,
) -> None:
    await set_feature_status(job_id, "shot_classifier", "processing")
    try:
        result = await _shot_classifier_service.run(
            artifacts,
            video_path=video_path,
            video_url=video_url,
        )
        await store_result(
            job_id,
            "shot_classifier",
            FeatureResult(status="done", result=result.model_dump()),
        )
    except FeatureError as exc:
        logger.error("shot_classifier failed for {}: {}", job_id, exc)
        await store_result(
            job_id,
            "shot_classifier",
            FeatureResult(status="failed", error=str(exc)),
        )
    except Exception as exc:
        logger.exception("Unexpected shot_classifier failure for {}", job_id)
        await store_result(
            job_id,
            "shot_classifier",
            FeatureResult(status="failed", error=str(exc)),
        )


async def process_job(
    job_id: str,
    selected_features: list[str],
    source_video_path: str,
    filename: str,
    calibration_payload: dict[str, Any],
) -> None:
    logger.info("Queued job {} with features {}", job_id, selected_features)
    implemented_features = [
        feature_name
        for feature_name in selected_features
        if feature_name
        in {"bowler_performance", "action_legality", "shot_classifier", "shot_similarity"}
    ]
    if not implemented_features:
        logger.info("No implemented background features selected for {}", job_id)
        return

    calibration = CalibrationData.model_validate(calibration_payload)
    source_path = Path(source_video_path)
    job_dir = Path(tempfile.mkdtemp(prefix=f"crickai_{job_id}_"))
    safe_name = Path(filename or source_path.name or "upload.mp4").name or "upload.mp4"
    video_path = job_dir / safe_name
    loop = asyncio.get_running_loop()

    try:
        await loop.run_in_executor(None, partial(shutil.copy2, source_path, video_path))
        artifacts = await _preprocessor.run(
            video_path,
            calibration,
            require_ball_path=bool(
                {"bowler_performance", "shot_classifier", "shot_similarity"}
                & set(implemented_features)
            ),
        )
    except Exception as exc:
        logger.error("Preprocessor failed for {}: {}", job_id, exc)
        for feature_name in implemented_features:
            await store_result(
                job_id,
                feature_name,
                FeatureResult(status="failed", error=f"Preprocessor failed: {exc}"),
            )
    else:
        if "bowler_performance" in implemented_features:
            fps = _derive_fps(artifacts.ball_path)
            await run_bowler_performance(job_id, artifacts, calibration, fps, safe_name)
        if "action_legality" in implemented_features:
            await run_action_legality(job_id, artifacts, safe_name)
        classifier_result: ShotClassifierResult | None = None
        classifier_error: Exception | None = None
        classifier_needed = bool({"shot_classifier", "shot_similarity"} & set(implemented_features))
        if classifier_needed:
            if "shot_classifier" in implemented_features:
                await set_feature_status(job_id, "shot_classifier", "processing")
            try:
                classifier_result = await _compute_shot_classifier_result(
                    artifacts,
                    video_path,
                    safe_name,
                )
            except FeatureError as exc:
                classifier_error = exc
                logger.error("shot_classifier prerequisite failed for {}: {}", job_id, exc)
                if "shot_classifier" in implemented_features:
                    await store_result(
                        job_id,
                        "shot_classifier",
                        FeatureResult(status="failed", error=str(exc)),
                    )
            except Exception as exc:
                classifier_error = exc
                logger.exception("Unexpected shot_classifier prerequisite failure for {}", job_id)
                if "shot_classifier" in implemented_features:
                    await store_result(
                        job_id,
                        "shot_classifier",
                        FeatureResult(status="failed", error=str(exc)),
                    )
            else:
                if "shot_classifier" in implemented_features:
                    await store_result(
                        job_id,
                        "shot_classifier",
                        FeatureResult(status="done", result=classifier_result.model_dump()),
                    )

        if "shot_similarity" in implemented_features:
            if classifier_result is None:
                message = "Shot similarity requires a shot classification result."
                if classifier_error is not None:
                    message = f"Shot similarity prerequisite failed: {classifier_error}"
                await store_result(
                    job_id,
                    "shot_similarity",
                    FeatureResult(status="failed", error=message),
                )
            else:
                await run_shot_similarity(
                    job_id,
                    artifacts,
                    safe_name,
                    classifier_result.predicted_shot,
                )
    finally:
        await loop.run_in_executor(None, partial(_cleanup_path, job_dir))
        await loop.run_in_executor(None, partial(_cleanup_path, source_path))
