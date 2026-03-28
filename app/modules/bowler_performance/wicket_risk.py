from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

from loguru import logger

from app.modules.bowler_performance.models import (
    DeliveryFeatures,
    WicketRiskBand,
    WicketRiskPrediction,
)

MODEL_NAME = "wicket_probability_regressor"
MODEL_VERSION = "synthetic_rf_v1"
MODEL_PATH = Path(__file__).resolve().parent / "weights" / "wicket_probability_regressor.pkl"

LOW_RISK_MAX = 0.33
MEDIUM_RISK_MAX = 0.66


def _risk_band(probability: float) -> WicketRiskBand:
    if probability < LOW_RISK_MAX:
        return WicketRiskBand.LOW
    if probability < MEDIUM_RISK_MAX:
        return WicketRiskBand.MEDIUM
    return WicketRiskBand.HIGH


@lru_cache(maxsize=1)
def _load_model_artifact() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        logger.warning("Wicket risk model weights not found at {}", MODEL_PATH)
        return None

    try:
        with MODEL_PATH.open("rb") as handle:
            artifact = pickle.load(handle)
    except Exception as exc:  # pragma: no cover - depends on local model environment
        logger.warning("Failed to load wicket risk model from {}: {}", MODEL_PATH, exc)
        return None

    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        logger.warning("Wicket risk model artifact at {} is malformed", MODEL_PATH)
        return None
    return artifact


def predict_wicket_risk(
    delivery_features: DeliveryFeatures,
) -> WicketRiskPrediction | None:
    artifact = _load_model_artifact()
    if artifact is None:
        return None

    feature_columns = artifact.get("metadata", {}).get("feature_columns")
    pipeline = artifact["pipeline"]
    if not isinstance(feature_columns, list) or not feature_columns:
        logger.warning("Wicket risk model metadata is missing feature_columns")
        return None

    payload = delivery_features.model_dump(by_alias=True)
    feature_row = [payload.get(column) for column in feature_columns]

    try:
        probability = float(pipeline.predict([feature_row])[0])
    except Exception as exc:  # pragma: no cover - depends on sklearn runtime
        logger.warning("Wicket risk inference failed: {}", exc)
        return None

    probability = max(0.0, min(1.0, probability))
    return WicketRiskPrediction(
        probability=probability,
        percentage=round(probability * 100.0, 1),
        risk_band=_risk_band(probability),
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )
