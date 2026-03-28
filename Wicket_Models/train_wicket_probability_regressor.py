from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except ImportError as exc:  # pragma: no cover - import guard for local environments
    raise SystemExit(
        "scikit-learn is required to train this model. "
        "Install the optional shot-classifier dependencies before running this script."
    ) from exc


CATEGORICAL_COLUMNS = [
    "batterMode",
    "hasBatContact",
    "contactMethod",
    "trajectoryReliable",
    "lengthClass",
    "lineBucket",
    "paceBucket",
]

NUMERIC_COLUMNS = [
    "fps",
    "releaseFrameIdx",
    "bounceFrameIdx",
    "contactFrameIdx",
    "releaseTimestampS",
    "bounceTimestampS",
    "contactTimestampS",
    "releaseToBounceMs",
    "bounceToContactMs",
    "releaseToContactMs",
    "preBounceDetectionCount",
    "postBounceDetectionCount",
    "selectedTrackDetectionCount",
    "inlierCount",
    "bouncePitchX",
    "bouncePitchY",
    "trackingConfidence",
    "releaseConfidence",
    "contactScore",
    "releaseHeightM",
    "contactHeightM",
    "releasePitchX",
    "contactPitchX",
    "preBounceLateralDelta",
    "postBounceLateralDelta",
    "approachToStumps",
]

FEATURE_COLUMNS = [*CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS]
_CATEGORICAL_INDICES = list(range(len(CATEGORICAL_COLUMNS)))
_NUMERIC_INDICES = list(range(len(CATEGORICAL_COLUMNS), len(FEATURE_COLUMNS)))

TARGET_COLUMN = "wicketProbability"
EXCLUDED_COLUMNS = {
    "deliveryId",
    "videoURL",
    "syntheticSource",
    "wicketLabel",
    TARGET_COLUMN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the first wicket probability regression model."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("Synthetic_Dataset/outputs/wicket_probability_dataset.csv"),
        help="Path to the synthetic training dataset CSV.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("Wicket_Models/outputs/wicket_probability_regressor.pkl"),
        help="Path to save the trained model artifact.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("Wicket_Models/outputs/wicket_probability_regressor_metrics.json"),
        help="Path to save evaluation metrics.",
    )
    parser.add_argument(
        "--feature-importances-out",
        type=Path,
        default=Path("Wicket_Models/outputs/wicket_probability_feature_importances.json"),
        help="Path to save feature importances.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split size.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=7,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def _coerce_numeric(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def load_rows(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    prepared: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if key in NUMERIC_COLUMNS or key == TARGET_COLUMN:
                converted[key] = _coerce_numeric(value)
            elif key in CATEGORICAL_COLUMNS:
                converted[key] = value if value != "" else None
            else:
                converted[key] = value
        prepared.append(converted)
    return prepared


def split_features_and_target(
    rows: list[dict[str, Any]],
) -> tuple[list[list[Any]], list[float]]:
    features: list[list[Any]] = []
    target: list[float] = []

    for row in rows:
        feature_row = [row.get(column) for column in FEATURE_COLUMNS]
        features.append(feature_row)
        target_value = row[TARGET_COLUMN]
        if target_value is None:
            raise ValueError("Target column contains missing values.")
        target.append(float(target_value))

    return features, target


def build_feature_row(row: dict[str, Any]) -> list[Any]:
    return [row.get(column) for column in FEATURE_COLUMNS]


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                _CATEGORICAL_INDICES,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                _NUMERIC_INDICES,
            ),
        ],
        remainder="drop",
    )

    regressor = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=7,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor),
        ]
    )


def evaluate_model(
    model: Pipeline,
    features: list[list[Any]],
    target: list[float],
) -> dict[str, float]:
    predictions = model.predict(features)
    mae = mean_absolute_error(target, predictions)
    rmse = math.sqrt(mean_squared_error(target, predictions))
    r2 = r2_score(target, predictions)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def extract_feature_importances(model: Pipeline) -> list[dict[str, float | str]]:
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    regressor: RandomForestRegressor = model.named_steps["model"]

    categorical_pipeline: Pipeline = preprocessor.named_transformers_["categorical"]
    encoder: OneHotEncoder = categorical_pipeline.named_steps["encoder"]

    categorical_names = encoder.get_feature_names_out(CATEGORICAL_COLUMNS).tolist()
    transformed_names = [*categorical_names, *NUMERIC_COLUMNS]
    importances = regressor.feature_importances_.tolist()

    ranked = sorted(
        (
            {
                "feature": feature_name,
                "importance": float(importance),
            }
            for feature_name, importance in zip(transformed_names, importances, strict=True)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return ranked


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.dataset)
    features, target = split_features_and_target(rows)

    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, x_valid, y_valid)
    feature_importances = extract_feature_importances(model)
    artifact = {
        "model_type": "RandomForestRegressor",
        "target": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "metrics": metrics,
        "top_feature_importances": feature_importances[:20],
        "random_state": args.random_state,
    }

    save_pickle(
        args.model_out,
        {
            "pipeline": model,
            "metadata": artifact,
        },
    )
    save_json(args.metrics_out, artifact)
    save_json(
        args.feature_importances_out,
        {
            "model_type": "RandomForestRegressor",
            "target": TARGET_COLUMN,
            "feature_importances": feature_importances,
        },
    )

    print(f"Saved model to {args.model_out}")
    print(f"Saved metrics to {args.metrics_out}")
    print(f"Saved feature importances to {args.feature_importances_out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
