# CrickAI API

FastAPI backend for a cricket training net application. The project is structured
as a modular monolith where shared preprocessing runs once per delivery and
produces in-memory artifacts for downstream feature modules.

## Current Status

The project skeleton is in place and importable. The API layer, storage layer,
shared models, Dockerfile, `uv`-managed dependency configuration, linting, and
test setup are already bootstrapped.

The preprocessor module currently implements:

- Video standardization with `ffmpeg` to `720x1280`
- Batter presence detection from calibration-derived ROI sampling
- Release frame detection using `releaseClassifier.onnx` + PoseNet
- Raw 2D ball-path extraction from the release frame onward using `ballDetection.onnx`

The preprocessor currently returns these artifacts through
`app.models.artifacts.VideoArtifacts`:

- `release_frame`
- `ball_path` as `list[BallDetection]`
- `bat_contact_frame` placeholder for the next task

## Preprocessor Flow

`VideoPreprocessor.run(video_path, calibration)` currently executes this order:

1. Standardize the input video if needed
2. Detect batter mode and derive batter ROI
3. Build `DeliveryContext`
4. Detect the release frame
5. Track the raw ball path
6. Return `VideoArtifacts`

Shared detector/model instances use module-level singleton getters so expensive
models are loaded once per process lifetime.

## Project Layout

```text
app/
├── api/                        # FastAPI routers
├── models/                     # Shared schemas/dataclasses
├── modules/
│   ├── preprocessor/           # Shared artifact extraction
│   ├── bowler_performance/     # Placeholder module
│   ├── action_legality/        # Placeholder module
│   ├── shot_classifier/        # Placeholder module
│   └── shot_similarity/        # Placeholder module
└── storage/                    # Redis-backed storage adapters
```

## Dependencies

The repo uses `uv` and Python 3.11.

Useful commands:

```bash
uv sync --all-extras
uv run ruff check .
uv run pytest
uv run uvicorn app.main:app --reload
```

For the current preprocessor pipeline, the local environment also needs:

- `ffmpeg` available on `PATH`, or `FFMPEG_PATH` set to a valid binary
- Preprocessor model files under `app/modules/preprocessor/weights/`
  - `releaseClassifier.onnx`
  - `posenet.tflite`
  - `ballDetection.onnx`

## Local Verification

`Test_Scripts/` contains local validation assets and helpers.

- `run_preprocessor_check.py`
  - Parses calibration `.txt` files exported from the app
  - Runs the preprocessor on the sample videos
  - Saves annotated release frames to `Test_Scripts/preprocessor_outputs/`
  - Writes a summary file at `Test_Scripts/preprocessor_outputs/summary.json`

Run it with:

```bash
uv run python Test_Scripts/run_preprocessor_check.py
```

The current sample set has already been verified locally with these outcomes:

- `batter_video_01.mp4` -> `batter_mode=present`
- `batter_video_02.mp4` -> `batter_mode=present`
- `bowler_video_01.mp4` -> `batter_mode=none`

## Quality Checks

The current codebase is green on:

- `uv run ruff check .`
- `uv run pytest`

At the latest checkpoint, the repository test suite passes with `34` tests.
