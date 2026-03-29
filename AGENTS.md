# CrickAI API - Codex agent instructions

## Repository Snapshot

This repo is a FastAPI backend for cricket training-net analysis.

Current implementation status:

- Fully implemented path: `/analyze` -> preprocessor -> `bowler_performance`
  -> Redis job status.
- Stub modules only: `action_legality`, `shot_classifier`,
  `shot_similarity`.
- Background execution currently uses FastAPI `BackgroundTasks`, not Celery
  workers.
- Calibration and results are stored in Redis with a TTL of 3600 seconds.
- The `features` form field on `POST /analyze` is currently a comma-separated
  string, not a JSON array.

If any code or documentation claims all four feature modules already fan out in
parallel, treat that as stale.

## Project Overview

The repo is organized as a modular monolith where shared preprocessing runs
once per uploaded video and produces in-memory artifacts for downstream feature
modules.

```text
app/
  modules/
    preprocessor/         # shared artifact extraction
    bowler_performance/   # implemented feature module
    action_legality/      # stub
    shot_classifier/      # stub
    shot_similarity/      # stub
  api/                    # FastAPI routers only
  storage/                # Redis adapters
  models/                 # shared schemas
  main.py
```

## Environment And Tooling

- Python 3.11+, managed with `uv`
- FastAPI + Pydantic v2
- Redis for calibration and job-status storage
- OpenCV headless for video processing
- ONNX Runtime and TFLite for the preprocessor models
- `loguru` for logging

Useful commands:

```bash
uv sync --all-extras
uv run uvicorn app.main:app --reload
uv run pytest
uv run ruff check .
uv run mypy app/
```

## Working Agreements

- After modifying any Python file, run `uv run ruff check .` and fix all
  errors.
- After modifying any file under `app/`, run `uv run pytest` and confirm no
  regressions.
- Never use `pip`. Use `uv add <pkg>`, `uv add <pkg> --optional <group>`, or
  `uv add <pkg> --dev`.
- Ask before adding a new production dependency.
- Do not add new top-level packages outside the existing module layout without
  confirmation.

## Architecture Rules

### Module boundaries

Each module under `app/modules/` should remain self-contained:

- `service.py`: plain Python class, no FastAPI imports
- `models.py`: Pydantic models or module-local dataclasses

The `app/api/` layer is the only place where FastAPI types such as `APIRouter`,
`UploadFile`, `Depends`, `HTTPException`, and `BackgroundTasks` should appear.

### Current request flow

The active `/analyze` path currently works like this:

1. Parse the `calibration` form field as JSON into `CalibrationData`.
2. Parse `features` as a comma-separated string.
3. Read the uploaded file into memory.
4. Store calibration in Redis under `calib:{job_id}`.
5. Initialize Redis job status under `results:{job_id}`.
6. Queue `process_job(...)` via FastAPI `BackgroundTasks`.
7. `process_job(...)` writes the file into a temp directory, runs the
   preprocessor, derives FPS from the ball path, and runs
   `BowlerPerformanceAnalyzer`.

Important limitation: the current background path only executes
`bowler_performance`. If that feature is not selected, the background job logs
that no implemented feature was selected and returns.

### Shared artifact contract

The preprocessor returns `VideoArtifacts`:

```python
@dataclass(slots=True)
class VideoArtifacts:
    release_frame: np.ndarray
    ball_path: list[BallDetection]
    bat_contact_frame: np.ndarray | None
```

Current feature ownership in this repo:

- `bowler_performance` consumes all three artifacts plus calibration.
- The other three feature modules are not implemented yet and should not be
  documented as active consumers.

### Preprocessor behavior

`VideoPreprocessor.run(...)` currently implements:

1. Video standardization with `ffmpeg`
2. Batter ROI derivation from calibration keypoints
3. Batter-presence detection using PoseNet over sampled frames
4. Release-frame detection using `releaseClassifier.onnx`
5. Ball-path tracking using `ballDetection.onnx`
6. Optional bat-contact detection using audio impact + ball-velocity refinement

Do not document or add a hit-classifier stage unless you also add the source
code for it. The current repo does not contain a live `hit_classifier.py`
module.

### Job status contract

Keep this shape stable unless the client contract changes with it:

```python
class FeatureResult(BaseModel):
    status: Literal["pending", "processing", "done", "failed"]
    result: dict | None = None
    error: str | None = None

class JobStatus(BaseModel):
    job_id: str
    overall_status: Literal["pending", "processing", "done", "partial"]
    bowler_performance: FeatureResult
    action_legality: FeatureResult
    shot_classifier: FeatureResult
    shot_similarity: FeatureResult
```

Be aware that the current implementation may leave `overall_status` at
`processing` when `bowler_performance` is done and the other three features are
still pending.

### Storage notes

- Calibration storage: `app/storage/calibration.py`
- Results storage: `app/storage/results.py`
- Both currently use Redis only
- S3/GCS is not wired into the active code path yet

`get_calibration(job_id)` exists, but the current background flow passes
`CalibrationData` in process instead of reading it back from Redis.

## Dependency Groups

Current optional dependency groups in `pyproject.toml`:

- `bat-contact`
- `ml-base`
- `ml-base-cpu`
- `ml-base-gpu`
- `bowler-performance`
- `action-legality`
- `shot-classifier`
- `shot-similarity`
- `dev`

Do not move ML packages into the core dependency list without a strong reason.

## Code Style

- Use Pydantic v2 APIs: `model_validate`, `model_dump`, `model_dump_json`
- Type-annotate every function signature
- Use `loguru`, not `logging`
- Prefer named exceptions from `app/exceptions.py`
- Return `None` for optional or missing resources where that is already the
  established contract

## Testing Conventions

- Tests mirror the app structure under `tests/`
- Use `factory-boy` factories from `tests/conftest.py`
- Mock Redis at the storage layer
- Use `pytest-asyncio` with `asyncio_mode = "auto"`
- Do not require a GPU in test coverage

## Ask Before Doing

- Adding a new production dependency
- Changing the `/analyze` request contract
- Wiring new persistence outside the Redis result-store pattern
- Splitting modules into separate services
- Claiming Celery, S3, or the remaining feature modules are already active when
  they are not
