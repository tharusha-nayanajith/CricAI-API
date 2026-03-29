# CrickAI API - Project context for Claude

## What this repo is

FastAPI backend for a cricket training-net application. A client uploads a
video plus calibration data, the backend preprocesses the delivery once, and
feature modules consume the shared artifacts inside the same process.

## Current Architecture

- Public endpoints:
  - `GET /health`
  - `POST /analyze`
  - `GET /results/{job_id}`
- Active background path:
  - FastAPI `BackgroundTasks`
  - Redis-backed calibration storage
  - Redis-backed job-status storage
  - Shared preprocessor
  - `bowler_performance` analysis
  - `action_legality` analysis
  - `shot_similarity` analysis
- Still stubbed:
  - `shot_classifier`

Important: Celery is a dependency, but it is not wired into the active request
flow yet.

## High-value files

```text
app/
  main.py                         # FastAPI app and router wiring
  config.py                       # REDIS_URL, S3_BUCKET, AWS_REGION
  exceptions.py                   # Full repo exception types
  api/
    analyze.py                    # POST /analyze
    results.py                    # GET /results/{job_id}
  models/
    calibration.py                # CalibrationData and Keypoint
    artifacts.py                  # VideoArtifacts
    job.py                        # FeatureResult and JobStatus
  storage/
    calibration.py                # Redis calibration helpers
    results.py                    # Redis job-status helpers
  modules/
    preprocessor/
      service.py                  # VideoPreprocessor
      release_detector.py         # releaseClassifier.onnx + PoseNet
      batter_detector.py          # ROI derivation + batter presence
      ball_tracker.py             # ballDetection.onnx
      bat_contact_detector.py     # audio impact + velocity refinement
      models.py
      constants.py
      weights/
    bowler_performance/
      service.py                  # BowlerPerformanceAnalyzer
      camera.py                   # intrinsic/extrinsic matrices
      ransac.py                   # BallPathCleaner
      metrics.py                  # speed, swing, bounce, length
      pitch_coordinates.py        # world-to-pitch coordinate mapping
      trajectory.py               # trajectory interpolation and helpers
      models.py
    action_legality/
      assets/
      service.py                  # legality model + MediaPipe pose extraction
      models.py
    shot_similarity/
      assets/
      service.py                  # contact-frame pose matching
      models.py
```

## API contract

### `POST /analyze`

`multipart/form-data`:

- `video`: uploaded file
- `calibration`: JSON string for `CalibrationData`
- `features`: optional comma-separated string

Current accepted feature names:

- `bowler_performance`
- `action_legality`
- `shot_classifier`
- `shot_similarity`

Current behavior:

- The endpoint stores calibration in Redis under `calib:{job_id}`
- The endpoint initializes job state under `results:{job_id}`
- The active background flow executes:
  - `bowler_performance`
  - `action_legality`
  - `shot_similarity`
- `shot_classifier` remains pending because it is still a stub

### `GET /results/{job_id}`

Returns:

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

## Preprocessor pipeline

`VideoPreprocessor.run(video_path, calibration, require_ball_path=True)`
currently does this:

1. Standardize the video to `720x1280` with `ffmpeg` when needed.
2. Derive a batter ROI from calibration stump channels `0-5`.
   Fallback: use projected striker stump keypoints when calibration
   reprojection is more reliable than detected keypoints.
3. Sample 5 frames between 15% and 50% of the clip and use PoseNet voting to
   decide `BatterMode.PRESENT` vs `BatterMode.NONE`.
4. Detect the release frame with a 3-frame `releaseClassifier.onnx` window and
   annotate the bowling arm using PoseNet.
5. Track the ball from the release frame with `ballDetection.onnx`.
6. When a batter is present, estimate contact from the audio impact peak and
   refine it using ball-velocity changes near the audio frame.

Returned artifacts:

```python
@dataclass(slots=True)
class VideoArtifacts:
    release_frame: np.ndarray
    ball_path: list[BallDetection]
    bat_contact_frame: np.ndarray | None
    release_point: ReleasePoint | None = None
```

If `require_ball_path=False`, the preprocessor returns an empty `ball_path` and
skips contact extraction.

## Bowler performance pipeline

`BowlerPerformanceAnalyzer.run(...)` currently:

1. Cleans the raw ball path with `BallPathCleaner`.
2. Builds intrinsic and extrinsic camera matrices from `CalibrationData`.
3. Unprojects inlier detections into world coordinates.
4. Runs sanity checks on the reconstructed trajectory.
5. Converts world points into pitch coordinates.
6. Computes speed, swing, bounce point, and length classification.
7. Builds optional `ballTrack` and `cameraCalibration` payloads for the result.

## Action legality pipeline

`ActionLegalityService.run(...)` currently:

1. Reuses the preprocessor release frame.
2. Extracts pose landmarks with MediaPipe Pose.
3. Falls back to the MediaPipe Tasks PoseLandmarker API when the classic
   `mp.solutions` API is unavailable.
4. Attempts GPU delegate initialization first on the Tasks path and falls back
   to CPU if GPU support is unavailable.
5. Applies the legacy normalization and scaler values.
6. Runs the imported legality model from
   `app/modules/action_legality/assets/bowler_model.h5`.

Runtime note:

- Set `MEDIAPIPE_POSE_TASK_PATH` or place the Tasks model at
  `app/modules/action_legality/assets/pose_landmarker.task` when deploying with
  the Tasks API path.

## Shot similarity pipeline

`ShotSimilarityService.run(...)` currently:

1. Reuses the shared preprocessor `bat_contact_frame`.
2. Extracts pose landmarks from the contact frame.
3. Compares them against the local reference library in
   `app/modules/shot_similarity/assets/golden_frames.json`.
4. Returns the best matched player, shot type, similarity, and coaching
   feedback.

Important:

- The checked-in reference library is empty until real golden references are
  added.
- This module does not port the old standalone YOLO plus audio impact pipeline.
  Impact ownership stays in the shared preprocessor.

## Conventions that matter

- Keep FastAPI imports inside `app/api/` only.
- Keep module business logic in `app/modules/*/service.py`.
- Use Pydantic v2 APIs only.
- Use `loguru` for logging.
- Use `uv`, never `pip`.
- After changing Python files:
  - run `uv run ruff check .`
  - if anything under `app/` changed, also run `uv run pytest`

## Local setup

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Environment and runtime notes:

- Redis is required
- `ffmpeg` must be on `PATH`, set through `FFMPEG_PATH`, or placed at
  `tools/ffmpeg/bin/ffmpeg.exe`
- Preprocessor model files live in `app/modules/preprocessor/weights/`
- MediaPipe Tasks deployments need `pose_landmarker.task` and may use
  `MEDIAPIPE_POSE_TASK_PATH`

## Useful local scripts

```bash
uv run python Test_Scripts/run_preprocessor_check.py
uv run python Test_Scripts/run_bowler_performance_check.py
uv run python Test_Scripts/validate_calibration_stadium.py
```

Outputs are written into:

- `Test_Scripts/preprocessor_outputs/`
- `Test_Scripts/bowler_performance_outputs/`
- `Test_Scripts/calibration_validation_outputs/`

`Test_Scripts/` is gitignored and should be treated as local-only tooling.

## Known gaps

- `shot_classifier` is still a stub
- Celery is not wired
- S3/GCS persistence is not wired
- `golden_frames.json` still needs real reference data for shot similarity
