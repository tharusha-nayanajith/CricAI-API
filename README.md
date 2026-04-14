# CrickAI API

FastAPI backend for a cricket training net workflow. The repository is built as
a modular monolith: shared preprocessing runs once per upload, keeps video
artifacts in memory, and hands them to feature modules inside the same process.

## Current State

- Implemented end to end: `preprocessor`, `bowler_performance`, `/analyze`,
  `action_legality`, `shot_similarity`, `/results/{job_id}`, Redis-backed
  calibration storage, and Redis-backed job status storage.
- Still stubbed: `shot_classifier`.
- Background execution currently uses FastAPI `BackgroundTasks`. Celery is
  listed in dependencies but is not wired into the active request flow yet.
- The `/analyze` form field `features` currently accepts a comma-separated
  string such as
  `bowler_performance,action_legality,shot_classifier,shot_similarity`.

## Implemented Flow

1. `POST /analyze` validates the upload, parses calibration JSON, stores
   calibration in Redis, initializes a `JobStatus`, and queues background work.
2. The background job writes the upload into a temporary directory and runs the
   shared preprocessor.
3. The preprocessor standardizes the video, detects batter presence, finds the
   release frame, tracks the ball path, and extracts a bat-contact frame when a
   batter is present and audio detection succeeds.
4. `BowlerPerformanceAnalyzer` cleans the ball path with RANSAC, reconstructs
   3D world points from calibration data, maps them into pitch coordinates, and
   computes speed, swing, bounce, and length metrics.
5. `ActionLegalityService` uses the preprocessor release frame, extracts pose
   landmarks, normalizes them with the legacy feature pipeline, and runs the
   imported legality model.
6. `ShotSimilarityService` uses the preprocessor bat-contact frame, extracts
   pose landmarks, and compares them against a local golden-shot reference
   library.
7. `GET /results/{job_id}` returns the latest `JobStatus` snapshot from Redis.

## Project Layout

```text
app/
  api/
    analyze.py
    results.py
  models/
    artifacts.py
    calibration.py
    job.py
  modules/
    preprocessor/
      ball_tracker.py
      batter_detector.py
      bat_contact_detector.py
      constants.py
      models.py
      release_detector.py
      service.py
      weights/
    bowler_performance/
      camera.py
      metrics.py
      models.py
      pitch_coordinates.py
      ransac.py
      service.py
    action_legality/
      assets/
      models.py
      service.py
    shot_classifier/
      models.py
      service.py
    shot_similarity/
      assets/
      models.py
      service.py
  storage/
    calibration.py
    results.py
  config.py
  exceptions.py
  main.py
tests/
  api/
  modules/
    bowler_performance/
    preprocessor/
  storage/
Test_Scripts/
```

## Environment

- Python 3.11+
- `uv` for dependency management
- Redis for calibration and job-status storage
- `ffmpeg` on `PATH`, or `FFMPEG_PATH` set, or a binary at
  `tools/ffmpeg/bin/ffmpeg.exe`

Install the default app dependencies:

```bash
uv sync
```

For local ML feature development, install the relevant extras instead of
`--all-extras`:

```bash
uv sync --extra ml-base --extra bat-contact --extra bowler-performance
```

Useful commands:

```bash
uv run uvicorn app.main:app --reload
uv run pytest
uv run ruff check .
uv run mypy app/
```

Deployment help:

- Google Compute Engine guide: [deploy/compute-engine.md](/home/tharu/projects/Final_CrickAI_Backend/deploy/compute-engine.md)
- `systemd` unit: [deploy/crickai-api.service](/home/tharu/projects/Final_CrickAI_Backend/deploy/crickai-api.service)
- `nginx` site config: [deploy/nginx-crickai.conf](/home/tharu/projects/Final_CrickAI_Backend/deploy/nginx-crickai.conf)
- production env example: [deploy/.env.gce.example](/home/tharu/projects/Final_CrickAI_Backend/deploy/.env.gce.example)

Relevant environment variables:

- `REDIS_URL`
- `S3_BUCKET`
- `AWS_REGION`
- `FFMPEG_PATH`
- `MEDIAPIPE_POSE_TASK_PATH`

## API Contract

### `POST /analyze`

`multipart/form-data` fields:

- `video`: uploaded video file
- `calibration`: JSON string matching `CalibrationData`
- `features`: optional comma-separated feature list

Example:

```bash
curl -X POST http://localhost:8000/analyze ^
  -F "video=@Test_Scripts/batter_video_01.mp4" ^
  -F "calibration={...}" ^
  -F "features=bowler_performance,action_legality"
```

Response:

```json
{
  "job_id": "9c29c0fa-9f85-4c0d-a2a4-0b1dca8b2c8d"
}
```

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

Current behavior note:

- `bowler_performance`, `action_legality`, and `shot_similarity` are executed
  by the active background flow.
- `shot_classifier` is still a placeholder and remains `pending`.

## Preprocessor Details

`VideoPreprocessor.run(video_path, calibration, require_ball_path=True)`
currently does the following:

1. Standardize the input to `720x1280` with `ffmpeg` when needed.
2. Derive a batter ROI from calibration stump channels `0-5`, with a fallback
   to projected stump keypoints when reprojection is more reliable.
3. Sample 5 frames between 15% and 50% of the clip and use PoseNet to classify
   `BatterMode.PRESENT` versus `BatterMode.NONE`.
4. Detect the release frame with `releaseClassifier.onnx` plus PoseNet-based
   arm annotation.
5. Track the ball with `ballDetection.onnx` using a 3-frame sliding window.
6. When a batter is present, estimate bat contact from the audio impact peak
   and refine that frame with ball-velocity changes near the audio estimate.

The preprocessor returns `app.models.artifacts.VideoArtifacts`:

- `release_frame`
- `ball_path`
- `bat_contact_frame`
- `release_point`

If `require_ball_path=False`, the preprocessor skips ball tracking and bat
contact detection and returns an empty `ball_path`.

## Action Legality

`ActionLegalityService` is wired into `/analyze` and consumes the release frame
from the shared preprocessor output.

Runtime details:

- Uses the current preprocessor release frame instead of the old standalone
  release-frame detector.
- Runs MediaPipe Pose extraction through either:
  - the classic `mp.solutions.pose.Pose` API when available, or
  - the MediaPipe Tasks PoseLandmarker API.
- For Tasks API deployments, place a model at
  `app/modules/action_legality/assets/pose_landmarker.task`, or set
  `MEDIAPIPE_POSE_TASK_PATH` to an absolute file path.
- The Tasks path attempts GPU delegate initialization first and falls back to
  CPU if the GPU delegate is unavailable.
- The normalized landmark vector is standardized with the legacy scaler values
  stored in `app/modules/action_legality/assets/scaler.json` and scored by
  `app/modules/action_legality/assets/bowler_model.h5`.

Returned result fields include:

- `verdict`
- `illegal_probability`
- `legal_probability`
- `confidence`
- `release_frame_index`
- `release_timestamp_s`
- `release_confidence`
- `selected_landmarks`
- `normalized_keypoints`
- `video_url`

## Shot Similarity

`ShotSimilarityService` is wired into `/analyze` and compares the batter pose
at contact against a local reference library.

Runtime details:

- Uses `bat_contact_frame` from the shared preprocessor pipeline.
- Does not port the old standalone YOLO plus audio impact detector. Impact
  detection remains owned by the shared preprocessor.
- Extracts pose landmarks from the contact frame and compares them to reference
  poses stored in
  `app/modules/shot_similarity/assets/golden_frames.json`.
- The checked-in reference library is intentionally empty until real reference
  shots are added.

Returned result fields include:

- `similarity_percentage`
- `matched_player`
- `shot_type`
- `keypoints_detected`
- `confidence`
- `feedback`
- `compared_frame`
- `video_url`

## Local Validation Scripts

Local helper scripts may exist under `Test_Scripts/`, but that directory is
gitignored and is not part of the committed application contract.

Useful entry points:

```bash
uv run python Test_Scripts/run_preprocessor_check.py
uv run python Test_Scripts/run_bowler_performance_check.py
uv run python Test_Scripts/validate_calibration_stadium.py
```

Generated artifacts are written under:

- `Test_Scripts/preprocessor_outputs/`
- `Test_Scripts/bowler_performance_outputs/`
- `Test_Scripts/calibration_validation_outputs/`

## Verification

At the time of this documentation update:

```bash
uv run ruff check .
```

`pytest` may still depend on the local Python interpreter layout and optional ML
dependencies present on the machine where it is run.
