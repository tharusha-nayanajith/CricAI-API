# FullTrack AI — Project context for Claude

## What this is

Cricket training net application. A Flutter mobile app records net sessions,
runs on-device camera calibration, then uploads videos to a FastAPI backend
for ML-powered analysis. Four analysis features are planned; the backend
preprocessor (shared pipeline feeding all four) is the current focus.

---

## Tech stack

### Backend
| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Framework | FastAPI + Pydantic v2 |
| Task queue | Celery + Redis |
| Package manager | uv (never pip) |
| ML runtime | ONNX Runtime (GPU → CPU fallback) + TFLite |
| CV | OpenCV headless (`opencv-python-headless`) |
| DL framework | PyTorch + TorchVision |
| Pose estimation | MediaPipe + PoseNet (posenet.tflite) |
| Audio analysis | moviepy + scipy + librosa |
| Storage | S3 / GCS (boto3) |
| Logging | loguru (never stdlib logging) |
| Linting | ruff |
| Type checking | mypy |
| Testing | pytest + pytest-asyncio + httpx + factory-boy |

### Flutter app (client)
- Runs on-device camera calibration before upload
- Sends video + calibration JSON as `multipart/form-data`
- Polls `GET /results/{job_id}` for partial results per feature

---

## Folder structure

```
app/
├── main.py                        # FastAPI app, mounts routers, GET /health
├── config.py                      # pydantic-settings Settings (REDIS_URL, S3_BUCKET …)
├── exceptions.py                  # FullTrackError, PreprocessingError,
│                                  # CalibrationError, FeatureError
├── api/
│   ├── analyze.py                 # POST /analyze — thin router only
│   └── results.py                 # GET /results/{job_id}
├── models/
│   ├── job.py                     # FeatureResult, JobStatus
│   ├── calibration.py             # CalibrationData, Keypoint
│   └── artifacts.py               # VideoArtifacts
├── storage/
│   ├── calibration.py             # store_calibration / get_calibration (Redis)
│   └── results.py                 # store_result / get_job_status
└── modules/
    ├── preprocessor/              # ← current focus
    │   ├── constants.py           # all magic numbers and model paths
    │   ├── models.py              # BallDetection, ReleasePoint, BatterROI,
    │   │                          # BatterMode, DeliveryContext,
    │   │                          # BatContactResult, ContactMethod
    │   ├── service.py             # VideoPreprocessor class + singletons
    │   ├── release_detector.py    # ReleaseDetector (posenet + releaseClassifier.onnx)
    │   ├── batter_detector.py     # BatterDetector (reuses posenet interpreter)
    │   ├── ball_tracker.py        # BallTracker (ballDetection.onnx)
    │   ├── audio_spike_detector.py# AudioSpikeDetector (moviepy + scipy)
    │   ├── hit_classifier.py      # HitClassifier (hitClassifier.onnx)
    │   ├── bat_contact_detector.py# BatContactDetector (fuses audio + classifier)
    │   └── weights/               # .onnx and .tflite model files (not in git)
    │       ├── releaseClassifier.onnx
    │       ├── posenet.tflite
    │       ├── ballDetection.onnx
    │       └── hitClassifier.onnx
    ├── bowler_performance/        # feature module
    │   ├── models.py              # BowlerPerformanceResult, BouncePoint, LengthClass
    │   ├── camera.py              # calibration matrices + pixel-to-ground unprojection
    │   ├── ransac.py              # BallPathCleaner + bounce detection
    │   ├── metrics.py             # speed, swing, bounce point, length band
    │   └── service.py             # BowlerPerformanceAnalyzer
    ├── action_legality/           # feature module (not yet implemented)
    ├── shot_classifier/           # feature module (not yet implemented)
    └── shot_similarity/           # feature module (not yet implemented)

tests/
└── modules/
    ├── preprocessor/
    │   ├── test_standardize.py
    │   ├── test_release_detector.py
    │   ├── test_batter_detector.py
    │   ├── test_ball_tracker.py
    │   ├── test_audio_spike_detector.py
    │   ├── test_hit_classifier.py
    │   └── test_bat_contact_detector.py
    └── bowler_performance/
        ├── test_camera.py
        ├── test_ransac.py
        ├── test_metrics.py
        └── test_service.py
```

---

## Key models and schemas

### CalibrationData (sent by Flutter on every upload)
```python
class Keypoint(BaseModel):
    x: float
    y: float
    score: float
    channel_index: int   # 0–5 = batter end stumps, 6–11 = bowler end stumps

class CalibrationData(BaseModel):
    image_size: tuple[int, int]
    fov: float
    yaw: float
    position: tuple[float, float, float]
    principal_point: tuple[float, float]
    rotation: tuple[float, float, float]
    score: float
    detected_channels: int
    total_detections: int
    keypoints: list[Keypoint]
```

### VideoArtifacts (output of the preprocessor)
```python
@dataclass
class VideoArtifacts:
    release_frame:     np.ndarray             # BGR, annotated
    ball_path:         list[BallDetection]    # raw 2D path, no smoothing
    bat_contact_frame: np.ndarray | None      # None when no batter
```

### BowlerPerformanceResult (stored under FeatureResult.result)
```python
class BouncePoint(BaseModel):
    x_metres: float
    z_metres: float

class BowlerPerformanceResult(BaseModel):
    speed_kmh: float
    swing_metres: float
    bounce_point: BouncePoint | None
    length_class: LengthClass | None
    confidence: float
    inlier_count: int
    raw_speed_ms: float
```

### JobStatus (polled by Flutter)
```python
class FeatureResult(BaseModel):
    status: Literal["pending", "processing", "done", "failed"]
    result: dict | None = None
    error:  str  | None = None

class JobStatus(BaseModel):
    job_id:             str
    overall_status:     Literal["pending", "processing", "done", "partial"]
    bowler_performance: FeatureResult
    action_legality:    FeatureResult
    shot_classifier:    FeatureResult
    shot_similarity:    FeatureResult
```

---

## Preprocessor pipeline — current implementation

`VideoPreprocessor.run(video_path, calibration)` executes in this exact order:

```
1. standardize_video()
   └─ ffmpeg resize to 720 × 1280, libx264, crf 18, preset fast
   └─ skip if already 720 × 1280

2. batter_detector.detect()
   └─ derive BatterROI from calibration keypoints channels 0–5
      · stump bounding box × 3, min floor 80 × 120 px
      · centroid-centred, clamped to frame bounds
   └─ sample 5 frames at 15%–50% of video duration
   └─ run PoseNet on ROI crop of each frame
   └─ majority vote ≥ 3/5 → BatterMode.PRESENT else BatterMode.NONE

3. release_detector.process_frame()  (per-frame loop)
   └─ PoseNet keypoints → releaseClassifier.onnx → release probability
   └─ first frame above threshold → ReleasePoint

4. ball_tracker.track()
   └─ starts at release_frame_idx (2-frame warm-up buffer)
   └─ ballDetection.onnx 3-frame sliding window
   └─ collect BallDetection if conf ≥ 0.30
   └─ termination:
      · PRESENT mode → ball enters batter ROI
      · NONE mode    → MAX_BALL_TRACK_FRAMES (80) or early stop heuristic
   └─ raises PreprocessingError if < 3 detections

5. bat_contact_detector.detect()   [only when BatterMode.PRESENT]
   └─ audio: moviepy extract → highpass 1000 Hz → smooth → find peak
   └─ search window: audio_frame ± 15 frames
   └─ fallback (no audio): first frame ball enters batter ROI as anchor
   └─ hit classifier: slide 3-frame window, score triplets, argmax
   └─ returns BatContactResult with method tag (audio+classifier | classifier_only)
```

---

## Key constants (`app/modules/preprocessor/constants.py`)

```python
STANDARDIZED_WIDTH  = 720
STANDARDIZED_HEIGHT = 1280

# Ball tracking
BALL_CONF_RAW_THRESHOLD   = 0.30
MAX_BALL_TRACK_FRAMES     = 80
BALL_EARLY_STOP_CONF      = 0.20
BALL_EARLY_STOP_Y         = 200
BALL_EARLY_STOP_MIN_FRAME = 10

# Batter detection
BATTER_ROI_SCALE_FACTOR   = 3      # stump bbox × 3
BATTER_ROI_MIN_WIDTH      = 80
BATTER_ROI_MIN_HEIGHT     = 120
BATTER_POSE_THRESHOLD     = 0.30
BATTER_SAMPLE_COUNT       = 5
BATTER_SAMPLE_WINDOW_START= 0.15
BATTER_SAMPLE_WINDOW_END  = 0.50
BATTER_VOTE_THRESHOLD     = 3

# Bat contact
AUDIO_SEARCH_WINDOW_FRAMES = 15
HIT_CLASSIFIER_INPUT_SIZE  = 512
HIT_CLASSIFIER_THRESHOLD   = 0.50
AUDIO_HIGHPASS_CUTOFF_HZ   = 1000.0
AUDIO_SMOOTHING_WINDOW     = 512
AUDIO_MIN_PROMINENCE       = 0.20
AUDIO_MIN_PEAK_DISTANCE_S  = 0.1
```

---

## Coding conventions

### Module boundary rule
Every `modules/<name>/service.py` is a plain Python class — **zero FastAPI
imports**. Only `app/api/` touches FastAPI. This keeps every module
extractable to its own microservice if needed later.

### Singleton model loader pattern
Models are loaded once per process lifetime, never per-request:
```python
_detector: Optional[SomeDetector] = None

def get_detector() -> SomeDetector:
    global _detector
    if _detector is None:
        _detector = SomeDetector(MODEL_PATH)
        _detector.load_models()
    return _detector
```

### Shared model instances — never load twice
PoseNet is shared between `ReleaseDetector` and `BatterDetector`.
`BatterDetector.__init__` receives the tflite interpreter as a constructor
argument. `ReleaseDetector` exposes it as `self.posenet_interpreter`.

### Async + blocking inference
All service methods touching I/O are `async`. CPU/GPU inference (blocking)
is dispatched via:
```python
await asyncio.get_event_loop().run_in_executor(None, blocking_fn, *args)
```

### Error handling
- Return `None` for optional/missing resources (calibration not found, no
  audio track, ball never enters ROI)
- Raise named exceptions from `app/exceptions.py` for unexpected failures
- `AudioSpikeDetector.detect()` **never raises** — always returns
  `float | None`, logs warnings internally

### Pydantic v2 only
Use `model_validate`, `model_dump`, `model_dump_json`.
Never use v1 aliases `.dict()`, `.parse_obj()`.

### Calibration storage
`CalibrationData` is stored in Redis under `calib:{job_id}` with TTL 3600s.
Only `bowler_performance` module may read it. All other modules are forbidden
from importing `app/storage/calibration.py`.

### Dependency management
```bash
uv add <pkg>                          # core
uv add <pkg> --optional <group>       # feature-specific
uv add <pkg> --dev                    # dev tools
# never: pip install
```

### Logging
`from loguru import logger` everywhere. Never `import logging`.

---

## Dependency groups (`pyproject.toml`)

| Group | Key packages |
|---|---|
| core | fastapi, uvicorn, pydantic, celery, redis, opencv-python-headless, numpy, boto3, loguru, moviepy, scipy |
| ml-base-cpu | torch, torchvision, onnxruntime, mediapipe |
| ml-base-gpu | torch, torchvision, onnxruntime-gpu, mediapipe |
| bowler-performance | scipy, filterpy, pykalman |
| action-legality | pose-format, trimesh |
| shot-classifier | scikit-learn, joblib |
| shot-similarity | faiss-cpu / faiss-gpu, annoy |
| bat-contact | librosa |
| dev | pytest, pytest-asyncio, httpx, ruff, mypy, factory-boy |

Production Docker install:
```bash
uv sync \
  --extra ml-base-gpu \
  --extra bowler-performance \
  --extra action-legality \
  --extra shot-classifier \
  --extra shot-similarity \
  --extra bat-contact \
  --no-dev --frozen
```

---

## Architecture decision: modular monolith

Single process, single container, shared GPU. Chosen because:
- ML models load once and are shared across all four features
- Video artifacts (numpy arrays) pass in-memory with zero copy
- One GPU serves all features without multiplying VRAM usage

**Planned extraction**: if `bowler_performance` becomes a bottleneck under
load, it is the first candidate to extract to a dedicated GPU service.
The clean `service.py` boundary makes this a wrapper swap, not a rewrite.

---

## API contract (do not change without updating Flutter client)

```
POST /analyze
  body: multipart/form-data
    video:       UploadFile
    calibration: str  (CalibrationData as JSON)
    features:    str  (JSON array, default all four)
  response: { "job_id": str }

GET /results/{job_id}
  response: JobStatus
  404 if job_id unknown
```

---

## Next steps

The preprocessor pipeline is fully designed and being implemented task by
task. Current implementation status:

- [x] Project skeleton (`pyproject.toml`, routers, Pydantic schemas,
      storage layer, exception hierarchy)
- [x] `standardize_video()` — ffmpeg resize to 720 × 1280
- [x] `ReleaseDetector` — PoseNet + releaseClassifier.onnx
- [x] `BatterDetector` — calibration ROI + PoseNet majority vote
- [x] `BallTracker` — ballDetection.onnx raw 2D path
- [x] `AudioSpikeDetector` — moviepy + scipy audio spike
- [x] `HitClassifier` — hitClassifier.onnx 3-frame sliding window
- [x] `BatContactDetector` — audio + classifier fusion
- [x] Wire all preprocessor tests to pass green in CI
- [x] Implement `bowler_performance` module
      - consumes: release_frame + ball_path + bat_contact_frame + calibration
      - includes: RANSAC parabola fitting, calibration-based 3D reconstruction,
        speed / swing / bounce / length metrics
- [ ] Implement `action_legality` module
      - consumes: release_frame only
      - needs: elbow angle analysis from PoseNet keypoints
- [ ] Implement `shot_classifier` module
      - consumes: bat_contact_frame only
      - needs: shot type classification (drive, pull, sweep …)
- [ ] Implement `shot_similarity` module
      - consumes: bat_contact_frame only
      - needs: embedding extraction + vector similarity search (faiss)
- [ ] Celery task wiring — fan-out from preprocessor to four feature tasks
- [ ] Redis result store — partial JobStatus updates per feature
- [ ] WebSocket push endpoint (`/ws/{job_id}`) as alternative to polling
- [ ] End-to-end test with a real net session video
