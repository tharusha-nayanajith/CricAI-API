# FullTrack API — Codex agent instructions

## Project overview

FastAPI backend for a cricket training net application. Processes uploaded videos
and runs ML-powered analysis across four features: bowler performance analysis,
bowling action legality checking, cricket shot classification, and cricket shot
similarity matching.

Uses a modular monolith architecture where all features share a single process,
GPU, and in-memory video artifacts.

```
app/
├── modules/
│   ├── preprocessor/          # shared video artifact extraction
│   ├── bowler_performance/    # bowler performance analyzer
│   ├── action_legality/       # bowling action legality checker
│   ├── shot_classifier/       # cricket shot classifier
│   └── shot_similarity/       # cricket shot similarity
├── api/                       # thin FastAPI routers only — no business logic here
├── storage/                   # artifact store (S3/GCS) and calibration (Redis)
├── models/                    # shared Pydantic schemas
└── main.py
```

## Environment and tooling

- Python 3.11+, managed with **uv** — never use pip directly
- FastAPI + Pydantic v2
- Celery + Redis for background task queue
- OpenCV headless (`opencv-python-headless`) for video processing
- PyTorch + ONNX Runtime for ML inference

```bash
# install all deps including dev and ML groups
uv sync --all-extras

# run the dev server
uv run uvicorn app.main:app --reload

# run tests
uv run pytest

# lint
uv run ruff check .

# type check
uv run mypy app/
```

## Working agreements

- After modifying any Python file, run `uv run ruff check .` and fix all errors
  before considering the task done.
- After modifying any file under `app/`, run `uv run pytest` and confirm no
  regressions. If tests fail due to missing fixtures or test-only deps, flag it
  rather than skipping.
- Never install packages with pip. Always use `uv add <pkg>` for core deps,
  `uv add <pkg> --optional <group>` for feature-specific deps, or
  `uv add <pkg> --dev` for dev tools.
- Ask for confirmation before adding any new production dependency.
- Do not add new top-level packages outside the established module structure
  without asking first.

## Architecture rules — read before writing any code

### Module boundaries

Each module under `app/modules/` must be self-contained:

- `service.py` — plain Python class, **zero FastAPI imports**. Takes typed inputs,
  returns typed outputs. This is the extractable unit if a feature is ever split
  to its own service.
- `models.py` — Pydantic models scoped to that module's inputs/outputs.

The `app/api/` layer is the only place FastAPI (`APIRouter`, `Depends`,
`UploadFile`, etc.) should appear. Routers call module services — they contain
no business logic themselves.

### Calibration data is scoped to bowler performance only

`CalibrationData` is stored under `calib:{job_id}` in Redis and is only read
inside `app/modules/bowler_performance/service.py`. No other module may import
from `app/storage/calibration.py`. If calibration data appears in any other
module, treat it as a bug and revert it.

### Video artifacts are passed in-process

The preprocessor extracts three shared artifacts from every uploaded video:
ball release frame, ball path, and bat contact frame. These are numpy arrays
held in memory — do not serialize them to disk or object storage between the
preprocessor and feature modules. The whole point of the monolith is zero-copy
artifact passing. Object storage is only for persisting final results and
long-term archival.

### Feature-to-artifact mapping

| Module | Artifacts required |
|---|---|
| bowler_performance | release frame + ball path + bat contact frame + calibration |
| action_legality | release frame only |
| shot_classifier | bat contact frame only |
| shot_similarity | bat contact frame only |

If a module accesses an artifact it does not own according to this table, flag
it as a design violation.

### Job lifecycle

1. `POST /analyze` — validates upload, stores calibration in Redis, returns
   `{ job_id }` immediately.
2. Preprocessor background task runs — extracts shared artifacts.
3. Feature module tasks fan out in parallel once artifacts are ready.
4. Each module writes its result independently to the result store keyed by
   `job_id`.
5. `GET /results/{job_id}` — returns partial results as modules complete; each
   module has its own `status: pending | processing | done | failed`.

### JobStatus schema shape

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

Do not change this schema shape without updating the Flutter client contract.

## Dependency groups

| Group | Purpose | Install condition |
|---|---|---|
| *(core)* | FastAPI, Celery, Redis, OpenCV, NumPy, MoviePy, SciPy | Always |
| `bat-contact` | librosa audio helpers for bat contact detection | Bat contact pipeline |
| `ml-base` | PyTorch, ONNX Runtime, MediaPipe | All envs with inference |
| `ml-base-cpu` | CPU-only ONNX Runtime variant | CI / local no-GPU |
| `ml-base-gpu` | GPU ONNX Runtime variant | Production GPU instance |
| `bowler-performance` | scipy, filterpy, pykalman | Bowler performance module |
| `action-legality` | pose-format, trimesh | Action legality module |
| `shot-classifier` | scikit-learn, joblib | Shot classifier module |
| `shot-similarity` | faiss-cpu / faiss-gpu, annoy | Shot similarity module |
| `dev` | pytest, ruff, mypy, httpx, factory-boy | Development only |

Never add ML packages to the core dependency list. They belong in `ml-base` or
a feature-specific group.

## Code style

- Pydantic v2 — use `model_validate`, `model_dump`, `model_dump_json`. Never use
  v1 aliases like `.dict()` or `.parse_obj()`.
- All service methods that touch I/O (Redis, S3, database) must be `async`.
- Use `loguru` for logging — not the stdlib `logging` module.
- Type-annotate all function signatures. `mypy` must pass clean.
- Prefer explicit error types over bare `Exception`. Define custom exceptions in
  `app/exceptions.py`.
- Return `None` rather than raising when an optional resource (e.g. calibration
  for a job) is not found. Raise only for unexpected failure states.

## Testing conventions

- Tests live in `tests/` mirroring the `app/` structure:
  `tests/modules/bowler_performance/test_service.py`, etc.
- Use `factory-boy` for fixture factories, not hand-rolled dicts.
- Mock external I/O (Redis, S3) at the storage layer — do not let tests hit real
  infrastructure.
- Use `pytest-asyncio` with `asyncio_mode = "auto"` (set in `pyproject.toml`).
- A test that patches ML model inference is acceptable — do not require a GPU in
  CI.

## Docker notes

Production Dockerfile installs with:

```bash
uv sync \
  --extra bat-contact \
  --extra ml-base-gpu \
  --extra bowler-performance \
  --extra action-legality \
  --extra shot-classifier \
  --extra shot-similarity \
  --no-dev --frozen
```

The `--frozen` flag is required — fail the build if `uv.lock` is out of sync
with `pyproject.toml`.

Use `opencv-python-headless` not `opencv-python` — the headless variant omits
GUI libraries that bloat the image and are never needed in a container.

## What to ask before doing

- Adding a new third-party ML model or dataset download — storage and licensing
  implications need review.
- Changing the `POST /analyze` request contract (field names, types) — Flutter
  client must be updated in sync.
- Splitting any module into a separate microservice — requires infra discussion.
- Adding any direct database writes outside the result store pattern.
