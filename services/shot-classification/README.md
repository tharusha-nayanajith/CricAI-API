# CricAI Coach — FastAPI Backend

Single FastAPI backend with modular routers, services, models, and utils.

## Project Layout

```
main.py
routers/
  bowling.py
  batting.py
  action.py
  similarity.py
services/
  bowling_service.py
  batting_service.py
  action_service.py
  similarity_service.py
models/
  schemas.py
utils/
  math_utils.py
  logging_utils.py
requirements.txt
```

## Endpoints
- POST `/api/bowling/analyze`
- POST `/api/batting/classify`
- POST `/api/action/validate`
- POST `/api/similarity/compare`
- GET  `/api/health`

## Quickstart

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows Git Bash
# For PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

Run the server (from repo root):

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open docs: http://localhost:8000/docs

## Team Workflow
- Each developer owns a feature pair: `routers/<feature>.py` + `services/<feature>_service.py`.
- All code runs in one FastAPI app (`main.py`), no inter-service calls needed.
- Shared schemas live in `models/schemas.py` and helpers in `utils/`.
- Prefer adding/using typed request/response models in `schemas.py` to keep consistency.

## Minimal Example Payloads

```jsonc
POST /api/bowling/analyze
{
  "measurements": [72.5, 80.2, 78.9],
  "hand": "right",
  "notes": "nets session"
}

POST /api/batting/classify
{
  "features": [0.4, 0.6, 0.7]
}

POST /api/action/validate
{
  "action_name": "cover_drive",
  "features": [0.2, 0.5, 0.9]
}

POST /api/similarity/compare
{
  "a": [0.1, 0.2, 0.3],
  "b": [0.1, 0.25, 0.35],
  "metric": "cosine"
}
```

## Notes
- CORS is permissive for development; tighten for production.
- Update logging via `LOG_LEVEL` env var.
