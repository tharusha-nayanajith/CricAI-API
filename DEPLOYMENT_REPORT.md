# CrickAI Deployment Report

Prepared on: 2026-04-30  
Project: `crickai`  
Repository inspected: `/mnt/crickai-data/crickai`

## Executive Summary

CrickAI is a containerized backend platform for cricket video analysis that processes uploaded delivery footage and returns structured performance outputs through a web API. The current implementation is built on FastAPI for request handling, Celery for asynchronous job execution, Redis for transient job state and task brokering, and PostgreSQL for user, entitlement, and quota data.

The core processing design is centered around one shared preprocessor. Every uploaded video first passes through this preprocessor, which standardizes the video and extracts common analysis artifacts such as release-frame data, ball path data, bat-contact data, and release-point information. After preprocessing is complete, the processed output is consumed by four feature modules: shot similarity, shot classification, bowler performance, and legality checker.

The deployment design is suitable for a staged or production-like environment provided that the API service and background worker are deployed together and connected to the same Redis and PostgreSQL services. Repository inspection confirms that the application includes working entry points for health checks, authentication, video submission, job-status retrieval, session results, presentation output, and RevenueCat webhook handling.

The main deployment considerations are operational rather than structural. In particular, the runtime environment must provide `ffmpeg`, preserve required ML model assets, and define how generated artifacts are retained beyond container-local storage. Subject to those controls, the current system is deployable and can support end-to-end asynchronous cricket-analysis workflows.

## Architecture Overview

The deployed CrickAI system follows a service-oriented backend pattern with a clear separation between synchronous API handling and asynchronous video-processing work.

Core architecture components:

- FastAPI application for HTTP endpoints and request validation
- Celery worker for background video-analysis execution
- Redis for Celery broker/backend, job-state storage, and SSE event publication
- PostgreSQL for persistent user and subscription-related data
- Local or mounted filesystem storage for generated artifacts and temporary video processing
- Embedded ML assets and model files required by the preprocessor and analysis modules

High-level processing flow:

1. A client authenticates and submits a video-analysis request to the FastAPI API.
2. The API validates calibration and feature inputs, stores initial job metadata, and queues a Celery task.
3. The Celery worker copies the uploaded media into a temporary workspace and runs the shared preprocessing pipeline first.
4. The preprocessor standardizes the uploaded video and extracts reusable artifacts needed by downstream analysis.
5. The four feature modules then consume the preprocessor output:
   `shot_similarity`, `shot_classifier`, `bowler_performance`, and `action_legality` (legality checker).
6. Each feature updates its own status and result in Redis as processing completes.
7. Clients retrieve status through polling or SSE and fetch presentation bundles or generated artifacts through result endpoints.

## 1. Deployment Summary

CrickAI is a FastAPI-based backend for cricket video analysis. The service accepts uploaded delivery videos, validates calibration data, dispatches background analysis jobs through Celery, stores job state in Redis, persists user and entitlement data in PostgreSQL, and exposes result, session, presentation, and authentication APIs.

The current deployment target is a containerized Python 3.11 service built from the repository `Dockerfile`. Background processing is separated from the HTTP API and is expected to run in at least one Celery worker process connected to the same Redis broker and result backend.

## 2. Application Scope in the Current Build

The inspected codebase currently exposes the following runtime capabilities:

- `POST /analyze` for single-video analysis jobs
- `POST /analyze/session` for multi-video session ingestion
- `GET /results/{job_id}` for job status retrieval
- `GET /results/{job_id}/events` for server-sent event status streaming
- `GET /results/{job_id}/artifacts/{feature_name}/{artifact_name}` for artifact access
- `GET /presentation/{job_id}` for presentation bundle generation
- `GET /sessions/{session_id}/results` for session result retrieval
- `POST /auth/register`, `/auth/login`, `/auth/refresh`, `/auth/logout`, and `GET /auth/me`
- `POST /webhooks/revenuecat` for subscription entitlement updates
- `GET /health` for service health checks

The deployed analysis pipeline includes one shared preprocessor and four feature modules.

Shared preprocessor:

- `preprocessor`

Feature modules:

- `bowler_performance`
- `action_legality`
- `shot_classifier`
- `shot_similarity`

Pipeline behavior:

1. The uploaded video is processed first by the shared preprocessor.
2. The preprocessor performs video normalization, batter detection, release-frame detection, ball tracking, and bat-contact extraction.
3. The preprocessor returns common video artifacts.
4. The four feature modules read those artifacts and produce their own outputs.
5. Each feature writes back its own job status and result.

## 3. Deployment Architecture

The current architecture inferred from the code is:

1. Clients authenticate against the FastAPI API and submit video-analysis requests.
2. The API stores calibration data and initializes job status in Redis.
3. The API copies uploaded video files to a temporary local path and queues a Celery task.
4. A Celery worker executes `app.tasks.process_video_job` and sends the video through the shared preprocessor first.
5. The preprocessor produces common artifacts including release frame, ball path, bat-contact frame, and release point.
6. The four feature modules then analyze that shared output:
   shot similarity, shot classification, bowler performance, and legality checker.
7. Each feature updates Redis with its own status and result, and job updates are published over Redis pub/sub for SSE consumers.
8. User data and entitlement state are stored in PostgreSQL through SQLAlchemy async models.
9. Generated image artifacts are written to the local filesystem under `CRICKAI_ARTIFACTS_DIR` or `/tmp/crickai-artifacts`.

## 4. Runtime Dependencies

The current code requires the following core services and runtime dependencies:

- Python `3.11`
- FastAPI and Uvicorn
- Celery
- Redis
- PostgreSQL
- `ffmpeg` available on `PATH` or configured via `FFMPEG_PATH`
- ML runtime packages from the enabled `uv` extras

Configured application integrations include:

- JWT-based authentication
- RevenueCat webhook verification
- Google Gemini / Google GenAI configuration
- Local artifact storage

## 5. Environment Configuration

The following environment variables are defined or consumed in the current code:

- `REDIS_URL`
- `REDIS_SOCKET_CONNECT_TIMEOUT`
- `REDIS_SOCKET_TIMEOUT`
- `DATABASE_URL`
- `S3_BUCKET`
- `AWS_REGION`
- `S3_PLAYBACK_PREFIX`
- `S3_PRESIGN_TTL_SECONDS`
- `JWT_SECRET`
- `JWT_ALGORITHM`
- `REVENUECAT_WEBHOOK_SECRET`
- `AI_PROVIDER`
- `AI_MODEL`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`
- `GOOGLE_GENAI_USE_VERTEXAI`
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `SHOT_SIMILARITY_REFERENCE_DIR`
- `SHOT_SIMILARITY_REFERENCE_PLAYER_NAME`
- `CRICKAI_ARTIFACTS_DIR`
- `MEDIAPIPE_POSE_TASK_PATH`
- `FFMPEG_PATH`

## 6. Container Build and Startup

The repository includes a Dockerfile that:

- uses `python:3.11-slim`
- installs dependencies with `uv sync --frozen`
- enables the extras `bat-contact`, `ml-base-gpu`, `bowler-performance`, `action-legality`, `shot-classifier`, and `shot-similarity`
- starts the API with `uv run uvicorn app.main:app --host 0.0.0.0 --port 8000`

The application also requires a separate worker startup command for background jobs. Based on the current code, the worker should run Celery against `app.celery_app.celery_app`.

Example worker command:

```bash
uv run celery -A app.celery_app.celery_app worker -Q analysis --loglevel=info
```

## 7. Deployment Procedure

The deployment process for the current implementation is:

1. Build the application container from the repository Dockerfile.
2. Provision Redis for job state, pub/sub events, calibration storage, and Celery broker/backend.
3. Provision PostgreSQL for user accounts, entitlement state, and quota tracking.
4. Supply runtime secrets and configuration through environment variables.
5. Start the FastAPI API container.
6. Start at least one Celery worker container or process using the same code version and environment.
7. Ensure `ffmpeg` and all required model assets are present in the runtime environment.
8. Run smoke tests against `/health`, authentication endpoints, and a known sample `POST /analyze` flow.

## 8. Verification Performed From Repository Inspection

This report is based on code and configuration inspection, not on a live production rollout. The following deployment-relevant elements were verified in the repository:

- FastAPI application entry point in `app/main.py`
- Celery task wiring in `app/tasks.py` and `app/celery_app.py`
- Redis-backed job status and SSE event publication in `app/storage/results.py`
- PostgreSQL-backed user persistence in `app/storage/database.py`
- Local artifact path handling in `app/storage/artifacts.py`
- Auth, session, presentation, result, and webhook endpoints under `app/api/`
- Docker build definition in `Dockerfile`
- Dependency and extras configuration in `pyproject.toml`

## 9. Known Deployment Risks and Gaps

The current codebase contains several operational points that should be recorded in the deployment report:

1. The README is partially outdated. It still states that Celery is not wired into the active request flow, but `app/api/analyze.py` now dispatches Celery tasks through `process_video_job.delay(...)`.
2. The Dockerfile does not install system packages such as `ffmpeg`. Since the preprocessor requires `ffmpeg`, runtime images must provide it another way or analysis jobs will fail.
3. Playback video storage is not currently backed by S3 in the inspected code. `app/storage/video.py` stores a local path in Redis, even though S3-related environment variables exist. This should be treated as a gap between configuration intent and actual implementation.
4. Generated result artifacts are stored on the local filesystem under `/tmp/crickai-artifacts` by default. In multi-instance or ephemeral-container deployments, artifact availability may be lost unless persistent shared storage is mounted.
5. Database schema initialization is performed through `Base.metadata.create_all()` on startup. This is acceptable for early-stage deployment but is not a substitute for a managed migration workflow in production.
6. The container currently installs `ml-base-gpu`, which may not be compatible with all deployment environments and may be unnecessary on CPU-only infrastructure.
7. Several ML features depend on local model assets checked into the repository. Deployment packaging must preserve those assets exactly.

## 10. Recommended Production Controls

For a stable deployment, the following controls are recommended:

- run the API and Celery worker as separate services
- attach persistent storage for artifacts if artifact URLs must remain available after job completion
- install and validate `ffmpeg` in the final runtime image
- add explicit startup checks for Redis, PostgreSQL, and critical ML assets
- replace startup `create_all()` with controlled schema migrations
- add central logging and worker failure alerting
- define retention and cleanup policies for temporary uploads and generated artifacts

## 11. Rollback Approach

If a deployment introduces regressions, rollback should be performed by:

1. redeploying the previous API container image
2. redeploying the matching previous Celery worker image
3. preserving Redis and PostgreSQL connectivity and credentials
4. validating `/health`, authentication, and one known analysis request after rollback

## 12. Conclusion

The CrickAI backend is deployable as a containerized FastAPI and Celery system with Redis and PostgreSQL dependencies. The repository contains the necessary application entry points, worker wiring, and runtime configuration to support deployment. However, production readiness still depends on resolving the current operational gaps around `ffmpeg` installation, artifact persistence, and the mismatch between configured S3 settings and the present local-path playback implementation.
