FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS uv

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./

RUN uv sync \
  --extra bat-contact \
  --extra ml-base-gpu \
  --extra bowler-performance \
  --extra action-legality \
  --extra shot-classifier \
  --extra shot-similarity \
  --no-dev --frozen

COPY . .

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
