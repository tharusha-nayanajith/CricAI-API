from fastapi import FastAPI

from app.api import analyze, auth, history, presentation, results, sessions, webhooks
from app.storage.database import dispose_database, init_database

app = FastAPI(title="CrickAI API")
app.include_router(analyze.router)
app.include_router(auth.router)
app.include_router(history.router)
app.include_router(presentation.router)
app.include_router(results.router)
app.include_router(sessions.router)
app.include_router(webhooks.router)


@app.on_event("startup")
async def startup() -> None:
    await init_database()


@app.on_event("shutdown")
async def shutdown() -> None:
    await dispose_database()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
