from fastapi import FastAPI

from app.api import analyze, results

app = FastAPI(title="CrickAI API")
app.include_router(analyze.router)
app.include_router(results.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
