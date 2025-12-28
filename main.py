from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
import uvicorn

from models.schemas import HealthResponse
from routers import bowling as bowling_router
from routers import batting as batting_router
from routers import action as action_router
from routers import similarity as similarity_router
from routers import stance_similarity as stance_similarity_router


app = FastAPI(title="CricAI Coach API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(bowling_router.router)
app.include_router(batting_router.router)
app.include_router(action_router.router)
app.include_router(similarity_router.router)
app.include_router(stance_similarity_router.router)
@app.get("/")
async def root():
    return {
        "message": "Welcome to CricAI API",
        "version": "1.0.0",
        "endpoints": {
            "batting": "/batting",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=time.time())

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )