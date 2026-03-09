from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import bowling

app = FastAPI(title="Bowler Legality Checker API", version="0.1.0")

# ✅ CORS settings for all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins temporarily for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(bowling.router, prefix="/api/bowling", tags=["bowling"])

@app.get("/")
def root():
    return {"message": "Bowler Legality Checker API running"}
