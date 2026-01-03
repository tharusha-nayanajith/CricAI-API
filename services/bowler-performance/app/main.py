"""
FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.api import api_router
from app.core.config import settings


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url=f"{settings.API_V1_STR}/docs",
        redoc_url=f"{settings.API_V1_STR}/redoc",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API router
    app.include_router(api_router, prefix=settings.API_V1_STR)
    
    # Startup event handler
    @app.on_event("startup")
    async def startup_event():
        """Initialize database and load YOLO model on startup"""
        print("Initializing application...")
        
        # Create database tables
        from app.db.database import init_db
        init_db()
        print("Database initialized")
        
        # Load YOLO model
        from app.services.vision_engine import vision_engine
        try:
            vision_engine.load_model()
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load YOLO model: {e}")
            print("Model will need to be loaded manually or endpoint will fail")

    return app


app = create_application()


@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Cricket Bowling Analysis API",
        "version": "0.1.0",
        "docs": f"{settings.API_V1_STR}/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True
    )
