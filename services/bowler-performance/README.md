# Cricket Bowling Analysis API

A production-ready FastAPI microservice for analyzing cricket bowling performance using computer vision and machine learning.

## Features (Planned)

- 🎥 **Video Analysis**: Upload bowling videos for automated analysis
- 🎯 **Ball Tracking**: YOLO-based object detection and trajectory tracking
- 📊 **Performance Metrics**: Calculate speed, line, length, swing, and spin
- 📐 **Camera Calibration**: Automatic stump detection for perspective correction
- 🚀 **Fast & Scalable**: Built with FastAPI for high performance

## Project Structure

```
bowler-performance/
│
├── app/
│   ├── main.py                # FastAPI application entry point
│   ├── api/v1/                # API endpoints (versioned)
│   ├── core/                  # Configuration and settings
│   ├── services/              # Business logic layer
│   ├── models/                # Data models and ML weights
│   └── utils/                 # Helper functions
│
├── .env                       # Environment variables
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env` file and adjust settings as needed:
- `MODEL_PATH`: Path to your YOLO model weights
- `PORT`: Server port (default: 8000)
- `UPLOAD_DIR`: Directory for uploaded files

### 4. Download YOLO Model

Place your YOLO model weights in `app/models/ml/`:

```bash
# Example: Download YOLOv8 nano model
# You'll need to train a custom model for cricket-specific objects
```

### 5. Run the Server

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or use the main.py directly
python -m app.main
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc

## API Endpoints

### Health Check
```
GET /api/v1/health
```

### Video Analysis
```
POST /api/v1/analysis/analyze-video
Content-Type: multipart/form-data

Body: video file
```

### Frame Analysis
```
POST /api/v1/analysis/analyze-frame
Content-Type: multipart/form-data

Body: image file
```

### Calibration
```
POST /api/v1/calibration/calibrate-stumps
Content-Type: multipart/form-data

Body: reference image with stumps
```

## Development Status

⚠️ **Current Status**: Boilerplate structure created

**Implemented**:
- ✅ Project structure
- ✅ FastAPI application setup
- ✅ API endpoint placeholders
- ✅ Configuration management
- ✅ Pydantic schemas

**TODO**:
- ⏳ YOLO model integration
- ⏳ Video processing pipeline
- ⏳ Ball trajectory calculation
- ⏳ Bowling metrics computation
- ⏳ Camera calibration logic
- ⏳ Unit tests
- ⏳ Docker containerization

## Technology Stack

- **Framework**: FastAPI 0.109+
- **ML/CV**: Ultralytics YOLO, OpenCV
- **Validation**: Pydantic v2
- **Server**: Uvicorn (ASGI)

## Contributing

This is a boilerplate structure. Implement the business logic in:
- `app/services/vision_engine.py` - YOLO detection
- `app/services/trajectory.py` - Ball tracking and metrics
- `app/api/v1/endpoints/` - API endpoint logic

## License

[Your License Here]

## Contact

[Your Contact Information]
