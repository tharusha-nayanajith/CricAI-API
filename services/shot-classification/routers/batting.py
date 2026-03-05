"""
Batting API Router  — with Field Setting + Shot Outcome
========================================================
New endpoints added:
  POST /batting/analyze-shot-with-field   — video + field placement → full analysis + outcome
  GET  /batting/default-field             — returns the default fielder positions
  GET  /batting/shot-zones                — returns hitting zones for all shot types (for UI)

Existing endpoints are UNCHANGED so nothing breaks.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse
import os
import json
import tempfile
from typing import Optional, List

from features.SHOT_CLASSIFICATION_SYSTEM.utils.config import (
    SUPPORTED_VIDEO_EXTENSIONS,
    supported_extensions_str,
)
from services.batting_service import get_batting_service

# ── NEW imports ───────────────────────────────────────────────────────────────
from features.SHOT_CLASSIFICATION_SYSTEM.utils.shot_outcome.field_schemas import (
    FieldSetting,
    FielderPosition,
)
from features.SHOT_CLASSIFICATION_SYSTEM.utils.shot_outcome.field_geometry import (
    SHOT_ZONES,
    polar_to_xy,
)
from features.SHOT_CLASSIFICATION_SYSTEM.utils.shot_outcome.shot_outcome_predictor import DEFAULT_FIELDERS


router = APIRouter(prefix="/batting", tags=["Batting Analysis"])


# ─────────────────────────────────────────────────────────────────────────────
# Existing endpoints — completely unchanged
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/shot-types")
async def get_shot_types():
    try:
        service    = get_batting_service()
        shot_types = service.get_shot_types()
        return {"success": True, "shot_types": shot_types,
                "message": "Available shot types retrieved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-shot")
async def analyze_shot(
    video: UploadFile = File(..., description="Cricket shot video"),
    intended_shot: str = Form(..., description="User's intended shot type"),
):
    temp_video_path = None
    try:
        if not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Invalid file type.")
        ext = os.path.splitext(video.filename)[1].lower()
        if ext not in SUPPORTED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid extension '{ext}'. Supported: {supported_extensions_str()}"
            )
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await video.read())
            temp_video_path = tmp.name

        service = get_batting_service()
        result  = service.analyze_shot(temp_video_path, intended_shot)
        return {"success": True, "data": result,
                "message": "Shot analysis completed successfully"}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)


@router.post("/batch-analyze")
async def batch_analyze_shots(
    videos: list[UploadFile] = File(...),
    intended_shots: str = Form(...),
):
    temp_paths = []
    try:
        intended_shot_list = [s.strip() for s in intended_shots.split(',')]
        if len(videos) != len(intended_shot_list):
            raise HTTPException(status_code=400,
                                detail="Number of videos must match intended shots")

        service = get_batting_service()
        results = []

        for video, intended_shot in zip(videos, intended_shot_list):
            if not video.content_type.startswith('video/'):
                raise HTTPException(status_code=400, detail="Invalid file type.")
            ext = os.path.splitext(video.filename)[1].lower()
            if ext not in SUPPORTED_VIDEO_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid extension '{ext}'. Supported: {supported_extensions_str()}"
                )
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(await video.read())
                temp_paths.append(tmp.name)
            result = service.analyze_shot(temp_paths[-1], intended_shot)
            results.append({"filename": video.filename, "analysis": result})

        avg_score    = sum(r['analysis']['intent_score'] for r in results) / len(results)
        correct_count = sum(1 for r in results if r['analysis']['is_correct'])
        return {
            "success": True, "results": results,
            "summary": {
                "total_shots": len(results),
                "average_intent_score": round(avg_score, 2),
                "correct_predictions": correct_count,
                "accuracy": round((correct_count / len(results)) * 100, 2),
            },
            "message": "Batch analysis completed successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")
    finally:
        for p in temp_paths:
            if os.path.exists(p):
                os.remove(p)


@router.get("/health")
async def health_check():
    try:
        get_batting_service()
        return {"success": True, "status": "healthy",
                "message": "Batting service is running"}
    except Exception as e:
        return {"success": False, "status": "unhealthy", "message": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# NEW endpoint 1 — analyze shot WITH field setting
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/analyze-shot-with-field")
async def analyze_shot_with_field(
    video: UploadFile = File(..., description="Cricket shot video"),
    intended_shot: str = Form(..., description="User's intended shot type"),
    field_setting: str = Form(
        ...,
        description=(
            "JSON string of field setting. "
            'Example: {"fielders":[{"role":"cover","x":0.6,"y":0.55},...], '
            '"batting_side":"right","match_format":"t20"}'
        ),
    ),
):
    """
    Analyze a cricket shot AND predict the run outcome based on field placement.

    The frontend should:
      1. Show a field-placement UI where the user drags 9-11 fielders
      2. Serialize the positions as JSON and send as `field_setting` form field
      3. Upload the video as `video`
      4. Use the `shot_outcome` key in the response for the result overlay

    Response includes everything from `/analyze-shot` PLUS:
    ```json
    {
      "shot_outcome": {
        "outcome":          "4",
        "runs":             4,
        "confidence":       0.82,
        "outcome_reason":   "Ball races to the boundary through the gap.",
        "elevation":        "low",
        "power_rating":     0.78,
        "timing_rating":    0.71,
        "landing_zone":     {"x": 0.52, "y": 0.75, "angle_deg": 34.8, "radius": 0.91},
        "fan_landing_points": [{"x": ..., "y": ...}, ...],
        "ball_trajectory":  {"pre_contact": [...], "post_contact": [...]},
        "fielder_involved": {"role": "cover", "x": 0.6, "y": 0.55, "distance_to_ball": 0.18},
        "shot_zone":        {"primary_angle": 35, "angle_spread": 20, "primary_radius": 0.92}
      }
    }
    ```
    """
    temp_video_path = None
    try:
        # ── Validate video ────────────────────────────────────────────────────
        if not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Upload a video.")
        ext = os.path.splitext(video.filename)[1].lower()
        if ext not in SUPPORTED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid extension '{ext}'. Supported: {supported_extensions_str()}"
            )

        # ── Parse field setting ───────────────────────────────────────────────
        try:
            field_dict = json.loads(field_setting)
            parsed_field = FieldSetting(**field_dict)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid field_setting JSON: {str(e)}"
            )

        # ── Save video ────────────────────────────────────────────────────────
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await video.read())
            temp_video_path = tmp.name

        # ── Analyse ───────────────────────────────────────────────────────────
        service = get_batting_service()
        result  = service.analyze_shot(
            temp_video_path,
            intended_shot,
            field_setting=parsed_field,   # ← pass to service
        )

        return {
            "success": True,
            "data":    result,
            "message": "Shot analysis with outcome prediction completed successfully",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)


# ─────────────────────────────────────────────────────────────────────────────
# NEW endpoint 2 — default field positions
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/default-field")
async def get_default_field(
    match_format: str = "t20",
    batting_side: str = "right",
):
    """
    Returns the default fielder positions for the given match format.
    The frontend uses this to pre-populate the field-placement UI.

    Returns list of:
    ```json
    {"role": "cover", "x": 0.6, "y": 0.55}
    ```
    Coordinates are normalised (−1..1). x+ = off side, y+ = straight.
    """
    fielders = [
        {"role": role, "x": x, "y": y}
        for role, x, y in DEFAULT_FIELDERS
    ]

    # Mirror for left-hander
    if batting_side.lower() == "left":
        fielders = [
            {"role": f["role"], "x": -f["x"], "y": f["y"]}
            for f in fielders
        ]

    return {
        "success":      True,
        "fielders":     fielders,
        "batting_side": batting_side,
        "match_format": match_format,
        "total":        len(fielders),
        "coordinate_info": {
            "x": "negative = leg side, positive = off side",
            "y": "positive = straight / toward bowler, negative = behind wicket",
            "scale": "1.0 = boundary rope",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# NEW endpoint 3 — shot zones (for UI heatmap / field overlay)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/shot-zones")
async def get_shot_zones(shot_type: Optional[str] = None):
    """
    Returns the statistical hitting zone(s) for shot types.
    The frontend can use this to show a coloured overlay on the field
    BEFORE the user uploads a video, so they understand where each
    shot typically goes.

    Query param `shot_type` filters to a single shot (optional).
    """
    def _zone_to_dict(name: str, zone) -> dict:
        cx, cy = polar_to_xy(zone.primary_angle, zone.primary_radius)
        return {
            "shot_type":      name,
            "primary_angle":  zone.primary_angle,
            "angle_spread":   zone.angle_spread,
            "primary_radius": zone.primary_radius,
            "radius_spread":  zone.radius_spread,
            "elevation":      zone.elevation,
            "center_x":       round(cx, 3),
            "center_y":       round(cy, 3),
        }

    if shot_type:
        shot_lower = shot_type.lower().strip()
        if shot_lower not in SHOT_ZONES:
            raise HTTPException(
                status_code=404,
                detail=f"Shot type '{shot_type}' not found. "
                       f"Available: {list(SHOT_ZONES.keys())}"
            )
        return {
            "success": True,
            "zone":    _zone_to_dict(shot_lower, SHOT_ZONES[shot_lower]),
        }

    zones = [_zone_to_dict(name, z) for name, z in SHOT_ZONES.items()]
    return {
        "success": True,
        "zones":   zones,
        "total":   len(zones),
    }