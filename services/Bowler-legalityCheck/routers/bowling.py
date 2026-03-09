from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
import traceback
import cv2
import base64
import json
import re
from groq import Groq
from pydantic import BaseModel
from typing import Optional
from services.bowling_service import get_model_and_infer, get_model_and_infer_video_with_frame

router = APIRouter()
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Groq client ──────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")


@router.post("/infer")
async def infer_bowling_image(file: UploadFile = File(...)):
    """Upload and analyze a bowling image"""
    print(f"📥 Received file: {file.filename}, content_type: {file.content_type}")
    
    if not file:
        raise HTTPException(400, "No file uploaded")
    
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(400, f"Invalid file type: {file.content_type}. Allowed: JPG/PNG")
    
    temp_name = uuid.uuid4().hex + "_" + (file.filename or "upload.jpg")
    temp_path = os.path.join(UPLOAD_DIR, temp_name)
    
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"💾 Saved to: {temp_path}")
        
        result = get_model_and_infer(temp_path)
        prob = result.get("prob_illegal", None)
        
        if prob is None:
            return {"status": "error", "message": "Pose not detected", "prob_illegal": None}
        
        label = "illegal" if prob >= 0.5 else "legal"
        return {"status": "ok", "label": label, "prob_illegal": float(prob)}
        
    except Exception as e:
        print(f"❌ Error processing: {str(e)}")
        traceback.print_exc()
        raise HTTPException(500, f"Processing error: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/infer-video")
async def infer_bowling_video(
    file: UploadFile = File(...),
    detection_method: str = "wrist_velocity"
):
    """Upload and analyze a bowling video"""
    print(f"📥 Received video: {file.filename}, content_type: {file.content_type}")
    
    if not file:
        raise HTTPException(400, "No file uploaded")
    
    allowed_types = [
        "video/mp4", "video/avi", "video/mov",
        "video/quicktime", "video/x-msvideo", "video/x-matroska"
    ]
    
    if file.content_type not in allowed_types:
        print(f"⚠️ Warning: Unexpected content type {file.content_type}")
    
    original_filename = file.filename or "upload.mp4"
    file_ext = os.path.splitext(original_filename)[1].lower()
    if not file_ext:
        file_ext = ".mp4"
    
    temp_name = uuid.uuid4().hex + file_ext
    temp_path = os.path.join(UPLOAD_DIR, temp_name)
    
    try:
        print(f"💾 Saving video to: {temp_path}")
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        if not os.path.exists(temp_path):
            raise Exception("Failed to save uploaded file")
        
        file_size = os.path.getsize(temp_path)
        print(f"✅ Video saved: {file_size} bytes")
        
        valid_methods = ["wrist_velocity", "wrist_deceleration", "arm_extension"]
        if detection_method not in valid_methods:
            print(f"⚠️ Invalid method '{detection_method}', using 'wrist_velocity'")
            detection_method = "wrist_velocity"
        
        print(f"🔄 Processing video with method: {detection_method}")
        result = get_model_and_infer_video_with_frame(temp_path, detection_method)
        
        print(f"📊 Processing result status: {result.get('status')}")
        
        if result.get("status") != "success":
            error_msg = result.get("error", "Could not process video")
            print(f"❌ Processing failed: {error_msg}")
            return {"status": "error", "message": error_msg, "prob_illegal": None}
        
        prob = result.get("prob_illegal")
        frame_idx = result.get("frame_index")
        keypoints = result.get("keypoints")
        release_frame = result.get("release_frame")
        
        if prob is None:
            print(f"❌ No probability returned")
            return {
                "status": "error",
                "message": "Could not detect pose in release frame",
                "frame_index": frame_idx,
                "prob_illegal": None
            }
        
        label = "illegal" if prob >= 0.5 else "legal"
        
        response = {
            "status": "ok",
            "label": label,
            "prob_illegal": float(prob),
            "frame_index": int(frame_idx) if frame_idx is not None else None,
            "detection_method": detection_method
        }
        
        if keypoints is not None:
            try:
                response["keypoints"] = keypoints.tolist() if hasattr(keypoints, 'tolist') else list(keypoints)
            except Exception as e:
                print(f"⚠️ Could not serialize keypoints: {e}")
        
        if release_frame is not None:
            try:
                _, buffer = cv2.imencode('.jpg', release_frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                response["release_frame_base64"] = frame_base64
                print(f"✅ Release frame encoded (size: {len(frame_base64)} chars)")
            except Exception as e:
                print(f"⚠️ Could not encode frame: {e}")
        
        print(f"✅ Analysis complete: {label} (prob={prob:.2f})")
        return response
        
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        raise HTTPException(500, error_msg)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"🗑️ Cleaned up: {temp_path}")
            except Exception as e:
                print(f"⚠️ Failed to cleanup {temp_path}: {e}")


# ── AI Coaching Endpoint ──────────────────────────────

class CoachingRequest(BaseModel):
    is_legal: bool
    confidence: float
    detection_method: str
    elbow_angle: Optional[float] = None


@router.post("/coaching")
async def get_coaching(req: CoachingRequest):
    """Generate personalized AI coaching using Groq (free)"""
    print(f"🤖 Coaching request: legal={req.is_legal}, confidence={req.confidence:.2f}")

    confidence_pct = f"{req.confidence * 100:.1f}"
    angle_info = (
        f"The measured elbow angle at release was {req.elbow_angle:.1f} degrees."
        if req.elbow_angle is not None
        else f"The exact elbow angle was not captured (detection method: {req.detection_method.replace('_', ' ')})."
    )
    verdict_str = "LEGAL" if req.is_legal else "ILLEGAL"

    prompt = f"""You are a professional cricket bowling coach AI. Analyze this bowling legality result and provide personalized coaching.

RESULT DATA:
- Verdict: {verdict_str}
- AI Confidence: {confidence_pct}%
- {angle_info}
- ICC Rule 24.3 limit: 15 degrees elbow straightening maximum

{"The bowler PASSED the legality check. Provide steps to MAINTAIN and IMPROVE their legal action." if req.is_legal else "The bowler FAILED the legality check. Their elbow exceeded the 15 degree limit. Provide specific corrective steps."}

Respond ONLY with valid JSON (no markdown, no backticks, no extra text) in this exact structure:
{{
  "verdict": "{verdict_str}",
  "summary": "1-2 sentence personalized summary based on their specific confidence score and angle",
  "targetAngle": "e.g. Keep below 12 degrees for safety margin",
  "estimatedWeeks": "e.g. 3-4 weeks with consistent practice",
  "steps": [
    {{
      "step": 1,
      "title": "Short action title",
      "description": "2-3 sentence specific instruction based on their data",
      "targetAngle": "optional angle target e.g. 10-12 degrees",
      "drillName": "optional drill name e.g. Wall Shadow Drill",
      "frequency": "optional e.g. 3x per week, 20 mins"
    }}
  ]
}}

{"Provide 3-4 maintenance steps." if req.is_legal else "Provide 4-5 corrective steps in order of importance."}
Base all advice on the specific confidence score of {confidence_pct}% and the angle data provided."""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        message = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.choices[0].message.content
        clean = re.sub(r'```json|```', '', raw).strip()
        result = json.loads(clean)
        print(f"✅ Coaching generated: {len(result.get('steps', []))} steps")
        return result

    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        raise HTTPException(500, f"AI response parse error: {str(e)}")
    except Exception as e:
        print(f"❌ Coaching error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(500, f"Coaching generation failed: {str(e)}")