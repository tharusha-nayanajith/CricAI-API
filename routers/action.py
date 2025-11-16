from fastapi import APIRouter

from models.schemas import ActionValidateRequest, ActionValidateResponse
from services.action_service import validate_action


router = APIRouter(prefix="/api/action", tags=["action"])


@router.post("/validate", response_model=ActionValidateResponse)
def validate(req: ActionValidateRequest) -> ActionValidateResponse:
    return validate_action(req.action_name, req.features)
