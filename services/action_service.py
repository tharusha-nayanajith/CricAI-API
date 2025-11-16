from models.schemas import ActionValidateResponse
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def validate_action(action_name: str, features: list[float]) -> ActionValidateResponse:
    errors: list[str] = []
    if not action_name:
        errors.append("Missing action name")
    if not features:
        errors.append("No features provided")
    if any(not isinstance(x, (int, float)) for x in features):
        errors.append("Features must be numeric")

    valid = len(errors) == 0
    if valid:
        logger.debug(f"Action '{action_name}' validated successfully")
    else:
        logger.info(f"Action '{action_name}' validation errors: {errors}")

    return ActionValidateResponse(valid=valid, errors=errors)
