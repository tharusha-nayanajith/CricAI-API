from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions import AuthenticationError
from app.modules.users.models import RevenueCatWebhookEvent, RevenueCatWebhookResult
from app.modules.users.service import UserService
from app.storage.database import get_db_session

router = APIRouter(prefix="/webhooks", tags=["webhooks"])
_user_service = UserService()


@router.post("/revenuecat", response_model=RevenueCatWebhookResult)
async def revenuecat_webhook(
    payload: RevenueCatWebhookEvent,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    authorization: Annotated[str | None, Header()] = None,
) -> RevenueCatWebhookResult:
    try:
        return await _user_service.handle_revenuecat_webhook(
            session,
            authorization=authorization,
            payload=payload.model_dump(),
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
