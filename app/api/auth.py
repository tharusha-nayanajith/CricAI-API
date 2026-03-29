from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.exceptions import AuthenticationError, ConflictError
from app.modules.users.models import (
    LoginRequest,
    LogoutRequest,
    RefreshTokenRequest,
    RegisterRequest,
    TokenPair,
    UserProfile,
)
from app.modules.users.service import UserService
from app.storage.database import get_db_session

router = APIRouter(prefix="/auth", tags=["auth"])
_user_service = UserService()


@router.post("/register", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> UserProfile:
    try:
        return await _user_service.register_user(
            session,
            email=payload.email,
            password=payload.password,
            full_name=payload.full_name,
        )
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except ConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/login", response_model=TokenPair)
async def login(
    payload: LoginRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> TokenPair:
    try:
        return await _user_service.login_user(session, payload)
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


@router.post("/refresh")
async def refresh(
    payload: RefreshTokenRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> dict[str, str | int]:
    try:
        token = await _user_service.refresh_access_token(session, payload)
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    return token.model_dump()


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(payload: LogoutRequest) -> None:
    try:
        await _user_service.logout(payload.refresh_token)
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


@router.get("/me", response_model=UserProfile)
async def me(current_user: Annotated[UserProfile, Depends(get_current_user)]) -> UserProfile:
    return current_user
