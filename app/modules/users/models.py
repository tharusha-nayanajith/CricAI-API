from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

TierName = Literal["basic", "coach", "academy"]
EntitlementStatus = Literal["inactive", "active", "canceled", "expired"]


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=1, max_length=255)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    access_token_expires_in: int = 900
    refresh_token_expires_in: int = 30 * 24 * 60 * 60


class AccessTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    access_token_expires_in: int = 900


class UserProfile(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    full_name: str
    created_at: datetime
    is_active: bool
    revenuecat_customer_id: str | None
    entitlement_status: EntitlementStatus
    entitlement_expires_at: datetime | None
    current_tier: TierName
    clips_used_this_month: int
    quota_reset_at: datetime


class RevenueCatWebhookEvent(BaseModel):
    event: dict


class RevenueCatWebhookResult(BaseModel):
    user_id: UUID
    event_type: str
    entitlement_status: EntitlementStatus
    current_tier: TierName
