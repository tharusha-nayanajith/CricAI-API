from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from jose import JWTError, jwt
from loguru import logger
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_redis, get_settings
from app.exceptions import AuthenticationError, AuthorizationError, ConflictError
from app.modules.users.models import (
    AccessTokenResponse,
    LoginRequest,
    RefreshTokenRequest,
    RevenueCatWebhookResult,
    TierName,
    TokenPair,
    UserProfile,
)
from app.storage.database import UserRecord

ACCESS_TOKEN_EXPIRES_MINUTES = 15
REFRESH_TOKEN_EXPIRES_DAYS = 30
PASSWORD_HASH_ITERATIONS = 600_000
REFRESH_TOKEN_PREFIX = "auth:refresh:"
UNLIMITED_QUOTA = -1
TIER_LIMITS: dict[TierName, int] = {
    "basic": 100,
    "coach": 500,
    "academy": UNLIMITED_QUOTA,
}


class UserService:
    async def register_user(
        self,
        session: AsyncSession,
        *,
        email: str,
        password: str,
        full_name: str,
    ) -> UserProfile:
        normalized_email = email.strip().lower()
        _validate_email(normalized_email)
        existing = await self._get_user_by_email(session, normalized_email)
        if existing is not None:
            raise ConflictError("A user with this email already exists.")

        user = UserRecord(
            email=normalized_email,
            hashed_password=_hash_password(password),
            full_name=full_name.strip(),
            quota_reset_at=_next_quota_reset(datetime.now(UTC)),
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        logger.info("Registered user {}", user.email)
        return UserProfile.model_validate(user)

    async def login_user(
        self,
        session: AsyncSession,
        payload: LoginRequest,
    ) -> TokenPair:
        user = await self._authenticate_user(session, payload.email, payload.password)
        return await self._issue_token_pair(user)

    async def refresh_access_token(
        self,
        session: AsyncSession,
        payload: RefreshTokenRequest,
    ) -> AccessTokenResponse:
        claims = self._decode_token(payload.refresh_token, expected_type="refresh")
        jti = claims.get("jti")
        user_id = claims.get("sub")
        if not isinstance(jti, str) or not isinstance(user_id, str):
            raise AuthenticationError("Invalid refresh token.")

        redis = get_redis()
        stored_user_id = await redis.get(_refresh_token_key(jti))
        if stored_user_id != user_id:
            raise AuthenticationError("Refresh token has been invalidated.")

        user = await self.get_user_by_id(session, UUID(user_id))
        if user is None or not user.is_active:
            raise AuthenticationError("User is not available.")

        return AccessTokenResponse(
            access_token=self._create_access_token(user.id),
        )

    async def logout(self, refresh_token: str) -> None:
        claims = self._decode_token(refresh_token, expected_type="refresh")
        jti = claims.get("jti")
        if not isinstance(jti, str):
            raise AuthenticationError("Invalid refresh token.")

        redis = get_redis()
        await redis.delete(_refresh_token_key(jti))

    async def get_user_by_id(self, session: AsyncSession, user_id: UUID) -> UserRecord | None:
        return await session.get(UserRecord, user_id)

    async def get_current_user_from_access_token(
        self,
        session: AsyncSession,
        token: str,
    ) -> UserProfile:
        claims = self._decode_token(token, expected_type="access")
        subject = claims.get("sub")
        if not isinstance(subject, str):
            raise AuthenticationError("Invalid access token.")

        user = await self.get_user_by_id(session, UUID(subject))
        if user is None or not user.is_active:
            raise AuthenticationError("User is not available.")
        return UserProfile.model_validate(user)

    async def require_entitlement(
        self,
        session: AsyncSession,
        user_id: UUID,
    ) -> UserProfile:
        user = await self.get_user_by_id(session, user_id)
        if user is None or not user.is_active:
            raise AuthenticationError("User is not available.")

        active = _has_active_entitlement(user.entitlement_status, user.entitlement_expires_at)
        if not active:
            raise AuthorizationError("An active subscription is required.")
        return UserProfile.model_validate(user)

    async def enforce_clip_quota(
        self,
        session: AsyncSession,
        user_id: UUID,
    ) -> UserProfile:
        user = await self.get_user_by_id(session, user_id)
        if user is None:
            raise AuthenticationError("User is not available.")

        now = datetime.now(UTC)
        if user.quota_reset_at <= now:
            user.clips_used_this_month = 0
            user.quota_reset_at = _next_quota_reset(now)

        limit = TIER_LIMITS.get(user.current_tier, TIER_LIMITS["basic"])
        if limit != UNLIMITED_QUOTA and user.clips_used_this_month >= limit:
            raise AuthorizationError("Monthly clip quota exceeded for the current tier.")

        user.clips_used_this_month += 1
        await session.commit()
        await session.refresh(user)
        return UserProfile.model_validate(user)

    async def handle_revenuecat_webhook(
        self,
        session: AsyncSession,
        *,
        authorization: str | None,
        payload: dict,
    ) -> RevenueCatWebhookResult:
        self._verify_revenuecat_authorization(authorization)
        event = payload.get("event", payload)
        if not isinstance(event, dict):
            raise AuthenticationError("RevenueCat payload is invalid.")

        event_type = str(event.get("type", "")).upper()
        customer_id = _extract_revenuecat_customer_id(event)
        user = await self._resolve_user_for_revenuecat_event(session, customer_id)
        if user is None:
            raise AuthenticationError("RevenueCat user could not be resolved.")

        if user.revenuecat_customer_id is None:
            user.revenuecat_customer_id = customer_id

        if event_type in {"INITIAL_PURCHASE", "RENEWAL"}:
            user.entitlement_status = "active"
        elif event_type == "CANCELLATION":
            user.entitlement_status = "canceled"
        elif event_type == "EXPIRATION":
            user.entitlement_status = "expired"
        else:
            raise AuthenticationError(f"Unsupported RevenueCat event type: {event_type}")

        user.current_tier = _derive_tier(event)
        user.entitlement_expires_at = _extract_expiration(event)
        await session.commit()
        await session.refresh(user)
        logger.info("Processed RevenueCat event {} for user {}", event_type, user.id)
        return RevenueCatWebhookResult(
            user_id=user.id,
            event_type=event_type,
            entitlement_status=user.entitlement_status,
            current_tier=user.current_tier,
        )

    async def _authenticate_user(
        self,
        session: AsyncSession,
        email: str,
        password: str,
    ) -> UserRecord:
        normalized_email = email.strip().lower()
        _validate_email(normalized_email)
        user = await self._get_user_by_email(session, normalized_email)
        if user is None or not _verify_password(password, user.hashed_password):
            raise AuthenticationError("Invalid email or password.")
        if not user.is_active:
            raise AuthenticationError("User account is inactive.")
        return user

    async def _issue_token_pair(self, user: UserRecord) -> TokenPair:
        refresh_jti = str(uuid4())
        access_token = self._create_access_token(user.id)
        refresh_token = self._create_refresh_token(user.id, refresh_jti)
        redis = get_redis()
        await redis.set(
            _refresh_token_key(refresh_jti),
            str(user.id),
            ex=REFRESH_TOKEN_EXPIRES_DAYS * 24 * 60 * 60,
        )
        return TokenPair(access_token=access_token, refresh_token=refresh_token)

    async def _get_user_by_email(self, session: AsyncSession, email: str) -> UserRecord | None:
        stmt: Select[tuple[UserRecord]] = select(UserRecord).where(UserRecord.email == email)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def _resolve_user_for_revenuecat_event(
        self,
        session: AsyncSession,
        customer_id: str,
    ) -> UserRecord | None:
        stmt: Select[tuple[UserRecord]] = select(UserRecord).where(
            UserRecord.revenuecat_customer_id == customer_id
        )
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        if user is not None:
            return user

        try:
            user_id = UUID(customer_id)
        except ValueError:
            return None
        return await self.get_user_by_id(session, user_id)

    def _create_access_token(self, user_id: UUID) -> str:
        now = datetime.now(UTC)
        settings = get_settings()
        payload = {
            "sub": str(user_id),
            "type": "access",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=ACCESS_TOKEN_EXPIRES_MINUTES)).timestamp()),
        }
        return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

    def _create_refresh_token(self, user_id: UUID, jti: str) -> str:
        now = datetime.now(UTC)
        settings = get_settings()
        payload = {
            "sub": str(user_id),
            "type": "refresh",
            "jti": jti,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(days=REFRESH_TOKEN_EXPIRES_DAYS)).timestamp()),
        }
        return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

    def _decode_token(self, token: str, *, expected_type: str) -> dict:
        settings = get_settings()
        try:
            payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        except JWTError as exc:
            raise AuthenticationError("Token is invalid or expired.") from exc
        if payload.get("type") != expected_type:
            raise AuthenticationError("Token type is invalid.")
        return payload

    def _verify_revenuecat_authorization(self, authorization: str | None) -> None:
        expected = get_settings().revenuecat_webhook_secret
        provided = (authorization or "").strip()
        if provided.startswith("Bearer "):
            provided = provided.removeprefix("Bearer ").strip()
        if not expected or not hmac.compare_digest(provided, expected):
            raise AuthenticationError("RevenueCat authorization is invalid.")


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )
    return (
        "pbkdf2_sha256"
        f"${PASSWORD_HASH_ITERATIONS}"
        f"${base64.urlsafe_b64encode(salt).decode('utf-8')}"
        f"${base64.urlsafe_b64encode(derived).decode('utf-8')}"
    )


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt_b64, digest_b64 = stored_hash.split("$", maxsplit=3)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False

    salt = base64.urlsafe_b64decode(salt_b64.encode("utf-8"))
    expected = base64.urlsafe_b64decode(digest_b64.encode("utf-8"))
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        int(iterations),
    )
    return hmac.compare_digest(derived, expected)


def _refresh_token_key(jti: str) -> str:
    return f"{REFRESH_TOKEN_PREFIX}{jti}"


def _validate_email(email: str) -> None:
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise AuthenticationError("Email address is invalid.")


def _next_quota_reset(now: datetime) -> datetime:
    now = now.astimezone(UTC)
    if now.month == 12:
        return datetime(now.year + 1, 1, 1, tzinfo=UTC)
    return datetime(now.year, now.month + 1, 1, tzinfo=UTC)


def _derive_tier(event: dict) -> TierName:
    candidates: list[str] = []
    for key in ("entitlement_id", "product_id", "period_type"):
        value = event.get(key)
        if isinstance(value, str):
            candidates.append(value.lower())
    entitlement_ids = event.get("entitlement_ids")
    if isinstance(entitlement_ids, list):
        candidates.extend(str(item).lower() for item in entitlement_ids)

    if any("academy" in candidate for candidate in candidates):
        return "academy"
    if any("coach" in candidate for candidate in candidates):
        return "coach"
    return "basic"


def _extract_expiration(event: dict) -> datetime | None:
    for key in ("expiration_at_ms", "expires_at_ms"):
        value = event.get(key)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
    for key in ("expiration_at", "expires_at"):
        value = event.get(key)
        if isinstance(value, str):
            normalized = value.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                continue
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
    return None


def _extract_revenuecat_customer_id(event: dict) -> str:
    for key in ("app_user_id", "original_app_user_id", "revenuecat_customer_id"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    aliases = event.get("aliases")
    if isinstance(aliases, list):
        for alias in aliases:
            if isinstance(alias, str) and alias.strip():
                return alias.strip()
    raise AuthenticationError("RevenueCat customer id is missing.")


def _has_active_entitlement(status: str, expires_at: datetime | None) -> bool:
    if status == "active":
        return True
    if status == "canceled" and expires_at is not None:
        return expires_at > datetime.now(UTC)
    return False
