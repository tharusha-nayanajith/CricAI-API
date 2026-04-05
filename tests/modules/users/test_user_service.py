from datetime import UTC, datetime, timedelta

import pytest

from app.modules.users.service import UserService
from app.storage.database import UserRecord, get_sessionmaker


@pytest.mark.asyncio
async def test_enforce_clip_quota_handles_naive_quota_reset_timestamp() -> None:
    service = UserService()

    async with get_sessionmaker()() as session:
        user = UserRecord(
            email="quota@example.com",
            hashed_password="pbkdf2_sha256$600000$c2FsdA==$ZGlnZXN0",
            full_name="Quota User",
            entitlement_status="active",
            current_tier="basic",
            clips_used_this_month=7,
            quota_reset_at=(datetime.now(UTC) - timedelta(days=1)).replace(tzinfo=None),
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

        profile = await service.enforce_clip_quota(session, user.id)

        assert profile.clips_used_this_month == 1
        assert profile.quota_reset_at.tzinfo == UTC

        refreshed_user = await session.get(UserRecord, user.id)
        assert refreshed_user is not None
        assert refreshed_user.clips_used_this_month == 1
        assert refreshed_user.quota_reset_at.tzinfo == UTC
