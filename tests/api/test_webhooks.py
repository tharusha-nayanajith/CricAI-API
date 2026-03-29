from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import select

from app.storage.database import UserRecord, get_sessionmaker


@pytest.mark.asyncio
async def test_revenuecat_webhook_updates_user_entitlement(test_client) -> None:
    register_response = await test_client.post(
        "/auth/register",
        json={
            "email": "revenuecat@example.com",
            "password": "Password123",
            "full_name": "Revenue Cat",
        },
    )
    user_id = register_response.json()["id"]
    expiration = int((datetime.now(UTC) + timedelta(days=30)).timestamp() * 1000)

    response = await test_client.post(
        "/webhooks/revenuecat",
        headers={"Authorization": "Bearer revenuecat-secret"},
        json={
            "event": {
                "type": "INITIAL_PURCHASE",
                "app_user_id": user_id,
                "entitlement_ids": ["coach_monthly"],
                "expiration_at_ms": expiration,
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["event_type"] == "INITIAL_PURCHASE"
    assert body["entitlement_status"] == "active"
    assert body["current_tier"] == "coach"

    async with get_sessionmaker()() as session:
        result = await session.execute(select(UserRecord).where(UserRecord.id == UUID(user_id)))
        user = result.scalar_one()

    assert user.revenuecat_customer_id == user_id
    assert user.entitlement_status == "active"
    assert user.current_tier == "coach"
