from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_register_login_refresh_logout_flow(test_client) -> None:
    register_response = await test_client.post(
        "/auth/register",
        json={
            "email": "user@example.com",
            "password": "Password123",
            "full_name": "Test User",
        },
    )

    assert register_response.status_code == 201
    assert register_response.json()["email"] == "user@example.com"

    login_response = await test_client.post(
        "/auth/login",
        json={
            "email": "user@example.com",
            "password": "Password123",
        },
    )

    assert login_response.status_code == 200
    tokens = login_response.json()
    assert "access_token" in tokens
    assert "refresh_token" in tokens

    me_response = await test_client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )

    assert me_response.status_code == 200
    assert me_response.json()["full_name"] == "Test User"

    refresh_response = await test_client.post(
        "/auth/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )

    assert refresh_response.status_code == 200
    refreshed = refresh_response.json()
    assert refreshed["access_token"] != ""

    logout_response = await test_client.post(
        "/auth/logout",
        json={"refresh_token": tokens["refresh_token"]},
    )

    assert logout_response.status_code == 204

    refresh_after_logout = await test_client.post(
        "/auth/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )

    assert refresh_after_logout.status_code == 401


@pytest.mark.asyncio
async def test_register_rejects_duplicate_email(test_client) -> None:
    first = await test_client.post(
        "/auth/register",
        json={
            "email": "duplicate@example.com",
            "password": "Password123",
            "full_name": "First User",
        },
    )
    second = await test_client.post(
        "/auth/register",
        json={
            "email": "duplicate@example.com",
            "password": "Password123",
            "full_name": "Second User",
        },
    )

    assert first.status_code == 201
    assert second.status_code == 409
