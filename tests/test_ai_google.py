from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.ai.google as google_ai
from app.config import Settings


def _settings(**overrides: object) -> Settings:
    base = {
        'redis_url': 'redis://localhost:6379/0',
        'database_url': 'sqlite+aiosqlite:///test.db',
        's3_bucket': 'test-bucket',
        'aws_region': 'us-east-1',
        's3_playback_prefix': 'deliveries',
        's3_presign_ttl_seconds': 3600,
        'jwt_secret': 'secret',
        'jwt_algorithm': 'HS256',
        'revenuecat_webhook_secret': 'webhook-secret',
        'ai_provider': 'google',
        'ai_model': 'gemini-2.5-flash',
        'gemini_api_key': None,
        'google_api_key': None,
        'google_genai_use_vertexai': False,
        'google_cloud_project': None,
        'google_cloud_location': 'global',
    }
    base.update(overrides)
    return Settings.model_construct(**base)


@pytest.fixture(autouse=True)
def clear_caches() -> None:
    google_ai.clear_google_genai_caches()
    yield
    google_ai.clear_google_genai_caches()


def test_google_genai_status_requires_api_key_outside_vertex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(google_ai.config_module, 'get_settings', lambda: _settings())
    monkeypatch.setattr(google_ai, '_import_google_genai', lambda: object())

    status = google_ai.get_google_genai_status()

    assert status.enabled is False
    assert status.reason == 'missing GEMINI_API_KEY or GOOGLE_API_KEY'


def test_google_genai_status_accepts_vertex_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        google_ai.config_module,
        'get_settings',
        lambda: _settings(
            google_genai_use_vertexai=True,
            google_cloud_project='crickai-prod',
            google_cloud_location='global',
        ),
    )
    monkeypatch.setattr(google_ai, '_import_google_genai', lambda: object())

    status = google_ai.get_google_genai_status()

    assert status.enabled is True
    assert status.uses_vertexai is True
    assert status.model == 'gemini-2.5-flash'


def test_google_genai_client_uses_vertex_client(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    fake_genai = SimpleNamespace(Client=FakeClient)
    monkeypatch.setattr(
        google_ai.config_module,
        'get_settings',
        lambda: _settings(
            google_genai_use_vertexai=True,
            google_cloud_project='crickai-prod',
            google_cloud_location='us-central1',
        ),
    )
    monkeypatch.setattr(google_ai, '_import_google_genai', lambda: fake_genai)

    client = google_ai.get_google_genai_client()

    assert client is not None
    assert calls == [{
        'vertexai': True,
        'project': 'crickai-prod',
        'location': 'us-central1',
    }]


def test_google_genai_client_uses_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    fake_genai = SimpleNamespace(Client=FakeClient)
    monkeypatch.setattr(
        google_ai.config_module,
        'get_settings',
        lambda: _settings(gemini_api_key='test-key'),
    )
    monkeypatch.setattr(google_ai, '_import_google_genai', lambda: fake_genai)

    client = google_ai.get_google_genai_client()

    assert client is not None
    assert calls == [{'api_key': 'test-key'}]
