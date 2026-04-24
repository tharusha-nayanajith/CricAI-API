from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
from typing import Any

import app.config as config_module


@dataclass(frozen=True)
class GoogleGenAIStatus:
    enabled: bool
    reason: str | None = None
    uses_vertexai: bool = False
    model: str = "gemini-2.5-flash"


def _get_settings() -> config_module.Settings:
    return config_module.get_settings()


def _resolve_api_key(settings: config_module.Settings) -> str | None:
    return settings.gemini_api_key or settings.google_api_key


def _import_google_genai() -> Any | None:
    try:
        return importlib.import_module("google.genai")
    except ModuleNotFoundError:
        return None


@lru_cache
def get_google_genai_status() -> GoogleGenAIStatus:
    settings = _get_settings()
    if settings.ai_provider.lower() != "google":
        return GoogleGenAIStatus(enabled=False, reason=f"unsupported AI_PROVIDER: {settings.ai_provider}", model=settings.ai_model)

    if _import_google_genai() is None:
        return GoogleGenAIStatus(enabled=False, reason="missing google-genai SDK", model=settings.ai_model)

    if settings.google_genai_use_vertexai:
        if not settings.google_cloud_project:
            return GoogleGenAIStatus(
                enabled=False,
                reason="missing GOOGLE_CLOUD_PROJECT for Vertex AI",
                uses_vertexai=True,
                model=settings.ai_model,
            )
        if not settings.google_cloud_location:
            return GoogleGenAIStatus(
                enabled=False,
                reason="missing GOOGLE_CLOUD_LOCATION for Vertex AI",
                uses_vertexai=True,
                model=settings.ai_model,
            )
        return GoogleGenAIStatus(enabled=True, uses_vertexai=True, model=settings.ai_model)

    if not _resolve_api_key(settings):
        return GoogleGenAIStatus(
            enabled=False,
            reason="missing GEMINI_API_KEY or GOOGLE_API_KEY",
            model=settings.ai_model,
        )

    return GoogleGenAIStatus(enabled=True, model=settings.ai_model)


@lru_cache
def get_google_genai_client() -> Any | None:
    status = get_google_genai_status()
    if not status.enabled:
        return None

    genai = _import_google_genai()
    if genai is None:
        return None

    settings = _get_settings()
    client_kwargs: dict[str, Any] = {}

    if status.uses_vertexai:
        client_kwargs.update(
            vertexai=True,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )
        api_key = _resolve_api_key(settings)
        if api_key:
            client_kwargs["api_key"] = api_key
    else:
        api_key = _resolve_api_key(settings)
        if api_key is None:
            return None
        client_kwargs["api_key"] = api_key

    return genai.Client(**client_kwargs)


def clear_google_genai_caches() -> None:
    get_google_genai_status.cache_clear()
    get_google_genai_client.cache_clear()
