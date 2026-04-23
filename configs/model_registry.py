from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ProviderPreset:
    key: str
    runtime_provider: str
    default_model: str
    base_url: str
    requires_api_key: bool


_PROVIDER_PRESETS: Dict[str, ProviderPreset] = {
    "local": ProviderPreset(
        key="local",
        runtime_provider="ollama",
        default_model="llama3.1:8b",
        base_url="http://localhost:11434",
        requires_api_key=False, 
    ),
    "openai": ProviderPreset(
        key="openai",
        runtime_provider="openai",
        default_model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        requires_api_key=True,
    ),
    "gemini": ProviderPreset(
        key="gemini",
        runtime_provider="openai",
        default_model="gemini-2.5-flash-lite",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        requires_api_key=True,
    ),
    "claude": ProviderPreset(
        key="claude",
        runtime_provider="anthropic",
        default_model="claude-sonnet-4-6",
        base_url="https://api.anthropic.com/v1",
        requires_api_key=True,
    ),
    "copilot": ProviderPreset(
        key="copilot",
        runtime_provider="openai",
        default_model="gpt-4.1",
        base_url="https://models.inference.ai.azure.com",
        requires_api_key=True,
    ),
}

_PROVIDER_ALIASES = {
    "ollama": "local",
    "anthropic": "claude",
}


def normalize_provider_key(provider: str) -> str:
    raw = (provider or "local").strip().lower()
    return _PROVIDER_ALIASES.get(raw, raw)


def resolve_provider_preset(provider: str) -> ProviderPreset:
    key = normalize_provider_key(provider)
    if key not in _PROVIDER_PRESETS:
        raise ValueError(
            f"Unsupported model provider: {provider}. Supported providers: {', '.join(sorted(_PROVIDER_PRESETS.keys()))}"
        )
    return _PROVIDER_PRESETS[key]


def supported_providers() -> list[str]:
    return sorted(_PROVIDER_PRESETS.keys())
