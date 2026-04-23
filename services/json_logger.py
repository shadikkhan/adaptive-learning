"""Simple structured JSON logger for backend services."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

_SENSITIVE_KEY_MARKERS = ("api_key", "authorization", "token", "password", "secret")
_LEVEL_ORDER = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


def _minimum_level() -> str:
    raw = (os.getenv("LOG_LEVEL", "INFO") or "INFO").strip().upper()
    return raw if raw in _LEVEL_ORDER else "INFO"


def _should_emit(level: str) -> bool:
    requested = (level or "INFO").strip().upper()
    effective = requested if requested in _LEVEL_ORDER else "INFO"
    return _LEVEL_ORDER[effective] >= _LEVEL_ORDER[_minimum_level()]


def _is_sensitive_key(key: str) -> bool:
    lowered = (key or "").lower()
    return any(marker in lowered for marker in _SENSITIVE_KEY_MARKERS)


def _sanitize(value: Any, key: str | None = None) -> Any:
    if key and _is_sensitive_key(key):
        if isinstance(value, str):
            return "<redacted>" if value.strip() else ""
        return "<redacted>" if value else value

    if isinstance(value, dict):
        return {k: _sanitize(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(v) for v in value]
    if isinstance(value, Exception):
        return str(value)
    return value


def log_event(event: str, *, level: str = "INFO", **fields: Any) -> None:
    if not _should_emit(level):
        return

    normalized_level = (level or "INFO").strip().upper()
    if normalized_level not in _LEVEL_ORDER:
        normalized_level = "INFO"

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "level": normalized_level,
        "event": event,
    }
    payload.update(_sanitize(fields))
    print(json.dumps(payload, ensure_ascii=True, default=str), flush=True)
