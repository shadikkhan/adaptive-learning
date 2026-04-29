"""Enterprise-friendly text-to-speech utilities for AgeXplain."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Optional

from gtts import gTTS

from configs.config import AUDIO_DIR
from services.json_logger import log_event

# Keep generated text bounded so TTS remains responsive.
MAX_TTS_CHARS = int(os.getenv("TTS_MAX_CHARS", "6000"))
AUDIO_RETENTION_SECONDS = int(os.getenv("AUDIO_RETENTION_SECONDS", "86400"))
AUDIO_MAX_FILES = int(os.getenv("AUDIO_MAX_FILES", "500"))
AUDIO_CLEANUP_INTERVAL_SECONDS = int(os.getenv("AUDIO_CLEANUP_INTERVAL_SECONDS", "300"))
_last_cleanup_ts = 0.0


def _backend_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_audio_dir() -> Path:
    audio_dir = _backend_root() / AUDIO_DIR
    audio_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir


def cleanup_audio_files(*, force: bool = False) -> dict:
    """Remove stale/overflow audio files based on TTL and max file count."""
    global _last_cleanup_ts

    now = time.time()
    if not force and (now - _last_cleanup_ts) < AUDIO_CLEANUP_INTERVAL_SECONDS:
        return {"deleted": 0, "checked": 0, "skipped": True}

    audio_dir = get_audio_dir()
    files = [p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp3"]
    deleted = 0

    # TTL-based cleanup
    for path in files:
        try:
            age_seconds = now - path.stat().st_mtime
            if age_seconds > AUDIO_RETENTION_SECONDS:
                path.unlink(missing_ok=True)
                deleted += 1
        except FileNotFoundError:
            continue
        except Exception as exc:
            log_event("tts.cleanup.error", level="WARNING", file=path.name, error=str(exc))

    # Count-based cleanup (oldest first)
    remaining = [p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp3"]
    overflow = max(0, len(remaining) - AUDIO_MAX_FILES)
    if overflow:
        remaining.sort(key=lambda p: p.stat().st_mtime)
        for path in remaining[:overflow]:
            try:
                path.unlink(missing_ok=True)
                deleted += 1
            except FileNotFoundError:
                continue
            except Exception as exc:
                log_event("tts.cleanup.error", level="WARNING", file=path.name, error=str(exc))

    _last_cleanup_ts = now
    result = {
        "deleted": deleted,
        "checked": len(files),
        "retention_seconds": AUDIO_RETENTION_SECONDS,
        "max_files": AUDIO_MAX_FILES,
        "skipped": False,
    }
    log_event("tts.cleanup", **result)
    return result


def _compact_for_tts(text: str) -> str:
    # Normalize whitespace only; let gTTS handle its own limits
    return " ".join((text or "").split())


def synthesize_tts_mp3(text: str, *, lang: str = "en") -> Optional[Path]:
    """Create an MP3 file for the provided text and return its path.

    Returns None when text is empty after normalization.
    Raises exceptions only for genuine TTS failures.
    """
    payload = _compact_for_tts(text)
    if not payload:
        return None

    cleanup_audio_files(force=False)

    audio_dir = get_audio_dir()
    file_name = f"{uuid.uuid4().hex}.mp3"
    target = audio_dir / file_name

    started = time.perf_counter()
    log_event("tts.start", file=file_name, chars=len(payload), lang=lang)
    tts = gTTS(text=payload, lang=lang, slow=False)
    tts.save(str(target))
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log_event("tts.end", file=file_name, exists=target.exists(), elapsed_ms=elapsed_ms)
    return target
