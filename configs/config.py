import os
from contextlib import contextmanager
from contextvars import ContextVar
from langchain_ollama import OllamaLLM

# Audio settings
AUDIO_DIR = "audio"

# LLM settings
LLM_MODEL = "llama3.1:8b"
LLM_TEMPERATURE = 0

_active_llm: ContextVar = ContextVar("active_llm", default=None)


def build_default_llm():
    return OllamaLLM(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )


_default_llm = None  # Lazy: only created when Ollama is actually needed


class LLMProxy:
    """Routes invoke() to request-scoped model, falling back to default Ollama."""

    def invoke(self, prompt: str):
        global _default_llm
        active = _active_llm.get()
        if active:
            return active.invoke(prompt)
        # Only build the Ollama default on first actual use (not at import time).
        if _default_llm is None:
            _default_llm = build_default_llm()
        return _default_llm.invoke(prompt)


@contextmanager
def use_request_llm(request_llm):
    token = _active_llm.set(request_llm)
    try:
        yield
    finally:
        _active_llm.reset(token)


# Backward-compatible symbol used by agents.
llm = LLMProxy()

# Difficulty mapping for prompts
DIFFICULTY_MAP = {
        "easy": "simple, straightforward questions",
        "medium": "moderate difficulty with some critical thinking",
        "hard": "challenging questions requiring deep understanding"
    }

# All settings loaded from config.json — no .env file needed
import json as _json
import os as _os
import pathlib as _pathlib

_here = _pathlib.Path(__file__).resolve()
_config_candidates = [
    _here.parent.parent / "config.json",         # adaptive-learning/config.json
    _here.parent.parent.parent / "config.json",  # fallback for alternate layouts
]
_config_path = next((p for p in _config_candidates if p.exists()), _config_candidates[0])
_root_config = _json.loads(_config_path.read_text())["backend"]

# Inject into os.environ so existing os.getenv() calls work unchanged
_os.environ.setdefault("SAFETY_STRICT_MODE",         str(_root_config["safety_strict_mode"]).lower())
_os.environ.setdefault("LOG_LEVEL",                   _root_config["log_level"])
_os.environ.setdefault("HTTP_LOG_SKIP_PATH_PREFIXES", _root_config["http_log_skip_path_prefixes"])
_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", str(_root_config["hf_hub_disable_progress_bars"]))
_os.environ.setdefault("TRANSFORMERS_VERBOSITY",      _root_config["transformers_verbosity"])

# CORS
CORS_ORIGINS       = _root_config["cors_origins"]
CORS_CREDENTIALS   = _root_config["cors_credentials"]
CORS_METHODS       = _root_config["cors_methods"]
CORS_HEADERS       = _root_config["cors_headers"]
CORS_EXPOSE_HEADERS = _root_config["cors_expose_headers"]

# Network/runtime endpoints
HOST = _root_config["host"]
PORT = int(_root_config["port"])
LOCAL_MODEL_BASE_URL = _root_config["local_model_base_url"]

# Provider endpoint defaults
PROVIDER_BASE_URLS = _root_config["provider_base_urls"]
OPENAI_BASE_URL = PROVIDER_BASE_URLS["openai"]
GEMINI_BASE_URL = PROVIDER_BASE_URLS["gemini"]
CLAUDE_BASE_URL = PROVIDER_BASE_URLS["claude"]
COPILOT_BASE_URL = PROVIDER_BASE_URLS["copilot"]

