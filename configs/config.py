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

# API CORS settings, needed for frontend-backend communication in case of local development
CORS_ORIGINS = ["*"]
CORS_CREDENTIALS = True
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]
CORS_EXPOSE_HEADERS = ["*"]

