from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, parse_qs, parse_qsl, urlencode, urlunparse

from langchain_ollama import OllamaLLM
from configs.model_registry import resolve_provider_preset
from services.json_logger import log_event


@dataclass
class RuntimeModelConfig:
    provider: str = "ollama"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


def _normalize_api_key(api_key: str) -> str:
    value = (api_key or "").strip()
    if value.lower().startswith("bearer "):
        return value[7:].strip()
    return value


def _strip_api_key_query_params(url: str) -> str:
    """Remove key-like query params so credentials are sent via a single header path."""
    try:
        parsed = urlparse(url or "")
        if not parsed.query:
            return url
        cleaned_items = []
        for k, v in parse_qsl(parsed.query, keep_blank_values=True):
            if k.strip().lower() in {"key", "api_key", "x-goog-api-key", "authorization"}:
                continue
            cleaned_items.append((k, v))
        new_query = urlencode(cleaned_items)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment)).rstrip("/")
    except Exception:
        return (url or "").rstrip("/")


class OpenAICompatibleLLM:
    """Minimal OpenAI-compatible Chat Completions client with LangChain-like invoke()."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0,
        require_api_key: bool = True,
        fallback_models: Optional[list[str]] = None,
        auth_style: str = "bearer",
        allow_auth_fallback: bool = True,
    ):
        if require_api_key and not api_key:
            raise ValueError("API key is required for OpenAI-compatible providers")
        self.model = model or "gpt-4o-mini"
        self.api_key = _normalize_api_key(api_key or "")
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.temperature = temperature
        self.fallback_models = [m for m in (fallback_models or []) if m]
        self.auth_style = auth_style
        self.allow_auth_fallback = allow_auth_fallback

    def _base_url_has_key_query(self) -> bool:
        try:
            parsed = urlparse(self.base_url)
            query = parse_qs(parsed.query)
            return bool(query.get("key") or query.get("api_key") or query.get("x-goog-api-key"))
        except Exception:
            return False

    def _model_candidates(self) -> list[str]:
        candidates = [self.model, *self.fallback_models]
        seen = set()
        ordered = []
        for m in candidates:
            if not m or m in seen:
                continue
            seen.add(m)
            ordered.append(m)
        return ordered

    @staticmethod
    def _append_query_key(url: str, api_key: str) -> str:
        parsed = urlparse(url or "")
        query = parse_qs(parsed.query)
        query["key"] = [api_key]
        flat = []
        for k, values in query.items():
            for v in values:
                flat.append((k, v))
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, urlencode(flat), parsed.fragment))

    def invoke(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        auth_styles = [self.auth_style]
        if self.auth_style == "gemini-auto":
            auth_styles = ["bearer", "x-goog-api-key", "query-key"]
        elif self.allow_auth_fallback and self.auth_style == "x-goog-api-key":
            auth_styles.append("bearer")
        elif self.allow_auth_fallback and self.auth_style == "bearer":
            auth_styles.append("x-goog-api-key")

        last_error = None
        for auth_style in auth_styles:
            mode_url = url
            headers = {"Content-Type": "application/json"}
            has_key_in_url = self._base_url_has_key_query()
            # If key is already in URL (?key=...), don't send any auth header to avoid duplicate credentials.
            if self.api_key and auth_style == "query-key":
                mode_url = self._append_query_key(url, self.api_key)
            elif self.api_key and not has_key_in_url:
                if auth_style == "x-goog-api-key":
                    headers["x-goog-api-key"] = self.api_key
                elif auth_style == "bearer":
                    headers["Authorization"] = f"Bearer {self.api_key}"

            switch_auth_style = False
            for model_name in self._model_candidates():
                started = time.perf_counter()
                log_event(
                    "model_provider.openai.start",
                    model=model_name,
                    url=mode_url,
                    prompt_chars=len(prompt or ""),
                    auth_style=auth_style,
                )
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                }
                body = json.dumps(payload).encode("utf-8")
                req = urllib_request.Request(mode_url, data=body, headers=headers, method="POST")
                try:
                    with urllib_request.urlopen(req, timeout=90) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        elapsed_ms = int((time.perf_counter() - started) * 1000)
                        log_event("model_provider.openai.success", model=model_name, status=resp.status, elapsed_ms=elapsed_ms, auth_style=auth_style)
                    try:
                        return (data["choices"][0]["message"]["content"] or "").strip()
                    except Exception as exc:
                        elapsed_ms = int((time.perf_counter() - started) * 1000)
                        log_event("model_provider.openai.parse_error", level="ERROR", model=model_name, elapsed_ms=elapsed_ms, auth_style=auth_style)
                        raise RuntimeError("Invalid response format from model provider") from exc
                except HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="ignore")
                    lower_detail = detail.lower()
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    log_event("model_provider.openai.http_error", level="ERROR", model=model_name, status=exc.code, elapsed_ms=elapsed_ms, auth_style=auth_style)

                    # Try next model candidate if model name is unsupported.
                    if exc.code == 404 and "model" in lower_detail:
                        last_error = RuntimeError(f"Model request failed ({exc.code}): {detail}")
                        continue

                    # If auth style is rejected, switch auth mode once before failing.
                    auth_style_mismatch = (
                        exc.code == 400
                        and (
                            "missing or invalid authorization header" in lower_detail
                            or "multiple authentication credentials" in lower_detail
                        )
                    )
                    if auth_style_mismatch and auth_style != auth_styles[-1]:
                        last_error = RuntimeError(f"Model request failed ({exc.code}): {detail}")
                        switch_auth_style = True
                        break

                    if exc.code == 401 and auth_style != auth_styles[-1]:
                        last_error = RuntimeError(f"Model request failed ({exc.code}): {detail}")
                        switch_auth_style = True
                        break

                    raise RuntimeError(f"Model request failed ({exc.code}): {detail}") from exc
                except URLError as exc:
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    log_event("model_provider.openai.url_error", level="ERROR", model=model_name, elapsed_ms=elapsed_ms, reason=str(exc.reason), auth_style=auth_style)
                    raise RuntimeError(f"Unable to reach model endpoint: {exc.reason}") from exc

            if switch_auth_style:
                continue

        if last_error:
            raise last_error
        raise RuntimeError("Model request failed without a recoverable fallback")


class GeminiLLM:
    """Minimal Gemini native API client (models/*:generateContent) with invoke()."""

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None, temperature: float = 0):
        if not api_key:
            raise ValueError("API key is required for Gemini provider")
        self.model = model or "gemini-2.5-flash"
        self.api_key = _normalize_api_key(api_key)
        raw_base = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        # Accept old /openai base URLs and normalize to native v1beta root.
        self.base_url = raw_base[:-7] if raw_base.endswith("/openai") else raw_base
        self.temperature = temperature

    @staticmethod
    def _fallback_models(primary_model: str) -> list[str]:
        candidates = [
            primary_model,
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            "gemini-flash-latest",
        ]
        seen = set()
        ordered = []
        for model in candidates:
            if not model or model in seen:
                continue
            seen.add(model)
            ordered.append(model)
        return ordered

    def invoke(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        last_error = None
        for model_name in self._fallback_models(self.model):
            url = f"{self.base_url}/models/{model_name}:generateContent"
            started = time.perf_counter()
            log_event("model_provider.gemini.start", model=model_name, url=url, prompt_chars=len(prompt or ""))
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": self.temperature,
                },
            }
            body = json.dumps(payload).encode("utf-8")
            req = urllib_request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib_request.urlopen(req, timeout=90) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    log_event("model_provider.gemini.success", model=model_name, status=resp.status, elapsed_ms=elapsed_ms)
                try:
                    candidates = data.get("candidates") or []
                    if not candidates:
                        raise RuntimeError("No candidates in Gemini response")
                    parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
                    text_blocks = [p.get("text", "") for p in parts if isinstance(p, dict)]
                    text = "\n".join(t for t in text_blocks if t).strip()
                    if not text:
                        raise RuntimeError("Empty text in Gemini response")
                    return text
                except Exception as exc:
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    log_event("model_provider.gemini.parse_error", level="ERROR", model=model_name, elapsed_ms=elapsed_ms)
                    raise RuntimeError("Invalid response format from Gemini provider") from exc
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                lower_detail = detail.lower()
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                log_event("model_provider.gemini.http_error", level="ERROR", model=model_name, status=exc.code, elapsed_ms=elapsed_ms)
                if exc.code in (400, 404) and "model" in lower_detail:
                    last_error = RuntimeError(f"Model request failed ({exc.code}): {detail}")
                    continue
                raise RuntimeError(f"Model request failed ({exc.code}): {detail}") from exc
            except URLError as exc:
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                log_event("model_provider.gemini.url_error", level="ERROR", model=model_name, elapsed_ms=elapsed_ms, reason=str(exc.reason))
                raise RuntimeError(f"Unable to reach model endpoint: {exc.reason}") from exc

        if last_error:
            raise last_error
        raise RuntimeError("Gemini request failed without a recoverable model fallback")


class AnthropicLLM:
    """Minimal Anthropic Messages API client with LangChain-like invoke()."""

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None, temperature: float = 0):
        if not api_key:
            raise ValueError("API key is required for Anthropic provider")
        self.model = model or "claude-sonnet-4-6"
        self.api_key = api_key.strip()
        self.base_url = (base_url or "https://api.anthropic.com/v1").rstrip("/")
        self.temperature = temperature

    @staticmethod
    def _fallback_models(primary_model: str) -> list[str]:
        # Keep first item as user-selected/default model, then safe fallbacks.
        candidates = [
            primary_model,
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-6",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
            "claude-3-7-sonnet-latest",
            "claude-3-5-haiku-latest",
        ]
        seen = set()
        ordered = []
        for model in candidates:
            if not model or model in seen:
                continue
            seen.add(model)
            ordered.append(model)
        return ordered

    def invoke(self, prompt: str) -> str:
        url = f"{self.base_url}/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        last_error = None
        for model_name in self._fallback_models(self.model):
            started = time.perf_counter()
            log_event("model_provider.anthropic.start", model=model_name, url=url, prompt_chars=len(prompt or ""))
            payload = {
                "model": model_name,
                "max_tokens": 1000,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            body = json.dumps(payload).encode("utf-8")
            req = urllib_request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib_request.urlopen(req, timeout=90) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    log_event("model_provider.anthropic.success", model=model_name, status=resp.status, elapsed_ms=elapsed_ms)

                parts = data.get("content") or []
                text_blocks = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text"]
                return "\n".join(t for t in text_blocks if t).strip()
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                log_event("model_provider.anthropic.http_error", level="ERROR", model=model_name, status=exc.code, elapsed_ms=elapsed_ms)
                # Only auto-retry when the model identifier is not found.
                if exc.code == 404 and "not_found_error" in detail and "model" in detail:
                    last_error = RuntimeError(f"Model request failed ({exc.code}): {detail}")
                    continue
                raise RuntimeError(f"Model request failed ({exc.code}): {detail}") from exc
            except URLError as exc:
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                log_event("model_provider.anthropic.url_error", level="ERROR", model=model_name, elapsed_ms=elapsed_ms, reason=str(exc.reason))
                raise RuntimeError(f"Unable to reach model endpoint: {exc.reason}") from exc
            except Exception as exc:
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                log_event("model_provider.anthropic.parse_error", level="ERROR", model=model_name, elapsed_ms=elapsed_ms)
                raise RuntimeError("Invalid response format from Anthropic provider") from exc

        if last_error:
            raise last_error
        raise RuntimeError("Anthropic request failed without a recoverable model fallback")


def build_runtime_llm(config: RuntimeModelConfig, default_model: str, default_temperature: float):
    preset = resolve_provider_preset(config.provider)
    provider = preset.runtime_provider
    resolved_model = (config.model or preset.default_model or default_model).strip()
    resolved_base_url = (config.base_url or preset.base_url or "").strip()
    api_key = _normalize_api_key(config.api_key or "")

    if preset.requires_api_key and not api_key:
        raise ValueError(f"API key is required for provider: {preset.key}")

    log_event(
        "model_provider.build",
        provider=preset.key,
        runtime_provider=provider,
        model=resolved_model,
        base_url=resolved_base_url or "<default>",
        api_key_set=bool(api_key),
    )

    if provider == "ollama":
        kwargs = {"model": resolved_model, "temperature": default_temperature}
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url
        return OllamaLLM(**kwargs)

    if preset.key == "gemini":
        return GeminiLLM(
            model=resolved_model,
            api_key=api_key,
            base_url=resolved_base_url,
            temperature=default_temperature,
        )

    if provider == "openai":
        fallback_models = None
        auth_style = "bearer"
        allow_auth_fallback = False
        if preset.key == "openai":
            fallback_models = [
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4-turbo",
            ]
        return OpenAICompatibleLLM(
            model=resolved_model,
            api_key=api_key,
            base_url=resolved_base_url,
            temperature=default_temperature,
            fallback_models=fallback_models,
            auth_style=auth_style,
            allow_auth_fallback=allow_auth_fallback,
        )

    if provider == "anthropic":
        return AnthropicLLM(
            model=resolved_model,
            api_key=api_key,
            base_url=resolved_base_url,
            temperature=default_temperature,
        )

    raise ValueError(f"Unsupported model provider: {provider}. Supported providers: ollama, openai, anthropic")
