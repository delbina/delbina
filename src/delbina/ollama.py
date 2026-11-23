"""Ready-made summarizers (stdlib only, no extra dependencies)."""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Callable

Summarizer = Callable[[list[str]], str]

PROMPT = (
    "Summarize the following chat turns as compactly as possible. "
    "Keep facts, decisions, names, numbers, file paths, errors and open questions. "
    "If a previous summary is given, merge it with the new turns into one summary. "
    "Drop greetings and filler.\n\n{chat}\n\nSummary:"
)


RETRY_STATUS = {429, 500, 502, 503, 504}
_sleep = time.sleep  # patched in tests


class RateLimiter:
    """Space calls to at most ``rpm`` requests per minute (thread-safe). 0 = no limit."""

    def __init__(self, rpm: float = 0):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self.interval
        if wait > 0:
            _sleep(wait)


def _retry_delay(err: urllib.error.HTTPError, attempt: int) -> float:
    """Server-suggested delay (Retry-After header, or Gemini's retryDelay), else exponential backoff."""
    try:
        header = err.headers.get("Retry-After") if err.headers else None
        if header:
            return min(float(header), 120.0)
        m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', err.read().decode("utf-8", "replace"))
        if m:
            return min(float(m.group(1)) + 1, 120.0)
    except Exception:
        pass
    return min(2.0 * 2 ** attempt, 60.0)


def _post(
    url: str,
    body: dict,
    headers: dict[str, str],
    timeout: float,
    retries: int = 0,
    throttle: Callable[[], None] | None = None,
    on_retry: Callable[[int, float, str], None] | None = None,
) -> dict:
    """POST JSON. Retries 429/5xx and network errors up to ``retries`` times.
    ``on_retry(attempt, wait_seconds, reason)`` is called before each retry."""
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        if throttle is not None:
            throttle()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUS or attempt == retries:
                raise
            wait, reason = _retry_delay(e, attempt), f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries:
                raise
            wait, reason = min(2.0 * 2 ** attempt, 60.0), type(getattr(e, "reason", e)).__name__
        if on_retry is not None:
            on_retry(attempt + 1, wait, reason)
        _sleep(wait)
    raise AssertionError("unreachable")


def reply_text(data: dict) -> str:
    """Text of a /chat/completions reply. Missing content (empty reply, safety
    block, tool call) gives "" instead of a KeyError."""
    try:
        message = data["choices"][0].get("message") or {}
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""
    return (message.get("content") or "").strip()


def finish_reason(data: dict) -> str:
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError):
        return "no choices"
    if (choice.get("message") or {}).get("tool_calls"):
        return "tool_calls"
    return str(choice.get("finish_reason") or "unknown")


def ollama_summarizer(
    model: str = "qwen2.5:1.5b",
    host: str = "http://localhost:11434",
    timeout: float = 120.0,
) -> Summarizer:
    """Summarizer that calls a local Ollama server's /api/generate endpoint."""
    url = host.rstrip("/") + "/api/generate"

    def summarize(turns: list[str]) -> str:
        prompt = PROMPT.format(chat="\n".join(f"- {t}" for t in turns))
        return _post(url, {"model": model, "prompt": prompt, "stream": False}, {}, timeout)["response"].strip()

    return summarize


def openai_summarizer(
    model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    api_key: str | None = None,
    timeout: float = 120.0,
    max_tokens: int = 400,
    reasoning_effort: str | None = None,
    retries: int = 0,
    throttle: Callable[[], None] | None = None,
    on_retry: Callable[[int, float, str], None] | None = None,
) -> Summarizer:
    """Summarizer for any OpenAI-compatible /chat/completions endpoint
    (OpenAI, Groq, OpenRouter, vLLM, LM Studio, llama.cpp, Ollama's /v1 ...).

    ``api_key`` defaults to the OPENAI_API_KEY environment variable.
    ``reasoning_effort`` is sent as-is when set (e.g. "minimal" for Gemini, "low" for o-series).
    ``retries`` retries 429/5xx with backoff; ``throttle`` is called before each request
    (pass a :class:`RateLimiter` to stay under a requests-per-minute limit).
    """
    url = base_url.rstrip("/") + "/chat/completions"
    key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    def summarize(turns: list[str]) -> str:
        prompt = PROMPT.format(chat="\n".join(f"- {t}" for t in turns))
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        data = _post(url, body, headers, timeout, retries, throttle, on_retry)
        return reply_text(data)

    return summarize
