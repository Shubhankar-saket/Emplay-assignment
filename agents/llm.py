"""OpenAI-compatible LLM client (Groq / OpenRouter): JSON mode, retries, timeouts, disk cache, usage log."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from core.config import get_settings
from core.logging import get_logger, log_event

log = get_logger("llm")


class LLMError(RuntimeError):
    pass


def _retry_after(exc: Exception) -> float | None:
    """Seconds the provider asked us to wait (429 Retry-After header), capped at 60."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        return min(float(headers.get("retry-after")), 60.0) + 1
    except (TypeError, ValueError):
        return None


class LLMClient:
    def __init__(self, cache_dir: Path = Path("data/llm_cache")) -> None:
        s = get_settings()
        if not s.llm_api_key:
            raise LLMError("LLM_API_KEY is not set (see .env.example)")
        self.settings = s
        self.client = OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key, timeout=s.llm_timeout_s, max_retries=0)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._lock = threading.Lock()

    def chat_json(self, system: str, user: str, fast: bool = False, temperature: float = 0.0) -> dict[str, Any]:
        """Return the model's reply parsed as a JSON object. Retries on timeouts, rate limits and bad JSON."""
        model = self.settings.llm_fast_model if fast else self.settings.llm_model
        key = hashlib.sha256(json.dumps([model, system, user, temperature]).encode()).hexdigest()
        cache = self.cache_dir / f"{key[:32]}.json"
        if cache.exists():
            with self._lock:
                self.usage["cached"] += 1
            return json.loads(cache.read_text(encoding="utf-8"))

        last_err: Exception | None = None
        for attempt in range(self.settings.llm_max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model, temperature=temperature, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                data = json.loads(resp.choices[0].message.content or "")
                if not isinstance(data, dict):
                    raise ValueError("reply is not a JSON object")
                with self._lock:
                    self.usage["calls"] += 1
                    if resp.usage:
                        self.usage["prompt_tokens"] += resp.usage.prompt_tokens
                        self.usage["completion_tokens"] += resp.usage.completion_tokens
                cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                log_event(log, "llm_call", model=model, attempt=attempt + 1,
                          tokens=resp.usage.total_tokens if resp.usage else None)
                return data
            except Exception as exc:  # timeouts, 429s, 5xx, invalid JSON
                last_err = exc
                if getattr(exc, "status_code", None) in (400, 401, 403, 404, 413):  # retrying cannot help
                    log_event(log, "llm_failed", level=40, model=model, error=repr(exc)[:300])
                    break
                wait = _retry_after(exc) or 5 * 2 ** attempt
                log_event(log, "llm_retry", level=30, model=model, attempt=attempt + 1, wait_s=wait,
                          error=repr(exc)[:300])
                time.sleep(wait)
        raise LLMError(f"LLM call failed: {last_err!r}"[:400])


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
