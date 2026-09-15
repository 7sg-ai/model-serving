"""
OpenAI-compatible chat client for Switchyard model targets.

Each ModelSpec may point at a different backend URL pool with its own
timeout, API key, and extra_body — independent deployment parameters.

Non-stream calls always use JSON bodies. Streaming uses shared SSE helpers
and must never be parsed with response.json().
"""
from __future__ import annotations

import itertools
import logging
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from .config import ModelSpec

logger = logging.getLogger(__name__)


class _UrlPool:
    def __init__(self, urls: List[str], strategy: str = "round_robin"):
        self.urls = [u.rstrip("/") for u in urls if u]
        self.strategy = (strategy or "round_robin").lower()
        self._lock = threading.Lock()
        self._cycle = itertools.cycle(self.urls) if self.urls else None
        self._failures: Dict[str, float] = {}
        self.cooldown_seconds = 30.0

    def mark_failure(self, url: str) -> None:
        with self._lock:
            self._failures[url] = time.time()

    def mark_success(self, url: str) -> None:
        with self._lock:
            self._failures.pop(url, None)

    def next(self) -> Optional[str]:
        if not self.urls:
            return None
        now = time.time()
        available = [
            u
            for u in self.urls
            if u not in self._failures
            or (now - self._failures[u]) >= self.cooldown_seconds
        ] or list(self.urls)
        if self.strategy == "random":
            return random.choice(available)
        if self.strategy == "first":
            return available[0]
        with self._lock:
            if self._cycle is None:
                return available[0]
            for _ in range(len(self.urls)):
                c = next(self._cycle)
                if c in available:
                    return c
            return available[0]


def _normalize_chat_url(url: str) -> str:
    url = (url or "").rstrip("/")
    if not url:
        return ""
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    if "/v1/" in url:
        # already a path under v1 — append if needed
        if url.endswith("completions") and not url.endswith("chat/completions"):
            return url
        return url if url.endswith("chat/completions") else url + "/chat/completions"
    return url + "/v1/chat/completions"


class ModelClient:
    """HTTP client bound to one ModelSpec (with optional multi-URL pool)."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        urls = list(spec.deployment.backend_urls)
        if not urls and spec.chat_url():
            urls = [spec.chat_url()]
        # Normalize all to chat/completions endpoints
        urls = [_normalize_chat_url(u) for u in urls if u]
        if spec.deployment.coordinator_url:
            urls = [_normalize_chat_url(spec.deployment.coordinator_url)] + [
                u for u in urls if u != _normalize_chat_url(spec.deployment.coordinator_url)
            ]
        self.pool = _UrlPool(urls, strategy=spec.deployment.backend_strategy)

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
        extra: Optional[Dict[str, Any]],
        model_id_override: Optional[str],
    ) -> Dict[str, Any]:
        temp = self.spec.temperature if temperature is None else temperature
        mt = self.spec.max_tokens if max_tokens is None else max_tokens
        payload: Dict[str, Any] = {
            "model": model_id_override or self.spec.id,
            "messages": messages,
            "temperature": temp,
            "max_tokens": mt,
            "stream": bool(stream),
        }
        if self.spec.top_p is not None:
            payload["top_p"] = self.spec.top_p
        for k, v in (self.spec.extra_body or {}).items():
            payload.setdefault(k, v)
        if extra:
            payload.update(extra)
        payload["stream"] = bool(stream)
        return payload

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self.spec.resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        extra: Optional[Dict[str, Any]] = None,
        model_id_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        POST chat/completions as non-stream JSON.

        ``stream`` is accepted for compatibility but always forced False.
        Use :meth:`chat_stream` for SSE.
        """
        if stream:
            logger.debug(
                "ModelClient.chat(stream=True) coerced to JSON; use chat_stream()"
            )
        url = self.pool.next()
        if not url:
            raise RuntimeError(
                f"No backend URL configured for model '{self.spec.name}' ({self.spec.id})"
            )

        payload = self._build_payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            extra=extra,
            model_id_override=model_id_override,
        )
        headers = self._headers()

        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.spec.request_timeout,
            )
            resp.raise_for_status()
            self.pool.mark_success(url)
            data = resp.json()
            if isinstance(data, dict):
                data.setdefault("switchyard_target", self.spec.name)
                data.setdefault("switchyard_model_id", self.spec.id)
            return data
        except Exception:
            self.pool.mark_failure(url)
            raise

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
        model_id_override: Optional[str] = None,
    ) -> requests.Response:
        """
        POST chat/completions with stream=true.

        Returns an open streaming Response. Caller must close it
        (flask_sse_from_upstream does this).
        """
        from shared.backends.streaming import post_chat_stream

        url = self.pool.next()
        if not url:
            raise RuntimeError(
                f"No backend URL configured for model '{self.spec.name}' ({self.spec.id})"
            )

        temp = self.spec.temperature if temperature is None else temperature
        mt = self.spec.max_tokens if max_tokens is None else max_tokens
        merged_extra: Dict[str, Any] = {}
        for k, v in (self.spec.extra_body or {}).items():
            if k != "stream":
                merged_extra[k] = v
        if extra:
            for k, v in extra.items():
                if k != "stream":
                    merged_extra[k] = v
        if self.spec.top_p is not None:
            merged_extra.setdefault("top_p", self.spec.top_p)

        try:
            upstream = post_chat_stream(
                url,
                messages,
                model=model_id_override or self.spec.id,
                temperature=float(temp),
                max_tokens=int(mt),
                api_key=self.spec.resolve_api_key() or "",
                timeout=float(self.spec.request_timeout),
                extra=merged_extra or None,
            )
            self.pool.mark_success(url)
            return upstream
        except Exception:
            self.pool.mark_failure(url)
            raise

    def extract_assistant_text(self, result: Dict[str, Any]) -> str:
        try:
            return result["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""


def chat_completion(
    spec: ModelSpec,
    messages: List[Dict[str, Any]],
    **kwargs: Any,
) -> Tuple[Dict[str, Any], str]:
    """Convenience: return (full_response, assistant_text)."""
    client = ModelClient(spec)
    result = client.chat(messages, **kwargs)
    return result, client.extract_assistant_text(result)
