"""
OpenAI-compatible chat/completions HTTP client.

Used by HF apps (vLLM-only) and reusable by NIM-style proxies.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)


def normalize_chat_url(url: str) -> str:
    """Normalize a base or full URL to .../v1/chat/completions."""
    url = (url or "").strip().rstrip("/")
    if not url:
        return ""
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    if "/v1/" in url:
        if url.endswith("chat/completions"):
            return url
        return url if url.endswith("chat/completions") else url + "/chat/completions"
    return url + "/v1/chat/completions"


def _split_csv(raw: str) -> List[str]:
    return [p.strip() for p in (raw or "").split(",") if p.strip()]


def resolve_vllm_backend_urls(
    existing: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Resolve ordered chat-completion backend URLs for HF/vLLM clients.

    Merges (first-seen order): coordinator, BACKEND_URLS family, VLLM_*, SGLANG_*, NIM_API_URL.
    """
    ordered: List[str] = []
    seen = set()

    def _add(raw: str) -> None:
        for part in _split_csv(raw):
            norm = normalize_chat_url(part)
            if norm and norm not in seen:
                seen.add(norm)
                ordered.append(norm)

    if existing:
        for u in existing:
            _add(u)

    _add(os.getenv("MODEL_COORDINATOR_URL", os.getenv("SHARD_COORDINATOR_URL", "")))
    _add(
        os.getenv(
            "BACKEND_URLS",
            os.getenv(
                "NIM_API_URLS",
                os.getenv("MODEL_BACKEND_URLS", os.getenv("MODEL_REPLICA_URLS", "")),
            ),
        )
    )
    _add(os.getenv("VLLM_API_URLS", os.getenv("VLLM_API_URL", "")))
    _add(os.getenv("SGLANG_API_URLS", os.getenv("SGLANG_API_URL", "")))
    _add(os.getenv("NIM_API_URL", ""))
    return ordered


def require_backend_urls(
    urls: Optional[Sequence[str]] = None,
    *,
    what: str = "vLLM",
) -> List[str]:
    """Return normalized backend URLs or raise with a clear startup error."""
    resolved = resolve_vllm_backend_urls(urls)
    if not resolved:
        raise RuntimeError(
            f"No {what} backend configured. Set one of: "
            "BACKEND_URLS, VLLM_API_URL, VLLM_API_URLS, SGLANG_API_URL, or MODEL_COORDINATOR_URL "
            "(OpenAI-compatible .../v1/chat/completions). "
            "Start a vLLM, SGLang, or NIM stack (see deploy/)."
        )
    return resolved


def extract_assistant_text(result: Dict[str, Any]) -> str:
    try:
        choice0 = result["choices"][0]
        if "message" in choice0:
            return choice0["message"].get("content") or ""
        return choice0.get("text") or ""
    except (KeyError, IndexError, TypeError):
        return ""


def chat_completions(
    url: str,
    messages: List[Dict[str, Any]],
    *,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    api_key: str = "",
    timeout: float = 300.0,
    extra: Optional[Dict[str, Any]] = None,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    POST one chat/completions request. Returns OpenAI-style JSON body.
    Raises requests.HTTPError / RequestException on failure.
    """
    endpoint = normalize_chat_url(url)
    if not endpoint:
        raise ValueError("chat_completions: empty url")

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if extra:
        payload.update(extra)

    headers = {"Content-Type": "application/json"}
    key = (api_key or os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))).strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    http = session or requests
    resp = http.post(endpoint, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected response type from {endpoint}: {type(data)}")
    return data


class OpenAIChatClient:
    """
    Pool-aware client: picks next URL via callback (e.g. ClusterInfo.next_backend).
    """

    def __init__(
        self,
        *,
        next_url: Callable[[], Optional[str]],
        model: str,
        api_key: str = "",
        timeout: float = 300.0,
        mark_success: Optional[Callable[[str], None]] = None,
        mark_failure: Optional[Callable[[str], None]] = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 1024,
    ):
        self.next_url = next_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.mark_success = mark_success
        self.mark_failure = mark_failure
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self._session = requests.Session()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        url = self.next_url()
        if not url:
            raise RuntimeError(
                "No inference backend available "
                "(BACKEND_URLS / VLLM_API_URL / SGLANG_API_URL empty or all cooling down)"
            )
        endpoint = normalize_chat_url(url)
        try:
            data = chat_completions(
                endpoint,
                messages,
                model=model or self.model,
                temperature=(
                    self.default_temperature if temperature is None else temperature
                ),
                max_tokens=(
                    self.default_max_tokens if max_tokens is None else max_tokens
                ),
                api_key=self.api_key,
                timeout=self.timeout,
                extra=extra,
                session=self._session,
            )
            if self.mark_success:
                self.mark_success(endpoint)
            return data, extract_assistant_text(data)
        except Exception:
            if self.mark_failure:
                self.mark_failure(endpoint)
            raise

    def usage_tuple(self, data: Dict[str, Any]) -> Tuple[int, int]:
        usage = data.get("usage") or {}
        return int(usage.get("prompt_tokens") or 0), int(
            usage.get("completion_tokens") or 0
        )
