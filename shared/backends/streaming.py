"""
OpenAI-compatible SSE (Server-Sent Events) streaming helpers.

Cline and most IDE clients send chat.completions with stream=true and expect
text/event-stream bodies. Proxies must not call response.json() on SSE.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

import requests

from .openai_chat import normalize_chat_url

logger = logging.getLogger(__name__)

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def extract_stream_delta_text(payload: Dict[str, Any]) -> str:
    """Pull assistant text delta from one OpenAI chat.completion.chunk object."""
    try:
        choice0 = payload["choices"][0]
        delta = choice0.get("delta") or {}
        content = delta.get("content")
        if content:
            return content if isinstance(content, str) else str(content)
        message = choice0.get("message") or {}
        content = message.get("content")
        if content:
            return content if isinstance(content, str) else str(content)
        text = choice0.get("text")
        if text:
            return text if isinstance(text, str) else str(text)
    except (KeyError, IndexError, TypeError):
        pass
    return ""


def parse_sse_data_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse one SSE line. Returns dict payload, None for [DONE]/empty/non-data."""
    line = (line or "").strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def post_chat_stream(
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
    headers: Optional[Dict[str, str]] = None,
) -> requests.Response:
    """
    POST chat/completions with stream=true. Returns an open streaming Response.
    Caller must close the response (generators below do this).
    """
    endpoint = normalize_chat_url(url)
    if not endpoint:
        raise ValueError("post_chat_stream: empty url")

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if extra:
        payload.update(extra)
        payload["stream"] = True

    hdrs = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if headers:
        hdrs.update(headers)
    key = (api_key or "").strip()
    if key and "Authorization" not in hdrs:
        hdrs["Authorization"] = f"Bearer {key}"

    http = session or requests
    connect_t = min(30.0, float(timeout) if timeout else 30.0)
    read_t = float(timeout) if timeout else 300.0
    resp = http.post(
        endpoint,
        json=payload,
        headers=hdrs,
        timeout=(connect_t, read_t),
        stream=True,
    )
    resp.raise_for_status()
    return resp


def iter_upstream_sse(
    upstream: requests.Response,
    *,
    on_delta: Optional[Callable[[str], None]] = None,
    ensure_done: bool = True,
) -> Iterator[bytes]:
    """Yield SSE bytes from upstream; optionally collect content deltas."""
    assistant_parts: List[str] = []
    line_carry = ""
    try:
        for raw in upstream.iter_content(chunk_size=1024):
            if not raw:
                continue
            try:
                text = raw.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
            if text:
                combined = line_carry + text
                lines = combined.splitlines(keepends=True)
                if lines and not lines[-1].endswith(("\n", "\r")):
                    line_carry = lines.pop()
                else:
                    line_carry = ""
                for line in lines:
                    payload = parse_sse_data_line(line.rstrip("\r\n"))
                    if payload is None:
                        continue
                    delta = extract_stream_delta_text(payload)
                    if delta:
                        assistant_parts.append(delta)
                        if on_delta:
                            try:
                                on_delta(delta)
                            except Exception:
                                logger.debug("on_delta failed", exc_info=True)
            yield raw

        if line_carry:
            payload = parse_sse_data_line(line_carry)
            if payload is not None:
                delta = extract_stream_delta_text(payload)
                if delta:
                    assistant_parts.append(delta)
                    if on_delta:
                        try:
                            on_delta(delta)
                        except Exception:
                            pass

        if ensure_done:
            yield b"data: [DONE]\n\n"
    finally:
        try:
            upstream.close()
        except Exception:
            pass
        try:
            upstream._collected_assistant_text = "".join(assistant_parts)  # type: ignore[attr-defined]
        except Exception:
            pass


def flask_sse_from_upstream(
    upstream: requests.Response,
    *,
    on_complete: Optional[Callable[[str], None]] = None,
    status: int = 200,
):
    """Flask Response that proxies upstream OpenAI SSE."""
    from flask import Response, stream_with_context

    collected: List[str] = []

    def _on_delta(delta: str) -> None:
        collected.append(delta)

    def generate() -> Iterator[bytes]:
        try:
            for chunk in iter_upstream_sse(upstream, on_delta=_on_delta, ensure_done=True):
                yield chunk
        finally:
            if on_complete:
                try:
                    on_complete("".join(collected))
                except Exception:
                    logger.debug("on_complete failed", exc_info=True)

    try:
        from flask import has_request_context

        body = stream_with_context(generate()) if has_request_context() else generate()
    except Exception:
        body = generate()

    return Response(
        body,
        status=status,
        mimetype="text/event-stream",
        headers=dict(SSE_HEADERS),
    )


def completion_to_sse_chunks(
    result: Dict[str, Any],
    *,
    model: Optional[str] = None,
    chunk_chars: int = 48,
) -> Iterator[str]:
    """Convert a non-stream chat.completion JSON body into SSE chunk lines."""
    created = int(result.get("created") or time.time())
    resp_id = result.get("id") or f"chatcmpl-{created}"
    model_out = model or result.get("model") or "unknown"
    text = ""
    finish_reason = "stop"
    try:
        choice0 = result["choices"][0]
        finish_reason = choice0.get("finish_reason") or "stop"
        msg = choice0.get("message") or {}
        text = msg.get("content") or choice0.get("text") or ""
    except (KeyError, IndexError, TypeError):
        text = ""

    role_chunk = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_out,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n"

    if not isinstance(text, str):
        text = str(text or "")

    step = max(1, int(chunk_chars))
    for i in range(0, len(text), step):
        piece = text[i : i + step]
        chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_out,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    end_chunk: Dict[str, Any] = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_out,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    if isinstance(result.get("usage"), dict):
        end_chunk["usage"] = result["usage"]
    yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def flask_sse_from_completion(
    result: Dict[str, Any],
    *,
    model: Optional[str] = None,
    on_complete: Optional[Callable[[str], None]] = None,
    chunk_chars: int = 48,
):
    """Flask SSE response synthesized from a full chat.completion dict."""
    from flask import Response, stream_with_context

    text = ""
    try:
        text = (result.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    except Exception:
        text = ""

    def generate() -> Iterator[str]:
        try:
            for line in completion_to_sse_chunks(
                result, model=model, chunk_chars=chunk_chars
            ):
                yield line
        finally:
            if on_complete:
                try:
                    on_complete(text if isinstance(text, str) else str(text))
                except Exception:
                    logger.debug("on_complete failed", exc_info=True)

    try:
        from flask import has_request_context

        body = stream_with_context(generate()) if has_request_context() else generate()
    except Exception:
        body = generate()

    return Response(
        body,
        status=200,
        mimetype="text/event-stream",
        headers=dict(SSE_HEADERS),
    )
