"""Flask handlers for POST /v1/messages and /v1/messages/count_tokens.

Inference is delegated to the proxy's existing chat_completions path. This
module only authenticates the Anthropic header, translates the body, and
writes Anthropic JSON or SSE back.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from .anthropic_convert import anthropic_to_openai, strip_reasoning_content
from .anthropic_response import anthropic_error, estimate_input_tokens, openai_to_anthropic
from .anthropic_stream import anthropic_message_to_sse, iter_openai_sse_as_anthropic
from .streaming import SSE_HEADERS, iter_upstream_sse


ChatFn = Callable[[Dict[str, Any]], Any]
ResolveFn = Callable[[Optional[str], bool], Optional[str]]
_ERROR = object()


def anthropic_api_key(req, expected: str) -> bool:
    """Accept x-api-key (Anthropic) or Authorization: Bearer (OpenAI)."""
    token = req.headers.get("x-api-key") or ""
    if not token:
        auth = req.headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            token = auth[7:]
    return bool(expected) and token == expected


def apply_model_hint(body: Dict[str, Any], resolve_hint: Optional[ResolveFn]) -> str:
    """Echo the requested id. Rewrite model only when the hint resolves.

    A real pinned spec is left for the router. haiku/sonnet/opus become the
    weak/strong role when resolve_hint returns that role and no spec matches.
    """
    requested = str(body.get("model") or "")
    if resolve_hint is None:
        return requested
    has_tools = isinstance(body.get("tools"), list) and len(body["tools"]) > 0
    resolved = resolve_hint(requested, has_tools)
    if resolved and resolved != requested:
        body["model"] = resolved
    return requested


def count_tokens_response(req, *, resolve_hint: Optional[ResolveFn] = None):
    from flask import jsonify

    body = req.json or {}
    if not isinstance(body, dict):
        payload, status = anthropic_error("Expected a JSON object.", error_type="invalid_request_error", status=400)
        return jsonify(payload), status
    apply_model_hint(body, resolve_hint)
    openai_body = anthropic_to_openai(body)
    return jsonify({"input_tokens": estimate_input_tokens(openai_body)})


def messages_response(
    req,
    chat: ChatFn,
    *,
    resolve_hint: Optional[ResolveFn] = None,
    default_model: str = "",
):
    """Handle one Anthropic Messages request. ``chat`` is the OpenAI path.

    ``chat(openai_body)`` returns either a Flask response (streaming) or an
    OpenAI chat.completion dict. Upstream OpenAI SSE is translated; a
    synthesized OpenAI SSE response is unwrapped via its completion when the
    caller sets ``response._anthropic_completion``.
    """
    from flask import Response, jsonify, stream_with_context

    body = req.json or {}
    if not isinstance(body, dict) or not body.get("messages"):
        payload, status = anthropic_error(
            "messages is required", error_type="invalid_request_error", status=400
        )
        return jsonify(payload), status

    requested = apply_model_hint(body, resolve_hint) or default_model
    stream = bool(body.get("stream"))
    openai_body = anthropic_to_openai(body)
    openai_body["stream"] = stream
    result = _chat_once(chat, openai_body)
    if isinstance(result, tuple) and result and result[0] == _ERROR:
        return result[1]

    if not stream:
        if not isinstance(result, dict):
            payload, status = anthropic_error(
                "Upstream returned a non-JSON completion.",
                error_type="api_error",
                status=502,
            )
            return jsonify(payload), status
        return jsonify(openai_to_anthropic(result, requested_model=requested))

    message_id = f"msg_{int(time.time())}"
    completion = getattr(result, "_anthropic_completion", None)
    if isinstance(completion, dict):
        message = openai_to_anthropic(completion, requested_model=requested)
        events = anthropic_message_to_sse(message)
    else:
        upstream = getattr(result, "_anthropic_upstream", None)
        source = iter_upstream_sse(upstream) if upstream is not None else _iter_response_body(result)
        events = iter_openai_sse_as_anthropic(
            source, requested_model=requested, message_id=message_id
        )

    def generate():
        for event in events:
            yield event

    return Response(
        stream_with_context(generate()),
        status=200,
        mimetype="text/event-stream",
        headers=dict(SSE_HEADERS),
    )


def _iter_response_body(result: Any):
    response = getattr(result, "response", result)
    body = getattr(response, "iter_encoded", None) or getattr(response, "iter_lines", None)
    if body is None:
        return iter(())
    return body()


def _chat_once(chat: ChatFn, openai_body: Dict[str, Any]):
    """Call the OpenAI path. Retry once without reasoning_content on a 400.

    NIM and vLLM reject that field. The retry stays on this handler so
    Cursor and other OpenAI clients keep sending it.
    """
    from flask import jsonify

    try:
        result = chat(openai_body)
    except Exception as exc:
        if _reasoning_rejected(exc) and _has_reasoning(openai_body):
            try:
                return chat(strip_reasoning_content(openai_body))
            except Exception as retry_exc:
                exc = retry_exc
        payload, status = anthropic_error(str(exc), error_type="api_error", status=502)
        return _ERROR, jsonify(payload), status
    if _reasoning_rejected(result) and _has_reasoning(openai_body):
        return chat(strip_reasoning_content(openai_body))
    return result


def _has_reasoning(body: Dict[str, Any]) -> bool:
    return any(
        isinstance(msg, dict) and msg.get("reasoning_content")
        for msg in body.get("messages") or []
    )


def _reasoning_rejected(result: Any) -> bool:
    status = getattr(result, "status_code", None)
    text = ""
    if isinstance(result, BaseException):
        text = str(result)
    else:
        text = str(getattr(result, "data", None) or getattr(result, "get_data", lambda: b"")() or "")
    blob = text.lower()
    rejected = "reasoning_content" in blob or "reasoning content" in blob
    if isinstance(result, BaseException):
        return rejected
    return status == 400 and rejected
