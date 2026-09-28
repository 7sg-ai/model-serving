"""
OpenAI-compatible tools / tool_calls helpers for IDE assistant proxies.

Cline, Cursor, and similar clients send `tools` + `tool_choice` and expect
assistant `tool_calls` (JSON and SSE) to round-trip unchanged. The IDE executes
tools; this server only forwards.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple



IDE_TOOL_NUDGE_TEXT = (
    "You are connected to an IDE coding agent with tools (read/list/search/edit files, "
    "run commands, etc.). When the user refers to code in the workspace, call the "
    "provided tools to inspect it yourself. Do not ask the user to paste code that "
    "tools can read. After tool results arrive, continue the task."
)


# Body fields the proxy owns. Everything else the client sends is forwarded.
_OWNED_BODY_FIELDS = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "stream",
        "user",
        "conversation_id",
        "session_id",
    }
)

_CONTEXT_ERROR_RE = re.compile(
    r"context length|maximum context|context window|context_length|"
    r"too many tokens|token limit|max[_ ]tokens|prompt is too long|"
    r"exceeds the (?:model'?s )?maximum|input is too long|"
    r"requested \d+ tokens|reduce the length",
    re.IGNORECASE,
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def request_has_tools(tools: Any) -> bool:
    return isinstance(tools, list) and len(tools) > 0


def messages_have_tool_protocol(messages: Optional[Sequence[Dict[str, Any]]]) -> bool:
    """True if history already includes tool calls or tool results."""
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "tool":
            return True
        if m.get("tool_calls"):
            return True
        if role == "assistant" and m.get("function_call"):
            return True
    return False


def messages_have_rich_content(messages: Optional[Sequence[Dict[str, Any]]]) -> bool:
    """True when any message content is a non-string part list (images, etc.)."""
    for m in messages or []:
        if isinstance(m, dict) and isinstance(m.get("content"), list):
            return True
    return False


def should_preserve_client_transcript(
    tools: Any,
    messages: Optional[Sequence[Dict[str, Any]]] = None,
) -> bool:
    """
    Client owns the transcript: do not inject a system prompt or merge memory.

    Covers tool schemas, in-progress tool loops, and multimodal parts.
    """
    return (
        request_has_tools(tools)
        or messages_have_tool_protocol(messages)
        or messages_have_rich_content(messages)
    )


def should_bypass_memory_merge(
    tools: Any,
    messages: Optional[Sequence[Dict[str, Any]]] = None,
) -> bool:
    """Avoid rewriting client history when a tool or vision turn is active."""
    return should_preserve_client_transcript(tools, messages)


def is_tool_continuation(messages: Optional[Sequence[Dict[str, Any]]]) -> bool:
    """
    True when the client is mid tool-loop and the serving model must not change.

    A fresh user turn (latest message role is user) is not a continuation, even
    if older history contains tool results.
    """
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    if not msgs:
        return False
    last = msgs[-1]
    role = last.get("role")
    if role == "tool":
        return True
    if role == "assistant" and (last.get("tool_calls") or last.get("function_call")):
        return True
    if role != "user" and messages_have_tool_protocol(msgs):
        return True
    return False


def should_skip_response_cache(
    tools: Any,
    messages: Optional[Sequence[Dict[str, Any]]] = None,
    stream: bool = False,
) -> bool:
    if stream:
        return True
    return should_bypass_memory_merge(tools, messages)


def resolve_max_tokens(
    body: Optional[Dict[str, Any]],
    default: int,
    *,
    cap: Optional[int] = None,
) -> int:
    """
    Honor max_tokens, else max_completion_tokens (newer OpenAI clients).

    `cap` is an exclusive upper bound (typically context_window - reserve).
    """
    data = body or {}
    raw = data.get("max_tokens")
    if raw is None:
        raw = data.get("max_completion_tokens")
    if raw is None:
        value = int(default)
    else:
        value = int(raw)
    if cap is not None:
        value = min(value, int(cap))
    return max(1, value)


def build_passthrough_extra(
    body: Optional[Dict[str, Any]],
    *,
    deny: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Forward client body fields the proxy does not own.

    Includes tools, tool_choice, stream_options, response_format, top_p,
    stop, reasoning fields, and any unknown vendor extensions. Drops fields
    the proxy sets itself so a client cannot override model routing or stream.
    """
    data = body or {}
    blocked = set(_OWNED_BODY_FIELDS)
    if deny:
        blocked.update(deny)
    extra: Dict[str, Any] = {}
    for key, value in data.items():
        if key in blocked or value is None:
            continue
        extra[key] = value
    return extra


def stream_options_want_usage(extra: Optional[Dict[str, Any]]) -> bool:
    opts = (extra or {}).get("stream_options")
    return isinstance(opts, dict) and bool(opts.get("include_usage"))


def stable_session_id(
    *,
    header_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    messages: Optional[Sequence[Dict[str, Any]]] = None,
    user: Optional[str] = None,
    fallback: str = "default",
) -> str:
    """
    Session key for Switchyard latch state.

    Prefer an explicit conversation id. Agents usually omit `user`, so fall
    back to a hash of the client system prompt (stable across a task) before
    using `user`, which would otherwise collapse every client onto one latch.
    """
    for candidate in (header_id, conversation_id):
        text = (candidate or "").strip()
        if text:
            return text
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") != "system":
            continue
        content = m.get("content")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        if isinstance(content, str) and content.strip():
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            return f"sys:{digest}"
    text = (user or "").strip()
    if text:
        return text
    return fallback


def build_tools_extra(
    tools: Any = None,
    tool_choice: Any = None,
    parallel_tool_calls: Any = None,
) -> Dict[str, Any]:
    """Fields to merge into the upstream chat.completions body."""
    extra: Dict[str, Any] = {}
    if request_has_tools(tools):
        extra["tools"] = tools
    if tool_choice is not None:
        extra["tool_choice"] = tool_choice
    if parallel_tool_calls is not None:
        extra["parallel_tool_calls"] = parallel_tool_calls
    return extra


def apply_ide_tool_nudge(
    messages: List[Dict[str, Any]],
    tools: Any = None,
    *,
    enabled: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Prepend a short system nudge when the client advertises tools.

    Controlled by IDE_TOOL_NUDGE (default false). Native IDE agents already
    ship their own system prompt; injecting one changes tool-choice behavior.
    """
    if not request_has_tools(tools):
        return list(messages or [])
    use = _env_bool("IDE_TOOL_NUDGE", False) if enabled is None else bool(enabled)
    if not use:
        return list(messages or [])

    out = list(messages or [])
    # Skip if an equivalent nudge is already present
    for m in out:
        if m.get("role") == "system":
            c = m.get("content") or ""
            if isinstance(c, str) and "IDE coding agent with tools" in c:
                return out

    out.insert(0, {"role": "system", "content": IDE_TOOL_NUDGE_TEXT})
    return out


def extract_tool_calls(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        msg = result["choices"][0].get("message") or {}
        tcs = msg.get("tool_calls")
        return list(tcs) if isinstance(tcs, list) else []
    except (KeyError, IndexError, TypeError):
        return []


def response_has_tool_calls(result: Optional[Dict[str, Any]]) -> bool:
    return bool(extract_tool_calls(result or {}))


def message_content_for_memory(result: Dict[str, Any]) -> str:
    """Assistant text only; empty when the model issued tool_calls without prose."""
    try:
        msg = result["choices"][0].get("message") or {}
        content = msg.get("content")
        if content:
            return content if isinstance(content, str) else str(content)
    except (KeyError, IndexError, TypeError):
        pass
    return ""


def _error_blob(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(payload)


def is_context_length_error(
    message: str = "",
    payload: Any = None,
    status: Optional[int] = None,
) -> bool:
    """True when an upstream failure is a context / token-limit overflow."""
    blob = " ".join(part for part in (message or "", _error_blob(payload)) if part)
    if not blob:
        return False
    if _CONTEXT_ERROR_RE.search(blob):
        return True
    if isinstance(payload, dict):
        err = payload.get("error") if isinstance(payload.get("error"), dict) else payload
        code = str((err or {}).get("code") or "")
        if code in ("context_length_exceeded", "context_length_exceeded_error"):
            return True
    if status == 413 and re.search(r"token|context|length", blob, re.IGNORECASE):
        return True
    return False


def openai_error_from_upstream(
    exc: BaseException,
    *,
    fallback_code: str = "upstream_error",
) -> Tuple[Dict[str, Any], int]:
    """
    Map an upstream chat failure to an OpenAI-style error body and HTTP status.

    Context overflows become `context_length_exceeded` / 400 so IDE clients
    can compact history. Other failures keep the upstream status when known.
    """
    status: Optional[int] = None
    payload: Any = None
    message = str(exc)
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            status = int(response.status_code)
        except (TypeError, ValueError):
            status = None
        try:
            payload = response.json()
        except Exception:
            try:
                message = response.text or message
            except Exception:
                pass
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and err.get("message"):
                message = str(err.get("message"))
            elif payload.get("message"):
                message = str(payload.get("message"))

    if is_context_length_error(message, payload, status):
        return (
            {
                "error": {
                    "message": message or "This model's maximum context length was exceeded.",
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                }
            },
            400,
        )

    http_status = status if status and 400 <= status < 600 else 500
    err_type = "invalid_request_error" if http_status < 500 else "server_error"
    code = fallback_code
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            err_type = str(err.get("type") or err_type)
            code = str(err.get("code") or code)
    return (
        {
            "error": {
                "message": message or "Upstream chat completion failed.",
                "type": err_type,
                "code": code,
            }
        },
        http_status,
    )


# Model id fragments known to support OpenAI-style tool calling well on
# NIM / vLLM (function-calling tuned instruct models).
TOOL_CAPABLE_HINTS = (
    "qwen3",
    "qwen2.5-coder",
    "qwen2.5-72b",
    "qwen2.5-32b",
    "qwen2.5-14b",
    "qwen2.5-7b",
    "llama-3.3",
    "llama-3.1-70b",
    "llama-3.1-405b",
    "mistral-large",
    "mixtral-8x22b",
    "deepseek-v3",
    "deepseek-r1",
    "deepseek-coder",
    "kimi-k2",
    "kimi-k3",
    "gpt-oss",
    "devstral",
    "codestral",
    "firefunction",
    "nemotron-4-340b",
    "command-r",
)

REASONING_HINTS = (
    "deepseek-r1",
    "qwen3",
    "qwq",
    "reasoning",
    "think",
    "kimi-k2",
    "kimi-k3",
    "gpt-oss",
)


def looks_tool_capable(model_id: str) -> bool:
    """Heuristic: does this model id look like a function-calling tuned model?"""
    mid = (model_id or "").lower()
    if not mid:
        return False
    return any(h in mid for h in TOOL_CAPABLE_HINTS)


def looks_reasoning_capable(model_id: str) -> bool:
    mid = (model_id or "").lower()
    if not mid:
        return False
    return any(h in mid for h in REASONING_HINTS)


def model_capabilities(
    model_id: str,
    *,
    tools: Optional[bool] = None,
    reasoning: Optional[bool] = None,
    route: bool = False,
) -> Dict[str, bool]:
    """Non-standard capability flags clients already tolerate on /v1/models."""
    return {
        "tools": True if route else (looks_tool_capable(model_id) if tools is None else bool(tools)),
        "reasoning": looks_reasoning_capable(model_id) if reasoning is None else bool(reasoning),
    }


def annotate_model_entry(entry: Dict[str, Any], *, route: bool = False) -> Dict[str, Any]:
    """Add capabilities (and context_window if missing) onto a /v1/models item."""
    model_id = str(entry.get("id") or "")
    is_route = route or entry.get("root") == "switchyard-route" or bool(entry.get("strategy"))
    caps = model_capabilities(model_id, route=is_route)
    existing = entry.get("capabilities")
    if isinstance(existing, dict):
        caps = {**caps, **{k: bool(v) for k, v in existing.items()}}
    entry["capabilities"] = caps
    if "context_window" not in entry and entry.get("effective_context_window"):
        entry["context_window"] = entry["effective_context_window"]
    return entry


def warn_if_not_tool_capable(model_id: str, *, app_name: str = "ide") -> None:
    if looks_tool_capable(model_id):
        return
    print(
        f"[ide-tools] WARNING: model '{model_id}' is not in the known tool-capable "
        f"list for {app_name}. IDE agents (Cline/Cursor) may ignore tool schemas and "
        f"ask the user to paste code instead. Consider a function-calling tuned model "
        f"(e.g. qwen2.5-coder / qwen3 / llama-3.3 / kimi / deepseek)."
    )
