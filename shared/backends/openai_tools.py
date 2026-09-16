"""
OpenAI-compatible tools / tool_calls helpers for IDE assistant proxies.

Cline, Cursor, and similar clients send `tools` + `tool_choice` and expect
assistant `tool_calls` (JSON and SSE) to round-trip unchanged. The IDE executes
tools; this server only forwards.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence


IDE_TOOL_NUDGE_TEXT = (
    "You are connected to an IDE coding agent with tools (read/list/search/edit files, "
    "run commands, etc.). When the user refers to code in the workspace, call the "
    "provided tools to inspect it yourself. Do not ask the user to paste code that "
    "tools can read. After tool results arrive, continue the task."
)


def _env_bool(name: str, default: bool = True) -> bool:
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


def should_bypass_memory_merge(
    tools: Any,
    messages: Optional[Sequence[Dict[str, Any]]] = None,
) -> bool:
    """
    Avoid rewriting client message history when a tool loop is active.
    Cline owns the transcript of tool_calls / tool results.
    """
    return request_has_tools(tools) or messages_have_tool_protocol(messages)


def should_skip_response_cache(
    tools: Any,
    messages: Optional[Sequence[Dict[str, Any]]] = None,
    stream: bool = False,
) -> bool:
    if stream:
        return True
    return should_bypass_memory_merge(tools, messages)


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
    Controlled by IDE_TOOL_NUDGE (default true).
    """
    if not request_has_tools(tools):
        return list(messages or [])
    use = _env_bool("IDE_TOOL_NUDGE", True) if enabled is None else bool(enabled)
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


def looks_tool_capable(model_id: str) -> bool:
    """Heuristic: does this model id look like a function-calling tuned model?"""
    mid = (model_id or "").lower()
    if not mid:
        return False
    return any(h in mid for h in TOOL_CAPABLE_HINTS)


def warn_if_not_tool_capable(model_id: str, *, app_name: str = "ide") -> None:
    if looks_tool_capable(model_id):
        return
    print(
        f"[ide-tools] WARNING: model '{model_id}' is not in the known tool-capable "
        f"list for {app_name}. IDE agents (Cline/Cursor) may ignore tool schemas and "
        f"ask the user to paste code instead. Consider a function-calling tuned model "
        f"(e.g. qwen2.5-coder / qwen3 / llama-3.3 / kimi / deepseek)."
    )
