"""Anthropic Messages → OpenAI chat request conversion."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# Claude Code model ids are tier hints, not backend ids.
_TIER_HINTS = (
    ("opus", "strong"),
    ("sonnet", "strong"),
    ("haiku", "weak"),
)


def claude_tier_hint(model_id: Optional[str]) -> Optional[str]:
    """Map a Claude-style model id to a Switchyard role, or None.

    opus/sonnet ask for the strong tier. haiku asks for the weak tier.
    """
    mid = (model_id or "").lower()
    if not mid:
        return None
    for needle, role in _TIER_HINTS:
        if needle in mid:
            return role
    return None


def block_text(content: Any) -> str:
    """Flatten Anthropic content to text. Drops thinking and cache markers."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float)):
        return str(content)
    if isinstance(content, dict):
        kind = content.get("type")
        if kind in (None, "text"):
            return str(content.get("text") or content.get("content") or "")
        if kind == "tool_result":
            return block_text(content.get("content"))
        return ""
    if isinstance(content, list):
        parts = [block_text(item) for item in content]
        return "\n".join(p for p in parts if p)
    return str(content)


def _tool_choice_to_openai(choice: Any) -> Any:
    if choice is None or isinstance(choice, str):
        return choice
    if not isinstance(choice, dict):
        return None
    kind = choice.get("type")
    mapped = {"auto": "auto", "any": "required", "none": "none"}.get(kind)
    if mapped:
        return mapped
    if kind == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


def _arguments_json(value: Any) -> str:
    """OpenAI function.arguments is a JSON string. Anthropic input is an object."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value if value is not None else {}, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _convert_message(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    role = msg.get("role") or "user"
    content = msg.get("content")
    if isinstance(content, str) or content is None:
        return [{"role": role, "content": content or ""}]
    if not isinstance(content, list):
        return [{"role": role, "content": block_text(content)}]

    text_parts: List[str] = []
    thinking_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text_parts.append(str(block.get("text") or ""))
        elif kind == "thinking":
            # signature is Anthropic-only. The text is what the next turn needs.
            thought = block.get("thinking")
            if isinstance(thought, str) and thought.strip():
                thinking_parts.append(thought)
        elif kind == "tool_use":
            # caller is not an OpenAI field. Claude Code already has it.
            tool_calls.append({
                "id": block.get("id") or f"toolu_{len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": block.get("name") or "",
                    "arguments": _arguments_json(block.get("input")),
                },
            })
        elif kind == "tool_result":
            tool_results.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id") or "",
                "content": block_text(block.get("content")),
            })
        # redacted_thinking, cache_control: drop. Sibling text is kept above.

    if role == "assistant":
        assistant: Dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(p for p in text_parts if p) or None,
        }
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        if thinking_parts:
            assistant["reasoning_content"] = "\n".join(thinking_parts)
        if assistant["content"] or tool_calls or thinking_parts:
            return [assistant]
        return []
    out: List[Dict[str, Any]] = []
    text = "\n".join(p for p in text_parts if p)
    if text:
        out.append({"role": "user", "content": text})
    out.extend(tool_results)
    return out


def _convert_tools(raw_tools: Any) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    for tool in raw_tools or []:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        # Hosted Anthropic tools (no name) are skipped above. MCP names
        # (mcp__server__tool) pass through unchanged. cache_control is dropped.
        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description") or "",
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return tools


def anthropic_to_openai(body: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Anthropic Messages request into an OpenAI chat body.

    Thinking becomes assistant reasoning_content so the next turn keeps it.
    cache_control, caller, signatures, and redacted_thinking stay off the
    body — NIM and vLLM 400 on those Anthropic-only fields.
    """
    messages: List[Dict[str, Any]] = []
    system = body.get("system")
    if system:
        text = block_text(system)
        if text:
            messages.append({"role": "system", "content": text})
    for msg in body.get("messages") or []:
        if isinstance(msg, dict):
            messages.extend(_convert_message(msg))

    out: Dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
        "max_tokens": body.get("max_tokens") or 4096,
        "stream": bool(body.get("stream")),
    }
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]
    tools = _convert_tools(body.get("tools"))
    if tools:
        out["tools"] = tools
    choice = _tool_choice_to_openai(body.get("tool_choice"))
    if choice is not None:
        out["tool_choice"] = choice
    if body.get("user"):
        out["user"] = body["user"]
    metadata = body.get("metadata")
    if isinstance(metadata, dict) and metadata.get("user_id"):
        out.setdefault("user", metadata["user_id"])
    return out


def strip_reasoning_content(body: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a chat body without reasoning_content.

    Used once when a backend 400s on that field. Other clients are untouched
    because this only runs on the Anthropic handler's retry.
    """
    cleaned: Dict[str, Any] = dict(body)
    messages = []
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict) or "reasoning_content" not in msg:
            messages.append(msg)
            continue
        messages.append({k: v for k, v in msg.items() if k != "reasoning_content"})
    cleaned["messages"] = messages
    return cleaned



