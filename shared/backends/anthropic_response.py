"""OpenAI chat.completion → Anthropic Messages response conversion."""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .anthropic_convert import block_text


_STOP_REASON = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}

_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>\s*", re.DOTALL | re.IGNORECASE)
_DIRECT_CALLER = {"type": "direct"}


def direct_caller() -> Dict[str, str]:
    """These models do not do Anthropic programmatic tool calling."""
    return dict(_DIRECT_CALLER)


def parse_tool_input(raw: Any) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Return (input, truncated). Truncated JSON must not become a tool_use."""
    if raw is None or raw == "":
        return {}, False
    if isinstance(raw, dict):
        return raw, False
    if not isinstance(raw, str):
        return {"value": raw}, False
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None, True
    if isinstance(parsed, dict):
        return parsed, False
    return {"value": parsed}, False


def repair_tool_input(raw: Any) -> Optional[Dict[str, Any]]:
    """Close a cut-off JSON string and brackets. Do not invent fields.

    Returns an object only when the repaired text parses. Empty ``{}`` is
    valid only if that is what the model actually wrote.
    """
    parsed, bad = parse_tool_input(raw)
    if not bad and isinstance(parsed, dict):
        return parsed
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    in_string = False
    escape = False
    stack: List[str] = []
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    if in_string:
        text += '"'
    text = text.rstrip().rstrip(",")
    for opener in reversed(stack):
        text += "}" if opener == "{" else "]"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


def tool_use_blocks(tool_calls: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Convert OpenAI tool_calls. Recover cut JSON. Keep sibling calls.

    A call that still will not parse is named in the dropped list and omitted.
    Never substitute ``{}`` for arguments the model did not write.
    """
    blocks: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        parsed = repair_tool_input(fn.get("arguments") if fn.get("arguments") is not None else "{}")
        name = str(fn.get("name") or "")
        if parsed is None:
            dropped.append(name or "tool")
            continue
        blocks.append({
            "type": "tool_use",
            "id": call.get("id") or "toolu_0",
            "name": name,
            "input": parsed,
            "caller": direct_caller(),
        })
    return blocks, dropped


def split_thinking(text: str) -> Tuple[str, str]:
    """Pull a leading <think> span out of assistant content."""
    if not text:
        return "", ""
    match = _THINK_RE.match(text)
    if not match:
        return "", text
    return match.group(1).strip(), text[match.end():]


def openai_to_anthropic(result: Dict[str, Any], *, requested_model: str) -> Dict[str, Any]:
    """Convert an OpenAI chat.completion into an Anthropic message.

    Echoes the model the client asked for. reasoning_content and a leading
    <think> span become a thinking block with no signature. Tool calls are
    marked caller.type=direct. A cut-off call is repaired or omitted; siblings
    that parsed are kept.
    """
    choice = (result.get("choices") or [{}])[0] or {}
    msg = choice.get("message") or {}
    blocks: List[Dict[str, Any]] = []
    reasoning = msg.get("reasoning_content")
    text = msg.get("content")
    if not isinstance(text, str):
        text = block_text(text) if text else ""
    inline, text = split_thinking(text)
    thought = reasoning.strip() if isinstance(reasoning, str) else ""
    if not thought:
        thought = inline
    if thought:
        blocks.append({"type": "thinking", "thinking": thought})
    if text:
        blocks.append({"type": "text", "text": text})

    tool_blocks, dropped = tool_use_blocks(msg.get("tool_calls"))
    blocks.extend(tool_blocks)

    finish = choice.get("finish_reason") or "stop"
    if tool_blocks and finish == "stop":
        finish = "tool_calls"
    if tool_blocks:
        stop_reason = "tool_use"
    elif dropped and finish == "length":
        stop_reason = "max_tokens"
    elif dropped:
        stop_reason = "end_turn"
        names = ", ".join(dropped)
        blocks.append({
            "type": "text",
            "text": f"Dropped an incomplete tool call ({names}). No arguments were invented.",
        })
    else:
        stop_reason = _STOP_REASON.get(finish, "end_turn")

    usage = result.get("usage") or {}
    body: Dict[str, Any] = {
        "id": result.get("id") or f"msg_{int(time.time())}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": blocks or [{"type": "text", "text": ""}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            "output_tokens": int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            ),
        },
    }
    switchyard = result.get("switchyard")
    if isinstance(switchyard, dict):
        body["switchyard"] = switchyard
    return body


def estimate_input_tokens(openai_body: Dict[str, Any]) -> int:
    """Local estimate over translated messages and tool schemas.

    Claude Code uses count_tokens to decide whether the next tool round fits.
    Omitting schemas under-counts and overflows the backend. ~4 chars/token.
    """
    blob = json.dumps(
        {
            "messages": openai_body.get("messages") or [],
            "tools": openai_body.get("tools") or [],
        },
        ensure_ascii=False,
    )
    return max(1, (len(blob) + 3) // 4)


def anthropic_error(message: str, *, error_type: str, status: int) -> Tuple[Dict[str, Any], int]:
    return (
        {"type": "error", "error": {"type": error_type, "message": message}},
        status,
    )
