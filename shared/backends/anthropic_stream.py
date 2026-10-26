"""Anthropic SSE emission and OpenAI SSE → Anthropic SSE translation."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterator, Optional

from .anthropic_response import direct_caller, repair_tool_input


_STOP_REASON = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}


def sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def anthropic_message_to_sse(message: Dict[str, Any]) -> Iterator[str]:
    """Emit Anthropic SSE for a finished Messages response.

    Tool arguments go out as one input_json_delta. Claude Code concatenates
    partial_json and parses it once at content_block_stop.
    """
    msg_id = message.get("id") or f"msg_{int(time.time())}"
    usage = message.get("usage") or {}
    yield sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": message.get("model") or "unknown",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": int(usage.get("input_tokens") or 0),
                "output_tokens": 0,
            },
        },
    })
    for index, block in enumerate(message.get("content") or []):
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "thinking":
            yield sse("content_block_start", {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "thinking", "thinking": ""},
            })
            if block.get("thinking"):
                yield sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": block["thinking"]},
                })
        elif kind == "text":
            yield sse("content_block_start", {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            })
            if block.get("text"):
                yield sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": block["text"]},
                })
        elif kind == "tool_use":
            yield sse("content_block_start", {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": block.get("id") or f"toolu_{index}",
                    "name": block.get("name") or "",
                    "input": {},
                    "caller": block.get("caller") or direct_caller(),
                },
            })
            yield sse("content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block.get("input") or {}, ensure_ascii=False),
                },
            })
        else:
            continue
        yield sse("content_block_stop", {"type": "content_block_stop", "index": index})
    yield sse("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": message.get("stop_reason") or "end_turn",
            "stop_sequence": None,
        },
        "usage": {"output_tokens": int(usage.get("output_tokens") or 0)},
    })
    yield sse("message_stop", {"type": "message_stop"})


class AnthropicStreamAssembler:
    """Buffer OpenAI tool-call fragments, then emit Anthropic events.

    OpenAI sends the function name on the first fragment and arguments later,
    keyed by tool_calls[].index. Hold each index until finish_reason, then
    emit one input_json_delta. Cut JSON is closed and reparsed. A call that
    still will not parse is omitted. Siblings that parsed are kept.
    """

    def __init__(self, *, requested_model: str, message_id: str) -> None:
        self.requested_model = requested_model
        self.message_id = message_id
        self.started = False
        self.input_tokens = 0
        self.output_tokens = 0
        self.next_index = 0
        self.text_open: Optional[int] = None
        self.finish_reason: Optional[str] = None
        self.tool_bufs: Dict[int, Dict[str, str]] = {}
        self.thinking_open: Optional[int] = None

    def start_events(self) -> Iterator[str]:
        if self.started:
            return
        self.started = True
        yield sse("message_start", {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "model": self.requested_model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
            },
        })

    def feed_chunk(self, chunk: Dict[str, Any]) -> Iterator[str]:
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.input_tokens = int(
                usage.get("prompt_tokens") or usage.get("input_tokens") or self.input_tokens
            )
            self.output_tokens = int(
                usage.get("completion_tokens") or usage.get("output_tokens") or self.output_tokens
            )
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            text = delta.get("content")
            if isinstance(text, str) and text:
                yield from self._text_delta(text)
            thought = delta.get("reasoning_content")
            if isinstance(thought, str) and thought:
                yield from self._thinking_delta(thought)
            for tc in delta.get("tool_calls") or []:
                if isinstance(tc, dict):
                    yield from self._buffer_tool(tc)

    def _thinking_delta(self, text: str) -> Iterator[str]:
        yield from self.start_events()
        if self.thinking_open is None:
            yield from self._close_text()
            self.thinking_open = self.next_index
            self.next_index += 1
            yield sse("content_block_start", {
                "type": "content_block_start",
                "index": self.thinking_open,
                "content_block": {"type": "thinking", "thinking": ""},
            })
        yield sse("content_block_delta", {
            "type": "content_block_delta",
            "index": self.thinking_open,
            "delta": {"type": "thinking_delta", "thinking": text},
        })

    def _text_delta(self, text: str) -> Iterator[str]:
        yield from self.start_events()
        if self.text_open is None:
            self.text_open = self.next_index
            self.next_index += 1
            yield sse("content_block_start", {
                "type": "content_block_start",
                "index": self.text_open,
                "content_block": {"type": "text", "text": ""},
            })
        yield sse("content_block_delta", {
            "type": "content_block_delta",
            "index": self.text_open,
            "delta": {"type": "text_delta", "text": text},
        })

    def _buffer_tool(self, tc: Dict[str, Any]) -> Iterator[str]:
        yield from self._close_thinking()
        yield from self._close_text()
        idx = int(tc.get("index") or 0)
        buf = self.tool_bufs.setdefault(idx, {"id": "", "name": "", "arguments": ""})
        if tc.get("id"):
            buf["id"] = tc["id"]
        fn = tc.get("function") or {}
        if isinstance(fn, dict):
            if fn.get("name"):
                buf["name"] = fn["name"]
            if fn.get("arguments"):
                buf["arguments"] += fn["arguments"]
        return
        yield  # pragma: no cover — keeps this a generator

    def _close_text(self) -> Iterator[str]:
        if self.text_open is None:
            return
        yield sse("content_block_stop", {"type": "content_block_stop", "index": self.text_open})
        self.text_open = None

    def _close_thinking(self) -> Iterator[str]:
        if self.thinking_open is None:
            return
        yield sse("content_block_stop", {
            "type": "content_block_stop",
            "index": self.thinking_open,
        })
        self.thinking_open = None

    def finish_events(self) -> Iterator[str]:
        yield from self.start_events()
        yield from self._close_thinking()
        yield from self._close_text()
        emitted_tool = False
        dropped = []
        for idx in sorted(self.tool_bufs):
            buf = self.tool_bufs[idx]
            parsed = repair_tool_input(buf.get("arguments") or "{}")
            if parsed is None:
                dropped.append(buf.get("name") or "tool")
                continue
            block_index = self.next_index
            self.next_index += 1
            emitted_tool = True
            yield sse("content_block_start", {
                "type": "content_block_start",
                "index": block_index,
                "content_block": {
                    "type": "tool_use",
                    "id": buf.get("id") or f"toolu_{idx}",
                    "name": buf.get("name") or "",
                    "input": {},
                    "caller": direct_caller(),
                },
            })
            yield sse("content_block_delta", {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(parsed, ensure_ascii=False),
                },
            })
            yield sse("content_block_stop", {
                "type": "content_block_stop",
                "index": block_index,
            })
        if emitted_tool:
            stop_reason = "tool_use"
        elif dropped and self.finish_reason == "length":
            stop_reason = "max_tokens"
        elif dropped or self.finish_reason == "tool_calls":
            stop_reason = "end_turn"
            if dropped:
                names = ", ".join(dropped)
                note = f"Dropped an incomplete tool call ({names}). No arguments were invented."
                yield from self._text_delta(note)
                yield from self._close_text()
        else:
            stop_reason = _STOP_REASON.get(self.finish_reason or "stop", "end_turn")
        yield sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": self.output_tokens},
        })
        yield sse("message_stop", {"type": "message_stop"})


def iter_openai_sse_as_anthropic(
    lines: Iterator[str],
    *,
    requested_model: str,
    message_id: Optional[str] = None,
) -> Iterator[str]:
    """Translate an OpenAI SSE text stream into Anthropic SSE events."""
    assembler = AnthropicStreamAssembler(
        requested_model=requested_model,
        message_id=message_id or f"msg_{int(time.time())}",
    )
    for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        for line in str(raw).splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk, dict):
                yield from assembler.feed_chunk(chunk)
    yield from assembler.finish_events()


