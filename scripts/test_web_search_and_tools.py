"""Lightweight tests for web search + tool_calls SSE (no network)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.tools.web_search import extract_search_query, parse_ddg_html
from shared.backends.streaming import completion_to_sse_chunks
from shared.backends.openai_tools import (
    apply_ide_tool_nudge,
    build_passthrough_extra,
    build_tools_extra,
    is_tool_continuation,
    openai_error_from_upstream,
    request_has_tools,
    resolve_max_tokens,
    should_bypass_memory_merge,
    should_preserve_client_transcript,
    stable_session_id,
)


def test_extract_search_query():
    assert extract_search_query("search the web for python 3.13") == "python 3.13"
    assert extract_search_query("/search nvidia nim") == "nvidia nim"
    assert extract_search_query("look up flask streaming") == "flask streaming"
    assert extract_search_query("hello there") is None


def test_parse_ddg_html():
    html = (
        '<div class="result">'
        '<a class="result__a" href="https://example.com/a">Alpha</a>'
        '<div class="result__snippet">Snippet A</div></div>'
    )
    rows = parse_ddg_html(html)
    assert len(rows) == 1
    assert rows[0].url == "https://example.com/a"
    assert "Alpha" in rows[0].title


def test_tools_extra_and_nudge():
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    assert request_has_tools(tools)
    extra = build_tools_extra(tools, "auto", None)
    assert extra["tools"] == tools
    assert extra["tool_choice"] == "auto"
    msgs = [{"role": "user", "content": "what does this code do?"}]
    out = apply_ide_tool_nudge(msgs, tools, enabled=True)
    assert out[0]["role"] == "system"
    assert "IDE coding agent" in out[0]["content"]
    untouched = apply_ide_tool_nudge(msgs, tools, enabled=False)
    assert untouched == msgs
    assert should_bypass_memory_merge(tools, msgs)


def test_passthrough_extra_keeps_unknown_fields():
    body = {
        "model": "switchyard",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "max_tokens": 128,
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
        "tool_choice": "auto",
        "stream_options": {"include_usage": True},
        "reasoning_effort": "low",
        "metadata": {"task": "1"},
    }
    extra = build_passthrough_extra(body)
    assert "model" not in extra and "messages" not in extra and "stream" not in extra
    assert extra["tools"][0]["function"]["name"] == "read_file"
    assert extra["stream_options"]["include_usage"] is True
    assert extra["reasoning_effort"] == "low"
    assert extra["metadata"] == {"task": "1"}
    assert resolve_max_tokens({"max_completion_tokens": 50}, 10, cap=1000) == 50


def test_preserve_transcript_and_continuation():
    image = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
    assert should_preserve_client_transcript(None, image)
    loop = [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    assert is_tool_continuation(loop)
    fresh = loop + [{"role": "user", "content": "now write tests"}]
    assert not is_tool_continuation(fresh)
    sid = stable_session_id(messages=[{"role": "system", "content": "agent prompt"}])
    assert sid.startswith("sys:")
    assert stable_session_id(user="missing") != "default"


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def test_context_length_error_mapping():
    exc = Exception("boom")
    exc.response = _Resp(
        400,
        {"error": {"message": "This model's maximum context length is 8192 tokens."}},
    )
    body, status = openai_error_from_upstream(exc)
    assert status == 400
    assert body["error"]["code"] == "context_length_exceeded"


def test_sse_tool_calls():
    result = {
        "id": "chatcmpl-1",
        "created": 1,
        "model": "test",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\": \"a.py\"}",
                            },
                        }
                    ],
                    "reasoning_content": "look at the file first",
                },
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    lines = list(completion_to_sse_chunks(result, model="test"))
    blob = "".join(lines)
    assert "tool_calls" in blob
    assert "read_file" in blob
    assert "reasoning_content" in blob
    assert "prompt_tokens" in blob
    assert "data: [DONE]" in blob
    found = False
    for line in lines:
        if line.startswith("data: ") and "[DONE]" not in line:
            obj = json.loads(line[6:])
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("tool_calls"):
                found = True
    assert found, blob


if __name__ == "__main__":
    test_extract_search_query()
    test_parse_ddg_html()
    test_tools_extra_and_nudge()
    test_passthrough_extra_keeps_unknown_fields()
    test_preserve_transcript_and_continuation()
    test_context_length_error_mapping()
    test_sse_tool_calls()
    print("all tests passed")
