"""Lightweight tests for web search + tool_calls SSE (no network)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.tools.web_search import extract_search_query, parse_ddg_html
from shared.backends.streaming import completion_to_sse_chunks
from shared.backends.anthropic_convert import anthropic_to_openai, strip_reasoning_content
from shared.backends.anthropic_response import openai_to_anthropic, repair_tool_input
from shared.backends.anthropic_stream import iter_openai_sse_as_anthropic
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


def test_thinking_round_trip_and_caller_stays_off_openai():
    body = {
        "model": "claude-sonnet",
        "max_tokens": 64,
        "messages": [{
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "check the file", "signature": "sig"},
                {"type": "redacted_thinking", "data": "secret"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "read_file",
                    "input": {"path": "a.py"},
                    "caller": {"type": "direct"},
                },
            ],
        }],
    }
    openai = anthropic_to_openai(body)
    msg = openai["messages"][0]
    assert msg["reasoning_content"] == "check the file"
    assert msg["content"] is None
    assert "signature" not in json.dumps(openai)
    assert "caller" not in json.dumps(openai)
    assert "redacted" not in json.dumps(openai)
    assert msg["tool_calls"][0]["function"]["name"] == "read_file"

    out = openai_to_anthropic({
        "id": "chatcmpl-1",
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "<think>inline</think>done",
                "reasoning_content": "from the field",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\": \"a.py\"}"},
                }],
            },
        }],
    }, requested_model="claude-sonnet")
    kinds = [b["type"] for b in out["content"]]
    assert kinds == ["thinking", "text", "tool_use"]
    assert out["content"][0]["thinking"] == "from the field"
    assert "signature" not in out["content"][0]
    assert out["content"][1]["text"] == "done"
    assert out["content"][2]["caller"] == {"type": "direct"}
    assert out["stop_reason"] == "tool_use"


def test_truncated_tool_json_keeps_siblings():
    cut = '{"path": "a.py", "note": "hel'
    assert repair_tool_input(cut) == {"path": "a.py", "note": "hel"}
    assert repair_tool_input('{"items": [1, 2') == {"items": [1, 2]}
    assert repair_tool_input("{not json") is None

    out = openai_to_anthropic({
        "choices": [{
            "finish_reason": "length",
            "message": {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_ok",
                        "function": {"name": "read_file", "arguments": "{\"path\": \"a.py\"}"},
                    },
                    {
                        "id": "call_cut",
                        "function": {"name": "write", "arguments": '{"path": "b.py", "text": "hi'},
                    },
                    {
                        "id": "call_dead",
                        "function": {"name": "bash", "arguments": "{not json"},
                    },
                ],
            },
        }],
    }, requested_model="claude-sonnet")
    uses = [b for b in out["content"] if b["type"] == "tool_use"]
    assert [b["name"] for b in uses] == ["read_file", "write"]
    assert uses[1]["input"] == {"path": "b.py", "text": "hi"}
    assert all(b["caller"]["type"] == "direct" for b in uses)
    assert out["stop_reason"] == "tool_use"
    assert not any(b.get("input") == {} and b["name"] == "bash" for b in uses)

    none = openai_to_anthropic({
        "choices": [{
            "finish_reason": "length",
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_dead",
                    "function": {"name": "bash", "arguments": "{not json"},
                }],
            },
        }],
    }, requested_model="claude-sonnet")
    assert none["stop_reason"] == "max_tokens"
    assert not any(b["type"] == "tool_use" for b in none["content"])

    finished = openai_to_anthropic({
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "call_dead",
                    "function": {"name": "bash", "arguments": "{not json"},
                }],
            },
        }],
    }, requested_model="claude-sonnet")
    assert finished["stop_reason"] == "end_turn"
    assert "bash" in finished["content"][-1]["text"]
    stripped = strip_reasoning_content({
        "messages": [{"role": "assistant", "content": "x", "reasoning_content": "y"}]
    })
    assert "reasoning_content" not in stripped["messages"][0]


def test_stream_recovers_cut_json_without_relabeling():
    lines = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {"reasoning_content": "plan"},
                "finish_reason": None,
            }]
        }),
        'data: ' + json.dumps({
            "choices": [{
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "write", "arguments": '{"path": "a.py"'},
                }]},
                "finish_reason": "tool_calls",
            }]
        }),
        "data: [DONE]",
    ]
    blob = "".join(iter_openai_sse_as_anthropic(iter(lines), requested_model="claude-sonnet"))
    assert "thinking_delta" in blob
    assert '"caller": {"type": "direct"}' in blob or '"caller":{"type":"direct"}' in blob
    assert "a.py" in blob
    assert '"stop_reason": "tool_use"' in blob
    assert "max_tokens" not in blob


if __name__ == "__main__":
    test_extract_search_query()
    test_parse_ddg_html()
    test_tools_extra_and_nudge()
    test_passthrough_extra_keeps_unknown_fields()
    test_preserve_transcript_and_continuation()
    test_context_length_error_mapping()
    test_sse_tool_calls()
    test_thinking_round_trip_and_caller_stays_off_openai()
    test_truncated_tool_json_keeps_siblings()
    test_stream_recovers_cut_json_without_relabeling()
    print("all tests passed")
