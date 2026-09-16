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
    build_tools_extra,
    request_has_tools,
    should_bypass_memory_merge,
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
    out = apply_ide_tool_nudge(msgs, tools)
    assert out[0]["role"] == "system"
    assert "IDE coding agent" in out[0]["content"]
    assert should_bypass_memory_merge(tools, msgs)


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
                },
            }
        ],
    }
    lines = list(completion_to_sse_chunks(result, model="test"))
    blob = "".join(lines)
    assert "tool_calls" in blob
    assert "read_file" in blob
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
    test_sse_tool_calls()
    print("all tests passed")
