"""OpenAI-compatible HTTP backends (vLLM, NIM, etc.)."""
from .openai_chat import (
    OpenAIChatClient,
    chat_completions,
    extract_assistant_text,
    normalize_chat_url,
    require_backend_urls,
    resolve_vllm_backend_urls,
)
from .openai_tools import (
    IDE_TOOL_NUDGE_TEXT,
    TOOL_CAPABLE_HINTS,
    apply_ide_tool_nudge,
    build_tools_extra,
    extract_tool_calls,
    looks_tool_capable,
    message_content_for_memory,
    messages_have_tool_protocol,
    request_has_tools,
    should_bypass_memory_merge,
    warn_if_not_tool_capable,
    should_skip_response_cache,
)
from .streaming import (
    flask_sse_from_completion,
    flask_sse_from_upstream,
    post_chat_stream,
)

__all__ = [
    "OpenAIChatClient",
    "chat_completions",
    "extract_assistant_text",
    "normalize_chat_url",
    "require_backend_urls",
    "resolve_vllm_backend_urls",
    "flask_sse_from_completion",
    "flask_sse_from_upstream",
    "post_chat_stream",
    "IDE_TOOL_NUDGE_TEXT",
    "TOOL_CAPABLE_HINTS",
    "apply_ide_tool_nudge",
    "looks_tool_capable",
    "warn_if_not_tool_capable",
    "build_tools_extra",
    "extract_tool_calls",
    "message_content_for_memory",
    "messages_have_tool_protocol",
    "request_has_tools",
    "should_bypass_memory_merge",
    "should_skip_response_cache",
]
