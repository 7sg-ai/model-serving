"""Anthropic Messages API adapter for the OpenAI-compatible IDE proxies.

Claude Code speaks ``POST /v1/messages``: tools are ``input_schema`` objects,
history carries ``tool_use`` / ``tool_result`` blocks, streams are Anthropic
SSE. This package only changes shape. Inference stays on the existing
Switchyard / chat-completions path. The requested model id is echoed back;
tier hints and tool-capable preference are applied by the caller first.
"""
from .anthropic_convert import anthropic_to_openai, claude_tier_hint
from .anthropic_response import (
    anthropic_error,
    estimate_input_tokens,
    openai_to_anthropic,
)
from .anthropic_stream import anthropic_message_to_sse, iter_openai_sse_as_anthropic

__all__ = [
    "anthropic_error",
    "anthropic_message_to_sse",
    "anthropic_to_openai",
    "claude_tier_hint",
    "estimate_input_tokens",
    "iter_openai_sse_as_anthropic",
    "openai_to_anthropic",
]
