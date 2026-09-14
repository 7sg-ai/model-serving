"""OpenAI-compatible HTTP backends (vLLM, NIM, etc.)."""
from .openai_chat import (
    OpenAIChatClient,
    chat_completions,
    extract_assistant_text,
    normalize_chat_url,
    require_backend_urls,
    resolve_vllm_backend_urls,
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
]
