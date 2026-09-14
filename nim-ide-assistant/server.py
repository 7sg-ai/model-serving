"""
NIM Model IDE Coding Assistant - OpenAI Compatible API Server
Can be connected to Cline or Cursor for coding assistance

Tuned for complicated coding workloads:
  - 1M context window
  - High max-token responses for multi-file generation
  - Response caching + conversation memory for faster follow-ups
  - NeMo Switchyard multi-model selection + escalation router
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import sys
import copy
from datetime import datetime

# Add parent directory to path for shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.auth import AuthManager
from shared.database import ChatHistory
from shared.cache import ResponseCache, ConversationMemory
from shared.backends import (
    flask_sse_from_completion,
    flask_sse_from_upstream,
    post_chat_stream,
)
from shared.multinode import (
    BackendPool,
    ClusterInfo,
    apply_flask_multinode,
    get_multinode_config,
    run_flask_app,
)
from shared.deploy import ensure_models_runtime
from shared.switchyard import (
    get_switchyard_config,
    get_switchyard_router,
    write_routes_toml,
)


app = Flask(__name__)
CORS(app)

# Multi-node / cluster configuration
MN_CFG = get_multinode_config(default_port=8080, app_name="nim-ide-assistant")
apply_flask_multinode(app, MN_CFG)
CLUSTER = ClusterInfo(MN_CFG, app_name="nim-ide-assistant")

# Configuration
NIM_API_URL = os.getenv("NIM_API_URL", "http://localhost:8000/v1/chat/completions")
MODEL_NAME = os.getenv("NIM_MODEL_NAME", "meta/llama-3.1-8b-instruct")
API_KEY = os.getenv("API_KEY", "nim-coding-assistant-key")
# Multi-model deploy happens after Switchyard config is loaded (see below).


# Model Parameters — defaults tuned for complicated coding use cases
# CONTEXT_WINDOW: total prompt budget the assistant will retain/trim to (1M)
# MAX_TOKENS: per-response generation limit (long multi-file code answers)
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "32768"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "1048576"))

# Cache / memory configuration
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "256"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
MEMORY_MAX_CONVERSATIONS = int(os.getenv("MEMORY_MAX_CONVERSATIONS", "128"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))

CODING_SYSTEM_PROMPT = (
    "You are an expert coding assistant optimized for complex, multi-file software "
    "engineering tasks. Provide clear, correct, production-quality code. Prefer "
    "complete solutions with necessary imports, error handling, and brief explanations. "
    "When editing existing code, preserve style and only change what is required."
)

# Initialize managers
auth_manager = AuthManager()
chat_history = ChatHistory()
response_cache = ResponseCache(
    max_size=CACHE_MAX_SIZE,
    ttl_seconds=CACHE_TTL_SECONDS,
    enabled=CACHE_ENABLED,
)
conversation_memory = ConversationMemory(
    context_window=CONTEXT_WINDOW,
    max_conversations=MEMORY_MAX_CONVERSATIONS,
)

# --- NeMo Switchyard (multi-model + escalation) ---
SY_CFG = get_switchyard_config(
    default_model_id=MODEL_NAME,
    default_backend_url=NIM_API_URL,
    owned_by="nim-switchyard",
    reload=True,
)
# Deploy every Switchyard tier (and default model) in parallel when needed.
_deploy_specs = list(SY_CFG.models) if SY_CFG.models else []
if not _deploy_specs:
    from shared.switchyard.config import DeploymentParams, ModelSpec

    _dep = DeploymentParams()
    if MN_CFG.backend_urls:
        _dep.backend_urls = list(MN_CFG.backend_urls)
    elif NIM_API_URL:
        _dep.backend_urls = [NIM_API_URL]
    _deploy_specs = [
        ModelSpec(name="default", id=MODEL_NAME, role="weak", deployment=_dep)
    ]
_HANDLES = ensure_models_runtime(
    "nim",
    _deploy_specs,
    app_name="nim-ide-assistant",
    default_backend_urls=MN_CFG.backend_urls or ([NIM_API_URL] if NIM_API_URL else []),
)
# Sync primary pool from deployed URLs
_all_urls = []
for _h in _HANDLES:
    _all_urls.extend(_h.urls or [])
if not _all_urls:
    for _m in _deploy_specs:
        _all_urls.extend(list(_m.deployment.backend_urls or []))
if _all_urls:
    # dedupe preserve order
    _seen = set()
    _uniq = []
    for u in _all_urls:
        if u not in _seen:
            _seen.add(u)
            _uniq.append(u)
    NIM_API_URL = _uniq[0]
    CLUSTER.backend_pool = BackendPool(_uniq, strategy=MN_CFG.backend_strategy)
    MN_CFG.backend_urls = list(CLUSTER.backend_pool.urls)
# Keep SY_CFG models in sync with deploy-filled URLs
if SY_CFG.models:
    SY_CFG.models = list(_deploy_specs)
print(
    f"Model runtime: engine=nim stacks={len(_HANDLES)} "
    f"models={[getattr(m, 'id', m) for m in _deploy_specs]} "
    f"backends={CLUSTER.backend_pool.all()}"
)
SY_ROUTER = get_switchyard_router(
    SY_CFG,
    fallback_backend=lambda: CLUSTER.next_backend() or NIM_API_URL,
    fallback_model_id=MODEL_NAME,
    reload=True,
)
if SY_CFG.enabled and SY_CFG.routes_toml_path:
    try:
        path = write_routes_toml(SY_CFG, SY_CFG.routes_toml_path)
        print(f"Switchyard routes.toml written: {path}")
    except Exception as exc:
        print(f"Switchyard routes.toml export failed: {exc}")


def verify_api_key(req):
    """Verify API key from request"""
    auth_header = req.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        return token == API_KEY
    return False


def ensure_system_message(messages):
    """Prepend coding system prompt when the client did not supply one."""
    if not messages or messages[0].get("role") != "system":
        return [{"role": "system", "content": CODING_SYSTEM_PROMPT}] + list(messages)
    return list(messages)


def merge_with_memory(conversation_id, messages, max_tokens, context_window=None):
    """
    Merge client messages with server-side conversation memory and trim
    to CONTEXT_WINDOW, reserving room for the completion.
    """
    messages = ensure_system_message(messages)
    window = context_window or CONTEXT_WINDOW
    original_window = conversation_memory.context_window
    conversation_memory.context_window = window
    try:
        if conversation_id:
            stored = conversation_memory.get(conversation_id)
            if stored:
                # Prefer full client history when it is longer; otherwise extend memory
                if len(messages) <= len(stored):
                    # Client may only send the latest turn — append new user msgs
                    known = {(m.get("role"), m.get("content")) for m in stored}
                    merged = list(stored)
                    for msg in messages:
                        key = (msg.get("role"), msg.get("content"))
                        if key not in known and msg.get("role") != "system":
                            merged.append(
                                {"role": msg["role"], "content": msg.get("content", "")}
                            )
                    messages = merged

        trimmed = conversation_memory.trim_to_context(
            messages,
            reserve_for_response=max_tokens,
        )

        if conversation_id:
            conversation_memory.set(conversation_id, trimmed)

        return trimmed
    finally:
        conversation_memory.context_window = original_window


def remember_exchange(conversation_id, messages, assistant_content):
    """Persist the latest user + assistant turn into conversation memory."""
    if not conversation_id or not assistant_content:
        return

    # Start from the trimmed prompt we just used
    history = list(messages)
    history.append({"role": "assistant", "content": assistant_content})
    conversation_memory.set(conversation_id, history)

    # Best-effort durable log (non-fatal if DB fails)
    try:
        # Use a stable numeric id derived from conversation string when possible
        conv_key = abs(hash(conversation_id)) % (10**9)
        chat_history.add_message(conv_key, "assistant", assistant_content)
    except Exception:
        pass


def _legacy_upstream_chat(messages, temperature, max_tokens, stream, model_name):
    """Single-backend NIM path (Switchyard disabled).

    When stream=True, returns a live requests.Response (SSE) that the caller
    must proxy; when stream=False, returns the parsed OpenAI JSON body.
    """
    backend = CLUSTER.next_backend() or NIM_API_URL
    try:
        if stream:
            upstream = post_chat_stream(
                backend,
                messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=float(REQUEST_TIMEOUT),
            )
            CLUSTER.backend_pool.mark_success(backend)
            return upstream
        nim_payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = requests.post(backend, json=nim_payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        CLUSTER.backend_pool.mark_success(backend)
        return response.json()
    except Exception:
        CLUSTER.backend_pool.mark_failure(backend)
        raise



@app.route("/v1/models", methods=["GET"])
def list_models():
    """List available models (OpenAI compatible)"""
    if not verify_api_key(request):
        return jsonify({"error": "Invalid API key"}), 401

    if SY_CFG.enabled:
        payload = SY_ROUTER.list_models_payload()
        # Ensure created timestamps
        for item in payload.get("data", []):
            item.setdefault("created", int(datetime.now().timestamp()))
        return jsonify(payload)

    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "nim",
                    "context_window": CONTEXT_WINDOW,
                    "max_tokens": DEFAULT_MAX_TOKENS,
                }
            ],
        }
    )


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """Handle chat completions (OpenAI compatible)"""
    if not verify_api_key(request):
        return jsonify({"error": "Invalid API key"}), 401

    data = request.json or {}
    messages = data.get("messages", [])
    temperature = data.get("temperature", DEFAULT_TEMPERATURE)
    max_tokens = int(data.get("max_tokens", DEFAULT_MAX_TOKENS))
    stream = data.get("stream", False)
    requested_model = data.get("model") or MODEL_NAME
    conversation_id = (
        request.headers.get("X-Conversation-ID")
        or data.get("user")
        or data.get("conversation_id")
    )

    # Cap max_tokens so prompt + completion fit the context window
    max_tokens = max(1, min(max_tokens, CONTEXT_WINDOW - 512))

    # Per-model context when Switchyard selects a specific tier
    ctx_window = CONTEXT_WINDOW
    if SY_CFG.enabled:
        spec = SY_ROUTER.resolve_spec(requested_model)
        if spec is None and SY_CFG.weak():
            spec = SY_CFG.weak()
        if spec:
            ctx_window = spec.context_window or CONTEXT_WINDOW
            max_tokens = max(1, min(max_tokens, ctx_window - 512))

    messages = merge_with_memory(
        conversation_id, messages, max_tokens, context_window=ctx_window
    )

    # Response cache (skip streaming — not cacheable as a single JSON body)
    cache_model = requested_model or MODEL_NAME
    if SY_CFG.enabled:
        cache_model = f"sy:{SY_CFG.strategy}:{cache_model}"
    cache_key = None
    if not stream:
        cache_key = response_cache.make_key(
            cache_model, messages, temperature, max_tokens
        )
        cached = response_cache.get(cache_key)
        if cached is not None:
            result = copy.deepcopy(cached)
            result["cached"] = True
            return jsonify(result)

    try:
        if stream:
            # Cline / Cursor send stream=true. Proxy SSE (or synthesize it after
            # Switchyard finishes a non-stream route decision).
            def _on_complete(text: str) -> None:
                remember_exchange(conversation_id, messages, text)

            if SY_CFG.enabled:
                # Escalation/capability need the full reply before judging —
                # run non-stream upstream then convert to SSE for the client.
                result, assistant_content, _served = SY_ROUTER.chat_completions(
                    messages,
                    requested_model=requested_model,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    session_id=conversation_id or "default",
                    stream=False,
                )
                if assistant_content:
                    remember_exchange(conversation_id, messages, assistant_content)
                model_out = (
                    (_served.id if _served else None)
                    or (result.get("model") if isinstance(result, dict) else None)
                    or requested_model
                    or MODEL_NAME
                )
                return flask_sse_from_completion(
                    result if isinstance(result, dict) else {"choices": [{"message": {"content": assistant_content or ""}, "finish_reason": "stop"}]},
                    model=model_out,
                )

            upstream = _legacy_upstream_chat(
                messages, temperature, max_tokens, True, MODEL_NAME
            )
            return flask_sse_from_upstream(upstream, on_complete=_on_complete)

        if SY_CFG.enabled:
            result, assistant_content, _served = SY_ROUTER.chat_completions(
                messages,
                requested_model=requested_model,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                session_id=conversation_id or "default",
                stream=False,
            )
            if assistant_content:
                remember_exchange(conversation_id, messages, assistant_content)
        else:
            result = _legacy_upstream_chat(
                messages, temperature, max_tokens, False, MODEL_NAME
            )
            try:
                assistant_content = result["choices"][0]["message"]["content"]
                remember_exchange(conversation_id, messages, assistant_content)
            except (KeyError, IndexError, TypeError):
                pass

        if cache_key is not None:
            response_cache.set(cache_key, result)

        if isinstance(result, dict):
            result = dict(result)
            result["cached"] = False
        return jsonify(result)

    except Exception as e:
        return (
            jsonify(
                {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": "nim_error",
                    }
                }
            ),
            500,
        )



@app.route("/v1/completions", methods=["POST"])
def completions():
    """Handle text completions (OpenAI compatible)"""
    if not verify_api_key(request):
        return jsonify({"error": "Invalid API key"}), 401

    data = request.json or {}
    prompt = data.get("prompt", "")
    temperature = data.get("temperature", DEFAULT_TEMPERATURE)
    max_tokens = int(data.get("max_tokens", DEFAULT_MAX_TOKENS))
    max_tokens = max(1, min(max_tokens, CONTEXT_WINDOW - 512))
    requested_model = data.get("model") or MODEL_NAME
    conversation_id = (
        request.headers.get("X-Conversation-ID")
        or data.get("user")
        or data.get("conversation_id")
        or "completion"
    )

    messages = [
        {"role": "system", "content": CODING_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    messages = conversation_memory.trim_to_context(
        messages, reserve_for_response=max_tokens
    )

    cache_model = requested_model or MODEL_NAME
    if SY_CFG.enabled:
        cache_model = f"sy:{SY_CFG.strategy}:{cache_model}"
    cache_key = response_cache.make_key(
        cache_model, messages, temperature, max_tokens, extra={"mode": "completion"}
    )
    cached = response_cache.get(cache_key)
    if cached is not None:
        result = copy.deepcopy(cached)
        result["cached"] = True
        return jsonify(result)

    try:
        if SY_CFG.enabled:
            nim_response, text, served = SY_ROUTER.chat_completions(
                messages,
                requested_model=requested_model,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                session_id=conversation_id,
            )
            model_out = (served.id if served else None) or requested_model or MODEL_NAME
        else:
            nim_response = _legacy_upstream_chat(
                messages, temperature, max_tokens, False, MODEL_NAME
            )
            text = nim_response["choices"][0]["message"]["content"]
            model_out = MODEL_NAME

        completion_response = {
            "id": nim_response.get(
                "id", "cmpl-" + str(int(datetime.now().timestamp()))
            ),
            "object": "text_completion",
            "created": nim_response.get(
                "created", int(datetime.now().timestamp())
            ),
            "model": model_out,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "finish_reason": (
                        nim_response.get("choices", [{}])[0].get("finish_reason", "stop")
                        if isinstance(nim_response.get("choices"), list)
                        else "stop"
                    ),
                }
            ],
            "usage": nim_response.get("usage", {}),
            "cached": False,
        }
        if isinstance(nim_response, dict) and nim_response.get("switchyard"):
            completion_response["switchyard"] = nim_response["switchyard"]

        response_cache.set(cache_key, completion_response)
        return jsonify(completion_response)

    except Exception as e:
        return (
            jsonify(
                {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": "nim_error",
                    }
                }
            ),
            500,
        )


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint (includes multi-node identity)"""
    return jsonify(
        CLUSTER.health(
            model=MODEL_NAME,
            context_window=CONTEXT_WINDOW,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            cache=response_cache.stats(),
            memory=conversation_memory.stats(),
            switchyard=SY_ROUTER.health_block(),
        )
    )


@app.route("/v1/cache", methods=["DELETE"])
def clear_cache():
    """Clear response cache (and optionally conversation memory)."""
    if not verify_api_key(request):
        return jsonify({"error": "Invalid API key"}), 401

    clear_memory = request.args.get("memory", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    response_cache.clear()
    if clear_memory:
        conversation_memory.clear()
        if SY_ROUTER.escalation and request.args.get("session"):
            SY_ROUTER.escalation.reset_session(request.args.get("session"))

    return jsonify(
        {
            "status": "cleared",
            "memory_cleared": clear_memory,
            "cache": response_cache.stats(),
            "memory": conversation_memory.stats(),
            "switchyard": SY_ROUTER.health_block(),
        }
    )


@app.route("/v1/switchyard/session", methods=["DELETE"])
def reset_switchyard_session():
    """Reset escalation latch/streak for a conversation session."""
    if not verify_api_key(request):
        return jsonify({"error": "Invalid API key"}), 401
    if not SY_CFG.enabled or not SY_ROUTER.escalation:
        return jsonify({"error": "Escalation router not active"}), 400
    data = request.json or {}
    session_id = (
        request.headers.get("X-Conversation-ID")
        or data.get("session_id")
        or data.get("conversation_id")
        or request.args.get("session")
        or "default"
    )
    SY_ROUTER.escalation.reset_session(session_id)
    return jsonify({"status": "reset", "session_id": session_id})


if __name__ == "__main__":
    print("Starting NIM IDE Assistant")
    print(f"Model: {MODEL_NAME}")
    print(f"API Key: {API_KEY}")
    print(f"Max Tokens: {DEFAULT_MAX_TOKENS}")
    print(f"Temperature: {DEFAULT_TEMPERATURE}")
    print(f"Context Window: {CONTEXT_WINDOW}")
    print(f"Cache Enabled: {CACHE_ENABLED} (size={CACHE_MAX_SIZE}, ttl={CACHE_TTL_SECONDS}s)")
    print(f"Conversation Memory: max {MEMORY_MAX_CONVERSATIONS} sessions")
    if SY_CFG.enabled:
        print(
            f"Switchyard: ON  strategy={SY_CFG.strategy}  route_id={SY_CFG.route_id}  "
            f"models={len(SY_CFG.models)}"
        )
        for m in SY_CFG.models:
            print(
                f"  - [{m.role}] {m.name} id={m.id} "
                f"backend={m.chat_url() or '(local/fallback)'} "
                f"deploy={m.deployment.to_dict()}"
            )
    else:
        print("Switchyard: OFF (set SWITCHYARD_ENABLED=true or configure models)")
    base = MN_CFG.public_url or f"http://localhost:{MN_CFG.node_port or 8080}"
    print("\nConfigure your IDE with:")
    print(f"  Base URL: {base}/v1")
    print(f"  API Key: {API_KEY}")
    if SY_CFG.enabled:
        print(f"  Model (route): {SY_CFG.route_id}")
        for m in SY_CFG.selectable_models():
            print(f"  Model (direct): {m.id}")
    else:
        print(f"  Model: {MODEL_NAME}")
    run_flask_app(app, MN_CFG, default_port=8080, debug=not MN_CFG.enabled)
