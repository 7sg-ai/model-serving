"""
HuggingFace IDE Coding Assistant - OpenAI Compatible API Server
Connects to vLLM (OpenAI-compatible) backends for Cline/Cursor.

Inference is vLLM-only (no in-process Transformers weights).
Switchyard multi-model routing targets separate vLLM instances per tier.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import copy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.auth import AuthManager
from shared.database import ChatHistory
from shared.cache import (
    ConversationMemory,
    build_response_cache_for_model,
    is_moonshot_model,
)
from shared.backends import (
    OpenAIChatClient,
    flask_sse_from_completion,
    flask_sse_from_upstream,
    normalize_chat_url,
    post_chat_stream,
    require_backend_urls,
)

from shared.deploy import ensure_models_runtime
from shared.multinode import (
    BackendPool,
    ClusterInfo,
    apply_flask_multinode,
    describe_placement_for_logs,
    get_model_placement,
    get_multinode_config,
    run_flask_app,
)
from shared.switchyard import (
    ModelSpec,
    get_switchyard_config,
    get_switchyard_router,
    write_routes_toml,
)


app = Flask(__name__)
CORS(app)

# Multi-node / cluster configuration
MN_CFG = get_multinode_config(default_port=8081, app_name="hf-ide-assistant")
apply_flask_multinode(app, MN_CFG)
CLUSTER = ClusterInfo(MN_CFG, app_name="hf-ide-assistant")

# HF apps never load model weights locally — vLLM holds GPUs/weights
os.environ.setdefault("LOAD_MODEL_WEIGHTS", "false")

# Configuration
MODEL_NAME = os.getenv(
    "HF_MODEL_NAME", os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
)
API_KEY = os.getenv("API_KEY", "hf-coding-assistant-key")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
REQUEST_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", os.getenv("REQUEST_TIMEOUT", "300")))

DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "32768"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "1048576"))
EFFECTIVE_CONTEXT_WINDOW = int(
    os.getenv("EFFECTIVE_CONTEXT_WINDOW", str(CONTEXT_WINDOW))
)

CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "256"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
MEMORY_MAX_CONVERSATIONS = int(os.getenv("MEMORY_MAX_CONVERSATIONS", "128"))

CODING_SYSTEM_PROMPT = (
    "You are an expert coding assistant optimized for complex, multi-file software "
    "engineering tasks. Provide clear, correct, production-quality code. Prefer "
    "complete solutions with necessary imports, error handling, and brief explanations. "
    "When editing existing code, preserve style and only change what is required."
)

auth_manager = AuthManager()
chat_history = ChatHistory()
response_cache = build_response_cache_for_model(
    MODEL_NAME,
    max_size=CACHE_MAX_SIZE,
    ttl_seconds=CACHE_TTL_SECONDS,
    enabled=CACHE_ENABLED,
)
conversation_memory = ConversationMemory(
    context_window=CONTEXT_WINDOW,
    max_conversations=MEMORY_MAX_CONVERSATIONS,
)
USING_MOONSHOT = is_moonshot_model(MODEL_NAME)

PLACEMENT = get_model_placement(replica_backends=MN_CFG.backend_urls)

def _force_remote_switchyard_models(cfg) -> None:
    """HF path: every Switchyard tier must use remote vLLM URLs."""
    if not cfg.models:
        return
    default_urls = list(CLUSTER.backend_pool.urls) if CLUSTER.backend_pool.urls else []
    for m in cfg.models:
        m.deployment.load_weights_locally = False
        if not m.deployment.backend_urls and not m.deployment.coordinator_url:
            if default_urls and len(cfg.models) == 1:
                m.deployment.backend_urls = list(default_urls)
        if m.deployment.backend_urls:
            m.deployment.backend_urls = [
                normalize_chat_url(u) for u in m.deployment.backend_urls if u
            ]


# Load Switchyard multi-model config first, then deploy all tiers in parallel.
SY_CFG = get_switchyard_config(
    default_model_id=MODEL_NAME,
    default_backend_url=(
        (MN_CFG.backend_urls[0] if MN_CFG.backend_urls else "")
        or (PLACEMENT.replica_backends[0] if PLACEMENT.replica_backends else "")
    ),
    owned_by="hf-switchyard",
    reload=True,
)
_force_remote_switchyard_models(SY_CFG)

_deploy_specs = list(SY_CFG.models) if SY_CFG.models else []
if not _deploy_specs:
    from shared.switchyard.config import DeploymentParams, ModelSpec

    _dep = DeploymentParams()
    _seed = list(MN_CFG.backend_urls or PLACEMENT.replica_backends or [])
    if _seed:
        _dep.backend_urls = _seed
    _deploy_specs = [
        ModelSpec(name="default", id=MODEL_NAME, role="weak", deployment=_dep)
    ]

_HANDLES = ensure_models_runtime(
    "vllm",
    _deploy_specs,
    app_name="hf-ide-assistant",
    api_key=VLLM_API_KEY,
    default_backend_urls=MN_CFG.backend_urls or PLACEMENT.replica_backends,
)
if SY_CFG.models:
    SY_CFG.models = list(_deploy_specs)
_force_remote_switchyard_models(SY_CFG)

_all_urls = []
for _h in _HANDLES:
    _all_urls.extend(_h.urls or [])
for _m in _deploy_specs:
    _all_urls.extend(list(_m.deployment.backend_urls or []))
_seen = set()
_uniq = []
for u in _all_urls:
    nu = normalize_chat_url(u) if u else ""
    if nu and nu not in _seen:
        _seen.add(nu)
        _uniq.append(nu)
PLACEMENT.replica_backends = list(require_backend_urls(_uniq or MN_CFG.backend_urls or PLACEMENT.replica_backends))
CLUSTER.backend_pool = BackendPool(
    PLACEMENT.replica_backends, strategy=MN_CFG.backend_strategy
)
MN_CFG.backend_urls = list(CLUSTER.backend_pool.urls)
print(
    f"Model runtime: engine=vllm stacks={len(_HANDLES)} "
    f"models={[getattr(m, 'id', m) for m in _deploy_specs]} "
    f"backends={CLUSTER.backend_pool.all()}"
)

VLLM_CLIENT = OpenAIChatClient(
    next_url=CLUSTER.next_backend,
    model=MODEL_NAME,
    api_key=VLLM_API_KEY,
    timeout=REQUEST_TIMEOUT,
    mark_success=CLUSTER.backend_pool.mark_success,
    mark_failure=CLUSTER.backend_pool.mark_failure,
    default_temperature=DEFAULT_TEMPERATURE,
    default_max_tokens=DEFAULT_MAX_TOKENS,
)

print("vLLM backends:", CLUSTER.backend_pool.all())
print("Placement:", describe_placement_for_logs(PLACEMENT))

if SY_CFG.enabled:
    missing = [
        m.name
        for m in SY_CFG.models
        if not m.chat_url()
        and not m.deployment.backend_urls
        and not m.deployment.coordinator_url
    ]
    if missing:
        raise RuntimeError(
            "Switchyard models missing vLLM backend_urls: "
            + ", ".join(missing)
            + ". Set deployment.backend_urls or BACKEND_URLS / VLLM_API_URL."
        )

SY_ROUTER = get_switchyard_router(
    SY_CFG,
    local_generate=None,
    fallback_backend=lambda: CLUSTER.next_backend(),
    fallback_model_id=MODEL_NAME,
    reload=True,
)
if SY_CFG.enabled and SY_CFG.routes_toml_path:
    try:
        path = write_routes_toml(SY_CFG, SY_CFG.routes_toml_path)
        print(f"Switchyard routes.toml written: {path}")
    except Exception as exc:
        print(f"Switchyard routes.toml export failed: {exc}")

_cache_stats = response_cache.stats()
print(
    f"Primary model id={MODEL_NAME} context={CONTEXT_WINDOW} "
    f"effective={EFFECTIVE_CONTEXT_WINDOW}"
)
_backend = _cache_stats.get("backend", "l1")
_extra = " (Mooncake response L2 for Moonshot/Kimi)" if USING_MOONSHOT else ""
print(f"Cache backend: {_backend}{_extra}")
if _cache_stats.get("mooncake_error"):
    print(f"Mooncake note: {_cache_stats['mooncake_error']}")


def verify_api_key(req):
    auth_header = req.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:] == API_KEY
    return False


def ensure_system_message(messages):
    if not messages or messages[0].get("role") != "system":
        return [{"role": "system", "content": CODING_SYSTEM_PROMPT}] + list(messages)
    return list(messages)


def merge_with_memory(conversation_id, messages, max_tokens, context_window=None):
    messages = ensure_system_message(messages)
    window = context_window or EFFECTIVE_CONTEXT_WINDOW
    original_window = conversation_memory.context_window
    conversation_memory.context_window = window
    try:
        if conversation_id:
            stored = conversation_memory.get(conversation_id)
            if stored and len(messages) <= len(stored):
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
            messages, reserve_for_response=max_tokens
        )
        if conversation_id:
            conversation_memory.set(conversation_id, trimmed)
        return trimmed
    finally:
        conversation_memory.context_window = original_window


def generate_response(messages, max_tokens, temperature, model_id=None):
    """Generate via vLLM OpenAI-compatible HTTP."""
    data, text = VLLM_CLIENT.chat(
        messages,
        model=model_id or MODEL_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    prompt_tokens, completion_tokens = VLLM_CLIENT.usage_tuple(data)
    return text, prompt_tokens, completion_tokens, data


def remember_exchange(conversation_id, messages, assistant_content):
    if not conversation_id or not assistant_content:
        return
    history = list(messages)
    history.append({"role": "assistant", "content": assistant_content})
    original_window = conversation_memory.context_window
    conversation_memory.context_window = EFFECTIVE_CONTEXT_WINDOW
    try:
        conversation_memory.set(conversation_id, history)
    finally:
        conversation_memory.context_window = original_window
    try:
        conv_key = abs(hash(conversation_id)) % (10**9)
        chat_history.add_message(conv_key, "assistant", assistant_content)
    except Exception:
        pass

@app.route("/v1/models", methods=["GET"])
def list_models():
    if not verify_api_key(request):
        return jsonify({"error": "Invalid API key"}), 401

    if SY_CFG.enabled:
        payload = SY_ROUTER.list_models_payload()
        for item in payload.get("data", []):
            item.setdefault("created", int(datetime.now().timestamp()))
            if item.get("id") == MODEL_NAME:
                item["effective_context_window"] = EFFECTIVE_CONTEXT_WINDOW
        return jsonify(payload)

    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "vllm",
                    "context_window": CONTEXT_WINDOW,
                    "effective_context_window": EFFECTIVE_CONTEXT_WINDOW,
                    "max_tokens": DEFAULT_MAX_TOKENS,
                }
            ],
        }
    )


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    if not verify_api_key(request):
        return jsonify({"error": "Invalid API key"}), 401

    data = request.json or {}
    messages = data.get("messages", [])
    temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))
    max_tokens = int(data.get("max_tokens", DEFAULT_MAX_TOKENS))
    max_tokens = max(1, min(max_tokens, EFFECTIVE_CONTEXT_WINDOW - 64))
    stream = bool(data.get("stream", False))
    requested_model = data.get("model") or MODEL_NAME

    conversation_id = (
        request.headers.get("X-Conversation-ID")
        or data.get("user")
        or data.get("conversation_id")
        or "default"
    )

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    ctx_window = EFFECTIVE_CONTEXT_WINDOW
    if SY_CFG.enabled:
        spec = SY_ROUTER.resolve_spec(requested_model) or SY_CFG.weak()
        if spec:
            ctx_window = min(
                EFFECTIVE_CONTEXT_WINDOW,
                spec.context_window or EFFECTIVE_CONTEXT_WINDOW,
            )
            max_tokens = max(1, min(max_tokens, ctx_window - 64))

    messages = merge_with_memory(
        conversation_id, messages, max_tokens, context_window=ctx_window
    )

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
            def _on_complete(text: str) -> None:
                remember_exchange(conversation_id, messages, text)

            if SY_CFG.enabled:
                outcome = SY_ROUTER.chat_completions_stream(
                    messages,
                    requested_model=requested_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    session_id=conversation_id,
                )
                if outcome.is_live_stream:
                    return flask_sse_from_upstream(
                        outcome.upstream, on_complete=_on_complete
                    )
                result = outcome.completion or {
                    "choices": [
                        {
                            "message": {"content": outcome.assistant_text or ""},
                            "finish_reason": "stop",
                        }
                    ]
                }
                if outcome.assistant_text:
                    remember_exchange(
                        conversation_id, messages, outcome.assistant_text
                    )
                model_out = (
                    (outcome.served.id if outcome.served else None)
                    or (result.get("model") if isinstance(result, dict) else None)
                    or requested_model
                    or MODEL_NAME
                )
                return flask_sse_from_completion(
                    result if isinstance(result, dict) else result,
                    model=model_out,
                )

            backend = CLUSTER.next_backend()
            if not backend:
                raise RuntimeError(
                    "No inference backend available "
                    "(BACKEND_URLS / VLLM_API_URL empty or all cooling down)"
                )
            try:
                upstream = post_chat_stream(
                    backend,
                    messages,
                    model=requested_model or MODEL_NAME,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=VLLM_API_KEY,
                    timeout=float(REQUEST_TIMEOUT),
                )
                CLUSTER.backend_pool.mark_success(backend)
            except Exception:
                CLUSTER.backend_pool.mark_failure(backend)
                raise
            return flask_sse_from_upstream(upstream, on_complete=_on_complete)

        if SY_CFG.enabled:
            result, response_text, served = SY_ROUTER.chat_completions(
                messages,
                requested_model=requested_model,
                temperature=temperature,
                max_tokens=max_tokens,
                session_id=conversation_id,
            )
            remember_exchange(conversation_id, messages, response_text)
            if isinstance(result, dict):
                result = dict(result)
                result["cached"] = False
            if cache_key is not None:
                response_cache.set(cache_key, result)
            return jsonify(result)

        response_text, prompt_tokens, completion_tokens, _raw = generate_response(
            messages, max_tokens, temperature, model_id=requested_model
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": "vllm_error",
                    }
                }
            ),
            500,
        )

    remember_exchange(conversation_id, messages, response_text)

    result = {
        "id": "chatcmpl-" + str(int(datetime.now().timestamp())),
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": requested_model or MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "cached": False,
    }

    if cache_key is not None:
        response_cache.set(cache_key, result)
    return jsonify(result)


@app.route("/v1/completions", methods=["POST"])
def completions():
    if not verify_api_key(request):
        return jsonify({"error": "Invalid API key"}), 401

    data = request.json or {}
    prompt = data.get("prompt", "")
    temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))
    max_tokens = int(data.get("max_tokens", DEFAULT_MAX_TOKENS))
    max_tokens = max(1, min(max_tokens, EFFECTIVE_CONTEXT_WINDOW - 64))
    requested_model = data.get("model") or MODEL_NAME
    conversation_id = (
        request.headers.get("X-Conversation-ID")
        or data.get("user")
        or data.get("conversation_id")
        or "completion"
    )

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    messages = [
        {"role": "system", "content": CODING_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    original_window = conversation_memory.context_window
    conversation_memory.context_window = EFFECTIVE_CONTEXT_WINDOW
    try:
        messages = conversation_memory.trim_to_context(
            messages, reserve_for_response=max_tokens
        )
    finally:
        conversation_memory.context_window = original_window

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
            chat_result, response_text, served = SY_ROUTER.chat_completions(
                messages,
                requested_model=requested_model,
                temperature=temperature,
                max_tokens=max_tokens,
                session_id=conversation_id,
            )
            usage = chat_result.get("usage", {}) if isinstance(chat_result, dict) else {}
            model_out = (served.id if served else None) or requested_model or MODEL_NAME
            result = {
                "id": "cmpl-" + str(int(datetime.now().timestamp())),
                "object": "text_completion",
                "created": int(datetime.now().timestamp()),
                "model": model_out,
                "choices": [
                    {
                        "text": response_text,
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
                "cached": False,
            }
            if isinstance(chat_result, dict) and chat_result.get("switchyard"):
                result["switchyard"] = chat_result["switchyard"]
            response_cache.set(cache_key, result)
            return jsonify(result)

        response_text, prompt_tokens, completion_tokens, _raw = generate_response(
            messages, max_tokens, temperature, model_id=requested_model
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": "vllm_error",
                    }
                }
            ),
            500,
        )

    result = {
        "id": "cmpl-" + str(int(datetime.now().timestamp())),
        "object": "text_completion",
        "created": int(datetime.now().timestamp()),
        "model": requested_model or MODEL_NAME,
        "choices": [
            {
                "text": response_text,
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "cached": False,
    }
    response_cache.set(cache_key, result)
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return (
        jsonify(
            CLUSTER.health(
                model=MODEL_NAME,
                inference_backend="vllm",
                loads_weights_locally=False,
                context_window=CONTEXT_WINDOW,
                effective_context_window=EFFECTIVE_CONTEXT_WINDOW,
                cache=response_cache.stats(),
                memory=conversation_memory.stats(),
                switchyard=SY_ROUTER.health_block(),
                placement=describe_placement_for_logs(PLACEMENT),
            )
        ),
        200,
    )


@app.route("/v1/cache", methods=["DELETE"])
def clear_cache():
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
    print("Starting HuggingFace IDE Assistant (vLLM backend)")
    print(f"Model: {MODEL_NAME}")
    print(f"API Key: {API_KEY}")
    print(f"Max Tokens: {DEFAULT_MAX_TOKENS}")
    print(f"Temperature: {DEFAULT_TEMPERATURE}")
    print(
        f"Context Window: {CONTEXT_WINDOW} (effective: {EFFECTIVE_CONTEXT_WINDOW})"
    )
    print(
        f"Cache Enabled: {CACHE_ENABLED} "
        f"(size={CACHE_MAX_SIZE}, ttl={CACHE_TTL_SECONDS}s)"
    )
    print(f"Cache backend: {response_cache.stats().get('backend', 'l1')}")
    if USING_MOONSHOT:
        print(
            "Moonshot model id -> app-level Mooncake response cache preferred (L1+L2)"
        )
        print(
            "Engine KV tiering (GPU working set + Mooncake store) is configured on vLLM."
        )
    print(f"Conversation Memory: max {MEMORY_MAX_CONVERSATIONS} sessions")
    backends = ", ".join(CLUSTER.backend_pool.all())
    print(f"vLLM backends ({MN_CFG.backend_strategy}): {backends}")
    print(
        f"MODEL_DEPLOY_MODE={MN_CFG.model_deploy_mode} "
        f"tp={MN_CFG.tensor_parallel_size}"
    )
    if SY_CFG.enabled:
        print(
            f"Switchyard: ON  strategy={SY_CFG.strategy}  "
            f"route_id={SY_CFG.route_id}  models={len(SY_CFG.models)}"
        )
        for m in SY_CFG.models:
            print(
                f"  - [{m.role}] {m.name} id={m.id} "
                f"backend={m.chat_url() or m.deployment.backend_urls} "
                f"deploy={m.deployment.to_dict()}"
            )
    else:
        print("Switchyard: OFF (set SWITCHYARD_ENABLED=true or configure models)")
    base = MN_CFG.public_url or f"http://localhost:{MN_CFG.node_port or 8081}"
    print("")
    print("Configure your IDE with:")
    print(f"  Base URL: {base}/v1")
    print(f"  API Key: {API_KEY}")
    if SY_CFG.enabled:
        print(f"  Model (route): {SY_CFG.route_id}")
        for m in SY_CFG.selectable_models():
            print(f"  Model (direct): {m.id}")
    else:
        print(f"  Model: {MODEL_NAME}")
    try:
        run_flask_app(app, MN_CFG, default_port=8081, debug=not MN_CFG.enabled)
    finally:
        close = getattr(response_cache, "close", None)
        if callable(close):
            close()
