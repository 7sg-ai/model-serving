"""
HuggingFace Model with WebRTC Voice Interface - Backend Server (vLLM backend)
Supports optional login, chat history, and multi-model selection.
"""
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import sys
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.auth import AuthManager
from shared.database import ChatHistory
from shared.backends import require_backend_urls
from shared.multinode import (
    BackendPool,
    ClusterInfo,
    apply_flask_multinode,
    get_multinode_config,
    run_socketio_app,
)
from shared.tools import get_web_access_tool, get_web_enricher
from shared.model_catalog import deploy_app_models, load_app_models

os.environ.setdefault("LOAD_MODEL_WEIGHTS", "false")

app = Flask(__name__)
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)
CORS(app, supports_credentials=True)

MN_CFG = get_multinode_config(default_port=5001, app_name="hf-webrtc-voice")
apply_flask_multinode(app, MN_CFG)
CLUSTER = ClusterInfo(MN_CFG, app_name="hf-webrtc-voice")

_socketio_kwargs = {"cors_allowed_origins": "*"}
if MN_CFG.use_redis and MN_CFG.redis_url:
    _socketio_kwargs["message_queue"] = MN_CFG.redis_url
socketio = SocketIO(app, **_socketio_kwargs)

DEFAULT_MODEL_NAME = os.getenv(
    "HF_MODEL_NAME", os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
)
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
VLLM_API_KEY = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))

default_backend = MN_CFG.backend_urls[0] if MN_CFG.backend_urls else ""
SY_CFG = load_app_models(
    default_model_id=DEFAULT_MODEL_NAME,
    default_backend_url=default_backend,
    owned_by="hf-webrtc-voice",
    reload=True,
)
CATALOG, _HANDLES = deploy_app_models(
    "vllm",
    SY_CFG,
    app_name="hf-webrtc-voice",
    api_key=VLLM_API_KEY,
    default_backend_urls=MN_CFG.backend_urls or CLUSTER.backend_pool.urls,
)
backends = require_backend_urls(
    CATALOG.all_backend_urls() or MN_CFG.backend_urls or CLUSTER.backend_pool.urls
)
CLUSTER.backend_pool = BackendPool(backends, strategy=MN_CFG.backend_strategy)
MN_CFG.backend_urls = backends
MODEL_NAME = CATALOG.default_id()

auth_manager = AuthManager()
chat_history = ChatHistory()
WEB = get_web_access_tool("hf-webrtc-voice")
ENRICH = get_web_enricher("hf-webrtc-voice")

print(f"vLLM voice models: {[m.id for m in CATALOG.models]} backends={backends}")
print("Model loaded via remote vLLM (no local weights).")

conversations = {}


@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.json or {}
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    if auth_manager.register_user(username, password):
        return jsonify({"message": "Registration successful"}), 201
    return jsonify({"error": "Username already exists"}), 409


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.json or {}
    username = data.get("username")
    password = data.get("password")
    session_token = auth_manager.login(username, password)
    if session_token:
        user_id = auth_manager.validate_session(session_token)
        session["session_token"] = session_token
        session["user_id"] = user_id
        session["username"] = username
        session.permanent = True
        return jsonify(
            {
                "message": "Login successful",
                "session_token": session_token,
                "username": username,
            }
        ), 200
    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    token = session.get("session_token")
    if token:
        auth_manager.logout(token)
    session.clear()
    return jsonify({"message": "Logged out"}), 200


@app.route("/api/auth/guest", methods=["POST"])
def guest():
    session["username"] = "Guest"
    session["user_id"] = None
    session.permanent = True
    return jsonify({"message": "Guest session", "username": "Guest"}), 200


@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify(CATALOG.list_payload()), 200


@app.route("/api/conversations", methods=["GET"])
def get_conversations():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    convs = chat_history.get_user_conversations(user_id, "hf-webrtc-voice")
    return jsonify({"conversations": convs}), 200


@app.route("/api/conversations/<int:conv_id>", methods=["GET"])
def get_conversation(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    messages = chat_history.get_conversation_history(conv_id)
    return jsonify({"messages": messages}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        CLUSTER.health(
            model=MODEL_NAME,
            inference_backend="vllm",
            loads_weights_locally=False,
            web_access=WEB.stats(),
            catalog=CATALOG.health_block(),
        )
    ), 200


def generate_response(prompt, prior_messages, web_context: str = "", model_id: str = ""):
    messages = []
    for msg in prior_messages or []:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ("user", "assistant", "system") and content:
            messages.append({"role": role, "content": content})
    model_prompt = prompt
    if web_context:
        model_prompt = (
            f"{web_context}\n\n"
            f"User question (may reference the URL(s) above):\n{prompt}"
        )
    messages.append({"role": "user", "content": model_prompt})
    try:
        text, _data = CATALOG.chat(
            messages,
            model=model_id or MODEL_NAME,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return text or "(empty response)"
    except Exception as e:
        return f"Error: {e}"


@socketio.on("connect")
def handle_connect():
    print("Client connected")
    emit("connected", {"message": "Connected to server"})


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")
    if request.sid in conversations:
        del conversations[request.sid]


@socketio.on("start_conversation")
def handle_start_conversation(data):
    data = data or {}
    user_id = session.get("user_id")
    if user_id:
        conv_id = chat_history.create_conversation(user_id, "hf-webrtc-voice")
    else:
        conv_id = None
    model_id = data.get("model") or CATALOG.default_id()
    conversations[request.sid] = {
        "id": conv_id,
        "messages": [],
        "model": model_id,
    }
    emit(
        "conversation_started",
        {"conversation_id": conv_id, "model": model_id},
    )


@socketio.on("voice_message")
def handle_voice_message(data):
    data = data or {}
    text = data.get("text", "")
    if not text:
        emit("error", {"message": "No text provided"})
        return

    if request.sid not in conversations:
        handle_start_conversation(data)

    conv = conversations[request.sid]
    if data.get("model"):
        conv["model"] = data["model"]
    model_id = conv.get("model") or CATALOG.default_id()

    prior = list(conv["messages"])
    conv["messages"].append({"role": "user", "content": text})

    if conv["id"]:
        chat_history.add_message(conv["id"], "user", text)

    session_key = str(conv.get("id") or request.sid)
    enrichment = ENRICH.enrich_user_text(text, session_id=session_key)
    web_context = enrichment.context
    if enrichment.search_results or enrichment.search_error:
        emit(
            "web_search",
            {
                "query": enrichment.query,
                "error": enrichment.search_error or None,
                "results": enrichment.ui_search_rows(),
            },
        )
    if enrichment.fetch_results:
        emit(
            "web_fetch",
            {
                "urls": enrichment.ui_fetch_rows(),
                "log_file": WEB.log_file_path,
            },
        )

    response = generate_response(
        text, prior, web_context=web_context, model_id=model_id
    )
    conv["messages"].append({"role": "assistant", "content": response})

    if conv["id"]:
        chat_history.add_message(conv["id"], "assistant", response)

    emit("ai_response", {"text": response, "model": model_id})


@socketio.on("text_message")
def handle_text_message(data):
    handle_voice_message(data)


if __name__ == "__main__":
    run_socketio_app(socketio, app, MN_CFG, default_port=5001, debug=not MN_CFG.enabled)
