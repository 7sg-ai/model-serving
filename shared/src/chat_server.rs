//! Shared Web Chat UI & REST API server (replaces Streamlit chat apps in Rust).

use crate::auth::AuthManager;
use crate::backends::openai_chat::require_backend_urls;
use crate::database::ChatHistory;
use crate::deploy::app_engine::resolve_app_engine;
use crate::model_catalog::{deploy_app_models, load_app_models, ModelCatalog};
use crate::multinode::{get_multinode_config, BackendPool, ClusterInfo, MultiNodeConfig};
use crate::tools::web_access::{get_web_access_tool, WebAccessTool};
use crate::tools::web_enrich::{get_web_enricher, WebEnricher};
use crate::{env_float, env_int};

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::env;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

#[derive(Clone)]
pub struct ChatServerConfig {
    pub family: String,
    pub app_name: String,
    pub default_port: u16,
    pub default_model: String,
    pub page_title: String,
    pub page_icon: String,
}

#[derive(Clone)]
pub struct ChatAppState {
    pub cfg: ChatServerConfig,
    pub mn_cfg: MultiNodeConfig,
    pub cluster: ClusterInfo,
    pub catalog: ModelCatalog,
    pub model_name: String,
    pub default_temperature: f64,
    pub default_max_tokens: usize,
    pub auth_manager: AuthManager,
    pub chat_history: ChatHistory,
    pub web: WebAccessTool,
    pub enrich: WebEnricher,
}

pub fn build_chat_state(cfg: ChatServerConfig) -> Arc<ChatAppState> {
    if cfg.family == "hf" && env::var("LOAD_MODEL_WEIGHTS").is_err() {
        env::set_var("LOAD_MODEL_WEIGHTS", "false");
    }
    let mut mn_cfg = get_multinode_config(cfg.default_port, &cfg.app_name);
    let mut cluster = ClusterInfo::new(mn_cfg.clone(), &cfg.app_name);
    let web = get_web_access_tool(&cfg.app_name);
    let enrich = get_web_enricher(&cfg.app_name);

    let default_model_name = if cfg.family == "hf" {
        env::var("HF_MODEL_NAME")
            .or_else(|_| env::var("VLLM_MODEL"))
            .unwrap_or_else(|_| cfg.default_model.clone())
    } else {
        env::var("NIM_MODEL_NAME").unwrap_or_else(|_| cfg.default_model.clone())
    };
    let default_temperature = env_float("TEMPERATURE", 0.7);
    let default_max_tokens = env_int("MAX_TOKENS", 1024).max(1) as usize;
    let api_key = env::var("VLLM_API_KEY")
        .or_else(|_| env::var("OPENAI_API_KEY"))
        .unwrap_or_default();

    let default_backend = mn_cfg
        .backend_urls
        .first()
        .cloned()
        .or_else(|| {
            if cfg.family == "nim" {
                Some(
                    env::var("NIM_API_URL")
                        .unwrap_or_else(|_| "http://localhost:8000/v1/chat/completions".to_string()),
                )
            } else {
                None
            }
        })
        .unwrap_or_default();

    let mut sy_cfg = load_app_models(&default_model_name, &default_backend, &cfg.app_name);
    let default_urls = if !mn_cfg.backend_urls.is_empty() {
        mn_cfg.backend_urls.clone()
    } else if !default_backend.is_empty() {
        vec![default_backend]
    } else {
        Vec::new()
    };

    let (catalog, _handles) = deploy_app_models(
        &resolve_app_engine(&cfg.family),
        &mut sy_cfg,
        &cfg.app_name,
        &api_key,
        &default_urls,
    );

    let mut backends = catalog.all_backend_urls();
    if backends.is_empty() {
        backends = require_backend_urls(Some(&default_urls), &cfg.family)
            .unwrap_or_else(|_| vec!["http://127.0.0.1:8000/v1/chat/completions".to_string()]);
    }
    cluster.backend_pool = BackendPool::new(backends.clone(), &mn_cfg.backend_strategy);
    mn_cfg.backend_urls = backends;
    let model_name = catalog.default_id();

    Arc::new(ChatAppState {
        cfg,
        mn_cfg,
        cluster,
        catalog,
        model_name,
        default_temperature,
        default_max_tokens,
        auth_manager: AuthManager::default(),
        chat_history: ChatHistory::default(),
        web,
        enrich,
    })
}

fn extract_token(headers: &HeaderMap, body: Option<&Value>) -> Option<String> {
    if let Some(auth) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
        if let Some(t) = auth.strip_prefix("Bearer ") {
            return Some(t.to_string());
        }
    }
    if let Some(cookie) = headers.get("cookie").and_then(|v| v.to_str().ok()) {
        for part in cookie.split(';') {
            if let Some((k, v)) = part.trim().split_once('=') {
                if k.trim() == "session_token" && !v.trim().is_empty() {
                    return Some(v.trim().to_string());
                }
            }
        }
    }
    body.and_then(|b| b.get("session_token"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

async fn handle_register(
    State(state): State<Arc<ChatAppState>>,
    Json(data): Json<Value>,
) -> impl IntoResponse {
    let username = data.get("username").and_then(|v| v.as_str()).unwrap_or("");
    let password = data.get("password").and_then(|v| v.as_str()).unwrap_or("");
    if username.is_empty() || password.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Username and password required"})),
        );
    }
    if state.auth_manager.register_user(username, password) {
        (
            StatusCode::CREATED,
            Json(json!({"message": "Registration successful"})),
        )
    } else {
        (
            StatusCode::CONFLICT,
            Json(json!({"error": "Username already exists"})),
        )
    }
}

async fn handle_login(
    State(state): State<Arc<ChatAppState>>,
    Json(data): Json<Value>,
) -> impl IntoResponse {
    let username = data.get("username").and_then(|v| v.as_str()).unwrap_or("");
    let password = data.get("password").and_then(|v| v.as_str()).unwrap_or("");
    if let Some(token) = state.auth_manager.login(username, password) {
        let user_id = state.auth_manager.validate_session(&token);
        let mut headers = HeaderMap::new();
        if let Ok(hv) = format!("session_token={token}; Path=/; HttpOnly").parse() {
            headers.insert("Set-Cookie", hv);
        }
        return (
            StatusCode::OK,
            headers,
            Json(json!({
                "message": "Login successful",
                "username": username,
                "user_id": user_id,
                "session_token": token,
            })),
        );
    }
    (
        StatusCode::UNAUTHORIZED,
        HeaderMap::new(),
        Json(json!({"error": "Invalid credentials"})),
    )
}

async fn handle_logout(
    State(state): State<Arc<ChatAppState>>,
    headers: HeaderMap,
    body: Option<Json<Value>>,
) -> impl IntoResponse {
    let b = body.map(|x| x.0);
    if let Some(tok) = extract_token(&headers, b.as_ref()) {
        state.auth_manager.logout(&tok);
    }
    (StatusCode::OK, Json(json!({"message": "Logged out"})))
}

async fn handle_models(State(state): State<Arc<ChatAppState>>) -> impl IntoResponse {
    (StatusCode::OK, Json(state.catalog.list_payload()))
}

async fn handle_conversations(
    State(state): State<Arc<ChatAppState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let user_id = extract_token(&headers, None)
        .and_then(|t| state.auth_manager.validate_session(&t));
    let Some(uid) = user_id else {
        return (StatusCode::OK, Json(json!({"conversations": []})));
    };
    let convs = state
        .chat_history
        .get_user_conversations(uid, Some(&state.cfg.app_name));
    (StatusCode::OK, Json(json!({"conversations": convs})))
}

async fn handle_get_conversation(
    State(state): State<Arc<ChatAppState>>,
    Path(conv_id): Path<i64>,
) -> impl IntoResponse {
    let messages = state.chat_history.get_conversation_history(conv_id);
    (StatusCode::OK, Json(json!({"messages": messages})))
}

async fn handle_chat(
    State(state): State<Arc<ChatAppState>>,
    headers: HeaderMap,
    Json(data): Json<Value>,
) -> impl IntoResponse {
    let prompt = data
        .get("prompt")
        .or_else(|| data.get("text"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if prompt.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "prompt is required"})),
        );
    }
    let model_id = data
        .get("model")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| state.model_name.clone());
    let temperature = data
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(state.default_temperature);
    let max_tokens = data
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(state.default_max_tokens);
    let mut history = data
        .get("messages")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if history.is_empty()
        || history
            .last()
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            != Some(&prompt)
    {
        history.push(json!({"role": "user", "content": prompt}));
    }

    let user_id = extract_token(&headers, Some(&data))
        .and_then(|t| state.auth_manager.validate_session(&t));
    let mut conv_id = data
        .get("conversation_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    if conv_id == 0 {
        if let Some(uid) = user_id {
            conv_id = state
                .chat_history
                .create_conversation(Some(uid), &state.cfg.app_name);
        }
    }
    if conv_id > 0 {
        state
            .chat_history
            .add_message(conv_id, "user", &prompt, None);
    }

    let st = state.clone();
    let p_clone = prompt.clone();
    let m_clone = model_id.clone();
    let out = tokio::task::spawn_blocking(move || {
        let session_key = if conv_id > 0 {
            conv_id.to_string()
        } else {
            "guest".to_string()
        };
        let (enriched_msgs, enrichment) =
            st.enrich
                .enrich_messages(&history, Some(&p_clone), &session_key);
        let reply = match st.catalog.chat(
            &enriched_msgs,
            Some(&m_clone),
            Some(temperature),
            Some(max_tokens),
        ) {
            Ok((text, _)) => {
                if text.is_empty() {
                    "(empty response)".to_string()
                } else {
                    text
                }
            }
            Err(e) => format!("Error querying model ({m_clone}): {e}"),
        };
        (reply, enrichment)
    })
    .await
    .unwrap_or_else(|e| (format!("Error: {e}"), Default::default()));

    let (reply, enrichment) = out;
    if conv_id > 0 {
        state
            .chat_history
            .add_message(conv_id, "assistant", &reply, None);
    }

    (
        StatusCode::OK,
        Json(json!({
            "response": reply,
            "model": model_id,
            "conversation_id": if conv_id > 0 { json!(conv_id) } else { Value::Null },
            "web_search": {
                "query": enrichment.query,
                "error": if enrichment.search_error.is_empty() { Value::Null } else { json!(enrichment.search_error) },
                "results": enrichment.ui_search_rows(),
            },
            "web_fetch": enrichment.ui_fetch_rows(),
        })),
    )
}

async fn handle_health(State(state): State<Arc<ChatAppState>>) -> impl IntoResponse {
    let extra = json!({
        "model": state.model_name,
        "web_access": state.web.stats(),
        "catalog": state.catalog.health_block(),
    });
    (StatusCode::OK, Json(state.cluster.health(extra)))
}

async fn handle_ui(State(state): State<Arc<ChatAppState>>) -> Html<String> {
    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{icon} {title}</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; display: flex; height: 100vh; background: #0e1117; color: #fafafa; }}
  #sidebar {{ width: 280px; background: #262730; padding: 20px; display: flex; flex-direction: column; gap: 12px; border-right: 1px solid #333; }}
  #main {{ flex: 1; display: flex; flex-direction: column; padding: 24px; overflow-y: auto; }}
  input, select, button {{ padding: 8px 12px; border-radius: 6px; border: 1px solid #444; background: #1e1e26; color: #fff; }}
  button {{ background: #ff4b4b; border: none; cursor: pointer; font-weight: 600; }}
  #messages {{ flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; margin-bottom: 16px; }}
  .msg {{ padding: 12px 16px; border-radius: 8px; max-width: 80%; white-space: pre-wrap; }}
  .user {{ background: #2b313e; align-self: flex-end; }}
  .assistant {{ background: #1c2029; align-self: flex-start; border: 1px solid #333; }}
  #chat-bar {{ display: flex; gap: 8px; }}
  #chat-input {{ flex: 1; }}
</style>
</head>
<body>
<div id="sidebar">
  <h3>{icon} {title}</h3>
  <label>Model <select id="model-select"></select></label>
  <label>Temperature <input type="number" id="temp" value="{temp}" step="0.1" min="0" max="2"></label>
  <label>Max Tokens <input type="number" id="max-tokens" value="{max_tokens}" step="128" min="64" max="32768"></label>
  <button onclick="clearChat()">Clear Chat History</button>
</div>
<div id="main">
  <h2>{icon} {title}</h2>
  <div id="messages"></div>
  <div id="chat-bar">
    <input id="chat-input" placeholder="What would you like to know?" onkeydown="if(event.key==='Enter')sendChat()">
    <button onclick="sendChat()">Send</button>
  </div>
</div>
<script>
let messages = [];
let conversationId = null;
async function init() {{
  const r = await fetch('/api/models');
  const d = await r.json();
  const sel = document.getElementById('model-select');
  (d.data || []).forEach(m => {{
    const o = document.createElement('option');
    o.value = m.id; o.textContent = m.label || m.id;
    if (m.id === d.default) o.selected = true;
    sel.appendChild(o);
  }});
}}
function addMsg(role, text) {{
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = text;
  const box = document.getElementById('messages');
  box.appendChild(d);
  box.scrollTop = box.scrollHeight;
}}
function clearChat() {{
  messages = []; conversationId = null;
  document.getElementById('messages').innerHTML = '';
}}
async function sendChat() {{
  const inp = document.getElementById('chat-input');
  const prompt = inp.value.trim();
  if (!prompt) return;
  inp.value = '';
  messages.push({{role: 'user', content: prompt}});
  addMsg('user', prompt);
  const r = await fetch('/api/chat', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{
      prompt,
      messages,
      conversation_id: conversationId,
      model: document.getElementById('model-select').value,
      temperature: parseFloat(document.getElementById('temp').value || '0.7'),
      max_tokens: parseInt(document.getElementById('max-tokens').value || '1024', 10)
    }})
  }});
  const data = await r.json();
  if (data.conversation_id) conversationId = data.conversation_id;
  const reply = data.response || data.error || '(empty)';
  messages.push({{role: 'assistant', content: reply}});
  addMsg('assistant', reply);
}}
init();
</script>
</body>
</html>"#,
        icon = state.cfg.page_icon,
        title = state.cfg.page_title,
        temp = state.default_temperature,
        max_tokens = state.default_max_tokens,
    );
    Html(html)
}

pub fn build_chat_router(state: Arc<ChatAppState>) -> Router {
    Router::new()
        .route("/", get(handle_ui))
        .route("/api/auth/register", post(handle_register))
        .route("/api/auth/login", post(handle_login))
        .route("/api/auth/logout", post(handle_logout))
        .route("/api/models", get(handle_models))
        .route("/api/conversations", get(handle_conversations))
        .route("/api/conversations/:id", get(handle_get_conversation))
        .route("/api/chat", post(handle_chat))
        .route("/health", get(handle_health))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

pub async fn run_chat_server(cfg: ChatServerConfig) -> anyhow::Result<()> {
    let state = build_chat_state(cfg);
    let host = if state.mn_cfg.node_host.is_empty() {
        "0.0.0.0".to_string()
    } else {
        state.mn_cfg.node_host.clone()
    };
    let port = if state.mn_cfg.node_port > 0 {
        state.mn_cfg.node_port
    } else {
        state.cfg.default_port
    };
    let addr = format!("{host}:{port}");
    println!("Starting {} on http://{addr}", state.cfg.app_name);
    let app = build_chat_router(state);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
