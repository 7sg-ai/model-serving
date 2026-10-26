//! Shared WebRTC / Voice backend server (REST + Socket.IO v4 via socketioxide).

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
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use parking_lot::Mutex;
use serde_json::{json, Value};
use socketioxide::{
    extract::{Data, SocketRef},
    SocketIo,
};
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::{cors::CorsLayer, services::ServeDir};

#[derive(Clone)]
pub struct VoiceServerConfig {
    pub family: String,
    pub app_name: String,
    pub default_port: u16,
    pub default_model: String,
    pub static_dir: PathBuf,
}

#[derive(Clone, Debug)]
pub struct ActiveVoiceConv {
    pub id: Option<i64>,
    pub messages: Vec<Value>,
    pub model: String,
}

#[derive(Clone)]
pub struct VoiceAppState {
    pub cfg: VoiceServerConfig,
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
    pub conversations: Arc<Mutex<HashMap<String, ActiveVoiceConv>>>,
}

pub fn build_voice_state(cfg: VoiceServerConfig) -> Arc<VoiceAppState> {
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

    Arc::new(VoiceAppState {
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
        conversations: Arc::new(Mutex::new(HashMap::new())),
    })
}

fn extract_token(headers: &HeaderMap) -> Option<String> {
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
    None
}

async fn handle_register(
    State(state): State<Arc<VoiceAppState>>,
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
    State(state): State<Arc<VoiceAppState>>,
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
    State(state): State<Arc<VoiceAppState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Some(tok) = extract_token(&headers) {
        state.auth_manager.logout(&tok);
    }
    (StatusCode::OK, Json(json!({"message": "Logged out"})))
}

async fn handle_models(State(state): State<Arc<VoiceAppState>>) -> impl IntoResponse {
    (StatusCode::OK, Json(state.catalog.list_payload()))
}

async fn handle_conversations(
    State(state): State<Arc<VoiceAppState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let Some(uid) = extract_token(&headers).and_then(|t| state.auth_manager.validate_session(&t)) else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "Not authenticated"})),
        );
    };
    let convs = state
        .chat_history
        .get_user_conversations(uid, Some(&state.cfg.app_name));
    (StatusCode::OK, Json(json!({"conversations": convs})))
}

async fn handle_get_conversation(
    State(state): State<Arc<VoiceAppState>>,
    Path(conv_id): Path<i64>,
) -> impl IntoResponse {
    let messages = state.chat_history.get_conversation_history(conv_id);
    (StatusCode::OK, Json(json!({"messages": messages})))
}

async fn handle_health(State(state): State<Arc<VoiceAppState>>) -> impl IntoResponse {
    let extra = json!({
        "model": state.model_name,
        "web_access": state.web.stats(),
        "catalog": state.catalog.health_block(),
    });
    (StatusCode::OK, Json(state.cluster.health(extra)))
}

fn handle_socket_message(s: SocketRef, data: Value, st_msg: Arc<VoiceAppState>) {
    let text = data
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if text.is_empty() {
        let _ = s.emit("error", &json!({"message": "No text provided"}));
        return;
    }
    let sid = s.id.to_string();
    let (history, model_id, conv_id) = {
        let mut guard = st_msg.conversations.lock();
        let entry = guard.entry(sid.clone()).or_insert_with(|| ActiveVoiceConv {
            id: None,
            messages: Vec::new(),
            model: st_msg.catalog.default_id(),
        });
        if let Some(m) = data.get("model").and_then(|v| v.as_str()) {
            if !m.is_empty() {
                entry.model = m.to_string();
            }
        }
        entry
            .messages
            .push(json!({"role": "user", "content": text}));
        (entry.messages.clone(), entry.model.clone(), entry.id)
    };

    if let Some(cid) = conv_id {
        st_msg.chat_history.add_message(cid, "user", &text, None);
    }

    tokio::spawn(async move {
        let st_bg = st_msg.clone();
        let text_bg = text.clone();
        let sid_bg = sid.clone();
        let m_bg = model_id.clone();
        let (reply, enrichment) = tokio::task::spawn_blocking(move || {
            let (enriched, enr) = st_bg
                .enrich
                .enrich_messages(&history, Some(&text_bg), &sid_bg);
            let r = match st_bg.catalog.chat(
                &enriched,
                Some(&m_bg),
                Some(st_bg.default_temperature),
                Some(st_bg.default_max_tokens),
            ) {
                Ok((t, _)) => {
                    if t.is_empty() {
                        "(empty response)".to_string()
                    } else {
                        t
                    }
                }
                Err(e) => format!("Error: {e}"),
            };
            (r, enr)
        })
        .await
        .unwrap_or_else(|e| (format!("Error: {e}"), Default::default()));

        if !enrichment.search_results.is_empty() || !enrichment.search_error.is_empty() {
            let _ = s.emit(
                "web_search",
                &json!({
                    "query": enrichment.query,
                    "error": if enrichment.search_error.is_empty() { Value::Null } else { json!(enrichment.search_error) },
                    "results": enrichment.ui_search_rows(),
                }),
            );
        }
        if !enrichment.fetch_results.is_empty() {
            let _ = s.emit(
                "web_fetch",
                &json!({
                    "urls": enrichment.ui_fetch_rows(),
                    "log_file": st_msg.web.log_file_path,
                }),
            );
        }

        {
            let mut guard = st_msg.conversations.lock();
            if let Some(entry) = guard.get_mut(&sid) {
                entry
                    .messages
                    .push(json!({"role": "assistant", "content": reply}));
            }
        }
        if let Some(cid) = conv_id {
            st_msg
                .chat_history
                .add_message(cid, "assistant", &reply, None);
        }
        let _ = s.emit("ai_response", &json!({"text": reply, "model": model_id}));
    });
}

fn setup_socketio(state: Arc<VoiceAppState>) -> socketioxide::layer::SocketIoLayer {
    let (layer, io) = SocketIo::new_layer();

    io.ns("/", move |socket: SocketRef| {
        let st = state.clone();
        let _ = socket.emit("connected", &json!({"message": "Connected to server"}));

        let st_start = st.clone();
        socket.on(
            "start_conversation",
            move |s: SocketRef, Data::<Value>(data)| {
                let model_id = data
                    .get("model")
                    .and_then(|v| v.as_str())
                    .filter(|m| !m.is_empty())
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| st_start.catalog.default_id());
                let sid = s.id.to_string();
                st_start.conversations.lock().insert(
                    sid,
                    ActiveVoiceConv {
                        id: None,
                        messages: Vec::new(),
                        model: model_id.clone(),
                    },
                );
                let _ = s.emit(
                    "conversation_started",
                    &json!({"conversation_id": Value::Null, "model": model_id}),
                );
            },
        );

        let st_voice = st.clone();
        socket.on("voice_message", move |s: SocketRef, Data::<Value>(data)| {
            handle_socket_message(s, data, st_voice.clone());
        });

        let st_text = st.clone();
        socket.on("text_message", move |s: SocketRef, Data::<Value>(data)| {
            handle_socket_message(s, data, st_text.clone());
        });

        let st_disc = st;
        socket.on_disconnect(move |s: SocketRef| {
            st_disc.conversations.lock().remove(&s.id.to_string());
        });
    });

    layer
}

pub fn build_voice_router(state: Arc<VoiceAppState>) -> Router {
    let sio_layer = setup_socketio(state.clone());
    let static_dir = state.cfg.static_dir.clone();
    Router::new()
        .route("/api/auth/register", post(handle_register))
        .route("/api/auth/login", post(handle_login))
        .route("/api/auth/logout", post(handle_logout))
        .route("/api/models", get(handle_models))
        .route("/api/conversations", get(handle_conversations))
        .route("/api/conversations/:id", get(handle_get_conversation))
        .route("/health", get(handle_health))
        .fallback_service(ServeDir::new(static_dir))
        .layer(sio_layer)
        .layer(CorsLayer::permissive())
        .with_state(state)
}

pub async fn run_voice_server(cfg: VoiceServerConfig) -> anyhow::Result<()> {
    let state = build_voice_state(cfg);
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
    println!("Starting {} on http://{addr} (Socket.IO)", state.cfg.app_name);
    let app = build_voice_router(state);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
