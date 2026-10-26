//! Shared OpenAI + Anthropic + Switchyard IDE assistant HTTP/SSE server.

use crate::auth::AuthManager;
use crate::backends::anthropic_axum::{
    anthropic_api_key, apply_model_hint, has_reasoning, reasoning_rejected,
};
use crate::backends::anthropic_convert::{anthropic_to_openai, strip_reasoning_content};
use crate::backends::anthropic_response::{
    anthropic_error, estimate_input_tokens, openai_to_anthropic,
};
use crate::backends::anthropic_stream::anthropic_message_to_sse;
use crate::backends::openai_chat::{
    chat_completions, extract_assistant_text, normalize_chat_url, require_backend_urls, usage_tuple,
};
use crate::backends::openai_tools::{
    annotate_model_entry, apply_ide_tool_nudge, build_passthrough_extra, message_content_for_memory,
    openai_error_from_upstream, resolve_max_tokens, should_bypass_memory_merge,
    should_preserve_client_transcript, should_skip_response_cache, stable_session_id,
    warn_if_not_tool_capable,
};
use crate::backends::streaming::completion_to_sse_chunks;
use crate::cache::{
    build_response_cache_for_model, is_moonshot_model, ConversationMemory, ResponseCache,
};
use crate::database::ChatHistory;
use crate::deploy::app_engine::resolve_app_engine;
use crate::deploy::ensure_models_runtime;
use crate::multinode::{
    describe_placement_for_logs, get_model_placement, get_multinode_config, BackendPool,
    ClusterInfo, MultiNodeConfig,
};
use crate::switchyard::config::{
    get_switchyard_config, DeploymentParams, ModelSpec, SwitchyardConfig,
};
use crate::switchyard::router::SwitchyardRouter;
use crate::switchyard::toml_export::write_routes_toml;
use crate::{env_bool, env_float, env_int};

use axum::{
    body::Body,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use chrono::Utc;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

pub const CODING_SYSTEM_PROMPT: &str = "You are an expert coding assistant optimized for complex, multi-file software engineering tasks. Provide clear, correct, production-quality code. Prefer complete solutions with necessary imports, error handling, and brief explanations of key design choices.";

#[derive(Clone)]
pub struct IdeServerConfig {
    pub family: String,
    pub app_name: String,
    pub default_port: u16,
    pub default_model: String,
    pub default_api_key: String,
}

#[derive(Clone)]
pub struct IdeAppState {
    pub cfg: IdeServerConfig,
    pub mn_cfg: MultiNodeConfig,
    pub cluster: ClusterInfo,
    pub model_name: String,
    pub api_key: String,
    pub upstream_api_key: String,
    pub request_timeout: f64,
    pub default_max_tokens: usize,
    pub default_temperature: f64,
    pub context_window: usize,
    pub effective_context_window: usize,
    pub auth_manager: AuthManager,
    pub chat_history: ChatHistory,
    pub response_cache: ResponseCache,
    pub conversation_memory: ConversationMemory,
    pub sy_cfg: SwitchyardConfig,
    pub sy_router: SwitchyardRouter,
}

pub fn build_ide_state(cfg: IdeServerConfig) -> Arc<IdeAppState> {
    if cfg.family == "hf" && env::var("LOAD_MODEL_WEIGHTS").is_err() {
        env::set_var("LOAD_MODEL_WEIGHTS", "false");
    }
    let mut mn_cfg = get_multinode_config(cfg.default_port, &cfg.app_name);
    let mut cluster = ClusterInfo::new(mn_cfg.clone(), &cfg.app_name);

    let model_name = if cfg.family == "hf" {
        env::var("HF_MODEL_NAME")
            .or_else(|_| env::var("VLLM_MODEL"))
            .unwrap_or_else(|_| cfg.default_model.clone())
    } else {
        env::var("NIM_MODEL_NAME").unwrap_or_else(|_| cfg.default_model.clone())
    };
    warn_if_not_tool_capable(&model_name, &cfg.app_name);

    let api_key = env::var("API_KEY").unwrap_or_else(|_| cfg.default_api_key.clone());
    let upstream_api_key = env::var("VLLM_API_KEY")
        .or_else(|_| env::var("OPENAI_API_KEY"))
        .unwrap_or_default();
    let request_timeout = env_float("VLLM_TIMEOUT", env_float("REQUEST_TIMEOUT", 300.0));
    let default_max_tokens = env_int("MAX_TOKENS", 32768).max(1) as usize;
    let default_temperature = env_float("TEMPERATURE", 0.3);
    let context_window = env_int("CONTEXT_WINDOW", 1048576).max(512) as usize;
    let effective_context_window =
        env_int("EFFECTIVE_CONTEXT_WINDOW", context_window as i64).max(512) as usize;

    let cache_enabled = env_bool("CACHE_ENABLED", true);
    let cache_max_size = env_int("CACHE_MAX_SIZE", 256).max(1) as usize;
    let cache_ttl_seconds = env_int("CACHE_TTL_SECONDS", 3600).max(1) as u64;
    let memory_max_conversations = env_int("MEMORY_MAX_CONVERSATIONS", 128).max(1) as usize;

    let response_cache = if cfg.family == "hf" {
        build_response_cache_for_model(
            &model_name,
            cache_max_size,
            cache_ttl_seconds,
            cache_enabled,
            None,
        )
    } else {
        ResponseCache::new(cache_max_size, cache_ttl_seconds, cache_enabled)
    };
    let conversation_memory =
        ConversationMemory::new(effective_context_window, memory_max_conversations);

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

    let mut sy_cfg = get_switchyard_config(&model_name, &default_backend, &cfg.app_name);
    let mut deploy_specs = sy_cfg.models.clone();
    if deploy_specs.is_empty() {
        let mut dep = DeploymentParams::default();
        if !mn_cfg.backend_urls.is_empty() {
            dep.backend_urls = mn_cfg.backend_urls.clone();
        } else if !default_backend.is_empty() {
            dep.backend_urls = vec![default_backend.clone()];
        }
        deploy_specs.push(ModelSpec {
            name: "default".to_string(),
            id: model_name.clone(),
            role: "weak".to_string(),
            deployment: dep,
            ..Default::default()
        });
    }

    let default_urls = if !mn_cfg.backend_urls.is_empty() {
        mn_cfg.backend_urls.clone()
    } else if !default_backend.is_empty() {
        vec![default_backend]
    } else {
        Vec::new()
    };

    let _handles = ensure_models_runtime(
        &resolve_app_engine(&cfg.family),
        &mut deploy_specs,
        &cfg.app_name,
        &upstream_api_key,
        &default_urls,
    );
    if !sy_cfg.models.is_empty() {
        sy_cfg.models = deploy_specs.clone();
    }

    let mut all_urls = Vec::new();
    for s in &deploy_specs {
        for u in &s.deployment.backend_urls {
            let norm = normalize_chat_url(u);
            if !norm.is_empty() && !all_urls.contains(&norm) {
                all_urls.push(norm);
            }
        }
    }
    if all_urls.is_empty() {
        all_urls = require_backend_urls(Some(&default_urls), &cfg.family)
            .unwrap_or_else(|_| vec!["http://127.0.0.1:8000/v1/chat/completions".to_string()]);
    }

    cluster.backend_pool = BackendPool::new(all_urls.clone(), &mn_cfg.backend_strategy);
    mn_cfg.backend_urls = all_urls.clone();
    cluster.model_placement = get_model_placement(Some(all_urls));

    let sy_router = SwitchyardRouter::new(sy_cfg.clone(), &model_name);
    if sy_cfg.enabled && !sy_cfg.routes_toml_path.is_empty() {
        let _ = write_routes_toml(&sy_cfg, Some(&sy_cfg.routes_toml_path));
    }

    println!("Placement: {}", describe_placement_for_logs(&cluster.model_placement));

    Arc::new(IdeAppState {
        cfg,
        mn_cfg,
        cluster,
        model_name,
        api_key,
        upstream_api_key,
        request_timeout,
        default_max_tokens,
        default_temperature,
        context_window,
        effective_context_window,
        auth_manager: AuthManager::default(),
        chat_history: ChatHistory::default(),
        response_cache,
        conversation_memory,
        sy_cfg,
        sy_router,
    })
}

fn verify_api_key(headers: &HeaderMap, expected: &str) -> bool {
    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if let Some(tok) = auth.strip_prefix("Bearer ") {
        return tok == expected;
    }
    false
}

fn headers_to_map(headers: &HeaderMap) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for (k, v) in headers {
        if let Ok(s) = v.to_str() {
            map.insert(k.as_str().to_string(), s.to_string());
        }
    }
    map
}

fn ensure_system_prompt(messages: &[Value]) -> Vec<Value> {
    if messages
        .iter()
        .any(|m| m.get("role").and_then(|v| v.as_str()) == Some("system"))
    {
        return messages.to_vec();
    }
    let mut out = Vec::with_capacity(messages.len() + 1);
    out.push(json!({"role": "system", "content": CODING_SYSTEM_PROMPT}));
    out.extend_from_slice(messages);
    out
}

fn merge_with_memory(session_id: &str, incoming: &[Value], mem: &ConversationMemory) -> Vec<Value> {
    if incoming.len() > 2 {
        return incoming.to_vec();
    }
    let stored = mem.get(session_id);
    if stored.is_empty() {
        return incoming.to_vec();
    }
    let mut sys_msgs: Vec<Value> = incoming
        .iter()
        .filter(|m| m.get("role").and_then(|v| v.as_str()) == Some("system"))
        .cloned()
        .collect();
    if sys_msgs.is_empty() {
        sys_msgs = stored
            .iter()
            .filter(|m| m.get("role").and_then(|v| v.as_str()) == Some("system"))
            .cloned()
            .collect();
    }
    let mut past_non_sys: Vec<Value> = stored
        .into_iter()
        .filter(|m| m.get("role").and_then(|v| v.as_str()) != Some("system"))
        .collect();
    let new_non_sys: Vec<Value> = incoming
        .iter()
        .filter(|m| m.get("role").and_then(|v| v.as_str()) != Some("system"))
        .cloned()
        .collect();
    if !past_non_sys.is_empty() && !new_non_sys.is_empty() && past_non_sys.last() == new_non_sys.first()
    {
        past_non_sys.pop();
    }
    sys_msgs.extend(past_non_sys);
    sys_msgs.extend(new_non_sys);
    sys_msgs
}

async fn handle_list_models(State(state): State<Arc<IdeAppState>>) -> impl IntoResponse {
    if state.sy_cfg.enabled {
        let mut payload = state
            .sy_router
            .list_models_payload(state.context_window, state.default_max_tokens);
        if let Some(arr) = payload.get_mut("data").and_then(|v| v.as_array_mut()) {
            for item in arr.iter_mut() {
                let is_route = item.get("id").and_then(|v| v.as_str()) == Some(&state.sy_cfg.route_id);
                *item = annotate_model_entry(item.clone(), is_route);
            }
        }
        return (StatusCode::OK, Json(payload));
    }
    let entry = annotate_model_entry(
        json!({
            "id": state.model_name,
            "object": "model",
            "created": Utc::now().timestamp(),
            "owned_by": state.cfg.app_name,
            "context_window": state.context_window,
            "effective_context_window": state.effective_context_window,
            "max_tokens": state.default_max_tokens,
        }),
        false,
    );
    (
        StatusCode::OK,
        Json(json!({
            "object": "list",
            "data": [entry],
        })),
    )
}

async fn handle_switchyard_info(State(state): State<Arc<IdeAppState>>) -> impl IntoResponse {
    (StatusCode::OK, Json(state.sy_router.health_block()))
}

enum ChatExecutionOutcome {
    Json(StatusCode, Value),
    Sse(Vec<String>, Value),
}

fn execute_openai_chat(
    state: &IdeAppState,
    headers: &HeaderMap,
    data: &Value,
) -> ChatExecutionOutcome {
    let raw_messages = data
        .get("messages")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let temperature = data
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(state.default_temperature);
    let max_tokens = resolve_max_tokens(
        data,
        state.default_max_tokens,
        Some(state.effective_context_window.saturating_sub(512).max(1)),
    );
    let stream = data
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let requested_model = data.get("model").and_then(|v| v.as_str());
    let tools = data.get("tools");
    let extra_passthrough = build_passthrough_extra(data);
    let extra_opt = if extra_passthrough
        .as_object()
        .map(|o| !o.is_empty())
        .unwrap_or(false)
    {
        Some(&extra_passthrough)
    } else {
        None
    };

    if raw_messages.is_empty() {
        return ChatExecutionOutcome::Json(
            StatusCode::BAD_REQUEST,
            json!({"error": "Messages required"}),
        );
    }

    let hmap = headers_to_map(headers);
    let session_id = stable_session_id(&hmap, data, &raw_messages);

    let messages = if should_preserve_client_transcript(tools, &raw_messages) {
        apply_ide_tool_nudge(&raw_messages, tools, None)
    } else {
        let mut m = ensure_system_prompt(&raw_messages);
        if !should_bypass_memory_merge(tools, &m) {
            m = merge_with_memory(&session_id, &m, &state.conversation_memory);
        }
        state.conversation_memory.trim_to_context(&m, max_tokens)
    };

    let cache_model_key = if state.sy_cfg.enabled {
        requested_model.unwrap_or(&state.sy_cfg.route_id)
    } else {
        &state.model_name
    };
    let skip_cache = should_skip_response_cache(tools, &messages);
    let cache_key = ResponseCache::make_key(
        cache_model_key,
        &messages,
        temperature,
        max_tokens,
        extra_opt,
    );

    if !skip_cache {
        if let Some(mut cached) = state.response_cache.get(&cache_key) {
            cached["cached"] = json!(true);
            let assistant_msg = extract_assistant_text(&cached);
            state.conversation_memory.set(&session_id, &messages);
            state
                .conversation_memory
                .append(&session_id, "assistant", &assistant_msg);
            if stream {
                let chunks = completion_to_sse_chunks(&cached, Some(cache_model_key), 48);
                return ChatExecutionOutcome::Sse(chunks, cached);
            }
            return ChatExecutionOutcome::Json(StatusCode::OK, cached);
        }
    }

    finish_openai_chat(
        state,
        &raw_messages,
        &messages,
        requested_model,
        &session_id,
        temperature,
        max_tokens,
        stream,
        skip_cache,
        &cache_key,
        extra_opt,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish_openai_chat(
    state: &IdeAppState,
    raw_messages: &[Value],
    messages: &[Value],
    requested_model: Option<&str>,
    session_id: &str,
    temperature: f64,
    max_tokens: usize,
    stream: bool,
    skip_cache: bool,
    cache_key: &str,
    extra_opt: Option<&Value>,
) -> ChatExecutionOutcome {
    let result_res: anyhow::Result<(Value, String, String)> = if state.sy_cfg.enabled {
        state
            .sy_router
            .route_chat(
                messages,
                requested_model,
                session_id,
                Some(temperature),
                Some(max_tokens),
                extra_opt,
            )
            .map(|(resp, text, served, _meta)| {
                let model_used = served
                    .map(|s| s.id)
                    .unwrap_or_else(|| state.model_name.clone());
                (resp, text, model_used)
            })
    } else {
        let url = state
            .cluster
            .next_backend(None)
            .unwrap_or_else(|| "http://127.0.0.1:8000/v1/chat/completions".to_string());
        match chat_completions(
            &url,
            messages,
            &state.model_name,
            temperature,
            max_tokens,
            &state.upstream_api_key,
            state.request_timeout,
            extra_opt,
        ) {
            Ok(data) => {
                state.cluster.backend_pool.mark_success(&url);
                let text = extract_assistant_text(&data);
                Ok((data, text, state.model_name.clone()))
            }
            Err(e) => {
                state.cluster.backend_pool.mark_failure(&url);
                Err(e)
            }
        }
    };

    match result_res {
        Ok((mut result, assistant_message, model_used)) => {
            if !result.is_object() {
                result = json!({});
            }
            if result.get("id").is_none() {
                result["id"] = json!(format!("chatcmpl-{}", Utc::now().timestamp()));
            }
            if result.get("object").is_none() {
                result["object"] = json!("chat.completion");
            }
            if result.get("created").is_none() {
                result["created"] = json!(Utc::now().timestamp());
            }
            if result.get("model").is_none() {
                result["model"] = json!(model_used);
            }
            result["cached"] = json!(false);
            let (pt, ct) = usage_tuple(&result);
            if result.get("usage").is_none() {
                result["usage"] = json!({
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": pt + ct,
                });
            }
            if !skip_cache {
                state.response_cache.set(cache_key, result.clone());
            }
            let mem_entry = result
                .get("choices")
                .and_then(|c| c.as_array())
                .and_then(|a| a.first())
                .and_then(|c0| c0.get("message"))
                .map(message_content_for_memory)
                .unwrap_or(assistant_message);
            state.conversation_memory.set(session_id, messages);
            state
                .conversation_memory
                .append(session_id, "assistant", &mem_entry);
            let conv_id = state
                .chat_history
                .create_conversation(None, &state.cfg.app_name);
            if let Some(last_msg) = raw_messages.last() {
                let u_text = last_msg
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                state.chat_history.add_message(conv_id, "user", u_text, None);
            }
            state
                .chat_history
                .add_message(conv_id, "assistant", &mem_entry, None);
            if stream {
                let chunks = completion_to_sse_chunks(&result, Some(&model_used), 48);
                ChatExecutionOutcome::Sse(chunks, result)
            } else {
                ChatExecutionOutcome::Json(StatusCode::OK, result)
            }
        }
        Err(e) => {
            let err_str = e.to_string();
            let (payload, status) = openai_error_from_upstream(&err_str, None, None, 502);
            ChatExecutionOutcome::Json(
                StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY),
                payload,
            )
        }
    }
}

fn sse_response(chunks: Vec<String>) -> Response {
    let body_str = chunks.join("");
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .header("X-Accel-Buffering", "no")
        .body(Body::from(body_str))
        .unwrap()
}

async fn handle_chat_completions(
    State(state): State<Arc<IdeAppState>>,
    headers: HeaderMap,
    Json(data): Json<Value>,
) -> Response {
    if !verify_api_key(&headers, &state.api_key) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "Invalid API key"}))).into_response();
    }
    let st = state.clone();
    let hdrs = headers.clone();
    let outcome = tokio::task::spawn_blocking(move || execute_openai_chat(&st, &hdrs, &data))
        .await
        .unwrap_or_else(|e| {
            ChatExecutionOutcome::Json(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"error": e.to_string()}),
            )
        });
    match outcome {
        ChatExecutionOutcome::Json(code, val) => (code, Json(val)).into_response(),
        ChatExecutionOutcome::Sse(chunks, _) => sse_response(chunks),
    }
}

async fn handle_anthropic_messages(
    State(state): State<Arc<IdeAppState>>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> Response {
    if !anthropic_api_key(&headers, &state.api_key) {
        let (p, code) = anthropic_error("Invalid API key", "authentication_error", 401);
        return (StatusCode::from_u16(code).unwrap(), Json(p)).into_response();
    }
    if !body.is_object()
        || body
            .get("messages")
            .and_then(|v| v.as_array())
            .map(|a| a.is_empty())
            .unwrap_or(true)
    {
        let (p, code) = anthropic_error("messages is required", "invalid_request_error", 400);
        return (StatusCode::from_u16(code).unwrap(), Json(p)).into_response();
    }
    let router = state.sy_router.clone();
    let requested = apply_model_hint(&mut body, |req, has_tools| {
        router.resolve_model_hint(Some(req), has_tools)
    });
    let req_model = if requested.is_empty() {
        state.model_name.clone()
    } else {
        requested
    };
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let mut openai_body = anthropic_to_openai(&body);
    openai_body["stream"] = json!(stream);

    let st = state.clone();
    let hdrs = headers.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        let out = execute_openai_chat(&st, &hdrs, &openai_body);
        if let ChatExecutionOutcome::Json(code, ref val) = out {
            let err_str = serde_json::to_string(val).unwrap_or_default();
            if reasoning_rejected(Some(code.as_u16()), &err_str) && has_reasoning(&openai_body) {
                let stripped = strip_reasoning_content(&openai_body);
                return execute_openai_chat(&st, &hdrs, &stripped);
            }
        }
        out
    })
    .await
    .unwrap_or_else(|e| {
        ChatExecutionOutcome::Json(StatusCode::BAD_GATEWAY, json!({"error": e.to_string()}))
    });

    match outcome {
        ChatExecutionOutcome::Json(code, val) => {
            if !code.is_success() {
                let (p, s) = anthropic_error(
                    &serde_json::to_string(&val).unwrap_or_default(),
                    "api_error",
                    code.as_u16(),
                );
                return (StatusCode::from_u16(s).unwrap(), Json(p)).into_response();
            }
            let anth = openai_to_anthropic(&val, &req_model);
            (StatusCode::OK, Json(anth)).into_response()
        }
        ChatExecutionOutcome::Sse(_chunks, completion) => {
            let anth_msg = openai_to_anthropic(&completion, &req_model);
            let sse_events = anthropic_message_to_sse(&anth_msg);
            sse_response(sse_events)
        }
    }
}

async fn handle_anthropic_count_tokens(
    State(state): State<Arc<IdeAppState>>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> Response {
    if !anthropic_api_key(&headers, &state.api_key) {
        let (p, code) = anthropic_error("Invalid API key", "authentication_error", 401);
        return (StatusCode::from_u16(code).unwrap(), Json(p)).into_response();
    }
    let router = state.sy_router.clone();
    let _ = apply_model_hint(&mut body, |req, has_tools| {
        router.resolve_model_hint(Some(req), has_tools)
    });
    let openai_body = anthropic_to_openai(&body);
    (
        StatusCode::OK,
        Json(json!({"input_tokens": estimate_input_tokens(&openai_body)})),
    )
        .into_response()
}

async fn handle_completions(
    State(state): State<Arc<IdeAppState>>,
    headers: HeaderMap,
    Json(data): Json<Value>,
) -> Response {
    if !verify_api_key(&headers, &state.api_key) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "Invalid API key"}))).into_response();
    }
    let prompt = data.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    if prompt.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "Prompt required"}))).into_response();
    }
    let chat_req = json!({
        "model": data.get("model").cloned().unwrap_or_else(|| json!(state.model_name)),
        "messages": [
            {"role": "system", "content": CODING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": data.get("temperature").cloned().unwrap_or_else(|| json!(state.default_temperature)),
        "max_tokens": data.get("max_tokens").cloned().unwrap_or_else(|| json!(state.default_max_tokens)),
        "stream": false,
    });
    let st = state.clone();
    let hdrs = headers.clone();
    let outcome = tokio::task::spawn_blocking(move || execute_openai_chat(&st, &hdrs, &chat_req))
        .await
        .unwrap_or_else(|e| {
            ChatExecutionOutcome::Json(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"error": e.to_string()}),
            )
        });
    match outcome {
        ChatExecutionOutcome::Json(code, val) if code.is_success() => {
            let text = extract_assistant_text(&val);
            (
                StatusCode::OK,
                Json(json!({
                    "id": format!("cmpl-{}", Utc::now().timestamp()),
                    "object": "text_completion",
                    "created": Utc::now().timestamp(),
                    "model": val.get("model").cloned().unwrap_or_else(|| json!(state.model_name)),
                    "choices": [{
                        "text": text,
                        "index": 0,
                        "logprobs": Value::Null,
                        "finish_reason": "stop",
                    }],
                    "usage": val.get("usage").cloned().unwrap_or_else(|| json!({})),
                })),
            )
                .into_response()
        }
        ChatExecutionOutcome::Json(code, val) => (code, Json(val)).into_response(),
        ChatExecutionOutcome::Sse(chunks, _) => sse_response(chunks),
    }
}

async fn handle_health(State(state): State<Arc<IdeAppState>>) -> impl IntoResponse {
    let mut extra = json!({
        "model": state.model_name,
        "context_window": state.context_window,
        "effective_context_window": state.effective_context_window,
        "max_tokens": state.default_max_tokens,
        "temperature": state.default_temperature,
        "cache": state.response_cache.stats(),
        "memory": state.conversation_memory.stats(),
        "switchyard": state.sy_router.health_block(),
    });
    if state.cfg.family == "hf" {
        extra["backend_type"] = json!("vllm");
        extra["moonshot_model"] = json!(is_moonshot_model(&state.model_name));
    }
    (StatusCode::OK, Json(state.cluster.health(extra)))
}

async fn handle_clear_cache(
    State(state): State<Arc<IdeAppState>>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if !verify_api_key(&headers, &state.api_key) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "Invalid API key"}))).into_response();
    }
    let clear_memory = params
        .get("memory")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);
    state.response_cache.clear();
    if clear_memory {
        state.conversation_memory.clear(None);
        if let (Some(esc), Some(sess)) = (&state.sy_router.escalation, params.get("session")) {
            esc.reset_session(sess);
        }
    }
    (
        StatusCode::OK,
        Json(json!({
            "status": "cleared",
            "memory_cleared": clear_memory,
            "cache": state.response_cache.stats(),
            "memory": state.conversation_memory.stats(),
            "switchyard": state.sy_router.health_block(),
        })),
    )
        .into_response()
}

async fn handle_reset_switchyard_session(
    State(state): State<Arc<IdeAppState>>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
    body: Option<Json<Value>>,
) -> Response {
    if !verify_api_key(&headers, &state.api_key) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "Invalid API key"}))).into_response();
    }
    let Some(esc) = &state.sy_router.escalation else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Escalation router not active"})),
        )
            .into_response();
    };
    let data = body.map(|b| b.0).unwrap_or_else(|| json!({}));
    let session_id = headers
        .get("x-conversation-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .or_else(|| {
            data.get("session_id")
                .or_else(|| data.get("conversation_id"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .or_else(|| params.get("session").cloned())
        .unwrap_or_else(|| "default".to_string());
    esc.reset_session(&session_id);
    (
        StatusCode::OK,
        Json(json!({"status": "reset", "session_id": session_id})),
    )
        .into_response()
}

pub fn build_ide_router(state: Arc<IdeAppState>) -> Router {
    Router::new()
        .route("/v1/models", get(handle_list_models))
        .route("/v1/switchyard", get(handle_switchyard_info))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/messages", post(handle_anthropic_messages))
        .route(
            "/v1/messages/count_tokens",
            post(handle_anthropic_count_tokens),
        )
        .route("/v1/completions", post(handle_completions))
        .route("/health", get(handle_health))
        .route("/v1/cache", delete(handle_clear_cache))
        .route(
            "/v1/switchyard/session",
            delete(handle_reset_switchyard_session),
        )
        .layer(CorsLayer::permissive())
        .with_state(state)
}

pub async fn run_ide_server(cfg: IdeServerConfig) -> anyhow::Result<()> {
    let state = build_ide_state(cfg);
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
    println!("Model: {}", state.model_name);
    println!("API Key: {}", state.api_key);
    let app = build_ide_router(state);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
