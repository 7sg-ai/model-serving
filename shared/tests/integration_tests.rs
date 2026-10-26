//! Comprehensive unit & integration tests for web search, OpenAI tools, SSE streaming,
//! Anthropic Messages compatibility, SQLite auth & chat history, and caching.

use model_serving_shared::auth::AuthManager;
use model_serving_shared::backends::anthropic_convert::{
    anthropic_to_openai, claude_tier_hint, strip_reasoning_content,
};
use model_serving_shared::backends::anthropic_response::{openai_to_anthropic, repair_tool_input};
use model_serving_shared::backends::anthropic_stream::iter_openai_sse_as_anthropic;
use model_serving_shared::backends::openai_tools::{
    apply_ide_tool_nudge, build_passthrough_extra, build_tools_extra, is_tool_continuation,
    openai_error_from_upstream, request_has_tools, resolve_max_tokens, should_bypass_memory_merge,
    should_preserve_client_transcript, stable_session_id,
};
use model_serving_shared::backends::streaming::completion_to_sse_chunks;
use model_serving_shared::cache::{ConversationMemory, ResponseCache};
use model_serving_shared::database::ChatHistory;
use model_serving_shared::tools::web_search::{extract_search_query, parse_ddg_html};
use serde_json::{json, Value};
use std::collections::HashMap;

#[test]
fn test_extract_search_query() {
    assert_eq!(
        extract_search_query("search the web for python 3.13").as_deref(),
        Some("python 3.13")
    );
    assert_eq!(
        extract_search_query("/search nvidia nim").as_deref(),
        Some("nvidia nim")
    );
    assert_eq!(
        extract_search_query("look up flask streaming").as_deref(),
        Some("flask streaming")
    );
    assert_eq!(extract_search_query("hello there"), None);
}

#[test]
fn test_parse_ddg_html() {
    let html = r#"<div class="result"><a class="result__a" href="https://example.com/a">Alpha</a><div class="result__snippet">Snippet A</div></div>"#;
    let rows = parse_ddg_html(html, 5);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].url, "https://example.com/a");
    assert!(rows[0].title.contains("Alpha"));
    assert_eq!(rows[0].snippet, "Snippet A");
}

#[test]
fn test_tools_extra_and_nudge() {
    let tools = json!([{"type": "function", "function": {"name": "read_file"}}]);
    assert!(request_has_tools(Some(&tools)));
    let tc = json!("auto");
    let extra = build_tools_extra(Some(&tools), Some(&tc), None);
    assert_eq!(extra["tools"], tools);
    assert_eq!(extra["tool_choice"], json!("auto"));

    let msgs = vec![json!({"role": "user", "content": "what does this code do?"})];
    let out = apply_ide_tool_nudge(&msgs, Some(&tools), Some(true));
    assert_eq!(out[0]["role"], json!("system"));
    assert!(out[0]["content"]
        .as_str()
        .unwrap()
        .contains("IDE coding agent"));
    let untouched = apply_ide_tool_nudge(&msgs, Some(&tools), Some(false));
    assert_eq!(untouched, msgs);
    assert!(should_bypass_memory_merge(Some(&tools), &msgs));
}

#[test]
fn test_passthrough_extra_keeps_unknown_fields() {
    let body = json!({
        "model": "switchyard",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
        "max_tokens": 128,
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
        "tool_choice": "auto",
        "stream_options": {"include_usage": true},
        "reasoning_effort": "low",
        "metadata": {"task": "1"},
    });
    let extra = build_passthrough_extra(&body);
    assert!(extra.get("model").is_none());
    assert!(extra.get("messages").is_none());
    assert!(extra.get("stream").is_none());
    assert_eq!(extra["tools"][0]["function"]["name"], json!("read_file"));
    assert_eq!(extra["stream_options"]["include_usage"], json!(true));
    assert_eq!(extra["reasoning_effort"], json!("low"));
    assert_eq!(extra["metadata"], json!({"task": "1"}));
    assert_eq!(
        resolve_max_tokens(&json!({"max_completion_tokens": 50}), 10, Some(1000)),
        50
    );
}

#[test]
fn test_preserve_transcript_and_continuation() {
    let image = vec![json!({
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "x"}}]
    })];
    assert!(should_preserve_client_transcript(None, &image));
    let loop_msgs = vec![
        json!({"role": "user", "content": "fix it"}),
        json!({"role": "assistant", "tool_calls": [{"id": "c1"}]}),
        json!({"role": "tool", "tool_call_id": "c1", "content": "ok"}),
    ];
    assert!(is_tool_continuation(&loop_msgs));
    let sid1 = stable_session_id(&HashMap::new(), &json!({}), &loop_msgs);
    let mut more = loop_msgs.clone();
    more.push(json!({"role": "assistant", "content": "done"}));
    let sid2 = stable_session_id(&HashMap::new(), &json!({}), &more);
    assert_eq!(sid1, sid2);
    assert_ne!(sid1, "default");
}

#[test]
fn test_context_length_error_mapping() {
    let (payload, status) = openai_error_from_upstream(
        "This model's maximum context length is 32768 tokens. However, you requested 40000 tokens.",
        Some(400),
        None,
        500,
    );
    assert_eq!(status, 400);
    assert_eq!(payload["error"]["code"], json!("context_length_exceeded"));
    assert_eq!(payload["error"]["type"], json!("invalid_request_error"));
}

#[test]
fn test_sse_tool_calls() {
    let completion = json!({
        "id": "chatcmpl-1",
        "created": 1,
        "model": "m",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning_content": "step 1",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"a.py\"}"},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    });
    let chunks = completion_to_sse_chunks(&completion, Some("m"), 48);
    let joined = chunks.join("");
    assert!(joined.contains("tool_calls"));
    assert!(joined.contains("read_file"));
    assert!(joined.contains("reasoning_content"));
    assert!(joined.contains("\"prompt_tokens\":10") || joined.contains("\"prompt_tokens\": 10"));
    assert!(joined.ends_with("data: [DONE]\n\n"));
}

#[test]
fn test_thinking_round_trip_and_caller_stays_off_openai() {
    let body = json!({
        "model": "claude-opus-4-6",
        "system": [
            {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}},
        ],
        "messages": [
            {"role": "user", "content": "inspect server.py"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "read it first", "signature": "sig_1"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "mcp__fs__read",
                        "input": {"path": "server.py"},
                        "caller": {"type": "direct"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "print(1)"},
                ],
            },
        ],
        "tools": [{
            "name": "mcp__fs__read",
            "description": "read",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            "cache_control": {"type": "ephemeral"},
        }],
    });
    let openai_body = anthropic_to_openai(&body);
    let dumped = serde_json::to_string(&openai_body).unwrap();
    assert!(!dumped.contains("cache_control"));
    assert!(!dumped.contains("signature"));
    assert!(!dumped.contains("caller"));
    let assistant = &openai_body["messages"][2];
    assert_eq!(assistant["reasoning_content"], json!("read it first"));
    assert_eq!(
        assistant["tool_calls"][0]["function"]["name"],
        json!("mcp__fs__read")
    );
    assert_eq!(openai_body["messages"][3]["role"], json!("tool"));
    assert_eq!(claude_tier_hint(Some("claude-opus-4-6")), Some("strong"));
}

#[test]
fn test_truncated_tool_json_keeps_siblings() {
    assert_eq!(
        repair_tool_input(Some(&json!("{\"path\": \"a.py\""))),
        Some(json!({"path": "a.py"}))
    );
    assert_eq!(repair_tool_input(Some(&json!("{not json"))), None);

    let out = openai_to_anthropic(
        &json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": "<think>check two files</think>Done.",
                    "tool_calls": [
                        {
                            "id": "call_ok",
                            "function": {"name": "read_file", "arguments": "{\"path\": \"a.py\"}"},
                        },
                        {
                            "id": "call_cut",
                            "function": {"name": "write", "arguments": "{\"path\": \"b.py\", \"text\": \"hi"},
                        },
                        {
                            "id": "call_dead",
                            "function": {"name": "bash", "arguments": "{not json"},
                        },
                    ],
                },
            }],
        }),
        "claude-sonnet",
    );
    let content = out["content"].as_array().unwrap();
    let uses: Vec<&Value> = content
        .iter()
        .filter(|b| b["type"] == json!("tool_use"))
        .collect();
    assert_eq!(uses.len(), 2);
    assert_eq!(uses[0]["name"], json!("read_file"));
    assert_eq!(uses[1]["name"], json!("write"));
    assert_eq!(uses[1]["input"], json!({"path": "b.py", "text": "hi"}));
    assert_eq!(out["stop_reason"], json!("tool_use"));

    let stripped = strip_reasoning_content(&json!({
        "messages": [{"role": "assistant", "content": "x", "reasoning_content": "y"}]
    }));
    assert!(stripped["messages"][0].get("reasoning_content").is_none());
}

#[test]
fn test_stream_recovers_cut_json_without_relabeling() {
    let lines = vec![
        format!(
            "data: {}",
            json!({
                "choices": [{
                    "delta": {"reasoning_content": "plan"},
                    "finish_reason": Value::Null,
                }]
            })
        ),
        format!(
            "data: {}",
            json!({
                "choices": [{
                    "delta": {"tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "write", "arguments": "{\"path\": \"a.py\""},
                    }]},
                    "finish_reason": "tool_calls",
                }]
            })
        ),
        "data: [DONE]".to_string(),
    ];
    let blob = iter_openai_sse_as_anthropic(&lines, "claude-sonnet", None).join("");
    assert!(blob.contains("thinking_delta"));
    assert!(
        blob.contains("\"caller\":{\"type\":\"direct\"}")
            || blob.contains("\"caller\": {\"type\": \"direct\"}")
    );
    assert!(blob.contains("a.py"));
    assert!(
        blob.contains("\"stop_reason\":\"tool_use\"")
            || blob.contains("\"stop_reason\": \"tool_use\"")
    );
}

#[test]
fn test_auth_and_chat_history_sqlite() {
    let tmp = std::env::temp_dir().join(format!("ms_test_{}", uuid::Uuid::new_v4()));
    let auth_db = tmp.join("users.db").to_string_lossy().into_owned();
    let chat_db = tmp.join("chat.db").to_string_lossy().into_owned();

    let auth = AuthManager::new(Some(&auth_db));
    assert!(auth.register_user("alice", "secret123"));
    assert!(!auth.register_user("alice", "other"));
    let tok = auth.login("alice", "secret123").expect("login token");
    let uid = auth.validate_session(&tok).expect("valid session");
    assert_eq!(auth.get_username(uid).as_deref(), Some("alice"));
    auth.logout(&tok);
    assert_eq!(auth.validate_session(&tok), None);

    let history = ChatHistory::new(Some(&chat_db));
    let cid = history.create_conversation(Some(uid), "hf-ide-assistant");
    history.add_message(cid, "user", "hello", None);
    history.add_message(cid, "assistant", "world", Some(&json!({"tokens": 2})));
    let msgs = history.get_conversation_history(cid);
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["content"], json!("hello"));
    assert_eq!(msgs[1]["content"], json!("world"));

    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn test_cache_and_memory() {
    let cache = ResponseCache::new(4, 60, true);
    let key = ResponseCache::make_key(
        "m",
        &[json!({"role": "user", "content": "hi"})],
        0.3,
        128,
        None,
    );
    assert!(cache.get(&key).is_none());
    cache.set(&key, json!({"ok": true}));
    assert_eq!(cache.get(&key), Some(json!({"ok": true})));

    let mem = ConversationMemory::new(1024, 8);
    mem.append("s1", "system", "sys");
    mem.append("s1", "user", "u1");
    mem.append("s1", "assistant", "a1");
    assert_eq!(mem.get("s1").len(), 3);
}

#[tokio::test]
async fn test_ide_axum_router_endpoints() {
    std::env::set_var("AUTO_DEPLOY_MODEL", "false");
    std::env::set_var("BACKEND_URLS", "http://127.0.0.1:8000/v1/chat/completions");
    let cfg = model_serving_shared::ide_server::IdeServerConfig {
        family: "hf".to_string(),
        app_name: "hf-ide-assistant".to_string(),
        default_port: 8081,
        default_model: "Qwen/Qwen2.5-Coder-32B-Instruct".to_string(),
        default_api_key: "test-key".to_string(),
    };
    let state = model_serving_shared::ide_server::build_ide_state(cfg);
    let app = model_serving_shared::ide_server::build_ide_router(state);

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let resp = app
        .clone()
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages/count_tokens")
                .header("x-api-key", "test-key")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_string(&json!({
                        "model": "claude-sonnet-4-6",
                        "messages": [{"role": "user", "content": "Hello world!"}]
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

