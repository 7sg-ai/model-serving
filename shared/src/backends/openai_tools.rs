//! OpenAI-compatible tools / tool_calls helpers for IDE assistant proxies.

use crate::env_bool;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub const IDE_TOOL_NUDGE_TEXT: &str = "You are connected to an IDE coding agent with tools (read/list/search/edit files, run commands, etc.). When the user refers to code in the workspace, call the provided tools to inspect it yourself. Do not ask the user to paste code that tools can read. After tool results arrive, continue the task.";

const OWNED_BODY_FIELDS: &[&str] = &[
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "stream",
    "user",
    "conversation_id",
    "session_id",
];

static CONTEXT_ERROR_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)context length|maximum context|context window|context_length|too many tokens|token limit|max[_ ]tokens|prompt is too long|exceeds the (?:model'?s )?maximum|input is too long|requested \d+ tokens|reduce the length",
    )
    .unwrap()
});

pub const TOOL_CAPABLE_HINTS: &[&str] = &[
    "qwen3",
    "qwen2.5-coder",
    "qwen2.5-72b",
    "qwen2.5-32b",
    "qwen2.5-14b",
    "qwen2.5-7b",
    "llama-3.3",
    "llama-3.1-70b",
    "llama-3.1-405b",
    "mistral-large",
    "mixtral-8x22b",
    "deepseek-v3",
    "deepseek-r1",
    "deepseek-coder",
    "kimi-k2",
    "kimi-k3",
    "gpt-oss",
    "devstral",
    "codestral",
    "firefunction",
    "nemotron-4-340b",
    "command-r",
];

pub const REASONING_HINTS: &[&str] = &[
    "deepseek-r1",
    "qwen3",
    "qwq",
    "reasoning",
    "think",
    "kimi-k2",
    "kimi-k3",
    "gpt-oss",
];

pub fn request_has_tools(tools: Option<&Value>) -> bool {
    tools
        .and_then(|t| t.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false)
}

pub fn messages_have_tool_protocol(messages: &[Value]) -> bool {
    for m in messages {
        let Some(obj) = m.as_object() else {
            continue;
        };
        let role = obj.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "tool" {
            return true;
        }
        if obj
            .get("tool_calls")
            .and_then(|tc| tc.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false)
        {
            return true;
        }
        if role == "assistant" && obj.get("function_call").map(|v| !v.is_null()).unwrap_or(false) {
            return true;
        }
    }
    false
}

pub fn messages_have_rich_content(messages: &[Value]) -> bool {
    messages
        .iter()
        .any(|m| m.get("content").map(|c| c.is_array()).unwrap_or(false))
}

pub fn should_preserve_client_transcript(tools: Option<&Value>, messages: &[Value]) -> bool {
    request_has_tools(tools)
        || messages_have_tool_protocol(messages)
        || messages_have_rich_content(messages)
}

pub fn should_bypass_memory_merge(tools: Option<&Value>, messages: &[Value]) -> bool {
    should_preserve_client_transcript(tools, messages)
}

pub fn is_tool_continuation(messages: &[Value]) -> bool {
    let Some(last) = messages.last() else {
        return false;
    };
    if last.get("role").and_then(|r| r.as_str()) == Some("tool") {
        return true;
    }
    for m in messages.iter().rev().take(4) {
        if m.get("role").and_then(|r| r.as_str()) == Some("tool") {
            return true;
        }
        if m.get("tool_calls")
            .and_then(|tc| tc.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false)
        {
            return true;
        }
    }
    false
}

pub fn apply_ide_tool_nudge(
    messages: &[Value],
    tools: Option<&Value>,
    enabled: Option<bool>,
) -> Vec<Value> {
    let active = enabled.unwrap_or_else(|| env_bool("IDE_TOOL_NUDGE", true));
    if !active || !request_has_tools(tools) {
        return messages.to_vec();
    }
    for m in messages {
        if m.get("role").and_then(|r| r.as_str()) == Some("system") {
            let content = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
            if content.contains("IDE coding agent with tools") {
                return messages.to_vec();
            }
        }
    }
    let mut out = Vec::with_capacity(messages.len() + 1);
    out.push(json!({
        "role": "system",
        "content": IDE_TOOL_NUDGE_TEXT,
    }));
    out.extend_from_slice(messages);
    out
}

pub fn build_tools_extra(
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
    parallel_tool_calls: Option<&Value>,
) -> Value {
    let mut out = serde_json::Map::new();
    if request_has_tools(tools) {
        if let Some(t) = tools {
            out.insert("tools".to_string(), t.clone());
        }
        if let Some(tc) = tool_choice {
            if !tc.is_null() {
                out.insert("tool_choice".to_string(), tc.clone());
            }
        }
        if let Some(ptc) = parallel_tool_calls {
            if !ptc.is_null() {
                out.insert("parallel_tool_calls".to_string(), ptc.clone());
            }
        }
    }
    Value::Object(out)
}

pub fn build_passthrough_extra(body: &Value) -> Value {
    let mut out = serde_json::Map::new();
    let Some(obj) = body.as_object() else {
        return Value::Object(out);
    };
    for (k, v) in obj {
        if OWNED_BODY_FIELDS.contains(&k.as_str()) || v.is_null() {
            continue;
        }
        if k == "tools" && !request_has_tools(Some(v)) {
            continue;
        }
        if (k == "tool_choice" || k == "parallel_tool_calls")
            && !request_has_tools(obj.get("tools"))
        {
            continue;
        }
        out.insert(k.clone(), v.clone());
    }
    Value::Object(out)
}

pub fn resolve_max_tokens(body: &Value, default: usize, cap: Option<usize>) -> usize {
    let raw = body
        .get("max_tokens")
        .or_else(|| body.get("max_completion_tokens"))
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(default);
    let mut value = raw.max(1);
    if let Some(c) = cap {
        value = value.min(c.max(1));
    }
    value
}

pub fn stream_options_want_usage(extra: Option<&Value>) -> bool {
    extra
        .and_then(|e| e.get("stream_options"))
        .and_then(|so| so.get("include_usage"))
        .and_then(|iu| iu.as_bool())
        .unwrap_or(false)
}

pub fn extract_tool_calls(result: &Value) -> Vec<Value> {
    result
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .and_then(|c0| c0.get("message"))
        .and_then(|m| m.get("tool_calls"))
        .and_then(|tc| tc.as_array())
        .cloned()
        .unwrap_or_default()
}

pub fn response_has_tool_calls(result: &Value) -> bool {
    if !extract_tool_calls(result).is_empty() {
        return true;
    }
    result
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .and_then(|c0| c0.get("finish_reason"))
        .and_then(|f| f.as_str())
        == Some("tool_calls")
}

pub fn should_skip_response_cache(tools: Option<&Value>, messages: &[Value]) -> bool {
    request_has_tools(tools) || messages_have_tool_protocol(messages)
}

pub fn message_content_for_memory(message: &Value) -> String {
    let content = message.get("content");
    let text = match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => {
            let mut joined = Vec::new();
            for p in parts {
                if let Some(s) = p.as_str() {
                    joined.push(s.to_string());
                } else if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                    joined.push(t.to_string());
                }
            }
            joined.join("\n")
        }
        _ => String::new(),
    };
    let tcs = message
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if !tcs.is_empty() {
        let mut names = Vec::new();
        for tc in &tcs {
            if let Some(n) = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
            {
                names.push(n.to_string());
            }
        }
        let summary = format!("[tool_calls: {}]", names.join(", "));
        return if !text.is_empty() {
            format!("{text}\n{summary}")
        } else {
            summary
        };
    }
    text
}

pub fn stable_session_id(
    headers: &HashMap<String, String>,
    data: &Value,
    messages: &[Value],
) -> String {
    for h in [
        "X-Conversation-ID",
        "x-conversation-id",
        "X-Session-ID",
        "x-session-id",
        "X-Task-ID",
        "x-task-id",
        "X-Cline-Task-Id",
        "x-cline-task-id",
    ] {
        if let Some(v) = headers.get(h) {
            if !v.trim().is_empty() {
                return v.trim().to_string();
            }
        }
    }
    for k in ["conversation_id", "session_id", "user"] {
        if let Some(v) = data.get(k).and_then(|v| v.as_str()) {
            if !v.trim().is_empty() {
                return v.trim().to_string();
            }
        }
    }
    if let Some(v) = data
        .get("metadata")
        .and_then(|m| {
            m.get("conversation_id")
                .or_else(|| m.get("session_id"))
                .or_else(|| m.get("task_id"))
                .or_else(|| m.get("user_id"))
        })
        .and_then(|v| v.as_str())
    {
        if !v.trim().is_empty() {
            return v.trim().to_string();
        }
    }
    let mut anchor_parts = Vec::new();
    for m in messages {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "system" || role == "user" {
            let content = m
                .get("content")
                .map(|c| {
                    if let Some(s) = c.as_str() {
                        s.to_string()
                    } else {
                        serde_json::to_string(c).unwrap_or_default()
                    }
                })
                .unwrap_or_default();
            anchor_parts.push(format!(
                "{role}:{}",
                content.chars().take(2000).collect::<String>()
            ));
            if role == "user" {
                break;
            }
        }
    }
    if !anchor_parts.is_empty() {
        let mut hasher = Sha256::new();
        hasher.update(anchor_parts.join("|").as_bytes());
        let digest = format!("{:x}", hasher.finalize());
        return format!("Task-{}", &digest[..16]);
    }
    "default".to_string()
}

pub fn is_context_length_error(status_code: Option<u16>, body: &str) -> bool {
    if let Some(code) = status_code {
        if !matches!(code, 400 | 413 | 422) {
            return false;
        }
    }
    if body.is_empty() {
        return false;
    }
    CONTEXT_ERROR_RE.is_match(body)
}

pub fn openai_error_from_upstream(
    err_str: &str,
    status_code: Option<u16>,
    body: Option<&str>,
    fallback_status: u16,
) -> (Value, u16) {
    let raw_body = body.unwrap_or(err_str);
    let mut message = raw_body.trim().to_string();
    let mut parsed_err_type: Option<String> = None;
    let mut parsed_code: Option<String> = None;

    if let Ok(val) = serde_json::from_str::<Value>(raw_body) {
        if let Some(err_obj) = val.get("error").and_then(|e| e.as_object()) {
            if let Some(m) = err_obj.get("message").and_then(|v| v.as_str()) {
                message = m.to_string();
            }
            if let Some(t) = err_obj.get("type").and_then(|v| v.as_str()) {
                parsed_err_type = Some(t.to_string());
            }
            if let Some(c) = err_obj.get("code").and_then(|v| v.as_str()) {
                parsed_code = Some(c.to_string());
            }
        } else if let Some(m) = val
            .get("message")
            .or_else(|| val.get("detail"))
            .and_then(|v| v.as_str())
        {
            message = m.to_string();
        }
    }
    if message.is_empty() {
        message = err_str.to_string();
    }

    if is_context_length_error(status_code, &message)
        || is_context_length_error(status_code, raw_body)
    {
        return (
            json!({
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "context_length_exceeded",
                }
            }),
            400,
        );
    }

    let http_status = match status_code {
        Some(s) if (400..600).contains(&s) => s,
        _ => fallback_status,
    };
    let err_type = parsed_err_type.unwrap_or_else(|| {
        if http_status < 500 {
            "invalid_request_error".to_string()
        } else {
            "server_error".to_string()
        }
    });
    let code = parsed_code.unwrap_or_else(|| Value::Null.to_string());
    let code_val = if code == "null" {
        Value::Null
    } else {
        json!(code)
    };
    (
        json!({
            "error": {
                "message": if message.is_empty() { "Upstream chat completion failed.".to_string() } else { message },
                "type": err_type,
                "code": code_val,
            }
        }),
        http_status,
    )
}

pub fn looks_tool_capable(model_id: &str) -> bool {
    let mid = model_id.to_ascii_lowercase();
    if mid.is_empty() {
        return false;
    }
    TOOL_CAPABLE_HINTS.iter().any(|h| mid.contains(h))
}

pub fn looks_reasoning_capable(model_id: &str) -> bool {
    let mid = model_id.to_ascii_lowercase();
    if mid.is_empty() {
        return false;
    }
    REASONING_HINTS.iter().any(|h| mid.contains(h))
}

pub fn model_capabilities(
    model_id: &str,
    tools: Option<bool>,
    reasoning: Option<bool>,
    route: bool,
) -> Value {
    json!({
        "tools": if route { true } else { tools.unwrap_or_else(|| looks_tool_capable(model_id)) },
        "reasoning": reasoning.unwrap_or_else(|| looks_reasoning_capable(model_id)),
    })
}

pub fn annotate_model_entry(mut entry: Value, route: bool) -> Value {
    let model_id = entry
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let is_route = route
        || entry.get("root").and_then(|v| v.as_str()) == Some("switchyard-route")
        || entry
            .get("strategy")
            .and_then(|v| v.as_str())
            .map(|s| !s.is_empty())
            .unwrap_or(false);
    let mut caps = model_capabilities(&model_id, None, None, is_route);
    if let (Some(caps_obj), Some(existing_obj)) = (
        caps.as_object_mut(),
        entry.get("capabilities").and_then(|v| v.as_object()),
    ) {
        for (k, v) in existing_obj {
            caps_obj.insert(k.clone(), json!(v.as_bool().unwrap_or(false)));
        }
    }
    if let Some(obj) = entry.as_object_mut() {
        obj.insert("capabilities".to_string(), caps);
        if !obj.contains_key("context_window") {
            if let Some(ecw) = obj.get("effective_context_window").cloned() {
                obj.insert("context_window".to_string(), ecw);
            }
        }
    }
    entry
}

pub fn warn_if_not_tool_capable(model_id: &str, app_name: &str) {
    if looks_tool_capable(model_id) {
        return;
    }
    println!(
        "[ide-tools] WARNING: model '{model_id}' is not in the known tool-capable list for {app_name}. IDE agents (Cline/Cursor) may ignore tool schemas and ask the user to paste code instead. Consider a function-calling tuned model (e.g. qwen2.5-coder / qwen3 / llama-3.3 / kimi / deepseek)."
    );
}
