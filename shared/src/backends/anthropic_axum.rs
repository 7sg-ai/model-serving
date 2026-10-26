//! Axum helpers for POST /v1/messages and /v1/messages/count_tokens.

use axum::http::HeaderMap;
use serde_json::Value;

pub fn anthropic_api_key(headers: &HeaderMap, expected: &str) -> bool {
    if expected.is_empty() {
        return false;
    }
    let token = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .or_else(|| {
            headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|auth| auth.strip_prefix("Bearer ").map(|s| s.to_string()))
        })
        .unwrap_or_default();
    token == expected
}

pub fn apply_model_hint<F>(body: &mut Value, resolve_hint: F) -> String
where
    F: FnOnce(&str, bool) -> Option<String>,
{
    let requested = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let has_tools = body
        .get("tools")
        .and_then(|v| v.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    if let Some(resolved) = resolve_hint(&requested, has_tools) {
        if !resolved.is_empty() && resolved != requested {
            body["model"] = Value::String(resolved);
        }
    }
    requested
}

pub fn has_reasoning(body: &Value) -> bool {
    body.get("messages")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter().any(|m| {
                m.get("reasoning_content")
                    .and_then(|v| v.as_str())
                    .map(|s| !s.is_empty())
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

pub fn reasoning_rejected(status_code: Option<u16>, err_text: &str) -> bool {
    let blob = err_text.to_ascii_lowercase();
    let rejected = blob.contains("reasoning_content") || blob.contains("reasoning content");
    match status_code {
        Some(code) => code == 400 && rejected,
        None => rejected,
    }
}
