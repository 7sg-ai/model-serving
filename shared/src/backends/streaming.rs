//! OpenAI-compatible SSE (Server-Sent Events) streaming helpers.

use crate::backends::openai_chat::normalize_chat_url;
use anyhow::{anyhow, Result};
use chrono::Utc;
use serde_json::{json, Value};
use std::time::Duration;

pub const SSE_HEADERS: &[(&str, &str)] = &[
    ("Cache-Control", "no-cache"),
    ("Connection", "keep-alive"),
    ("X-Accel-Buffering", "no"),
];

pub fn extract_stream_delta_text(payload: &Value) -> String {
    let Some(choice0) = payload
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
    else {
        return String::new();
    };
    if let Some(content) = choice0
        .get("delta")
        .and_then(|d| d.get("content"))
        .and_then(|c| c.as_str())
    {
        if !content.is_empty() {
            return content.to_string();
        }
    }
    if let Some(content) = choice0
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
    {
        if !content.is_empty() {
            return content.to_string();
        }
    }
    choice0
        .get("text")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_string()
}

pub fn parse_sse_data_line(line: &str) -> Option<Value> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with(':') || !trimmed.starts_with("data:") {
        return None;
    }
    let data = trimmed[5..].trim();
    if data.is_empty() || data == "[DONE]" {
        return None;
    }
    let obj: Value = serde_json::from_str(data).ok()?;
    if obj.is_object() {
        Some(obj)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
pub fn post_chat_stream_blocking(
    url: &str,
    messages: &[Value],
    model: &str,
    temperature: f64,
    max_tokens: usize,
    api_key: &str,
    timeout_secs: f64,
    extra: Option<&Value>,
) -> Result<reqwest::blocking::Response> {
    let endpoint = normalize_chat_url(url);
    if endpoint.is_empty() {
        return Err(anyhow!("post_chat_stream: empty url"));
    }
    let mut payload = json!({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": true,
    });
    if let (Some(base), Some(ex)) = (payload.as_object_mut(), extra.and_then(|v| v.as_object())) {
        for (k, v) in ex {
            base.insert(k.clone(), v.clone());
        }
        base.insert("stream".to_string(), json!(true));
    }

    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs_f64(timeout_secs.clamp(1.0, 30.0)))
        .timeout(Duration::from_secs_f64(timeout_secs.max(1.0)))
        .build()?;
    let mut req = client
        .post(&endpoint)
        .header("Content-Type", "application/json")
        .header("Accept", "text/event-stream")
        .json(&payload);
    let key = api_key.trim();
    if !key.is_empty() {
        req = req.header("Authorization", format!("Bearer {key}"));
    }
    let resp = req.send()?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(anyhow!("HTTP {}: {}", status.as_u16(), body));
    }
    Ok(resp)
}

pub fn completion_to_sse_chunks(
    result: &Value,
    model: Option<&str>,
    chunk_chars: usize,
) -> Vec<String> {
    let resp_id = result
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("chatcmpl-{}", Utc::now().timestamp()));
    let created = result
        .get("created")
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| Utc::now().timestamp());
    let model_out = model
        .or_else(|| result.get("model").and_then(|v| v.as_str()))
        .unwrap_or("unknown");

    let choice0 = result
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let msg = choice0.get("message").cloned().unwrap_or_else(|| json!({}));
    let text = msg
        .get("content")
        .and_then(|v| v.as_str())
        .or_else(|| choice0.get("text").and_then(|v| v.as_str()))
        .unwrap_or("");
    let reasoning = msg
        .get("reasoning_content")
        .or_else(|| msg.get("reasoning"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let tool_calls = msg
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let default_finish = if !tool_calls.is_empty() {
        "tool_calls"
    } else {
        "stop"
    };
    let finish_reason = choice0
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .unwrap_or(default_finish);

    let mut out = Vec::new();
    let role_chunk = json!({
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_out,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": Value::Null}],
    });
    out.push(format!("data: {}\n\n", role_chunk));

    let step = chunk_chars.max(1);
    if !reasoning.is_empty() {
        let chars: Vec<char> = reasoning.chars().collect();
        for piece in chars.chunks(step) {
            let s: String = piece.iter().collect();
            let chunk = json!({
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_out,
                "choices": [{"index": 0, "delta": {"reasoning_content": s}, "finish_reason": Value::Null}],
            });
            out.push(format!("data: {}\n\n", chunk));
        }
    }

    if !text.is_empty() {
        let chars: Vec<char> = text.chars().collect();
        for piece in chars.chunks(step) {
            let s: String = piece.iter().collect();
            let chunk = json!({
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_out,
                "choices": [{"index": 0, "delta": {"content": s}, "finish_reason": Value::Null}],
            });
            out.push(format!("data: {}\n\n", chunk));
        }
    }

    for (i, tc) in tool_calls.iter().enumerate() {
        let Some(tc_obj) = tc.as_object() else {
            continue;
        };
        let fn_obj = tc_obj.get("function");
        let delta_tc = json!({
            "index": tc_obj.get("index").and_then(|v| v.as_u64()).unwrap_or(i as u64),
            "id": tc_obj.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| format!("call_{i}")),
            "type": tc_obj.get("type").and_then(|v| v.as_str()).unwrap_or("function"),
            "function": {
                "name": fn_obj.and_then(|f| f.get("name")).and_then(|v| v.as_str()).unwrap_or(""),
                "arguments": fn_obj.and_then(|f| f.get("arguments")).and_then(|v| v.as_str()).unwrap_or(""),
            }
        });
        let chunk = json!({
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_out,
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [delta_tc]},
                "finish_reason": Value::Null,
            }],
        });
        out.push(format!("data: {}\n\n", chunk));
    }

    let mut end_chunk = json!({
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_out,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    });
    if let Some(usage) = result.get("usage") {
        if usage.is_object() {
            end_chunk["usage"] = usage.clone();
        }
    }
    out.push(format!("data: {}\n\n", end_chunk));
    out.push("data: [DONE]\n\n".to_string());
    out
}
