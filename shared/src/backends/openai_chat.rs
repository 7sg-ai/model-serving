//! OpenAI-compatible chat/completions HTTP client.

use crate::split_csv;
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::env;
use std::time::Duration;

pub fn normalize_chat_url(url: &str) -> String {
    let u = url.trim().trim_end_matches('/');
    if u.is_empty() {
        return String::new();
    }
    if u.ends_with("/chat/completions") {
        return u.to_string();
    }
    if u.ends_with("/v1") {
        return format!("{u}/chat/completions");
    }
    if u.contains("/v1/") {
        if u.ends_with("chat/completions") {
            return u.to_string();
        }
        if u.ends_with("completions") && !u.ends_with("chat/completions") {
            return u.to_string();
        }
        return format!("{u}/chat/completions");
    }
    format!("{u}/v1/chat/completions")
}

pub fn resolve_vllm_backend_urls(existing: Option<&[String]>) -> Vec<String> {
    let mut ordered = Vec::new();
    let mut seen = HashSet::new();

    let mut add = |raw: &str| {
        for part in split_csv(raw) {
            let norm = normalize_chat_url(&part);
            if !norm.is_empty() && seen.insert(norm.clone()) {
                ordered.push(norm);
            }
        }
    };

    if let Some(ex) = existing {
        for u in ex {
            add(u);
        }
    }

    let coord = env::var("MODEL_COORDINATOR_URL")
        .or_else(|_| env::var("SHARD_COORDINATOR_URL"))
        .unwrap_or_default();
    add(&coord);

    let backends = env::var("BACKEND_URLS")
        .or_else(|_| env::var("NIM_API_URLS"))
        .or_else(|_| env::var("MODEL_BACKEND_URLS"))
        .or_else(|_| env::var("MODEL_REPLICA_URLS"))
        .unwrap_or_default();
    add(&backends);

    let vllm = env::var("VLLM_API_URLS")
        .or_else(|_| env::var("VLLM_API_URL"))
        .unwrap_or_default();
    add(&vllm);

    let sglang = env::var("SGLANG_API_URLS")
        .or_else(|_| env::var("SGLANG_API_URL"))
        .unwrap_or_default();
    add(&sglang);

    let nim = env::var("NIM_API_URL").unwrap_or_default();
    add(&nim);

    ordered
}

pub fn require_backend_urls(existing: Option<&[String]>, what: &str) -> Result<Vec<String>> {
    let resolved = resolve_vllm_backend_urls(existing);
    if resolved.is_empty() {
        return Err(anyhow!(
            "No {what} backend configured. Set one of: BACKEND_URLS, VLLM_API_URL, VLLM_API_URLS, SGLANG_API_URL, or MODEL_COORDINATOR_URL (OpenAI-compatible .../v1/chat/completions). Start a vLLM, SGLang, or NIM stack (see deploy/)."
        ));
    }
    Ok(resolved)
}

pub fn extract_assistant_text(result: &Value) -> String {
    result
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .and_then(|choice0| {
            choice0
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .or_else(|| choice0.get("text").and_then(|t| t.as_str()))
        })
        .unwrap_or("")
        .to_string()
}

#[allow(clippy::too_many_arguments)]
pub fn chat_completions(
    url: &str,
    messages: &[Value],
    model: &str,
    temperature: f64,
    max_tokens: usize,
    api_key: &str,
    timeout_secs: f64,
    extra: Option<&Value>,
) -> Result<Value> {
    let endpoint = normalize_chat_url(url);
    if endpoint.is_empty() {
        return Err(anyhow!("chat_completions: empty url"));
    }

    let mut payload = json!({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": false,
    });
    if let (Some(base), Some(ex)) = (payload.as_object_mut(), extra.and_then(|v| v.as_object())) {
        for (k, v) in ex {
            if k != "stream" {
                base.insert(k.clone(), v.clone());
            }
        }
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs_f64(timeout_secs.max(1.0)))
        .build()?;
    let mut req = client
        .post(&endpoint)
        .header("Content-Type", "application/json")
        .json(&payload);

    let key = if !api_key.trim().is_empty() {
        api_key.trim().to_string()
    } else {
        env::var("VLLM_API_KEY")
            .or_else(|_| env::var("OPENAI_API_KEY"))
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    if !key.is_empty() {
        req = req.header("Authorization", format!("Bearer {key}"));
    }

    let resp = req.send()?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(anyhow!("HTTP {}: {}", status.as_u16(), text));
    }
    let data: Value = serde_json::from_str(&text)
        .map_err(|e| anyhow!("Invalid JSON from {endpoint}: {e}"))?;
    if !data.is_object() {
        return Err(anyhow!("Unexpected response type from {endpoint}"));
    }
    Ok(data)
}

pub fn usage_tuple(data: &Value) -> (usize, usize) {
    let usage = data.get("usage");
    let prompt = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let completion = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    (prompt, completion)
}
