//! OpenAI chat.completion -> Anthropic Messages response conversion.

use crate::backends::anthropic_convert::block_text;
use chrono::Utc;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Value};

static THINK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)^<think>\s*(.*?)\s*</think>\s*").unwrap());

pub fn direct_caller() -> Value {
    json!({"type": "direct"})
}

pub fn parse_tool_input(raw: Option<&Value>) -> (Option<Value>, bool) {
    let Some(r) = raw else {
        return (Some(json!({})), false);
    };
    match r {
        Value::Null => (Some(json!({})), false),
        Value::Object(_) => (Some(r.clone()), false),
        Value::String(s) => {
            if s.is_empty() {
                return (Some(json!({})), false);
            }
            match serde_json::from_str::<Value>(s) {
                Ok(Value::Object(m)) => (Some(Value::Object(m)), false),
                Ok(other) => (Some(json!({"value": other})), false),
                Err(_) => (None, true),
            }
        }
        other => (Some(json!({"value": other})), false),
    }
}

pub fn repair_tool_input(raw: Option<&Value>) -> Option<Value> {
    let (parsed, bad) = parse_tool_input(raw);
    if !bad {
        if let Some(v) = parsed {
            if v.is_object() {
                return Some(v);
            }
        }
    }
    let s = raw.and_then(|v| v.as_str())?.trim();
    if s.is_empty() {
        return None;
    }
    let mut text = s.to_string();
    let mut in_string = false;
    let mut escape = false;
    let mut stack: Vec<char> = Vec::new();
    for ch in text.chars() {
        if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
        } else if ch == '{' || ch == '[' {
            stack.push(ch);
        } else if (ch == '}' && stack.last() == Some(&'{'))
            || (ch == ']' && stack.last() == Some(&'['))
        {
            stack.pop();
        }
    }
    if in_string {
        text.push('"');
    }
    text = text.trim_end().trim_end_matches(',').to_string();
    for opener in stack.into_iter().rev() {
        text.push(if opener == '{' { '}' } else { ']' });
    }
    let parsed: Value = serde_json::from_str(&text).ok()?;
    if parsed.is_object() {
        Some(parsed)
    } else {
        Some(json!({"value": parsed}))
    }
}

pub fn tool_use_blocks(tool_calls: Option<&Value>) -> (Vec<Value>, Vec<String>) {
    let mut blocks = Vec::new();
    let mut dropped = Vec::new();
    let Some(arr) = tool_calls.and_then(|v| v.as_array()) else {
        return (blocks, dropped);
    };
    for call in arr {
        let Some(obj) = call.as_object() else {
            continue;
        };
        let fn_obj = obj.get("function");
        let args = fn_obj.and_then(|f| f.get("arguments"));
        let default_empty = json!("{}");
        let parsed = repair_tool_input(Some(args.unwrap_or(&default_empty)));
        let name = fn_obj
            .and_then(|f| f.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let Some(input) = parsed else {
            dropped.push(if name.is_empty() {
                "tool".to_string()
            } else {
                name.to_string()
            });
            continue;
        };
        let id = obj
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("toolu_{}", blocks.len()));
        blocks.push(json!({
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": input,
            "caller": direct_caller(),
        }));
    }
    (blocks, dropped)
}

pub fn split_thinking(text: &str) -> (String, String) {
    if text.is_empty() {
        return (String::new(), String::new());
    }
    if let Some(m) = THINK_RE.captures(text) {
        let full = m.get(0).unwrap();
        let inner = m.get(1).map(|g| g.as_str().trim()).unwrap_or("");
        return (inner.to_string(), text[full.end()..].to_string());
    }
    (String::new(), text.to_string())
}

pub fn openai_to_anthropic(result: &Value, requested_model: &str) -> Value {
    let choice = result
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let msg = choice.get("message").cloned().unwrap_or_else(|| json!({}));
    let mut blocks = Vec::new();

    let reasoning = msg
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let raw_text = match msg.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(v) if !v.is_null() => block_text(v),
        _ => String::new(),
    };
    let (inline_think, text) = split_thinking(&raw_text);
    let thought = if !reasoning.is_empty() {
        reasoning
    } else {
        inline_think
    };
    if !thought.is_empty() {
        blocks.push(json!({"type": "thinking", "thinking": thought}));
    }
    if !text.is_empty() {
        blocks.push(json!({"type": "text", "text": text}));
    }

    let (tool_blocks, dropped) = tool_use_blocks(msg.get("tool_calls"));
    let has_tool_blocks = !tool_blocks.is_empty();
    blocks.extend(tool_blocks);

    let mut finish = choice
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("stop");
    if has_tool_blocks && finish == "stop" {
        finish = "tool_calls";
    }
    let stop_reason = if has_tool_blocks {
        "tool_use"
    } else if !dropped.is_empty() && finish == "length" {
        "max_tokens"
    } else if !dropped.is_empty() {
        let names = dropped.join(", ");
        blocks.push(json!({
            "type": "text",
            "text": format!("Dropped an incomplete tool call ({names}). No arguments were invented."),
        }));
        "end_turn"
    } else {
        match finish {
            "stop" | "content_filter" => "end_turn",
            "tool_calls" => "tool_use",
            "length" => "max_tokens",
            _ => "end_turn",
        }
    };

    if blocks.is_empty() {
        blocks.push(json!({"type": "text", "text": ""}));
    }

    let usage = result.get("usage");
    let input_tokens = usage
        .and_then(|u| u.get("prompt_tokens").or_else(|| u.get("input_tokens")))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output_tokens = usage
        .and_then(|u| u.get("completion_tokens").or_else(|| u.get("output_tokens")))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let mut body = json!({
        "id": result.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| format!("msg_{}", Utc::now().timestamp())),
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    });
    if let Some(sy) = result.get("switchyard") {
        if sy.is_object() {
            body["switchyard"] = sy.clone();
        }
    }
    body
}

pub fn estimate_input_tokens(openai_body: &Value) -> usize {
    let blob = serde_json::to_string(&json!({
        "messages": openai_body.get("messages").cloned().unwrap_or_else(|| json!([])),
        "tools": openai_body.get("tools").cloned().unwrap_or_else(|| json!([])),
    }))
    .unwrap_or_default();
    blob.len().div_ceil(4).max(1)
}

pub fn anthropic_error(message: &str, error_type: &str, status: u16) -> (Value, u16) {
    (
        json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            }
        }),
        status,
    )
}
