//! Anthropic Messages -> OpenAI chat request conversion.

use serde_json::{json, Value};

const TIER_HINTS: &[(&str, &str)] = &[
    ("opus", "strong"),
    ("sonnet", "strong"),
    ("haiku", "weak"),
];

pub fn claude_tier_hint(model_id: Option<&str>) -> Option<&'static str> {
    let mid = model_id.unwrap_or("").to_ascii_lowercase();
    if mid.is_empty() {
        return None;
    }
    for (needle, role) in TIER_HINTS {
        if mid.contains(needle) {
            return Some(*role);
        }
    }
    None
}

pub fn block_text(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Object(obj) => {
            let kind = obj.get("type").and_then(|v| v.as_str());
            match kind {
                None | Some("text") => obj
                    .get("text")
                    .or_else(|| obj.get("content"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                Some("tool_result") => obj
                    .get("content")
                    .map(block_text)
                    .unwrap_or_default(),
                _ => String::new(),
            }
        }
        Value::Array(items) => {
            let parts: Vec<String> = items
                .iter()
                .map(block_text)
                .filter(|p| !p.is_empty())
                .collect();
            parts.join("\n")
        }
    }
}

fn tool_choice_to_openai(choice: &Value) -> Option<Value> {
    match choice {
        Value::Null => None,
        Value::String(s) => Some(Value::String(s.clone())),
        Value::Object(obj) => {
            let kind = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match kind {
                "auto" => Some(json!("auto")),
                "any" => Some(json!("required")),
                "none" => Some(json!("none")),
                "tool" => obj.get("name").and_then(|v| v.as_str()).map(|name| {
                    json!({
                        "type": "function",
                        "function": {"name": name},
                    })
                }),
                _ => None,
            }
        }
        _ => None,
    }
}

fn convert_message(msg: &Value) -> Vec<Value> {
    let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
    let content = msg.get("content").unwrap_or(&Value::Null);
    if content.is_null() {
        return vec![json!({"role": role, "content": ""})];
    }
    if let Some(s) = content.as_str() {
        return vec![json!({"role": role, "content": s})];
    }
    let Some(blocks) = content.as_array() else {
        return vec![json!({"role": role, "content": block_text(content)})];
    };

    let mut text_parts = Vec::new();
    let mut thinking_parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut tool_results = Vec::new();

    for block in blocks {
        if let Some(s) = block.as_str() {
            text_parts.push(s.to_string());
            continue;
        }
        let Some(obj) = block.as_object() else {
            continue;
        };
        let kind = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match kind {
            "text" => {
                let t = obj.get("text").and_then(|v| v.as_str()).unwrap_or("");
                text_parts.push(t.to_string());
            }
            "thinking" => {
                if let Some(thought) = obj.get("thinking").and_then(|v| v.as_str()) {
                    if !thought.trim().is_empty() {
                        thinking_parts.push(thought.to_string());
                    }
                }
            }
            "tool_use" => {
                let id = obj
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("call_{}", tool_calls.len()));
                let name = obj.get("name").and_then(|v| v.as_str()).unwrap_or("");
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments_json(obj.get("input")),
                    }
                }));
            }
            "tool_result" => {
                let call_id = obj
                    .get("tool_use_id")
                    .or_else(|| obj.get("id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("call_0");
                tool_results.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": obj.get("content").map(block_text).unwrap_or_default(),
                }));
            }
            _ => {}
        }
    }

    if role == "assistant" {
        let joined_text: String = text_parts
            .into_iter()
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        let content_val = if joined_text.is_empty() {
            Value::Null
        } else {
            json!(joined_text)
        };
        let mut assistant = json!({
            "role": "assistant",
            "content": content_val,
        });
        if !tool_calls.is_empty() {
            assistant["tool_calls"] = Value::Array(tool_calls.clone());
        }
        if !thinking_parts.is_empty() {
            assistant["reasoning_content"] = json!(thinking_parts.join("\n"));
        }
        if !joined_text.is_empty() || !tool_calls.is_empty() || !thinking_parts.is_empty() {
            return vec![assistant];
        }
        return Vec::new();
    }

    let mut out = Vec::new();
    let text: String = text_parts
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if !text.is_empty() {
        out.push(json!({"role": "user", "content": text}));
    }
    out.extend(tool_results);
    out
}

fn arguments_json(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(s)) => s.clone(),
        Some(v) if !v.is_null() => serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string()),
        _ => "{}".to_string(),
    }
}

fn convert_tools(raw_tools: Option<&Value>) -> Vec<Value> {
    let mut tools = Vec::new();
    let Some(arr) = raw_tools.and_then(|v| v.as_array()) else {
        return tools;
    };
    for tool in arr {
        let Some(name) = tool.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        if name.is_empty() {
            continue;
        }
        let desc = tool.get("description").and_then(|v| v.as_str()).unwrap_or("");
        let params = tool
            .get("input_schema")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        tools.push(json!({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": params,
            }
        }));
    }
    tools
}

pub fn anthropic_to_openai(body: &Value) -> Value {
    let mut messages = Vec::new();
    if let Some(sys) = body.get("system") {
        let text = block_text(sys);
        if !text.is_empty() {
            messages.push(json!({"role": "system", "content": text}));
        }
    }
    if let Some(arr) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in arr {
            if msg.is_object() {
                messages.extend(convert_message(msg));
            }
        }
    }

    let max_tokens = body
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096);
    let stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let mut out = json!({
        "model": body.get("model").cloned().unwrap_or(Value::Null),
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    });
    if let Some(t) = body.get("temperature") {
        if !t.is_null() {
            out["temperature"] = t.clone();
        }
    }
    if let Some(tp) = body.get("top_p") {
        if !tp.is_null() {
            out["top_p"] = tp.clone();
        }
    }
    if let Some(stop) = body.get("stop_sequences") {
        if stop.as_array().map(|a| !a.is_empty()).unwrap_or(false) {
            out["stop"] = stop.clone();
        }
    }
    let tools = convert_tools(body.get("tools"));
    if !tools.is_empty() {
        out["tools"] = Value::Array(tools);
    }
    if let Some(tc) = body.get("tool_choice").and_then(tool_choice_to_openai) {
        out["tool_choice"] = tc;
    }
    if let Some(u) = body.get("user") {
        if !u.is_null() {
            out["user"] = u.clone();
        }
    } else if let Some(uid) = body
        .get("metadata")
        .and_then(|m| m.get("user_id"))
        .filter(|v| !v.is_null())
    {
        out["user"] = uid.clone();
    }
    out
}

pub fn strip_reasoning_content(body: &Value) -> Value {
    let mut cleaned = body.clone();
    if let Some(arr) = body.get("messages").and_then(|v| v.as_array()) {
        let mut msgs = Vec::with_capacity(arr.len());
        for m in arr {
            if let Some(obj) = m.as_object() {
                if obj.contains_key("reasoning_content") {
                    let mut copy = obj.clone();
                    copy.remove("reasoning_content");
                    msgs.push(Value::Object(copy));
                    continue;
                }
            }
            msgs.push(m.clone());
        }
        cleaned["messages"] = Value::Array(msgs);
    }
    cleaned
}
