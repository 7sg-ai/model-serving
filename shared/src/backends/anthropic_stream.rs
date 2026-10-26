//! Anthropic SSE emission and OpenAI SSE -> Anthropic SSE translation.

use crate::backends::anthropic_response::{direct_caller, repair_tool_input};
use chrono::Utc;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub fn sse(event: &str, data: &Value) -> String {
    format!(
        "event: {event}\ndata: {}\n\n",
        serde_json::to_string(data).unwrap_or_else(|_| "{}".to_string())
    )
}

pub fn anthropic_message_to_sse(message: &Value) -> Vec<String> {
    let msg_id = message
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("msg_{}", Utc::now().timestamp()));
    let usage = message.get("usage");
    let input_tokens = usage
        .and_then(|u| u.get("input_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output_tokens = usage
        .and_then(|u| u.get("output_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let mut out = Vec::new();
    out.push(sse(
        "message_start",
        &json!({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": message.get("model").and_then(|v| v.as_str()).unwrap_or("unknown"),
                "content": [],
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}
            }
        }),
    ));

    if let Some(blocks) = message.get("content").and_then(|v| v.as_array()) {
        for (index, block) in blocks.iter().enumerate() {
            let Some(obj) = block.as_object() else {
                continue;
            };
            let kind = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match kind {
                "thinking" => {
                    out.push(sse(
                        "content_block_start",
                        &json!({"type": "content_block_start", "index": index, "content_block": {"type": "thinking", "thinking": ""}}),
                    ));
                    if let Some(t) = obj.get("thinking").and_then(|v| v.as_str()) {
                        if !t.is_empty() {
                            out.push(sse(
                                "content_block_delta",
                                &json!({"type": "content_block_delta", "index": index, "delta": {"type": "thinking_delta", "thinking": t}}),
                            ));
                        }
                    }
                    out.push(sse("content_block_stop", &json!({"type": "content_block_stop", "index": index})));
                }
                "text" => {
                    out.push(sse(
                        "content_block_start",
                        &json!({"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}}),
                    ));
                    if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
                        if !t.is_empty() {
                            out.push(sse(
                                "content_block_delta",
                                &json!({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": t}}),
                            ));
                        }
                    }
                    out.push(sse("content_block_stop", &json!({"type": "content_block_stop", "index": index})));
                }
                "tool_use" => {
                    out.push(sse(
                        "content_block_start",
                        &json!({
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {
                                "type": "tool_use",
                                "id": obj.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| format!("toolu_{index}")),
                                "name": obj.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                                "input": {},
                                "caller": obj.get("caller").cloned().unwrap_or_else(direct_caller),
                            }
                        }),
                    ));
                    let partial_json = serde_json::to_string(obj.get("input").unwrap_or(&json!({})))
                        .unwrap_or_else(|_| "{}".to_string());
                    out.push(sse(
                        "content_block_delta",
                        &json!({"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": partial_json}}),
                    ));
                    out.push(sse("content_block_stop", &json!({"type": "content_block_stop", "index": index})));
                }
                _ => {}
            }
        }
    }

    out.push(sse(
        "message_delta",
        &json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": message.get("stop_reason").and_then(|v| v.as_str()).unwrap_or("end_turn"),
                "stop_sequence": Value::Null,
            },
            "usage": {"output_tokens": output_tokens}
        }),
    ));
    out.push(sse("message_stop", &json!({"type": "message_stop"})));
    out
}

#[derive(Clone, Debug, Default)]
struct ToolBuf {
    id: String,
    name: String,
    arguments: String,
}

pub struct AnthropicStreamAssembler {
    pub requested_model: String,
    pub message_id: String,
    started: bool,
    next_index: usize,
    thinking_open: Option<usize>,
    text_open: Option<usize>,
    tool_bufs: BTreeMap<usize, ToolBuf>,
    finish_reason: Option<String>,
    output_tokens: u64,
}

impl AnthropicStreamAssembler {
    pub fn new(requested_model: &str, message_id: Option<&str>) -> Self {
        Self {
            requested_model: requested_model.to_string(),
            message_id: message_id
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("msg_{}", Utc::now().timestamp())),
            started: false,
            next_index: 0,
            thinking_open: None,
            text_open: None,
            tool_bufs: BTreeMap::new(),
            finish_reason: None,
            output_tokens: 0,
        }
    }

    fn start_events(&mut self) -> Vec<String> {
        if self.started {
            return Vec::new();
        }
        self.started = true;
        vec![sse(
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.requested_model,
                    "content": [],
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }
            }),
        )]
    }

    fn close_thinking(&mut self) -> Vec<String> {
        if let Some(idx) = self.thinking_open.take() {
            vec![sse(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": idx}),
            )]
        } else {
            Vec::new()
        }
    }

    fn close_text(&mut self) -> Vec<String> {
        if let Some(idx) = self.text_open.take() {
            vec![sse(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": idx}),
            )]
        } else {
            Vec::new()
        }
    }

    fn thinking_delta(&mut self, thought: &str) -> Vec<String> {
        if thought.is_empty() {
            return Vec::new();
        }
        let mut out = self.start_events();
        out.extend(self.close_text());
        let idx = match self.thinking_open {
            Some(i) => i,
            None => {
                let i = self.next_index;
                self.next_index += 1;
                self.thinking_open = Some(i);
                out.push(sse(
                    "content_block_start",
                    &json!({
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {"type": "thinking", "thinking": ""},
                    }),
                ));
                i
            }
        };
        out.push(sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "thinking_delta", "thinking": thought},
            }),
        ));
        out
    }

    fn text_delta(&mut self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut out = self.start_events();
        out.extend(self.close_thinking());
        let idx = match self.text_open {
            Some(i) => i,
            None => {
                let i = self.next_index;
                self.next_index += 1;
                self.text_open = Some(i);
                out.push(sse(
                    "content_block_start",
                    &json!({
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {"type": "text", "text": ""},
                    }),
                ));
                i
            }
        };
        out.push(sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "text_delta", "text": text},
            }),
        ));
        out
    }
}

impl AnthropicStreamAssembler {
    pub fn feed_chunk(&mut self, chunk: &Value) -> Vec<String> {
        let mut out = self.start_events();
        if let Some(ot) = chunk
            .get("usage")
            .and_then(|u| u.get("completion_tokens").or_else(|| u.get("output_tokens")))
            .and_then(|v| v.as_u64())
        {
            self.output_tokens = ot;
        }
        let Some(choice) = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
        else {
            return out;
        };
        if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
            if !fr.is_empty() {
                self.finish_reason = Some(fr.to_string());
            }
        }
        let Some(delta) = choice.get("delta").and_then(|d| d.as_object()) else {
            return out;
        };
        if let Some(rc) = delta
            .get("reasoning_content")
            .or_else(|| delta.get("reasoning"))
            .and_then(|v| v.as_str())
        {
            out.extend(self.thinking_delta(rc));
        }
        if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
            out.extend(self.text_delta(content));
        }
        if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for (pos, tc) in tcs.iter().enumerate() {
                let idx = tc
                    .get("index")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize)
                    .unwrap_or(pos);
                let entry = self.tool_bufs.entry(idx).or_default();
                if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                    if !id.is_empty() {
                        entry.id = id.to_string();
                    }
                }
                if let Some(fn_obj) = tc.get("function") {
                    if let Some(name) = fn_obj.get("name").and_then(|v| v.as_str()) {
                        if !name.is_empty() {
                            entry.name.push_str(name);
                        }
                    }
                    if let Some(args) = fn_obj.get("arguments").and_then(|v| v.as_str()) {
                        entry.arguments.push_str(args);
                    }
                }
            }
        }
        out
    }
}

impl AnthropicStreamAssembler {
    pub fn finish_events(&mut self) -> Vec<String> {
        let mut out = self.start_events();
        out.extend(self.close_thinking());
        out.extend(self.close_text());
        let mut emitted_tool = false;
        let mut dropped = Vec::new();
        let keys: Vec<usize> = self.tool_bufs.keys().copied().collect();
        for idx in keys {
            let buf = self.tool_bufs.get(&idx).unwrap().clone();
            let raw_args = if buf.arguments.is_empty() {
                json!("{}")
            } else {
                json!(buf.arguments)
            };
            let Some(parsed) = repair_tool_input(Some(&raw_args)) else {
                dropped.push(if buf.name.is_empty() {
                    "tool".to_string()
                } else {
                    buf.name
                });
                continue;
            };
            let block_index = self.next_index;
            self.next_index += 1;
            emitted_tool = true;
            out.push(sse(
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": if buf.id.is_empty() { format!("toolu_{idx}") } else { buf.id },
                        "name": buf.name,
                        "input": {},
                        "caller": direct_caller(),
                    }
                }),
            ));
            out.push(sse(
                "content_block_delta",
                &json!({
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": serde_json::to_string(&parsed).unwrap_or_else(|_| "{}".to_string()),
                    }
                }),
            ));
            out.push(sse(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": block_index}),
            ));
        }

        let stop_reason = if emitted_tool {
            "tool_use"
        } else if !dropped.is_empty() && self.finish_reason.as_deref() == Some("length") {
            "max_tokens"
        } else if !dropped.is_empty() || self.finish_reason.as_deref() == Some("tool_calls") {
            if !dropped.is_empty() {
                let names = dropped.join(", ");
                let note = format!(
                    "Dropped an incomplete tool call ({names}). No arguments were invented."
                );
                out.extend(self.text_delta(&note));
                out.extend(self.close_text());
            }
            "end_turn"
        } else {
            match self.finish_reason.as_deref().unwrap_or("stop") {
                "stop" | "content_filter" => "end_turn",
                "tool_calls" => "tool_use",
                "length" => "max_tokens",
                _ => "end_turn",
            }
        };

        out.push(sse(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": Value::Null},
                "usage": {"output_tokens": self.output_tokens}
            }),
        ));
        out.push(sse("message_stop", &json!({"type": "message_stop"})));
        out
    }
}

pub fn iter_openai_sse_as_anthropic(
    lines: &[String],
    requested_model: &str,
    message_id: Option<&str>,
) -> Vec<String> {
    let mut assembler = AnthropicStreamAssembler::new(requested_model, message_id);
    let mut out = Vec::new();
    for raw in lines {
        for line in raw.lines() {
            let trimmed = line.trim();
            if !trimmed.starts_with("data:") {
                continue;
            }
            let data = trimmed[5..].trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            if let Ok(chunk) = serde_json::from_str::<Value>(data) {
                if chunk.is_object() {
                    out.extend(assembler.feed_chunk(&chunk));
                }
            }
        }
    }
    out.extend(assembler.finish_events());
    out
}
