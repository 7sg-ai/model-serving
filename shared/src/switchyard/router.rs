//! Unified Switchyard router used by NIM / HF IDE assistants.

use crate::backends::anthropic_convert::claude_tier_hint;
use crate::backends::openai_tools::{
    is_tool_continuation, looks_tool_capable, model_capabilities, request_has_tools,
};
use crate::switchyard::client::ModelClient;
use crate::switchyard::config::{ModelSpec, SwitchyardConfig};
use crate::switchyard::escalation::{build_escalation_router, EscalationRouter};
use anyhow::{anyhow, Result};
use chrono::Utc;
use parking_lot::RwLock;
use rand::seq::SliceRandom;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

pub struct StreamOutcome {
    pub upstream: Option<reqwest::blocking::Response>,
    pub completion: Option<Value>,
    pub assistant_text: String,
    pub served: Option<ModelSpec>,
    pub meta: Value,
    pub synthesize_sse: bool,
}

impl StreamOutcome {
    pub fn is_live_stream(&self) -> bool {
        self.upstream.is_some()
    }
}

#[derive(Clone, Debug)]
pub struct SwitchyardRouter {
    pub cfg: SwitchyardConfig,
    pub fallback_model_id: String,
    pub escalation: Option<EscalationRouter>,
    sticky: Arc<RwLock<HashMap<String, String>>>,
}

impl SwitchyardRouter {
    pub fn new(cfg: SwitchyardConfig, fallback_model_id: &str) -> Self {
        let escalation = if cfg.enabled && cfg.strategy == "escalation" {
            build_escalation_router(&cfg)
        } else {
            None
        };
        Self {
            cfg,
            fallback_model_id: fallback_model_id.to_string(),
            escalation,
            sticky: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn client(&self, spec: &ModelSpec) -> ModelClient {
        ModelClient::new(spec.clone())
    }

    fn remember_spec(&self, session_id: &str, spec: Option<&ModelSpec>) {
        if session_id.is_empty() {
            return;
        }
        if let Some(s) = spec {
            self.sticky
                .write()
                .insert(session_id.to_string(), s.name.clone());
        }
    }

    fn sticky_spec(&self, session_id: &str, models: &[ModelSpec]) -> Option<ModelSpec> {
        let name = self.sticky.read().get(session_id).cloned()?;
        models
            .iter()
            .find(|s| s.name == name || s.id == name)
            .cloned()
    }

    pub fn hinted_role(&self, requested_model: Option<&str>) -> Option<&'static str> {
        let req = requested_model.unwrap_or("");
        if req.is_empty() || self.cfg.by_id(req).is_some() {
            return None;
        }
        claude_tier_hint(Some(req))
    }

    pub fn resolve_model_hint(&self, requested_model: Option<&str>, has_tools: bool) -> Option<String> {
        let req = requested_model.unwrap_or("");
        if self.cfg.by_id(req).is_some() {
            return None;
        }
        let role = claude_tier_hint(Some(req));
        let mut spec = match role {
            Some("strong") => self.cfg.strong().cloned(),
            Some("weak") => self.cfg.weak().cloned(),
            _ => None,
        };
        if has_tools {
            if let Some(ref s) = spec {
                if !looks_tool_capable(&s.id) {
                    if let Some(tool_spec) = self.first_tool_capable() {
                        spec = Some(tool_spec);
                    }
                }
            } else if let Some(tool_spec) = self.first_tool_capable() {
                spec = Some(tool_spec);
            }
        }
        spec.map(|s| s.id)
    }

    fn first_tool_capable(&self) -> Option<ModelSpec> {
        let selectable = self.cfg.selectable_models();
        for tier in [self.cfg.strong(), self.cfg.weak()] {
            if let Some(s) = tier {
                if looks_tool_capable(&s.id) {
                    return Some(s.clone());
                }
            }
        }
        selectable.into_iter().find(|s| looks_tool_capable(&s.id))
    }

    fn prefer_tool_spec(&self, spec: Option<ModelSpec>, extra: Option<&Value>) -> Option<ModelSpec> {
        let s = spec?;
        let tools = extra.and_then(|e| e.get("tools"));
        if !request_has_tools(tools) || looks_tool_capable(&s.id) {
            return Some(s);
        }
        self.first_tool_capable().or(Some(s))
    }
}

impl SwitchyardRouter {
    pub fn list_models_payload(&self, default_context_window: usize, default_max_tokens: usize) -> Value {
        let ts = Utc::now().timestamp();
        if !self.cfg.enabled {
            return json!({
                "object": "list",
                "data": [{
                    "id": self.fallback_model_id,
                    "object": "model",
                    "created": ts,
                    "owned_by": self.cfg.owned_by,
                    "context_window": default_context_window,
                    "max_tokens": default_max_tokens,
                    "capabilities": model_capabilities(&self.fallback_model_id, None, None, false),
                }]
            });
        }

        let mut data = Vec::new();
        data.push(json!({
            "id": self.cfg.route_id,
            "object": "model",
            "created": ts,
            "owned_by": self.cfg.owned_by,
            "root": "switchyard-route",
            "strategy": self.cfg.strategy,
            "context_window": self.cfg.models.iter().map(|m| m.context_window).max().unwrap_or(default_context_window),
            "max_tokens": self.cfg.models.iter().map(|m| m.max_tokens).max().unwrap_or(default_max_tokens),
            "targets": self.cfg.models.iter().map(|m| &m.id).collect::<Vec<_>>(),
            "capabilities": model_capabilities(&self.cfg.route_id, None, None, true),
        }));

        if self.cfg.expose_models {
            for m in self.cfg.selectable_models() {
                data.push(json!({
                    "id": m.id,
                    "name": m.name,
                    "role": m.role,
                    "object": "model",
                    "created": ts,
                    "owned_by": self.cfg.owned_by,
                    "context_window": m.context_window,
                    "max_tokens": m.max_tokens,
                    "temperature": m.temperature,
                    "deployment": m.deployment.to_dict(),
                    "capabilities": model_capabilities(&m.id, None, None, false),
                }));
            }
        }

        json!({
            "object": "list",
            "data": data,
        })
    }

    pub fn health_block(&self) -> Value {
        let mut out = self.cfg.to_dict();
        if let (Some(obj), Some(esc)) = (out.as_object_mut(), &self.escalation) {
            obj.insert("escalation_stats".to_string(), esc.stats());
        }
        out
    }

    pub fn resolve_requested_spec(&self, requested_model: Option<&str>) -> Option<ModelSpec> {
        let req = requested_model.unwrap_or("").trim();
        if req.is_empty()
            || req == self.cfg.route_id
            || matches!(req.to_ascii_lowercase().as_str(), "switchyard" | "auto" | "router" | "default")
        {
            return None;
        }
        if !self.cfg.allow_model_select {
            return None;
        }
        self.cfg
            .by_id(req)
            .or_else(|| self.cfg.by_name(req))
            .cloned()
    }

    fn call_spec(
        &self,
        spec: &ModelSpec,
        messages: &[Value],
        temperature: Option<f64>,
        max_tokens: Option<usize>,
        extra: Option<&Value>,
        route: &str,
    ) -> Result<(Value, String, ModelSpec, Value)> {
        let temp = temperature.unwrap_or(spec.temperature);
        let mt = max_tokens
            .unwrap_or(spec.max_tokens)
            .min(spec.context_window.saturating_sub(512).max(1))
            .max(1);
        let client = self.client(spec);
        let mut resp = client.chat(messages, Some(temp), Some(mt), extra, None)?;
        let text = client.extract_assistant_text(&resp);
        let meta = json!({
            "route": route,
            "served_by": spec.name,
            "model_id": spec.id,
            "role": spec.role,
        });
        resp["switchyard"] = meta.clone();
        Ok((resp, text, spec.clone(), meta))
    }
}

impl SwitchyardRouter {
    pub fn route_chat(
        &self,
        messages: &[Value],
        requested_model: Option<&str>,
        session_id: &str,
        temperature: Option<f64>,
        max_tokens: Option<usize>,
        extra: Option<&Value>,
    ) -> Result<(Value, String, Option<ModelSpec>, Value)> {
        if let Some(pinned) = self.resolve_requested_spec(requested_model) {
            self.remember_spec(session_id, Some(&pinned));
            let (resp, text, served, meta) =
                self.call_spec(&pinned, messages, temperature, max_tokens, extra, "pinned")?;
            return Ok((resp, text, Some(served), meta));
        }

        let freeze = is_tool_continuation(messages);

        if self.cfg.strategy == "escalation" {
            if let Some(esc) = &self.escalation {
                if !freeze && self.hinted_role(requested_model) == Some("strong") {
                    if let Some(strong) = self.cfg.strong() {
                        let chosen = self.prefer_tool_spec(Some(strong.clone()), extra).unwrap();
                        self.remember_spec(session_id, Some(&chosen));
                        let (resp, text, served, meta) = self.call_spec(
                            &chosen,
                            messages,
                            temperature,
                            max_tokens,
                            extra,
                            "hint",
                        )?;
                        return Ok((resp, text, Some(served), meta));
                    }
                }
                if freeze {
                    let locked = if esc.is_session_latched(session_id) {
                        esc.strong.clone()
                    } else {
                        self.sticky_spec(session_id, &self.cfg.models)
                            .unwrap_or_else(|| esc.weak.clone())
                    };
                    let (resp, text, served, meta) = self.call_spec(
                        &locked,
                        messages,
                        temperature,
                        max_tokens,
                        extra,
                        "escalation_frozen",
                    )?;
                    return Ok((resp, text, Some(served), meta));
                }
                let res = esc.route(messages, session_id, temperature, max_tokens, extra)?;
                self.remember_spec(session_id, Some(&res.model));
                let meta = res.to_meta();
                return Ok((res.response, res.assistant_text, Some(res.model), meta));
            }
        }

        if self.cfg.strategy == "random" {
            let models = self.cfg.selectable_models();
            let chosen = if freeze {
                self.sticky_spec(session_id, &models)
                    .or_else(|| models.first().cloned())
            } else {
                models.choose(&mut rand::thread_rng()).cloned()
            };
            let spec = self
                .prefer_tool_spec(chosen, extra)
                .ok_or_else(|| anyhow!("No models configured for random routing"))?;
            self.remember_spec(session_id, Some(&spec));
            let (resp, text, served, meta) =
                self.call_spec(&spec, messages, temperature, max_tokens, extra, "random")?;
            return Ok((resp, text, Some(served), meta));
        }

        let default_spec = self
            .prefer_tool_spec(self.cfg.default_model().cloned(), extra)
            .ok_or_else(|| anyhow!("No models configured"))?;
        self.remember_spec(session_id, Some(&default_spec));
        let (resp, text, served, meta) = self.call_spec(
            &default_spec,
            messages,
            temperature,
            max_tokens,
            extra,
            "passthrough",
        )?;
        Ok((resp, text, Some(served), meta))
    }

    pub fn route_chat_stream(
        &self,
        messages: &[Value],
        requested_model: Option<&str>,
        session_id: &str,
        temperature: Option<f64>,
        max_tokens: Option<usize>,
        extra: Option<&Value>,
    ) -> Result<StreamOutcome> {
        let (resp, text, served, meta) = self.route_chat(
            messages,
            requested_model,
            session_id,
            temperature,
            max_tokens,
            extra,
        )?;
        Ok(StreamOutcome {
            upstream: None,
            completion: Some(resp),
            assistant_text: text,
            served,
            meta,
            synthesize_sse: true,
        })
    }
}
