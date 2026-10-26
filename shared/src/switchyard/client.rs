//! OpenAI-compatible chat client for Switchyard model targets.

use crate::backends::openai_chat::{extract_assistant_text, normalize_chat_url};
use crate::backends::streaming::post_chat_stream_blocking;
use crate::multinode::BackendPool;
use crate::switchyard::config::ModelSpec;
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct ModelClient {
    pub spec: ModelSpec,
    pub pool: BackendPool,
}

impl ModelClient {
    pub fn new(spec: ModelSpec) -> Self {
        let mut urls: Vec<String> = spec
            .deployment
            .backend_urls
            .iter()
            .map(|u| normalize_chat_url(u))
            .filter(|u| !u.is_empty())
            .collect();
        if urls.is_empty() {
            let cu = spec.chat_url();
            if !cu.is_empty() {
                urls.push(normalize_chat_url(&cu));
            }
        }
        if !spec.deployment.coordinator_url.is_empty() {
            let coord = normalize_chat_url(&spec.deployment.coordinator_url);
            if !coord.is_empty() {
                urls.retain(|u| u != &coord);
                urls.insert(0, coord);
            }
        }
        let pool = BackendPool::new(urls, &spec.deployment.backend_strategy);
        Self { spec, pool }
    }

    fn build_payload(
        &self,
        messages: &[Value],
        temperature: Option<f64>,
        max_tokens: Option<usize>,
        stream: bool,
        extra: Option<&Value>,
        model_id_override: Option<&str>,
    ) -> Value {
        let temp = temperature.unwrap_or(self.spec.temperature);
        let mt = max_tokens.unwrap_or(self.spec.max_tokens);
        let mut payload = json!({
            "model": model_id_override.unwrap_or(&self.spec.id),
            "messages": messages,
            "temperature": temp,
            "max_tokens": mt,
            "stream": stream,
        });
        if let Some(tp) = self.spec.top_p {
            payload["top_p"] = json!(tp);
        }
        if let Some(base) = payload.as_object_mut() {
            if let Some(eb) = self.spec.extra_body.as_object() {
                for (k, v) in eb {
                    if k != "stream" {
                        base.insert(k.clone(), v.clone());
                    }
                }
            }
            if let Some(ex) = extra.and_then(|v| v.as_object()) {
                for (k, v) in ex {
                    if k != "stream" {
                        base.insert(k.clone(), v.clone());
                    }
                }
            }
            base.insert("stream".to_string(), json!(stream));
        }
        payload
    }

    pub fn chat(
        &self,
        messages: &[Value],
        temperature: Option<f64>,
        max_tokens: Option<usize>,
        extra: Option<&Value>,
        model_id_override: Option<&str>,
    ) -> Result<Value> {
        let Some(url) = self.pool.next() else {
            return Err(anyhow!(
                "No backend URL configured for model '{}' ({})",
                self.spec.name,
                self.spec.id
            ));
        };
        let payload = self.build_payload(
            messages,
            temperature,
            max_tokens,
            false,
            extra,
            model_id_override,
        );

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs_f64(self.spec.request_timeout.max(1.0)))
            .build()?;
        let mut req = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&payload);
        for (k, v) in &self.spec.headers {
            req = req.header(k, v);
        }
        if let Some(key) = self.spec.resolve_api_key() {
            if !key.is_empty() && !self.spec.headers.contains_key("Authorization") {
                req = req.header("Authorization", format!("Bearer {key}"));
            }
        }

        let resp = match req.send() {
            Ok(r) => r,
            Err(e) => {
                self.pool.mark_failure(&url);
                return Err(e.into());
            }
        };
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if !status.is_success() {
            self.pool.mark_failure(&url);
            return Err(anyhow!("HTTP {}: {}", status.as_u16(), text));
        }
        self.pool.mark_success(&url);
        let mut data: Value = serde_json::from_str(&text)?;
        if let Some(obj) = data.as_object_mut() {
            obj.entry("switchyard_target".to_string())
                .or_insert_with(|| json!(self.spec.name));
            obj.entry("switchyard_model_id".to_string())
                .or_insert_with(|| json!(self.spec.id));
        }
        Ok(data)
    }

    pub fn chat_stream(
        &self,
        messages: &[Value],
        temperature: Option<f64>,
        max_tokens: Option<usize>,
        extra: Option<&Value>,
        model_id_override: Option<&str>,
    ) -> Result<reqwest::blocking::Response> {
        let Some(url) = self.pool.next() else {
            return Err(anyhow!(
                "No backend URL configured for model '{}' ({})",
                self.spec.name,
                self.spec.id
            ));
        };
        let temp = temperature.unwrap_or(self.spec.temperature);
        let mt = max_tokens.unwrap_or(self.spec.max_tokens);
        let mut merged_extra = serde_json::Map::new();
        if let Some(eb) = self.spec.extra_body.as_object() {
            for (k, v) in eb {
                if k != "stream" {
                    merged_extra.insert(k.clone(), v.clone());
                }
            }
        }
        if let Some(ex) = extra.and_then(|v| v.as_object()) {
            for (k, v) in ex {
                if k != "stream" {
                    merged_extra.insert(k.clone(), v.clone());
                }
            }
        }
        if let Some(tp) = self.spec.top_p {
            merged_extra
                .entry("top_p".to_string())
                .or_insert_with(|| json!(tp));
        }
        let extra_val = Value::Object(merged_extra);
        match post_chat_stream_blocking(
            &url,
            messages,
            model_id_override.unwrap_or(&self.spec.id),
            temp,
            mt,
            &self.spec.resolve_api_key().unwrap_or_default(),
            self.spec.request_timeout,
            Some(&extra_val),
        ) {
            Ok(resp) => {
                self.pool.mark_success(&url);
                Ok(resp)
            }
            Err(e) => {
                self.pool.mark_failure(&url);
                Err(e)
            }
        }
    }

    pub fn extract_assistant_text(&self, result: &Value) -> String {
        extract_assistant_text(result)
    }
}
