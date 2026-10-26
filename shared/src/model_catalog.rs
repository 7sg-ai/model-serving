//! Multi-model catalog for all applications.

use crate::deploy::{ensure_models_runtime, RuntimeHandle};
use crate::switchyard::client::ModelClient;
use crate::switchyard::config::{
    load_switchyard_config, DeploymentParams, ModelSpec, SwitchyardConfig,
};
use anyhow::Result;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::env;

pub fn load_app_models(
    default_model_id: &str,
    default_backend_url: &str,
    owned_by: &str,
) -> SwitchyardConfig {
    let mut cfg = load_switchyard_config(default_model_id, default_backend_url, owned_by);
    if cfg.models.is_empty() && !default_model_id.is_empty() {
        let mut dep = DeploymentParams::default();
        if !default_backend_url.is_empty() {
            dep.backend_urls = vec![default_backend_url.trim_end_matches('/').to_string()];
        }
        cfg.models.push(ModelSpec {
            name: "default".to_string(),
            id: default_model_id.to_string(),
            role: "weak".to_string(),
            deployment: dep,
            ..Default::default()
        });
    }
    if !cfg.models.is_empty() && !cfg.enabled {
        let explicit = env::var("SWITCHYARD_ENABLED").unwrap_or_default();
        if explicit.trim().is_empty() {
            cfg.enabled = true;
            if (cfg.strategy.is_empty() || cfg.strategy == "escalation")
                && !(cfg.weak().is_some() && cfg.strong().is_some())
            {
                cfg.strategy = "passthrough".to_string();
            }
        }
    }
    cfg
}

pub fn deploy_key(spec: &ModelSpec) -> String {
    let dep = &spec.deployment;
    let mut urls: Vec<String> = dep
        .backend_urls
        .iter()
        .map(|u| u.trim_end_matches('/').to_string())
        .filter(|u| !u.is_empty())
        .collect();
    urls.sort();
    if !urls.is_empty() {
        return format!("urls:{}", urls.join(","));
    }
    let cvd = dep.cuda_visible_devices.trim();
    let image = dep.nim_image.trim();
    let tp = dep.tensor_parallel_size.max(1);
    let mode = dep.deploy_mode.trim().to_ascii_lowercase();
    format!("id:{}|img:{image}|cvd:{cvd}|tp:{tp}|mode:{mode}", spec.id)
}

#[derive(Clone, Debug)]
pub struct ModelCatalog {
    pub cfg: SwitchyardConfig,
    pub engine: String,
    pub app_name: String,
    pub api_key: String,
    pub models: Vec<ModelSpec>,
    pub handles: Vec<RuntimeHandle>,
}

impl ModelCatalog {
    pub fn new(
        cfg: SwitchyardConfig,
        engine: &str,
        app_name: &str,
        api_key: &str,
        handles: Vec<RuntimeHandle>,
    ) -> Self {
        let mut models = cfg.selectable_models();
        if models.is_empty() {
            models = cfg.models.clone();
        }
        Self {
            cfg,
            engine: engine.to_string(),
            app_name: app_name.to_string(),
            api_key: api_key.to_string(),
            models,
            handles,
        }
    }

    pub fn multi(&self) -> bool {
        self.models.len() > 1
    }

    pub fn default_spec(&self) -> ModelSpec {
        self.cfg
            .default_model()
            .cloned()
            .or_else(|| self.models.first().cloned())
            .unwrap_or_default()
    }

    pub fn default_id(&self) -> String {
        self.default_spec().id
    }

    pub fn resolve(&self, key: Option<&str>) -> ModelSpec {
        let k = key.unwrap_or("").trim();
        if k.is_empty() {
            return self.default_spec();
        }
        for m in &self.models {
            if m.id == k || m.name == k {
                return m.clone();
            }
        }
        self.default_spec()
    }
}

impl ModelCatalog {
    pub fn choices_for_ui(&self) -> Vec<Value> {
        let mut out = Vec::new();
        for m in &self.models {
            let label = if self.multi() && !m.name.is_empty() && m.name != "default" && m.name != m.id
            {
                format!("{} ({})", m.id, m.name)
            } else {
                m.id.clone()
            };
            out.push(json!({
                "id": m.id,
                "name": m.name,
                "label": label,
                "role": m.role,
            }));
        }
        if out.is_empty() {
            let d = self.default_spec();
            out.push(json!({
                "id": d.id,
                "name": d.name,
                "label": d.id,
                "role": d.role,
            }));
        }
        out
    }

    pub fn client_for(&self, key: Option<&str>) -> ModelClient {
        let spec = self.resolve(key);
        ModelClient::new(spec)
    }

    pub fn chat(
        &self,
        messages: &[Value],
        model: Option<&str>,
        temperature: Option<f64>,
        max_tokens: Option<usize>,
    ) -> Result<(String, Value)> {
        let client = self.client_for(model);
        let data = client.chat(messages, temperature, max_tokens, None, None)?;
        let text = client.extract_assistant_text(&data);
        Ok((text, data))
    }

    pub fn list_payload(&self) -> Value {
        json!({
            "object": "list",
            "default": self.default_id(),
            "data": self.choices_for_ui(),
        })
    }

    pub fn health_block(&self) -> Value {
        json!({
            "multi_model": self.multi(),
            "default": self.default_id(),
            "models": self.models.iter().map(|m| m.to_dict()).collect::<Vec<_>>(),
            "deploy": self.handles.iter().map(|h| h.to_dict()).collect::<Vec<_>>(),
        })
    }

    pub fn all_backend_urls(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for m in &self.models {
            for u in &m.deployment.backend_urls {
                let trimmed = u.trim().trim_end_matches('/').to_string();
                if !trimmed.is_empty() && seen.insert(trimmed.clone()) {
                    out.push(trimmed);
                }
            }
            if !m.deployment.coordinator_url.is_empty() {
                let u = m.deployment.coordinator_url.trim_end_matches('/').to_string();
                if !u.is_empty() && seen.insert(u.clone()) {
                    out.push(u);
                }
            }
        }
        out
    }
}

pub fn deploy_app_models(
    engine: &str,
    cfg: &mut SwitchyardConfig,
    app_name: &str,
    api_key: &str,
    default_backend_urls: &[String],
) -> (ModelCatalog, Vec<RuntimeHandle>) {
    let handles = ensure_models_runtime(
        engine,
        &mut cfg.models,
        app_name,
        api_key,
        default_backend_urls,
    );
    let catalog = ModelCatalog::new(cfg.clone(), engine, app_name, api_key, handles.clone());
    (catalog, handles)
}
