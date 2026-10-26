//! Multi-model Switchyard configuration.

use crate::{env_bool, env_int, split_csv};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::env;
use std::fs;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeploymentParams {
    pub backend_urls: Vec<String>,
    pub backend_strategy: String,
    pub load_weights_locally: bool,
    pub coordinator_url: String,
    pub deploy_mode: String,
    pub tensor_parallel_size: usize,
    pub pipeline_parallel_size: usize,
    pub model_parallel_size: usize,
    pub shard_rank: usize,
    pub shard_world_size: usize,
    pub shard_group: String,
    pub cuda_visible_devices: String,
    pub device_map: String,
    pub max_memory: String,
    pub max_memory_per_gpu: String,
    pub max_memory_cpu: String,
    pub torch_dtype: String,
    pub trust_remote_code: Option<bool>,
    pub offload_folder: String,
    pub nim_image: String,
    pub nim_port: u16,
    pub gpu_count: usize,
    pub node_selector: String,
    pub extra: Value,
}

impl Default for DeploymentParams {
    fn default() -> Self {
        Self {
            backend_urls: Vec::new(),
            backend_strategy: "round_robin".to_string(),
            load_weights_locally: false,
            coordinator_url: String::new(),
            deploy_mode: "replica".to_string(),
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            model_parallel_size: 1,
            shard_rank: 0,
            shard_world_size: 1,
            shard_group: String::new(),
            cuda_visible_devices: String::new(),
            device_map: String::new(),
            max_memory: String::new(),
            max_memory_per_gpu: String::new(),
            max_memory_cpu: String::new(),
            torch_dtype: String::new(),
            trust_remote_code: None,
            offload_folder: String::new(),
            nim_image: String::new(),
            nim_port: 0,
            gpu_count: 0,
            node_selector: String::new(),
            extra: json!({}),
        }
    }
}

impl DeploymentParams {
    pub fn primary_backend(&self) -> String {
        if !self.coordinator_url.is_empty() {
            return self.coordinator_url.trim_end_matches('/').to_string();
        }
        self.backend_urls
            .first()
            .map(|u| u.trim_end_matches('/').to_string())
            .unwrap_or_default()
    }

    pub fn to_dict(&self) -> Value {
        let mut out = serde_json::Map::new();
        if !self.backend_urls.is_empty() {
            out.insert("backend_urls".to_string(), json!(self.backend_urls));
        }
        if !self.backend_strategy.is_empty() {
            out.insert("backend_strategy".to_string(), json!(self.backend_strategy));
        }
        if self.load_weights_locally {
            out.insert("load_weights_locally".to_string(), json!(true));
        }
        if !self.coordinator_url.is_empty() {
            out.insert("coordinator_url".to_string(), json!(self.coordinator_url));
        }
        if !self.deploy_mode.is_empty() {
            out.insert("deploy_mode".to_string(), json!(self.deploy_mode));
        }
        if self.tensor_parallel_size > 0 {
            out.insert(
                "tensor_parallel_size".to_string(),
                json!(self.tensor_parallel_size),
            );
        }
        if self.pipeline_parallel_size > 0 {
            out.insert(
                "pipeline_parallel_size".to_string(),
                json!(self.pipeline_parallel_size),
            );
        }
        if self.model_parallel_size > 0 {
            out.insert(
                "model_parallel_size".to_string(),
                json!(self.model_parallel_size),
            );
        }
        if self.shard_world_size > 0 {
            out.insert("shard_world_size".to_string(), json!(self.shard_world_size));
        }
        if !self.cuda_visible_devices.is_empty() {
            out.insert(
                "cuda_visible_devices".to_string(),
                json!(self.cuda_visible_devices),
            );
        }
        if !self.nim_image.is_empty() {
            out.insert("nim_image".to_string(), json!(self.nim_image));
        }
        if self.nim_port > 0 {
            out.insert("nim_port".to_string(), json!(self.nim_port));
        }
        if self.gpu_count > 0 {
            out.insert("gpu_count".to_string(), json!(self.gpu_count));
        }
        if let Some(ex) = self.extra.as_object() {
            if !ex.is_empty() {
                out.insert("extra".to_string(), self.extra.clone());
            }
        }
        Value::Object(out)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelSpec {
    pub name: String,
    pub id: String,
    pub role: String,
    pub base_url: String,
    pub api_key: String,
    pub api_key_env: String,
    pub format: String,
    pub context_window: usize,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub request_timeout: f64,
    pub extra_body: Value,
    pub headers: HashMap<String, String>,
    pub deployment: DeploymentParams,
    pub description: String,
    pub tags: Vec<String>,
}

impl Default for ModelSpec {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            id: String::new(),
            role: "weak".to_string(),
            base_url: String::new(),
            api_key: String::new(),
            api_key_env: String::new(),
            format: "openai_chat".to_string(),
            context_window: 1048576,
            max_tokens: 32768,
            temperature: 0.3,
            top_p: None,
            request_timeout: 300.0,
            extra_body: json!({}),
            headers: HashMap::new(),
            deployment: DeploymentParams::default(),
            description: String::new(),
            tags: Vec::new(),
        }
    }
}

impl ModelSpec {
    pub fn resolve_api_key(&self) -> Option<String> {
        if !self.api_key.is_empty() {
            return Some(self.api_key.clone());
        }
        if !self.api_key_env.is_empty() {
            if let Ok(v) = env::var(&self.api_key_env) {
                if !v.is_empty() {
                    return Some(v);
                }
            }
        }
        None
    }

    pub fn chat_url(&self) -> String {
        let mut raw = if !self.base_url.is_empty() {
            self.base_url.clone()
        } else {
            self.deployment.primary_backend()
        };
        raw = raw.trim().trim_end_matches('/').to_string();
        if raw.is_empty() {
            return String::new();
        }
        if raw.ends_with("/chat/completions") {
            return raw;
        }
        if raw.ends_with("/v1") {
            return format!("{raw}/chat/completions");
        }
        format!("{raw}/v1/chat/completions")
    }

    pub fn openai_base_url(&self) -> String {
        let mut raw = if !self.base_url.is_empty() {
            self.base_url.clone()
        } else {
            self.deployment.primary_backend()
        };
        raw = raw.trim().trim_end_matches('/').to_string();
        if raw.is_empty() {
            return String::new();
        }
        if let Some(stripped) = raw.strip_suffix("/chat/completions") {
            raw = stripped.to_string();
        }
        if !raw.ends_with("/v1") {
            raw = format!("{raw}/v1");
        }
        raw
    }

    pub fn to_dict(&self) -> Value {
        json!({
            "name": self.name,
            "id": self.id,
            "role": self.role,
            "format": self.format,
            "chat_url": if self.chat_url().is_empty() { Value::Null } else { json!(self.chat_url()) },
            "openai_base_url": if self.openai_base_url().is_empty() { Value::Null } else { json!(self.openai_base_url()) },
            "api_key_configured": self.resolve_api_key().is_some(),
            "api_key_env": if self.api_key_env.is_empty() { Value::Null } else { json!(self.api_key_env) },
            "context_window": self.context_window,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "request_timeout": self.request_timeout,
            "extra_body": if self.extra_body.as_object().map(|o| !o.is_empty()).unwrap_or(false) { self.extra_body.clone() } else { Value::Null },
            "deployment": self.deployment.to_dict(),
            "description": if self.description.is_empty() { Value::Null } else { json!(self.description) },
            "tags": self.tags,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EscalationSettings {
    pub confirmations: usize,
    pub recent_turn_window: usize,
    pub window_message_chars: usize,
    pub max_output_tokens: usize,
    pub prompt: String,
    pub session_ttl_seconds: u64,
    pub max_sessions: usize,
    pub fail_open: bool,
}

impl Default for EscalationSettings {
    fn default() -> Self {
        Self {
            confirmations: 2,
            recent_turn_window: 10,
            window_message_chars: 1800,
            max_output_tokens: 400,
            prompt: String::new(),
            session_ttl_seconds: 7200,
            max_sessions: 256,
            fail_open: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SwitchyardConfig {
    pub enabled: bool,
    pub strategy: String,
    pub route_id: String,
    pub expose_models: bool,
    pub allow_model_select: bool,
    pub models: Vec<ModelSpec>,
    pub escalation: EscalationSettings,
    pub external_url: String,
    pub external_api_key_env: String,
    pub routes_toml_path: String,
    pub owned_by: String,
}

impl Default for SwitchyardConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: "escalation".to_string(),
            route_id: "switchyard/agent".to_string(),
            expose_models: true,
            allow_model_select: true,
            models: Vec::new(),
            escalation: EscalationSettings::default(),
            external_url: String::new(),
            external_api_key_env: String::new(),
            routes_toml_path: String::new(),
            owned_by: "switchyard".to_string(),
        }
    }
}

impl SwitchyardConfig {
    pub fn by_name(&self, name: &str) -> Option<&ModelSpec> {
        self.models.iter().find(|m| m.name == name)
    }

    pub fn by_id(&self, model_id: &str) -> Option<&ModelSpec> {
        self.models
            .iter()
            .find(|m| m.id == model_id || m.name == model_id)
    }

    pub fn by_role(&self, role: &str) -> Option<&ModelSpec> {
        let r = role.to_ascii_lowercase();
        self.models.iter().find(|m| m.role.to_ascii_lowercase() == r)
    }

    pub fn weak(&self) -> Option<&ModelSpec> {
        self.by_role("weak")
            .or_else(|| self.by_name("weak"))
            .or_else(|| self.models.first())
    }

    pub fn strong(&self) -> Option<&ModelSpec> {
        self.by_role("strong").or_else(|| self.by_name("strong"))
    }

    pub fn judge(&self) -> Option<&ModelSpec> {
        self.by_role("judge")
            .or_else(|| self.by_name("judge"))
            .or_else(|| self.weak())
    }

    pub fn default_model(&self) -> Option<&ModelSpec> {
        self.weak()
            .or_else(|| self.strong())
            .or_else(|| self.models.first())
    }

    pub fn selectable_models(&self) -> Vec<ModelSpec> {
        let mut out: Vec<ModelSpec> = self
            .models
            .iter()
            .filter(|m| m.role != "judge" || self.models.len() == 1)
            .cloned()
            .collect();
        if out.is_empty() {
            out = self.models.clone();
        }
        out
    }

    pub fn to_dict(&self) -> Value {
        json!({
            "enabled": self.enabled,
            "strategy": self.strategy,
            "route_id": self.route_id,
            "expose_models": self.expose_models,
            "allow_model_select": self.allow_model_select,
            "external_url": if self.external_url.is_empty() { Value::Null } else { json!(self.external_url) },
            "escalation": {
                "confirmations": self.escalation.confirmations,
                "recent_turn_window": self.escalation.recent_turn_window,
                "window_message_chars": self.escalation.window_message_chars,
                "max_output_tokens": self.escalation.max_output_tokens,
                "session_ttl_seconds": self.escalation.session_ttl_seconds,
                "fail_open": self.escalation.fail_open,
            },
            "models": self.models.iter().map(|m| m.to_dict()).collect::<Vec<_>>(),
        })
    }
}

pub fn deployment_from_dict(raw: Option<&Value>, parent: Option<&Value>) -> DeploymentParams {
    let d = raw.unwrap_or(&Value::Null);
    let p = parent.unwrap_or(&Value::Null);
    let get = |k: &str| -> Option<&Value> { d.get(k).or_else(|| p.get(k)) };
    let get_str = |k: &str, def: &str| -> String {
        get(k)
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| def.to_string())
    };
    let get_usize = |k: &str, def: usize| -> usize {
        get(k)
            .and_then(|v| {
                v.as_u64()
                    .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
            })
            .map(|n| n as usize)
            .unwrap_or(def)
    };

    let mut urls = Vec::new();
    for key in ["backend_urls", "backends", "urls"] {
        if let Some(val) = get(key) {
            if let Some(arr) = val.as_array() {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        if !s.trim().is_empty() {
                            urls.push(s.trim().to_string());
                        }
                    }
                }
            } else if let Some(s) = val.as_str() {
                urls.extend(split_csv(s));
            }
            if !urls.is_empty() {
                break;
            }
        }
    }
    if urls.is_empty() {
        for key in ["backend_url", "url", "base_url", "endpoint"] {
            let s = get_str(key, "");
            if !s.is_empty() {
                urls = split_csv(&s);
                break;
            }
        }
    }

    let tp = get_usize("tensor_parallel_size", get_usize("tp", 1)).max(1);
    let pp = get_usize("pipeline_parallel_size", get_usize("pp", 1)).max(1);
    let mp = get_usize("model_parallel_size", (tp * pp).max(1)).max(1);

    let known = [
        "backend_urls",
        "backends",
        "urls",
        "backend_url",
        "url",
        "base_url",
        "endpoint",
        "backend_strategy",
        "load_weights_locally",
        "coordinator_url",
        "deploy_mode",
        "tensor_parallel_size",
        "tp",
        "pipeline_parallel_size",
        "pp",
        "model_parallel_size",
        "shard_rank",
        "shard_world_size",
        "shard_group",
        "cuda_visible_devices",
        "gpu_devices",
        "device_map",
        "max_memory",
        "max_memory_per_gpu",
        "max_memory_cpu",
        "torch_dtype",
        "trust_remote_code",
        "offload_folder",
        "nim_image",
        "nim_port",
        "gpu_count",
        "node_selector",
    ];
    let mut extra = serde_json::Map::new();
    if let Some(obj) = d.as_object() {
        for (k, v) in obj {
            if !known.contains(&k.as_str()) {
                extra.insert(k.clone(), v.clone());
            }
        }
    }

    DeploymentParams {
        backend_urls: urls,
        backend_strategy: get_str("backend_strategy", "round_robin").to_ascii_lowercase(),
        load_weights_locally: get("load_weights_locally")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        coordinator_url: get_str("coordinator_url", ""),
        deploy_mode: get_str("deploy_mode", "replica").to_ascii_lowercase(),
        tensor_parallel_size: tp,
        pipeline_parallel_size: pp,
        model_parallel_size: mp,
        shard_rank: get_usize("shard_rank", 0),
        shard_world_size: get_usize("shard_world_size", mp.max(1)).max(1),
        shard_group: get_str("shard_group", ""),
        cuda_visible_devices: get_str("cuda_visible_devices", &get_str("gpu_devices", "")),
        device_map: get_str("device_map", ""),
        max_memory: get_str("max_memory", ""),
        max_memory_per_gpu: get_str("max_memory_per_gpu", ""),
        max_memory_cpu: get_str("max_memory_cpu", ""),
        torch_dtype: get_str("torch_dtype", ""),
        trust_remote_code: get("trust_remote_code").and_then(|v| v.as_bool()),
        offload_folder: get_str("offload_folder", ""),
        nim_image: get_str("nim_image", ""),
        nim_port: get_usize("nim_port", 0) as u16,
        gpu_count: get_usize("gpu_count", 0),
        node_selector: get_str("node_selector", ""),
        extra: Value::Object(extra),
    }
}

pub fn model_spec_from_dict(raw: &Value, index: usize) -> ModelSpec {
    let name = raw
        .get("name")
        .or_else(|| raw.get("key"))
        .or_else(|| raw.get("target"))
        .or_else(|| raw.get("role"))
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("model-{index}"));
    let model_id = raw
        .get("id")
        .or_else(|| raw.get("model"))
        .or_else(|| raw.get("model_id"))
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| name.clone());
    let mut role = raw
        .get("role")
        .or_else(|| raw.get("tier"))
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| name.to_ascii_lowercase());
    if !matches!(role.as_str(), "weak" | "strong" | "judge" | "general") {
        role = if matches!(name.to_ascii_lowercase().as_str(), "weak" | "strong" | "judge") {
            name.to_ascii_lowercase()
        } else {
            "general".to_string()
        };
    }

    let deployment = deployment_from_dict(raw.get("deployment"), Some(raw));
    let base_url = raw
        .get("base_url")
        .or_else(|| raw.get("url"))
        .or_else(|| raw.get("backend_url"))
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| deployment.primary_backend());

    ModelSpec {
        name,
        id: model_id,
        role,
        base_url,
        api_key: raw
            .get("api_key")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string(),
        api_key_env: raw
            .get("api_key_env")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string(),
        format: raw
            .get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("openai_chat")
            .trim()
            .to_string(),
        context_window: raw
            .get("context_window")
            .and_then(|v| v.as_u64())
            .unwrap_or(1048576) as usize,
        max_tokens: raw
            .get("max_tokens")
            .or_else(|| raw.get("max_output_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(32768) as usize,
        temperature: raw
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.3),
        top_p: raw.get("top_p").and_then(|v| v.as_f64()),
        request_timeout: raw
            .get("request_timeout")
            .or_else(|| raw.get("timeout"))
            .and_then(|v| v.as_f64())
            .unwrap_or(300.0),
        extra_body: raw.get("extra_body").cloned().unwrap_or_else(|| json!({})),
        headers: HashMap::new(),
        deployment,
        description: raw
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        tags: Vec::new(),
    }
}

pub fn load_switchyard_config(
    default_model_id: &str,
    default_backend_url: &str,
    owned_by: &str,
) -> SwitchyardConfig {
    let mut file_data = json!({});
    if let Ok(path) = env::var("SWITCHYARD_CONFIG").or_else(|_| env::var("SWITCHYARD_MODELS_FILE"))
    {
        let p = path.trim();
        if !p.is_empty() {
            if let Ok(raw) = fs::read_to_string(p) {
                if let Ok(val) = serde_json::from_str::<Value>(&raw) {
                    if val.is_array() {
                        file_data = json!({"models": val});
                    } else if val.is_object() {
                        file_data = val;
                    }
                }
            }
        }
    }
    if let Ok(inline) = env::var("SWITCHYARD_MODELS") {
        let s = inline.trim();
        if !s.is_empty() {
            if let Ok(val) = serde_json::from_str::<Value>(s) {
                if val.is_array() {
                    file_data = json!({"models": val});
                } else if val.is_object() {
                    file_data = val;
                }
            }
        }
    }

    let mut models_raw = file_data
        .get("models")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    if models_raw.is_empty() {
        for tier in ["weak", "strong", "judge"] {
            let prefix = format!("SWITCHYARD_{}_", tier.to_ascii_uppercase());
            let id = env::var(format!("{prefix}MODEL"))
                .or_else(|_| env::var(format!("{prefix}ID")))
                .unwrap_or_default();
            let url = env::var(format!("{prefix}URL"))
                .or_else(|_| env::var(format!("{prefix}BACKEND_URL")))
                .unwrap_or_default();
            if !id.trim().is_empty() || !url.trim().is_empty() {
                models_raw.push(json!({
                    "name": tier,
                    "role": tier,
                    "id": if id.trim().is_empty() { tier } else { id.trim() },
                    "base_url": url.trim(),
                }));
            }
        }
    }

    let external = env::var("SWITCHYARD_SERVER_URL")
        .or_else(|_| env::var("SWITCHYARD_EXTERNAL_URL"))
        .unwrap_or_default()
        .trim()
        .to_string();

    let explicit = env::var("SWITCHYARD_ENABLED").unwrap_or_default();
    let enabled = if !explicit.trim().is_empty() {
        matches!(
            explicit.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    } else {
        !models_raw.is_empty()
            || !external.is_empty()
            || file_data
                .get("enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
    };

    let mut strategy = env::var("SWITCHYARD_STRATEGY")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            file_data
                .get("strategy")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "escalation".to_string())
        .trim()
        .to_ascii_lowercase();
    if matches!(strategy.as_str(), "escalate" | "escalation_router") {
        strategy = "escalation".to_string();
    }

    let route_id = env::var("SWITCHYARD_ROUTE_ID")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            file_data
                .get("route_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "switchyard/agent".to_string());

    let mut models: Vec<ModelSpec> = models_raw
        .iter()
        .enumerate()
        .map(|(i, m)| model_spec_from_dict(m, i))
        .collect();

    if enabled && models.is_empty() && !default_model_id.is_empty() {
        let mut dep = DeploymentParams::default();
        if !default_backend_url.is_empty() {
            dep.backend_urls = vec![default_backend_url.trim_end_matches('/').to_string()];
        }
        models.push(ModelSpec {
            name: "default".to_string(),
            id: default_model_id.to_string(),
            role: "weak".to_string(),
            deployment: dep,
            ..Default::default()
        });
    }

    let esc_raw = file_data.get("escalation");
    let confirmations = env_int(
        "SWITCHYARD_CONFIRMATIONS",
        esc_raw
            .and_then(|e| e.get("confirmations"))
            .and_then(|v| v.as_i64())
            .unwrap_or(2),
    )
    .max(1) as usize;

    SwitchyardConfig {
        enabled,
        strategy: if !external.is_empty() && models.is_empty() {
            "external".to_string()
        } else {
            strategy
        },
        route_id,
        expose_models: env_bool("SWITCHYARD_EXPOSE_MODELS", true),
        allow_model_select: env_bool("SWITCHYARD_ALLOW_MODEL_SELECT", true),
        models,
        escalation: EscalationSettings {
            confirmations,
            ..Default::default()
        },
        external_url: external,
        external_api_key_env: env::var("SWITCHYARD_EXTERNAL_API_KEY_ENV").unwrap_or_default(),
        routes_toml_path: env::var("SWITCHYARD_ROUTES_TOML").unwrap_or_default(),
        owned_by: owned_by.to_string(),
    }
}

pub fn get_switchyard_config(
    default_model_id: &str,
    default_backend_url: &str,
    owned_by: &str,
) -> SwitchyardConfig {
    load_switchyard_config(default_model_id, default_backend_url, owned_by)
}
