//! Export SwitchyardConfig to a NeMo Switchyard routes.toml for switchyard-server.

use crate::switchyard::config::SwitchyardConfig;
use regex::Regex;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

fn toml_str(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

fn safe_key(name: &str) -> String {
    let re = Regex::new(r"[^A-Za-z0-9_]").unwrap();
    let mut key = re
        .replace_all(if name.is_empty() { "model" } else { name }, "_")
        .into_owned();
    if key
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
    {
        key = format!("m_{key}");
    }
    if key.is_empty() {
        "model".to_string()
    } else {
        key
    }
}

pub fn export_routes_toml(cfg: &SwitchyardConfig) -> String {
    let mut lines = vec![
        "# Auto-generated from model-serving Switchyard config".to_string(),
        "schema_version = 1".to_string(),
        "".to_string(),
    ];

    let mut clients: BTreeMap<String, (String, String, String)> = BTreeMap::new();
    let mut target_client: BTreeMap<String, String> = BTreeMap::new();
    let proto_re = Regex::new(r"https?://").unwrap();

    for m in &cfg.models {
        let mut base = m.openai_base_url();
        if base.is_empty() {
            base = "http://127.0.0.1:8000/v1".to_string();
        }
        let stripped = proto_re.replace_all(&base, "");
        let client_key = safe_key(&stripped.replace(['/', '.', ':'], "_"));
        clients.entry(client_key.clone()).or_insert_with(|| {
            let fmt = if m.format.is_empty() {
                "openai_chat".to_string()
            } else {
                m.format.clone()
            };
            (fmt, base, m.api_key_env.clone())
        });
        target_client.insert(m.name.clone(), client_key);
    }

    for (ckey, (fmt, base_url, api_key_env)) in &clients {
        lines.push(format!("[llm_clients.{ckey}]"));
        lines.push(format!("format = {}", toml_str(fmt)));
        lines.push(format!("base_url = {}", toml_str(base_url)));
        if !api_key_env.is_empty() {
            lines.push(format!("api_key_env = {}", toml_str(api_key_env)));
        }
        lines.push("max_retries = 2".to_string());
        lines.push("".to_string());
    }

    for m in &cfg.models {
        let tkey = safe_key(&m.name);
        let ckey = target_client.get(&m.name).cloned().unwrap_or_default();
        lines.push(format!("[targets.{tkey}]"));
        lines.push(format!("id = {}", toml_str(&m.id)));
        lines.push(format!("llm_client = {}", toml_str(&ckey)));
        lines.push("".to_string());
    }

    let weak = cfg.weak();
    let strong = cfg.strong();
    let judge = cfg.judge();
    let route_key = safe_key(&cfg.route_id.replace('/', "_"));

    if cfg.strategy == "escalation" && weak.is_some() && strong.is_some() {
        let w = weak.unwrap();
        let s = strong.unwrap();
        let j = judge.unwrap_or(w);
        lines.push(format!("[routes.{route_key}]"));
        lines.push(format!("id = {}", toml_str(&cfg.route_id)));
        lines.push("type = \"llm_classifier\"".to_string());
        lines.push("mode = \"escalation\"".to_string());
        lines.push(format!("classifier_target = {}", toml_str(&safe_key(&j.name))));
        lines.push(format!("strong_target = {}", toml_str(&safe_key(&s.name))));
        lines.push(format!("weak_target = {}", toml_str(&safe_key(&w.name))));
        lines.push(format!("max_output_tokens = {}", cfg.escalation.max_output_tokens));
        lines.push("".to_string());
    } else if let Some(m) = cfg.default_model().or_else(|| cfg.models.first()) {
        lines.push(format!("[routes.{route_key}]"));
        lines.push(format!("id = {}", toml_str(&cfg.route_id)));
        lines.push("type = \"passthrough\"".to_string());
        lines.push(format!("target = {}", toml_str(&safe_key(&m.name))));
        lines.push("".to_string());
    }

    for m in cfg.selectable_models() {
        let pkey = safe_key(&format!("direct_{}", m.name));
        lines.push(format!("[routes.{pkey}]"));
        lines.push(format!("id = {}", toml_str(&m.id)));
        lines.push("type = \"passthrough\"".to_string());
        lines.push(format!("target = {}", toml_str(&safe_key(&m.name))));
        if m.context_window > 0 {
            lines.push(format!("context_window = {}", m.context_window));
        }
        lines.push("".to_string());
    }

    format!("{}\n", lines.join("\n").trim_end())
}

pub fn write_routes_toml(cfg: &SwitchyardConfig, path: Option<&str>) -> String {
    let raw_path = path
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .or_else(|| {
            if !cfg.routes_toml_path.is_empty() {
                Some(cfg.routes_toml_path.clone())
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join("switchyard-routes.toml")
                .to_string_lossy()
                .into_owned()
        });
    let pb = PathBuf::from(&raw_path);
    if let Some(parent) = pb.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
    }
    let content = export_routes_toml(cfg);
    let _ = fs::write(&pb, content);
    raw_path
}
