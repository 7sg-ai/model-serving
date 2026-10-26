//! Inference engines shared by HF and NIM apps.

use regex::Regex;
use std::env;

pub const ENGINES: &[&str] = &["nim", "vllm", "sglang"];
pub const VLLM_IMAGE: &str = "vllm/vllm-openai:latest";
pub const SGLANG_IMAGE: &str = "lmsysorg/sglang:latest-runtime";
pub const SGLANG_CONTAINER_PORT: u16 = 30000;
pub const NIM_CATALOG_IMAGE: &str = "nvcr.io/nim/meta/llama-3.1-8b-instruct:latest";
pub const NIM_VLLM_MODEL_FREE_IMAGE: &str = "nvcr.io/nim/nvidia/vllm-model-free-nim:latest";
pub const NIM_SGLANG_MODEL_FREE_IMAGE: &str = "nvcr.io/nim/nvidia/sglang-model-free-nim:latest";

const MODEL_FREE_MARKERS: &[&str] = &[
    "model-free",
    "modelfree",
    "sglang-model-free",
    "vllm-model-free",
];

pub fn normalize_engine(engine: Option<&str>, default: &str) -> String {
    let mut val = engine
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(default)
        .trim()
        .to_ascii_lowercase();
    if matches!(val.as_str(), "sg" | "sgl") {
        val = "sglang".to_string();
    }
    if !ENGINES.contains(&val.as_str()) {
        "vllm".to_string()
    } else {
        val
    }
}

pub fn is_model_free_nim_image(image: &str) -> bool {
    let low = image.trim().to_ascii_lowercase();
    if low.is_empty() {
        return false;
    }
    MODEL_FREE_MARKERS.iter().any(|m| low.contains(m))
}

pub fn nim_model_path(model_id: &str, explicit: &str) -> String {
    let raw = explicit.trim();
    if !raw.is_empty() {
        return raw.to_string();
    }
    let model = model_id.trim();
    if model.is_empty() {
        return String::new();
    }
    if model.contains("://") || model.starts_with('/') {
        return model.to_string();
    }
    format!("hf://{model}")
}

pub fn served_model_name(model_id: &str, explicit: &str) -> String {
    let mut name = if !explicit.trim().is_empty() {
        explicit.trim().to_string()
    } else {
        model_id.trim().to_string()
    };
    if let Some(stripped) = name.strip_prefix("hf://") {
        name = stripped.to_string();
    }
    name
}

pub fn default_image(engine: &str) -> String {
    let eng = normalize_engine(Some(engine), "vllm");
    if eng == "sglang" {
        return env::var("SGLANG_IMAGE")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| SGLANG_IMAGE.to_string());
    }
    if eng == "vllm" {
        return env::var("VLLM_IMAGE")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| VLLM_IMAGE.to_string());
    }
    if let Ok(explicit) = env::var("NIM_IMAGE") {
        if !explicit.trim().is_empty() {
            return explicit.trim().to_string();
        }
    }
    let backend = env::var("NIM_BACKEND")
        .or_else(|_| env::var("NIM_ENGINE"))
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if matches!(backend.as_str(), "sglang" | "sg" | "sgl") {
        return NIM_SGLANG_MODEL_FREE_IMAGE.to_string();
    }
    if matches!(
        backend.as_str(),
        "vllm" | "model-free" | "modelfree" | "model_free"
    ) {
        return NIM_VLLM_MODEL_FREE_IMAGE.to_string();
    }
    NIM_CATALOG_IMAGE.to_string()
}

pub fn default_model_id(engine: &str) -> String {
    let eng = normalize_engine(Some(engine), "vllm");
    if eng == "nim" {
        return env::var("NIM_MODEL_NAME")
            .unwrap_or_else(|_| "meta/llama-3.1-8b-instruct".to_string());
    }
    let key = if eng == "sglang" {
        "SGLANG_MODEL"
    } else {
        "VLLM_MODEL"
    };
    env::var(key)
        .or_else(|_| env::var("HF_MODEL_NAME"))
        .unwrap_or_else(|_| "meta-llama/Llama-3.1-8B-Instruct".to_string())
}

pub fn default_port(engine: &str, fallback: u16) -> u16 {
    let eng = normalize_engine(Some(engine), "vllm");
    let name = match eng.as_str() {
        "nim" => "NIM_PORT",
        "sglang" => "SGLANG_PORT",
        _ => "VLLM_PORT",
    };
    env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<u16>().ok())
        .unwrap_or(fallback)
}

pub fn container_port(engine: &str, image: &str) -> u16 {
    let eng = normalize_engine(Some(engine), "vllm");
    if eng == "sglang" && !is_model_free_nim_image(image) {
        if let Ok(raw) = env::var("SGLANG_CONTAINER_PORT") {
            if let Ok(p) = raw.trim().parse::<u16>() {
                return p;
            }
        }
        return SGLANG_CONTAINER_PORT;
    }
    8000
}

pub fn service_prefix(engine: &str) -> String {
    normalize_engine(Some(engine), "vllm")
}

pub fn yaml_quote(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

pub fn slug(value: &str) -> String {
    let re = Regex::new(r"[^a-zA-Z0-9_.-]+").unwrap();
    let s = re
        .replace_all(value.trim().to_ascii_lowercase().as_str(), "-")
        .trim_matches('-')
        .to_string();
    if s.is_empty() {
        "app".to_string()
    } else {
        s
    }
}
