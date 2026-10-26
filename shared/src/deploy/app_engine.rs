//! Resolve which inference engine an app should auto-deploy.

use crate::deploy::engines::normalize_engine;
use std::env;

pub fn resolve_app_engine(family: &str) -> String {
    let fam = family.trim().to_ascii_lowercase();
    if let Ok(explicit) = env::var("INFERENCE_ENGINE").or_else(|_| env::var("MODEL_ENGINE")) {
        if !explicit.trim().is_empty() {
            return normalize_engine(Some(&explicit), "vllm");
        }
    }
    if fam == "nim" {
        return "nim".to_string();
    }
    let backend = env::var("HF_BACKEND")
        .or_else(|_| env::var("HF_ENGINE"))
        .unwrap_or_else(|_| "vllm".to_string())
        .trim()
        .to_ascii_lowercase();
    if matches!(backend.as_str(), "sglang" | "sg" | "sgl") {
        "sglang".to_string()
    } else {
        "vllm".to_string()
    }
}
