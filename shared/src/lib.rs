pub mod chat_server;
pub mod ide_server;
pub mod voice_server;

pub mod auth;
pub mod backends;
pub mod cache;
pub mod database;
pub mod deploy;
pub mod model_catalog;
pub mod multinode;
pub mod switchyard;
pub mod tools;

use std::env;

pub fn env_bool(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(raw) => {
            let s = raw.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => default,
    }
}

pub fn env_int(name: &str, default: i64) -> i64 {
    match env::var(name) {
        Ok(raw) => {
            let s = raw.trim();
            if s.is_empty() {
                default
            } else {
                s.parse::<i64>().unwrap_or(default)
            }
        }
        Err(_) => default,
    }
}

pub fn env_float(name: &str, default: f64) -> f64 {
    match env::var(name) {
        Ok(raw) => {
            let s = raw.trim();
            if s.is_empty() {
                default
            } else {
                s.parse::<f64>().unwrap_or(default)
            }
        }
        Err(_) => default,
    }
}

pub fn split_csv(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}
