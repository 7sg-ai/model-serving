//! Backend health probes.

use anyhow::{anyhow, Result};
use std::thread;
use std::time::{Duration, Instant};
use url::Url;

pub fn models_url_from_chat_url(chat_url: &str) -> String {
    let u = chat_url.trim().trim_end_matches('/');
    if let Some(base) = u.strip_suffix("/chat/completions") {
        return format!("{base}/models");
    }
    if u.ends_with("/v1") {
        return format!("{u}/models");
    }
    if u.contains("/v1/") {
        if let Some((base, _)) = u.rsplit_once('/') {
            return format!("{base}/models");
        }
    }
    format!("{u}/v1/models")
}

pub fn probe_backend(chat_url: &str, timeout_secs: f64, api_key: &str) -> bool {
    let url = models_url_from_chat_url(chat_url);
    let Ok(client) = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs_f64(timeout_secs.max(1.0)))
        .build()
    else {
        return false;
    };
    let mut req = client.get(&url);
    if !api_key.trim().is_empty() {
        req = req.header("Authorization", format!("Bearer {}", api_key.trim()));
    }
    if let Ok(resp) = req.send() {
        if resp.status().as_u16() < 500 {
            return true;
        }
    }
    if let Ok(parsed) = Url::parse(chat_url) {
        if let Some(host) = parsed.host_str() {
            let port_part = parsed.port().map(|p| format!(":{p}")).unwrap_or_default();
            let root = format!("{}://{host}{port_part}/health", parsed.scheme());
            if let Ok(resp) = client.get(&root).send() {
                return resp.status().as_u16() < 500;
            }
        }
    }
    false
}

pub fn wait_until_healthy(
    urls: &[String],
    wait_seconds: u64,
    interval_secs: u64,
    api_key: &str,
) -> Result<Vec<String>> {
    let mut pending: Vec<String> = urls
        .iter()
        .filter(|u| !u.trim().is_empty())
        .cloned()
        .collect();
    if pending.is_empty() {
        return Ok(Vec::new());
    }
    let deadline = Instant::now() + Duration::from_secs(wait_seconds.max(1));
    let mut healthy = Vec::new();
    while !pending.is_empty() && Instant::now() < deadline {
        pending.retain(|u| {
            if probe_backend(u, 5.0, api_key) {
                healthy.push(u.clone());
                false
            } else {
                true
            }
        });
        if !pending.is_empty() {
            thread::sleep(Duration::from_secs(interval_secs.max(1)));
        }
    }
    if !pending.is_empty() {
        return Err(anyhow!(
            "Timed out waiting for backends to become healthy: {:?}",
            pending
        ));
    }
    Ok(healthy)
}

pub fn any_healthy(urls: &[String], api_key: &str) -> bool {
    urls.iter()
        .filter(|u| !u.trim().is_empty())
        .any(|u| probe_backend(u, 3.0, api_key))
}
