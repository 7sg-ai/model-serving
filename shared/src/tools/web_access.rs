//! Internet access tool for chat and voice apps.

use crate::{env_bool, env_int};
use chrono::Utc;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

static URL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)https?://[^\s<>'\x22\)\]\}]+" ).unwrap());
static SIZE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)^\s*([0-9]+(?:\.[0-9]+)?)\s*([kmgt]i?b?)\s*$").unwrap());
static SCRIPT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<(?:script|style|noscript)[^>]*>.*?</(?:script|style|noscript)>").unwrap());
static TITLE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<title[^>]*>(.*?)</title>").unwrap());
static TAG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"<[^>]+>").unwrap());

const TRAIL_PUNCT: &[char] = &[
    '.', ',', ';', ':', '!', '?', ')', ']', '}', '\'', '"', '…',
];

pub fn parse_size(value: &str, default_bytes: u64) -> u64 {
    let raw = value.trim();
    if raw.is_empty() {
        return default_bytes;
    }
    if let Ok(n) = raw.parse::<u64>() {
        return n;
    }
    let Some(caps) = SIZE_RE.captures(raw) else {
        return default_bytes;
    };
    let Ok(amount) = caps.get(1).unwrap().as_str().parse::<f64>() else {
        return default_bytes;
    };
    let unit = caps.get(2).unwrap().as_str().to_ascii_lowercase();
    let mult: f64 = match unit.as_str() {
        "k" | "ki" | "kib" => 1024.0,
        "m" | "mi" | "mib" => 1024.0 * 1024.0,
        "g" | "gi" | "gib" => 1024.0 * 1024.0 * 1024.0,
        "t" | "ti" | "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        "kb" => 1000.0,
        "mb" => 1_000_000.0,
        "gb" => 1_000_000_000.0,
        "tb" => 1_000_000_000_000.0,
        _ => return default_bytes,
    };
    (amount * mult) as u64
}

pub fn format_size(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const KIB: f64 = 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.2} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.2} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

pub fn extract_urls(text: &str) -> Vec<String> {
    let mut seen = Vec::new();
    for m in URL_RE.find_iter(text) {
        let u = m.as_str().trim_end_matches(TRAIL_PUNCT).to_string();
        if !u.is_empty() && !seen.contains(&u) {
            seen.push(u);
        }
    }
    seen
}

pub fn html_to_text(raw_html: &str) -> (String, String) {
    let title = TITLE_RE
        .captures(raw_html)
        .and_then(|c| c.get(1))
        .map(|m| {
            let t = TAG_RE.replace_all(m.as_str(), " ");
            html_escape::decode_html_entities(&t)
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();

    let no_scripts = SCRIPT_RE.replace_all(raw_html, " ");
    let no_tags = TAG_RE.replace_all(&no_scripts, " ");
    let decoded = html_escape::decode_html_entities(&no_tags);
    let text = decoded.split_whitespace().collect::<Vec<_>>().join(" ");
    (title, text)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FetchResult {
    pub url: String,
    pub ok: bool,
    pub status_code: u16,
    pub title: String,
    pub text: String,
    pub raw_length: usize,
    pub content_sha256: String,
    pub log_path: String,
    pub error: Option<String>,
}

impl FetchResult {
    pub fn prompt_block(&self, max_chars: usize) -> String {
        if !self.ok {
            return format!(
                "[URL: {}] (fetch failed: {})",
                self.url,
                self.error.as_deref().unwrap_or("unknown error")
            );
        }
        let snippet: String = self.text.chars().take(max_chars.max(100)).collect();
        format!(
            "[URL: {}]\nTitle: {}\nContent:\n{}",
            self.url,
            if self.title.is_empty() {
                "(none)"
            } else {
                &self.title
            },
            snippet
        )
    }
}

#[derive(Default, Debug)]
struct AccessState {
    terminated: bool,
    terminate_reason: String,
    fetches: u64,
    failures: u64,
}

#[derive(Clone, Debug)]
pub struct WebAccessTool {
    pub app_name: String,
    pub enabled: bool,
    pub log_dir: PathBuf,
    pub log_file_path: String,
    pub log_max_bytes: u64,
    pub timeout: f64,
    pub max_bytes: usize,
    pub max_prompt_chars: usize,
    pub max_urls_per_turn: usize,
    state: Arc<Mutex<AccessState>>,
}

impl WebAccessTool {
    pub fn new(app_name: &str) -> Self {
        let enabled = env_bool("WEB_ACCESS_ENABLED", true);
        let shared = env::var("SHARED_DATA_DIR").unwrap_or_else(|_| "shared/database".to_string());
        let log_dir_raw = env::var("WEB_ACCESS_LOG_DIR").unwrap_or_else(|_| {
            Path::new(&shared)
                .join("web_access_logs")
                .to_string_lossy()
                .into_owned()
        });
        let log_dir = PathBuf::from(&log_dir_raw);
        let _ = fs::create_dir_all(&log_dir);
        let log_file_path = env::var("WEB_ACCESS_LOG_FILE").unwrap_or_else(|_| {
            log_dir
                .join(format!("{app_name}.web_access.jsonl"))
                .to_string_lossy()
                .into_owned()
        });
        let max_str = env::var("WEB_ACCESS_LOG_MAX_BYTES")
            .or_else(|_| env::var("WEB_ACCESS_LOG_LIMIT"))
            .unwrap_or_else(|_| "1GiB".to_string());
        let log_max_bytes = parse_size(&max_str, 1024 * 1024 * 1024);

        Self {
            app_name: app_name.to_string(),
            enabled,
            log_dir,
            log_file_path,
            log_max_bytes,
            timeout: env::var("WEB_ACCESS_TIMEOUT")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(15.0),
            max_bytes: env_int("WEB_ACCESS_MAX_BYTES", 2 * 1024 * 1024).max(1024) as usize,
            max_prompt_chars: env_int("WEB_ACCESS_MAX_PROMPT_CHARS", 12000).max(500) as usize,
            max_urls_per_turn: env_int("WEB_ACCESS_MAX_URLS_PER_TURN", 5).max(1) as usize,
            state: Arc::new(Mutex::new(AccessState::default())),
        }
    }

    pub fn terminated(&self) -> bool {
        self.state.lock().terminated
    }

    fn log_usage_bytes(&self) -> u64 {
        fs::metadata(&self.log_file_path)
            .map(|m| m.len())
            .unwrap_or(0)
    }

    fn write_log(&self, entry: &Value) -> bool {
        let line = format!("{}\n", serde_json::to_string(entry).unwrap_or_default());
        let current = self.log_usage_bytes();
        if current + line.len() as u64 > self.log_max_bytes {
            let mut st = self.state.lock();
            st.terminated = true;
            st.terminate_reason = format!(
                "Web access log limit reached ({} / {})",
                format_size(current),
                format_size(self.log_max_bytes)
            );
            return false;
        }
        if let Some(parent) = Path::new(&self.log_file_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_file_path)
        {
            let _ = file.write_all(line.as_bytes());
            true
        } else {
            false
        }
    }
}

impl WebAccessTool {
    pub fn fetch_url(&self, url: &str, session_id: &str) -> FetchResult {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs_f64(self.timeout.max(1.0)))
            .build();
        let Ok(client) = client else {
            return FetchResult {
                url: url.to_string(),
                ok: false,
                status_code: 0,
                title: String::new(),
                text: String::new(),
                raw_length: 0,
                content_sha256: String::new(),
                log_path: self.log_file_path.clone(),
                error: Some("client build error".to_string()),
            };
        };

        match client
            .get(url)
            .header(
                "User-Agent",
                "Mozilla/5.0 (compatible; ModelServingWebAccess/1.0)",
            )
            .send()
        {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let ok = resp.status().is_success();
                let raw_body = resp.text().unwrap_or_default();
                let raw_length = raw_body.len();
                let mut hasher = Sha256::new();
                hasher.update(raw_body.as_bytes());
                let sha = format!("{:x}", hasher.finalize());
                let (title, text) = html_to_text(&raw_body);

                let entry = json!({
                    "ts": Utc::now().to_rfc3339(),
                    "app": self.app_name,
                    "session_id": session_id,
                    "url": url,
                    "ok": ok,
                    "status_code": status,
                    "title": title,
                    "raw_length": raw_length,
                    "content_sha256": sha,
                    "full_text": text,
                });
                self.write_log(&entry);
                let mut st = self.state.lock();
                st.fetches += 1;
                if !ok {
                    st.failures += 1;
                }
                FetchResult {
                    url: url.to_string(),
                    ok,
                    status_code: status,
                    title,
                    text,
                    raw_length,
                    content_sha256: sha,
                    log_path: self.log_file_path.clone(),
                    error: if ok { None } else { Some(format!("HTTP {status}")) },
                }
            }
            Err(e) => {
                self.state.lock().failures += 1;
                FetchResult {
                    url: url.to_string(),
                    ok: false,
                    status_code: 0,
                    title: String::new(),
                    text: String::new(),
                    raw_length: 0,
                    content_sha256: String::new(),
                    log_path: self.log_file_path.clone(),
                    error: Some(e.to_string()),
                }
            }
        }
    }

    pub fn fetch_urls(&self, urls: &[String], session_id: &str) -> Vec<FetchResult> {
        if !self.enabled || self.terminated() {
            return Vec::new();
        }
        urls.iter()
            .take(self.max_urls_per_turn)
            .map(|u| self.fetch_url(u, session_id))
            .collect()
    }

    pub fn stats(&self) -> Value {
        let usage = self.log_usage_bytes();
        let st = self.state.lock();
        json!({
            "enabled": self.enabled && !st.terminated,
            "terminated": st.terminated,
            "terminate_reason": if st.terminate_reason.is_empty() { Value::Null } else { json!(st.terminate_reason) },
            "app_name": self.app_name,
            "log_dir": self.log_dir.to_string_lossy(),
            "log_file": self.log_file_path,
            "log_usage_bytes": usage,
            "log_max_bytes": self.log_max_bytes,
            "log_usage": format_size(usage),
            "log_max": format_size(self.log_max_bytes),
            "fetches": st.fetches,
            "failures": st.failures,
            "timeout": self.timeout,
            "max_bytes": self.max_bytes,
            "max_prompt_chars": self.max_prompt_chars,
        })
    }
}

pub fn get_web_access_tool(app_name: &str) -> WebAccessTool {
    WebAccessTool::new(app_name)
}
