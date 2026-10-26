//! DuckDuckGo HTML web search for chat and voice apps.

use crate::{env_bool, env_int};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use std::time::Duration;
use url::Url;

static SEARCH_PREFIX_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)^\s*/search\s+(.+)$").unwrap());

static SEARCH_INTENT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?is)\b(?:search\s+(?:the\s+)?(?:web|internet|online)\s+for|search\s+for|look\s+up|google\s+|bing\s+|find\s+(?:online|on\s+the\s+web)|what'?s\s+the\s+latest\s+on|latest\s+news\s+on|current\s+(?:status|price|version)\s+of)\b\s*(?P<q>.+)$",
    )
    .unwrap()
});

static STRIP_FILLER_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:help\s+me\s+)?").unwrap());

static TRAIL_PUNCT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[?.!]+$").unwrap());

static TAG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"<[^>]+>").unwrap());

static ANCHOR_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>"#)
        .unwrap()
});

static SNIPPET_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)<(?:a|div)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>"#)
        .unwrap()
});

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub rank: usize,
}

impl SearchResult {
    pub fn prompt_line(&self) -> String {
        let mut parts = vec![format!(
            "{}. {}",
            self.rank,
            if self.title.is_empty() {
                "(no title)"
            } else {
                &self.title
            }
        )];
        if !self.url.is_empty() {
            parts.push(format!("   URL: {}", self.url));
        }
        if !self.snippet.is_empty() {
            parts.push(format!("   {}", self.snippet));
        }
        parts.join("\n")
    }
}

pub fn extract_search_query(text: &str) -> Option<String> {
    let raw = text.trim();
    if raw.is_empty() {
        return None;
    }
    if let Some(caps) = SEARCH_PREFIX_RE.captures(raw) {
        let q = caps.get(1)?.as_str().trim();
        if !q.is_empty() {
            return Some(q.to_string());
        }
    }
    if let Some(caps) = SEARCH_INTENT_RE.captures(raw) {
        let mut q = caps.name("q")?.as_str().trim().to_string();
        q = STRIP_FILLER_RE.replace(&q, "").trim().to_string();
        q = q.trim_matches(&[' ', '\t', '\r', '\n', '"', '\''][..]).to_string();
        q = TRAIL_PUNCT_RE.replace(&q, "").trim().to_string();
        if q.len() >= 2 {
            return Some(q);
        }
    }
    None
}

fn unwrap_ddg_redirect(href: &str) -> String {
    let mut h = html_escape::decode_html_entities(href.trim()).into_owned();
    if h.is_empty() {
        return String::new();
    }
    if h.starts_with("//") {
        h = format!("https:{h}");
    }
    if let Ok(parsed) = Url::parse(&h) {
        let host = parsed.host_str().unwrap_or("");
        if host.contains("duckduckgo.com")
            && (parsed.path().starts_with("/l/")
                || parsed.query().unwrap_or("").contains("uddg="))
        {
            for (k, v) in parsed.query_pairs() {
                if k == "uddg" || k == "u" {
                    if !v.is_empty() {
                        return v.into_owned();
                    }
                }
            }
        }
        if matches!(parsed.scheme(), "http" | "https") && !host.is_empty() {
            return h;
        }
    }
    if h.starts_with("http://") || h.starts_with("https://") {
        return h;
    }
    String::new()
}

fn clean_html_fragment(s: &str) -> String {
    let without_tags = TAG_RE.replace_all(s, " ");
    let decoded = html_escape::decode_html_entities(&without_tags);
    decoded.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn parse_ddg_html(html_text: &str, max_results: usize) -> Vec<SearchResult> {
    if html_text.trim().is_empty() {
        return Vec::new();
    }
    let snippets: Vec<String> = SNIPPET_RE
        .captures_iter(html_text)
        .filter_map(|c| c.get(1).map(|m| clean_html_fragment(m.as_str())))
        .collect();

    let mut out = Vec::new();
    for (idx, caps) in ANCHOR_RE.captures_iter(html_text).enumerate() {
        if out.len() >= max_results.max(1) {
            break;
        }
        let href = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        let url = unwrap_ddg_redirect(href);
        if url.is_empty() {
            continue;
        }
        let title = caps
            .get(2)
            .map(|m| clean_html_fragment(m.as_str()))
            .unwrap_or_default();
        let snippet = snippets.get(idx).cloned().unwrap_or_default();
        out.push(SearchResult {
            title,
            url,
            snippet,
            rank: out.len() + 1,
        });
    }
    out
}

#[derive(Default, Debug)]
struct SearchState {
    searches: u64,
    failures: u64,
    last_error: String,
}

#[derive(Clone, Debug)]
pub struct WebSearchTool {
    pub app_name: String,
    pub enabled: bool,
    pub max_results: usize,
    pub timeout: f64,
    pub user_agent: String,
    pub endpoint: String,
    state: Arc<Mutex<SearchState>>,
}

impl WebSearchTool {
    pub fn new(app_name: &str) -> Self {
        let enabled = env_bool("WEB_SEARCH_ENABLED", env_bool("WEB_ACCESS_ENABLED", true));
        let max_results = env_int("WEB_SEARCH_MAX_RESULTS", 5).clamp(1, 10) as usize;
        let timeout = env::var("WEB_SEARCH_TIMEOUT")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(12.0);
        let user_agent = env::var("WEB_SEARCH_USER_AGENT").unwrap_or_else(|_| {
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36".to_string()
        });
        let endpoint = env::var("WEB_SEARCH_DDG_URL")
            .unwrap_or_else(|_| "https://html.duckduckgo.com/html/".to_string());
        Self {
            app_name: app_name.to_string(),
            enabled,
            max_results,
            timeout,
            user_agent,
            endpoint,
            state: Arc::new(Mutex::new(SearchState::default())),
        }
    }

    pub fn last_error(&self) -> String {
        self.state.lock().last_error.clone()
    }

    pub fn search(&self, query: &str, max_results: Option<usize>) -> Vec<SearchResult> {
        if !self.enabled {
            self.state.lock().last_error = "web search disabled".to_string();
            return Vec::new();
        }
        let q = query.trim();
        if q.is_empty() {
            self.state.lock().last_error = "empty query".to_string();
            return Vec::new();
        }
        let limit = max_results.unwrap_or(self.max_results).clamp(1, 10);
        let Ok(client) = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs_f64(self.timeout.max(1.0)))
            .build()
        else {
            return Vec::new();
        };
        let mut form = HashMap::new();
        form.insert("q", q);
        form.insert("b", "");
        form.insert("kl", "us-en");

        match client
            .post(&self.endpoint)
            .header("User-Agent", &self.user_agent)
            .header("Accept", "text/html,application/xhtml+xml")
            .header("Accept-Language", "en-US,en;q=0.9")
            .form(&form)
            .send()
        {
            Ok(resp) => {
                if !resp.status().is_success() {
                    let mut st = self.state.lock();
                    st.failures += 1;
                    st.last_error = format!("HTTP {}", resp.status().as_u16());
                    return Vec::new();
                }
                let html = resp.text().unwrap_or_default();
                let results = parse_ddg_html(&html, limit);
                let mut st = self.state.lock();
                st.searches += 1;
                if results.is_empty() {
                    st.failures += 1;
                    st.last_error = "no results parsed (blocked or empty SERP)".to_string();
                } else {
                    st.last_error.clear();
                }
                results
            }
            Err(e) => {
                let mut st = self.state.lock();
                st.failures += 1;
                st.last_error = e.to_string();
                Vec::new()
            }
        }
    }

    pub fn format_context(&self, query: &str, results: &[SearchResult], error: &str) -> String {
        let mut lines = vec![
            "The user asked for web search. Results from DuckDuckGo HTML:".to_string(),
            format!("Query: {query}"),
        ];
        if !error.is_empty() && results.is_empty() {
            lines.push(format!("Search error: {error}"));
            lines.push("Answer from model knowledge and note that live search failed.".to_string());
            return lines.join("\n");
        }
        if results.is_empty() {
            lines.push("No results returned.".to_string());
            return lines.join("\n");
        }
        lines.push("Results:".to_string());
        for r in results {
            lines.push(r.prompt_line());
        }
        lines.push("Use these results when answering. Cite titles/URLs when relevant.".to_string());
        lines.join("\n")
    }

    pub fn stats(&self) -> Value {
        let st = self.state.lock();
        json!({
            "enabled": self.enabled,
            "app_name": self.app_name,
            "endpoint": self.endpoint,
            "max_results": self.max_results,
            "timeout": self.timeout,
            "searches": st.searches,
            "failures": st.failures,
            "last_error": if st.last_error.is_empty() { Value::Null } else { json!(st.last_error) },
        })
    }
}

pub fn get_web_search_tool(app_name: &str) -> WebSearchTool {
    WebSearchTool::new(app_name)
}
