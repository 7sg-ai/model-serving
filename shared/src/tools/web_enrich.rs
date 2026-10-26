//! Orchestrate web search + URL fetch for chat/voice apps.

use crate::tools::web_access::{
    extract_urls, get_web_access_tool, FetchResult, WebAccessTool,
};
use crate::tools::web_search::{
    extract_search_query, get_web_search_tool, SearchResult, WebSearchTool,
};
use crate::{env_bool, env_int};
use serde_json::{json, Value};

#[derive(Clone, Debug, Default)]
pub struct WebEnrichment {
    pub query: String,
    pub search_results: Vec<SearchResult>,
    pub search_error: String,
    pub fetch_results: Vec<FetchResult>,
    pub context: String,
}

impl WebEnrichment {
    pub fn ok(&self) -> bool {
        !self.context.is_empty()
    }

    pub fn ui_search_rows(&self) -> Vec<Value> {
        self.search_results
            .iter()
            .map(|r| {
                json!({
                    "rank": r.rank,
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                })
            })
            .collect()
    }

    pub fn ui_fetch_rows(&self) -> Vec<Value> {
        self.fetch_results
            .iter()
            .map(|wr| {
                json!({
                    "url": wr.url,
                    "ok": wr.ok,
                    "status_code": wr.status_code,
                    "title": wr.title,
                    "log_path": wr.log_path,
                    "content_sha256": wr.content_sha256,
                    "raw_length": wr.raw_length,
                    "error": wr.error,
                })
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct WebEnricher {
    pub app_name: String,
    pub access: WebAccessTool,
    pub search: WebSearchTool,
    pub fetch_top_n: usize,
    pub fetch_search_pages: bool,
}

impl WebEnricher {
    pub fn new(app_name: &str) -> Self {
        Self {
            app_name: app_name.to_string(),
            access: get_web_access_tool(app_name),
            search: get_web_search_tool(app_name),
            fetch_top_n: env_int("WEB_SEARCH_FETCH_TOP", 2).max(0) as usize,
            fetch_search_pages: env_bool("WEB_SEARCH_FETCH_PAGES", true),
        }
    }

    pub fn enrich_user_text(&self, text: &str, session_id: &str) -> WebEnrichment {
        let mut out = WebEnrichment::default();
        if text.is_empty() {
            return out;
        }
        let mut blocks = Vec::new();

        if let Some(query) = extract_search_query(text) {
            if self.search.enabled {
                out.query = query.clone();
                let results = self.search.search(&query, None);
                out.search_results = results.clone();
                out.search_error = self.search.last_error();
                blocks.push(self.search.format_context(&query, &results, &out.search_error));

                if self.fetch_search_pages
                    && !results.is_empty()
                    && self.access.enabled
                    && !self.access.terminated()
                {
                    let top_urls: Vec<String> = results
                        .iter()
                        .take(self.fetch_top_n)
                        .map(|r| r.url.clone())
                        .filter(|u| !u.is_empty())
                        .collect();
                    if !top_urls.is_empty() {
                        let page_results = self.access.fetch_urls(&top_urls, session_id);
                        let page_blocks: Vec<String> = page_results
                            .iter()
                            .map(|r| r.prompt_block(self.access.max_prompt_chars))
                            .collect();
                        out.fetch_results.extend(page_results);
                        if !page_blocks.is_empty() {
                            blocks.push(format!(
                                "Fetched top search result page(s) for deeper context:\n\n{}",
                                page_blocks.join("\n\n")
                            ));
                        }
                    }
                }
            }
        }

        let explicit = extract_urls(text);
        if !explicit.is_empty() && self.access.enabled && !self.access.terminated() {
            let new_urls: Vec<String> = explicit
                .into_iter()
                .filter(|u| !out.fetch_results.iter().any(|r| &r.url == u))
                .collect();
            if !new_urls.is_empty() {
                let page_results = self.access.fetch_urls(&new_urls, session_id);
                let page_blocks: Vec<String> = page_results
                    .iter()
                    .map(|r| r.prompt_block(self.access.max_prompt_chars))
                    .collect();
                out.fetch_results.extend(page_results);
                if !page_blocks.is_empty() {
                    blocks.push(format!(
                        "The user referenced web URL(s). Full page contents were fetched and logged server-side. Use the following extracted content:\n\n{}",
                        page_blocks.join("\n\n")
                    ));
                }
            }
        }

        out.context = blocks.join("\n\n").trim().to_string();
        out
    }

    pub fn enrich_messages(
        &self,
        messages: &[Value],
        user_text: Option<&str>,
        session_id: &str,
    ) -> (Vec<Value>, WebEnrichment) {
        let resolved_text = user_text
            .map(|s| s.to_string())
            .or_else(|| {
                messages
                    .iter()
                    .rev()
                    .find(|m| m.get("role").and_then(|v| v.as_str()) == Some("user"))
                    .and_then(|m| m.get("content").and_then(|v| v.as_str()))
                    .map(|s| s.to_string())
            })
            .unwrap_or_default();

        let enrichment = self.enrich_user_text(&resolved_text, session_id);
        if enrichment.context.is_empty() {
            return (messages.to_vec(), enrichment);
        }
        let mut enriched = messages.to_vec();
        let insert_at = enriched
            .iter()
            .rposition(|m| m.get("role").and_then(|v| v.as_str()) == Some("user"))
            .unwrap_or(enriched.len());
        enriched.insert(
            insert_at,
            json!({
                "role": "system",
                "content": enrichment.context,
            }),
        );
        (enriched, enrichment)
    }

    pub fn stats(&self) -> Value {
        json!({
            "app_name": self.app_name,
            "search": self.search.stats(),
            "access": self.access.stats(),
            "fetch_top_n": self.fetch_top_n,
            "fetch_search_pages": self.fetch_search_pages,
        })
    }
}

pub fn get_web_enricher(app_name: &str) -> WebEnricher {
    WebEnricher::new(app_name)
}
