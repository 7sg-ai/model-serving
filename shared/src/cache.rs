//! Shared response cache (L1 LRU + optional Mooncake L2) and conversation memory.

use indexmap::IndexMap;
use parking_lot::RwLock;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
struct CacheEntry {
    value: Value,
    ts: Instant,
}

#[derive(Debug)]
struct CacheInner {
    store: IndexMap<String, CacheEntry>,
    hits: u64,
    misses: u64,
    mooncake_hits: u64,
    mooncake_misses: u64,
    mooncake_writes: u64,
}

pub fn is_moonshot_model(model_name: &str) -> bool {
    let name = model_name.trim().to_ascii_lowercase();
    if name.is_empty() {
        return false;
    }
    name.starts_with("moonshotai/")
        || name.starts_with("moonshot/")
        || name.contains("kimi")
        || name.starts_with("moonshotai")
}

fn canonicalize_value(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut sorted = BTreeMap::new();
            for (k, val) in map {
                sorted.insert(k.clone(), canonicalize_value(val));
            }
            let mut out = serde_json::Map::new();
            for (k, val) in sorted {
                out.insert(k, val);
            }
            Value::Object(out)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(canonicalize_value).collect()),
        other => other.clone(),
    }
}

#[derive(Clone, Debug)]
pub struct ResponseCache {
    pub max_size: usize,
    pub ttl_seconds: u64,
    pub enabled: bool,
    pub use_mooncake: bool,
    pub mooncake_enabled: bool,
    pub mooncake_error: Option<String>,
    pub key_prefix: String,
    inner: Arc<RwLock<CacheInner>>,
}

impl ResponseCache {
    pub fn new(max_size: usize, ttl_seconds: u64, enabled: bool) -> Self {
        Self {
            max_size: max_size.max(1),
            ttl_seconds: ttl_seconds.max(1),
            enabled,
            use_mooncake: false,
            mooncake_enabled: false,
            mooncake_error: None,
            key_prefix: "ide".to_string(),
            inner: Arc::new(RwLock::new(CacheInner {
                store: IndexMap::new(),
                hits: 0,
                misses: 0,
                mooncake_hits: 0,
                mooncake_misses: 0,
                mooncake_writes: 0,
            })),
        }
    }

    pub fn with_mooncake(
        max_size: usize,
        ttl_seconds: u64,
        enabled: bool,
        key_prefix: String,
    ) -> Self {
        let mut cache = Self::new(max_size, ttl_seconds, enabled);
        cache.use_mooncake = true;
        cache.key_prefix = key_prefix.trim_end_matches(':').to_string();
        if enabled {
            cache.mooncake_enabled = false;
            cache.mooncake_error =
                Some("mooncake-transfer-engine not connected (L1 fallback)".to_string());
        }
        cache
    }

    pub fn make_key(
        model: &str,
        messages: &[Value],
        temperature: f64,
        max_tokens: usize,
        extra: Option<&Value>,
    ) -> String {
        let rounded_temp = (temperature * 10000.0).round() / 10000.0;
        let mut payload = json!({
            "max_tokens": max_tokens,
            "messages": messages,
            "model": model,
            "temperature": rounded_temp,
        });
        if let Some(ex) = extra {
            if !ex.is_null() && ex.as_object().map(|o| !o.is_empty()).unwrap_or(true) {
                payload["extra"] = ex.clone();
            }
        }
        let canonical = canonicalize_value(&payload);
        let raw = serde_json::to_string(&canonical).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(raw.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn get(&self, key: &str) -> Option<Value> {
        if !self.enabled {
            return None;
        }
        let mut guard = self.inner.write();
        if let Some(entry) = guard.store.shift_remove(key) {
            if entry.ts.elapsed() <= Duration::from_secs(self.ttl_seconds) {
                let val = entry.value.clone();
                guard.store.insert(key.to_string(), entry);
                guard.hits += 1;
                return Some(val);
            }
        }
        if self.use_mooncake && self.mooncake_enabled {
            guard.mooncake_misses += 1;
        }
        guard.misses += 1;
        None
    }

    pub fn set(&self, key: &str, value: Value) {
        if !self.enabled {
            return;
        }
        let mut guard = self.inner.write();
        guard.store.shift_remove(key);
        guard.store.insert(
            key.to_string(),
            CacheEntry {
                value,
                ts: Instant::now(),
            },
        );
        while guard.store.len() > self.max_size {
            guard.store.shift_remove_index(0);
        }
    }

    pub fn clear(&self) {
        let mut guard = self.inner.write();
        guard.store.clear();
        guard.hits = 0;
        guard.misses = 0;
        guard.mooncake_hits = 0;
        guard.mooncake_misses = 0;
        guard.mooncake_writes = 0;
    }

    pub fn close(&self) {}

    pub fn stats(&self) -> Value {
        let guard = self.inner.read();
        let total = guard.hits + guard.misses;
        let hit_rate = if total > 0 {
            ((guard.hits as f64 / total as f64) * 10000.0).round() / 10000.0
        } else {
            0.0
        };
        let mut base = json!({
            "enabled": self.enabled,
            "size": guard.store.len(),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits": guard.hits,
            "misses": guard.misses,
            "hit_rate": hit_rate,
        });
        if self.use_mooncake {
            if let Some(obj) = base.as_object_mut() {
                obj.insert(
                    "backend".to_string(),
                    json!(if self.mooncake_enabled { "mooncake+l1" } else { "l1" }),
                );
                obj.insert("mooncake_enabled".to_string(), json!(self.mooncake_enabled));
                obj.insert("mooncake_error".to_string(), json!(self.mooncake_error));
                obj.insert("mooncake_hits".to_string(), json!(guard.mooncake_hits));
                obj.insert("mooncake_misses".to_string(), json!(guard.mooncake_misses));
                obj.insert("mooncake_writes".to_string(), json!(guard.mooncake_writes));
                obj.insert("key_prefix".to_string(), json!(self.key_prefix));
            }
        }
        base
    }
}

pub fn build_response_cache_for_model(
    model_name: &str,
    max_size: usize,
    ttl_seconds: u64,
    enabled: bool,
    force_mooncake: Option<bool>,
) -> ResponseCache {
    let mut use_mooncake = force_mooncake.unwrap_or_else(|| is_moonshot_model(model_name));
    if let Ok(flag) = env::var("MOONCAKE_CACHE") {
        let f = flag.trim().to_ascii_lowercase();
        if matches!(f.as_str(), "1" | "true" | "yes" | "on") {
            use_mooncake = true;
        } else if matches!(f.as_str(), "0" | "false" | "no" | "off") {
            use_mooncake = false;
        }
    }
    if use_mooncake {
        let prefix = env::var("MOONCAKE_KEY_PREFIX")
            .unwrap_or_else(|_| format!("hf-ide:{model_name}"));
        ResponseCache::with_mooncake(max_size, ttl_seconds, enabled, prefix)
    } else {
        ResponseCache::new(max_size, ttl_seconds, enabled)
    }
}

#[derive(Clone, Debug)]
pub struct ConversationMemory {
    pub context_window: usize,
    pub max_conversations: usize,
    conversations: Arc<RwLock<IndexMap<String, Vec<Value>>>>,
}

impl ConversationMemory {
    pub fn new(context_window: usize, max_conversations: usize) -> Self {
        Self {
            context_window: context_window.max(512),
            max_conversations: max_conversations.max(1),
            conversations: Arc::new(RwLock::new(IndexMap::new())),
        }
    }

    pub fn estimate_tokens(text: &str) -> usize {
        if text.is_empty() {
            0
        } else {
            (text.len() / 4).max(1)
        }
    }

    fn message_tokens(message: &Value) -> usize {
        let content = message
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        Self::estimate_tokens(content) + 4
    }

    pub fn get(&self, conversation_id: &str) -> Vec<Value> {
        let mut guard = self.conversations.write();
        if let Some(msgs) = guard.shift_remove(conversation_id) {
            let out = msgs.clone();
            guard.insert(conversation_id.to_string(), msgs);
            out
        } else {
            Vec::new()
        }
    }

    pub fn set(&self, conversation_id: &str, messages: &[Value]) {
        let trimmed = self.trim_to_context(messages, 0);
        let mut guard = self.conversations.write();
        guard.shift_remove(conversation_id);
        guard.insert(conversation_id.to_string(), trimmed);
        while guard.len() > self.max_conversations {
            guard.shift_remove_index(0);
        }
    }

    pub fn append(&self, conversation_id: &str, role: &str, content: &str) -> Vec<Value> {
        let mut guard = self.conversations.write();
        let mut history = guard.shift_remove(conversation_id).unwrap_or_default();
        history.push(json!({
            "role": role,
            "content": content,
        }));
        let trimmed = self.trim_to_context(&history, 0);
        guard.insert(conversation_id.to_string(), trimmed.clone());
        while guard.len() > self.max_conversations {
            guard.shift_remove_index(0);
        }
        trimmed
    }

    pub fn trim_to_context(&self, messages: &[Value], reserve_for_response: usize) -> Vec<Value> {
        if messages.is_empty() {
            return Vec::new();
        }
        let budget = self
            .context_window
            .saturating_sub(reserve_for_response)
            .max(512);

        let mut system_msgs = Vec::new();
        let mut other_msgs = Vec::new();
        for m in messages {
            if m.get("role").and_then(|r| r.as_str()) == Some("system") {
                system_msgs.push(m.clone());
            } else {
                other_msgs.push(m.clone());
            }
        }

        let system_tokens: usize = system_msgs.iter().map(Self::message_tokens).sum();
        if system_tokens >= budget {
            return system_msgs.last().cloned().into_iter().collect();
        }
        let remaining = budget - system_tokens;

        let mut kept_reversed = Vec::new();
        let mut used = 0;
        for msg in other_msgs.iter().rev() {
            let cost = Self::message_tokens(msg);
            if used + cost > remaining {
                break;
            }
            kept_reversed.push(msg.clone());
            used += cost;
        }
        kept_reversed.reverse();
        system_msgs.extend(kept_reversed);
        system_msgs
    }

    pub fn clear(&self, conversation_id: Option<&str>) {
        let mut guard = self.conversations.write();
        match conversation_id {
            Some(id) => {
                guard.shift_remove(id);
            }
            None => guard.clear(),
        }
    }

    pub fn stats(&self) -> Value {
        let guard = self.conversations.read();
        json!({
            "conversations": guard.len(),
            "max_conversations": self.max_conversations,
            "context_window": self.context_window,
        })
    }
}
