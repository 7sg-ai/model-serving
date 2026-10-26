//! Multi-node configuration, model placement, backend pool, and cluster health.

use crate::{env_bool, env_int, split_csv};
use parking_lot::Mutex;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::process;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub fn default_hostname() -> String {
    env::var("HOSTNAME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "node-local".to_string())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiNodeConfig {
    pub enabled: bool,
    pub node_id: String,
    pub node_host: String,
    pub node_port: u16,
    pub public_url: String,
    pub cluster_name: String,
    pub role: String,
    pub peers: Vec<String>,
    pub workers: usize,
    pub threads: usize,
    pub shared_data_dir: String,
    pub auth_db_path: String,
    pub chat_db_path: String,
    pub redis_url: String,
    pub use_redis: bool,
    pub sticky_sessions: bool,
    pub session_secret: String,
    pub backend_urls: Vec<String>,
    pub backend_strategy: String,
    pub model_deploy_mode: String,
    pub tensor_parallel_size: usize,
    pub pipeline_parallel_size: usize,
    pub model_parallel_size: usize,
    pub shard_rank: usize,
    pub shard_world_size: usize,
    pub shard_group: String,
    pub cuda_visible_devices: String,
    pub hf_device_map: String,
    pub hf_max_memory: String,
    pub model_shard_backends: Vec<String>,
    pub model_coordinator_url: String,
}

impl MultiNodeConfig {
    pub fn resolved_auth_db(&self) -> String {
        if !self.auth_db_path.is_empty() {
            return self.auth_db_path.clone();
        }
        Path::new(&self.shared_data_dir)
            .join("users.db")
            .to_string_lossy()
            .into_owned()
    }

    pub fn resolved_chat_db(&self) -> String {
        if !self.chat_db_path.is_empty() {
            return self.chat_db_path.clone();
        }
        Path::new(&self.shared_data_dir)
            .join("chat_history.db")
            .to_string_lossy()
            .into_owned()
    }

    pub fn to_dict(&self) -> Value {
        json!({
            "enabled": self.enabled,
            "node_id": self.node_id,
            "node_host": self.node_host,
            "node_port": self.node_port,
            "public_url": self.public_url,
            "cluster_name": self.cluster_name,
            "role": self.role,
            "peers": self.peers,
            "workers": self.workers,
            "threads": self.threads,
            "shared_data_dir": self.shared_data_dir,
            "auth_db": self.resolved_auth_db(),
            "chat_db": self.resolved_chat_db(),
            "use_redis": self.use_redis,
            "redis_configured": !self.redis_url.is_empty(),
            "sticky_sessions": self.sticky_sessions,
            "backend_urls": self.backend_urls,
            "backend_strategy": self.backend_strategy,
            "model_deploy_mode": self.model_deploy_mode,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "model_parallel_size": self.model_parallel_size,
            "shard_rank": self.shard_rank,
            "shard_world_size": self.shard_world_size,
            "shard_group": self.shard_group,
            "cuda_visible_devices": self.cuda_visible_devices,
            "hf_device_map": self.hf_device_map,
            "model_shard_backends": self.model_shard_backends,
            "model_coordinator_url": self.model_coordinator_url,
        })
    }
}

pub fn get_multinode_config(default_port: u16, app_name: &str) -> MultiNodeConfig {
    let peers_raw = env::var("CLUSTER_PEERS")
        .or_else(|_| env::var("PEERS"))
        .unwrap_or_default();
    let peers = split_csv(&peers_raw);

    let backends_raw = env::var("BACKEND_URLS")
        .or_else(|_| env::var("NIM_API_URLS"))
        .or_else(|_| env::var("MODEL_BACKEND_URLS"))
        .unwrap_or_default();
    let mut backend_urls = split_csv(&backends_raw);

    if let Ok(single) = env::var("NIM_API_URL") {
        let s = single.trim();
        if !s.is_empty() && !backend_urls.iter().any(|u| u == s) && backend_urls.is_empty() {
            backend_urls.push(s.to_string());
        }
    }
    for key in ["VLLM_API_URLS", "VLLM_API_URL", "SGLANG_API_URLS", "SGLANG_API_URL"] {
        if let Ok(raw) = env::var(key) {
            for p in split_csv(&raw) {
                if !backend_urls.contains(&p) {
                    backend_urls.push(p);
                }
            }
        }
    }

    let port = env_int("PORT", env_int("NODE_PORT", default_port as i64)) as u16;
    let host = env::var("HOST")
        .or_else(|_| env::var("NODE_HOST"))
        .unwrap_or_else(|_| "0.0.0.0".to_string());
    let node_id = env::var("NODE_ID")
        .unwrap_or_else(|_| format!("{app_name}-{}", default_hostname()));

    let enabled = env_bool(
        "MULTI_NODE",
        !peers.is_empty() || env_bool("CLUSTER_ENABLED", false),
    );
    let redis_url = env::var("REDIS_URL").unwrap_or_default().trim().to_string();
    let shared_dir = env::var("SHARED_DATA_DIR").unwrap_or_else(|_| "shared/database".to_string());

    let tp = env_int("TENSOR_PARALLEL_SIZE", env_int("TP_SIZE", 1)).max(1) as usize;
    let pp = env_int("PIPELINE_PARALLEL_SIZE", env_int("PP_SIZE", 1)).max(1) as usize;
    let mp_default = if tp > 1 || pp > 1 { (tp * pp) as i64 } else { 1 };
    let mp = env_int("MODEL_PARALLEL_SIZE", mp_default).max(1) as usize;
    let shard_backends = split_csv(&env::var("MODEL_SHARD_BACKENDS").unwrap_or_default());

    let mut deploy_mode = env::var("MODEL_DEPLOY_MODE")
        .or_else(|_| env::var("LLM_DEPLOY_MODE"))
        .unwrap_or_else(|_| "hybrid".to_string())
        .trim()
        .to_ascii_lowercase();
    if !matches!(deploy_mode.as_str(), "replica" | "sharded" | "hybrid") {
        deploy_mode = "hybrid".to_string();
    }

    let world = env_int(
        "SHARD_WORLD_SIZE",
        env_int("WORLD_SIZE", if mp > 1 { mp as i64 } else { 1 }),
    )
    .max(1) as usize;

    MultiNodeConfig {
        enabled,
        node_id,
        node_host: host,
        node_port: port,
        public_url: env::var("PUBLIC_URL")
            .or_else(|_| env::var("NODE_PUBLIC_URL"))
            .unwrap_or_default()
            .trim_end_matches('/')
            .to_string(),
        cluster_name: env::var("CLUSTER_NAME").unwrap_or_else(|_| "model-serving".to_string()),
        role: env::var("NODE_ROLE")
            .unwrap_or_else(|_| "all".to_string())
            .to_ascii_lowercase(),
        peers,
        workers: env_int("WEB_CONCURRENCY", env_int("WORKERS", 1)).max(1) as usize,
        threads: env_int("THREADS", 8).max(1) as usize,
        shared_data_dir: shared_dir,
        auth_db_path: env::var("AUTH_DB_PATH").unwrap_or_default(),
        chat_db_path: env::var("CHAT_DB_PATH").unwrap_or_default(),
        use_redis: env_bool("USE_REDIS", !redis_url.is_empty()),
        redis_url,
        sticky_sessions: env_bool("STICKY_SESSIONS", true),
        session_secret: env::var("SECRET_KEY")
            .or_else(|_| env::var("SESSION_SECRET"))
            .unwrap_or_else(|_| "change-me-in-production".to_string()),
        backend_urls,
        backend_strategy: env::var("BACKEND_STRATEGY")
            .unwrap_or_else(|_| "round_robin".to_string())
            .to_ascii_lowercase(),
        model_deploy_mode: deploy_mode,
        tensor_parallel_size: tp,
        pipeline_parallel_size: pp,
        model_parallel_size: mp,
        shard_rank: env_int("SHARD_RANK", env_int("RANK", 0)).max(0) as usize,
        shard_world_size: world,
        shard_group: env::var("SHARD_GROUP").unwrap_or_else(|_| "default".to_string()),
        cuda_visible_devices: env::var("CUDA_VISIBLE_DEVICES")
            .or_else(|_| env::var("GPU_DEVICES"))
            .unwrap_or_default(),
        hf_device_map: env::var("HF_DEVICE_MAP").unwrap_or_default(),
        hf_max_memory: env::var("HF_MAX_MEMORY").unwrap_or_default(),
        model_shard_backends: shard_backends,
        model_coordinator_url: env::var("MODEL_COORDINATOR_URL")
            .or_else(|_| env::var("SHARD_COORDINATOR_URL"))
            .unwrap_or_default()
            .trim()
            .to_string(),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelPlacement {
    pub deploy_mode: String,
    pub tensor_parallel_size: usize,
    pub pipeline_parallel_size: usize,
    pub model_parallel_size: usize,
    pub shard_rank: usize,
    pub shard_world_size: usize,
    pub shard_group: String,
    pub cuda_visible_devices: String,
    pub local_gpu_count: usize,
    pub device_map: Option<String>,
    pub max_memory: HashMap<String, String>,
    pub low_cpu_mem_usage: bool,
    pub offload_folder: String,
    pub torch_dtype: String,
    pub shard_backends: Vec<String>,
    pub replica_backends: Vec<String>,
    pub coordinator_url: String,
}

impl ModelPlacement {
    pub fn is_sharded(&self) -> bool {
        matches!(self.deploy_mode.as_str(), "sharded" | "hybrid")
            || self.tensor_parallel_size > 1
            || self.pipeline_parallel_size > 1
            || self.shard_world_size > 1
            || !self.shard_backends.is_empty()
    }

    pub fn is_replicated(&self) -> bool {
        matches!(self.deploy_mode.as_str(), "replica" | "hybrid")
            && self.replica_backends.len() > 1
    }

    pub fn loads_weights_locally(&self) -> bool {
        if !env_bool("LOAD_MODEL_WEIGHTS", true) {
            return false;
        }
        let role = env::var("NODE_ROLE")
            .unwrap_or_else(|_| "all".to_string())
            .to_ascii_lowercase();
        if role == "api" && (!self.replica_backends.is_empty() || !self.shard_backends.is_empty()) {
            return false;
        }
        true
    }

    pub fn to_dict(&self) -> Value {
        json!({
            "deploy_mode": self.deploy_mode,
            "is_sharded": self.is_sharded(),
            "is_replicated": self.is_replicated(),
            "loads_weights_locally": self.loads_weights_locally(),
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "model_parallel_size": self.model_parallel_size,
            "shard_rank": self.shard_rank,
            "shard_world_size": self.shard_world_size,
            "shard_group": self.shard_group,
            "cuda_visible_devices": if self.cuda_visible_devices.is_empty() { Value::Null } else { json!(self.cuda_visible_devices) },
            "local_gpu_count": self.local_gpu_count,
            "device_map": self.device_map,
            "max_memory": self.max_memory,
            "torch_dtype": if self.torch_dtype.is_empty() { Value::Null } else { json!(self.torch_dtype) },
            "shard_backends": self.shard_backends,
            "replica_backends": self.replica_backends,
            "coordinator_url": if self.coordinator_url.is_empty() { Value::Null } else { json!(self.coordinator_url) },
        })
    }
}

pub fn get_model_placement(replica_backends: Option<Vec<String>>) -> ModelPlacement {
    let mut mode = env::var("MODEL_DEPLOY_MODE")
        .or_else(|_| env::var("LLM_DEPLOY_MODE"))
        .unwrap_or_else(|_| "hybrid".to_string())
        .trim()
        .to_ascii_lowercase();
    if !matches!(mode.as_str(), "replica" | "sharded" | "hybrid") {
        mode = "hybrid".to_string();
    }
    let tp = env_int("TENSOR_PARALLEL_SIZE", env_int("TP_SIZE", 1)).max(1) as usize;
    let pp = env_int("PIPELINE_PARALLEL_SIZE", env_int("PP_SIZE", 1)).max(1) as usize;
    let mp = env_int(
        "MODEL_PARALLEL_SIZE",
        if tp > 1 || pp > 1 { (tp * pp) as i64 } else { 1 },
    )
    .max(1) as usize;
    let rank = env_int("SHARD_RANK", env_int("RANK", 0)).max(0) as usize;
    let world = env_int(
        "SHARD_WORLD_SIZE",
        env_int("WORLD_SIZE", if mp > 1 { mp as i64 } else { 1 }),
    )
    .max(1) as usize;
    let cvd = env::var("CUDA_VISIBLE_DEVICES")
        .or_else(|_| env::var("GPU_DEVICES"))
        .unwrap_or_default();
    let local_gpu_count = split_csv(&cvd).len();
    let shard_backends = split_csv(&env::var("MODEL_SHARD_BACKENDS").unwrap_or_default());
    let mut replicas = replica_backends.unwrap_or_default();
    if replicas.is_empty() {
        let raw = env::var("BACKEND_URLS")
            .or_else(|_| env::var("NIM_API_URLS"))
            .or_else(|_| env::var("MODEL_REPLICA_URLS"))
            .unwrap_or_default();
        replicas = split_csv(&raw);
    }
    let coordinator = env::var("MODEL_COORDINATOR_URL")
        .or_else(|_| env::var("SHARD_COORDINATOR_URL"))
        .unwrap_or_default()
        .trim()
        .to_string();

    ModelPlacement {
        deploy_mode: mode,
        tensor_parallel_size: tp,
        pipeline_parallel_size: pp,
        model_parallel_size: mp,
        shard_rank: rank,
        shard_world_size: world,
        shard_group: env::var("SHARD_GROUP").unwrap_or_else(|_| "default".to_string()),
        cuda_visible_devices: cvd,
        local_gpu_count,
        device_map: Some("auto".to_string()),
        max_memory: HashMap::new(),
        low_cpu_mem_usage: env_bool("HF_LOW_CPU_MEM_USAGE", true),
        offload_folder: env::var("HF_OFFLOAD_FOLDER").unwrap_or_default(),
        torch_dtype: env::var("HF_TORCH_DTYPE").unwrap_or_default(),
        shard_backends,
        replica_backends: replicas,
        coordinator_url: coordinator,
    }
}

pub fn describe_placement_for_logs(placement: &ModelPlacement) -> String {
    let mut parts = vec![
        format!("mode={}", placement.deploy_mode),
        format!("tp={}", placement.tensor_parallel_size),
        format!("pp={}", placement.pipeline_parallel_size),
        format!("rank={}/{}", placement.shard_rank, placement.shard_world_size),
        format!("gpus_local={}", placement.local_gpu_count),
        format!("device_map={}", placement.device_map.as_deref().unwrap_or("none")),
        format!("replicas={}", placement.replica_backends.len()),
        format!("shard_backends={}", placement.shard_backends.len()),
    ];
    if !placement.coordinator_url.is_empty() {
        parts.push(format!("coordinator={}", placement.coordinator_url));
    }
    parts.join(" ")
}

pub fn node_identity(cfg: &MultiNodeConfig) -> Value {
    json!({
        "node_id": cfg.node_id,
        "hostname": default_hostname(),
        "role": cfg.role,
        "cluster": cfg.cluster_name,
        "multi_node": cfg.enabled,
        "public_url": if cfg.public_url.is_empty() { Value::Null } else { json!(cfg.public_url) },
        "listen": if cfg.node_port > 0 { format!("{}:{}", cfg.node_host, cfg.node_port) } else { cfg.node_host.clone() },
        "workers": cfg.workers,
        "pid": process::id(),
        "model_deploy_mode": cfg.model_deploy_mode,
        "tensor_parallel_size": cfg.tensor_parallel_size,
        "pipeline_parallel_size": cfg.pipeline_parallel_size,
        "shard_rank": cfg.shard_rank,
        "shard_world_size": cfg.shard_world_size,
    })
}

#[derive(Debug)]
struct PoolState {
    cursor: usize,
    failures: HashMap<String, Instant>,
}

#[derive(Clone, Debug)]
pub struct BackendPool {
    pub urls: Vec<String>,
    pub strategy: String,
    pub cooldown_seconds: u64,
    state: Arc<Mutex<PoolState>>,
}

impl BackendPool {
    pub fn new(urls: Vec<String>, strategy: &str) -> Self {
        let cleaned: Vec<String> = urls
            .into_iter()
            .map(|u| u.trim().trim_end_matches('/').to_string())
            .filter(|u| !u.is_empty())
            .collect();
        let cooldown = env::var("BACKEND_COOLDOWN_SECONDS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(30);
        Self {
            urls: cleaned,
            strategy: if strategy.is_empty() {
                "round_robin".to_string()
            } else {
                strategy.to_ascii_lowercase()
            },
            cooldown_seconds: cooldown,
            state: Arc::new(Mutex::new(PoolState {
                cursor: 0,
                failures: HashMap::new(),
            })),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.urls.is_empty()
    }

    pub fn len(&self) -> usize {
        self.urls.len()
    }

    pub fn mark_failure(&self, url: &str) {
        let mut st = self.state.lock();
        st.failures
            .insert(url.trim_end_matches('/').to_string(), Instant::now());
    }

    pub fn mark_success(&self, url: &str) {
        let mut st = self.state.lock();
        st.failures.remove(url.trim_end_matches('/'));
    }

    pub fn next(&self) -> Option<String> {
        if self.urls.is_empty() {
            return None;
        }
        let mut st = self.state.lock();
        let cooldown = Duration::from_secs(self.cooldown_seconds);
        let mut available: Vec<String> = self
            .urls
            .iter()
            .filter(|u| {
                st.failures
                    .get(*u)
                    .map(|t| t.elapsed() >= cooldown)
                    .unwrap_or(true)
            })
            .cloned()
            .collect();
        if available.is_empty() {
            available = self.urls.clone();
        }
        if self.strategy == "random" {
            return available.choose(&mut rand::thread_rng()).cloned();
        }
        if self.strategy == "first" {
            return available.first().cloned();
        }
        for _ in 0..self.urls.len() {
            let idx = st.cursor % self.urls.len();
            st.cursor = st.cursor.wrapping_add(1);
            let candidate = &self.urls[idx];
            if available.contains(candidate) {
                return Some(candidate.clone());
            }
        }
        available.first().cloned()
    }

    pub fn all(&self) -> Vec<String> {
        self.urls.clone()
    }
}

#[derive(Clone, Debug)]
pub struct ClusterInfo {
    pub cfg: MultiNodeConfig,
    pub app_name: String,
    pub backend_pool: BackendPool,
    pub model_placement: ModelPlacement,
}

impl ClusterInfo {
    pub fn new(cfg: MultiNodeConfig, app_name: &str) -> Self {
        let backend_pool = BackendPool::new(cfg.backend_urls.clone(), &cfg.backend_strategy);
        let model_placement = get_model_placement(Some(cfg.backend_urls.clone()));
        Self {
            cfg,
            app_name: app_name.to_string(),
            backend_pool,
            model_placement,
        }
    }

    pub fn next_backend(&self, default: Option<&str>) -> Option<String> {
        if !self.model_placement.coordinator_url.is_empty() {
            return Some(
                self.model_placement
                    .coordinator_url
                    .trim_end_matches('/')
                    .to_string(),
            );
        }
        if let Some(u) = self.backend_pool.next() {
            return Some(u);
        }
        if !self.model_placement.shard_backends.is_empty()
            && !self.model_placement.loads_weights_locally()
        {
            return self
                .model_placement
                .shard_backends
                .first()
                .map(|s| s.trim_end_matches('/').to_string());
        }
        default.map(|s| s.to_string())
    }

    pub fn health(&self, extra: Value) -> Value {
        let mut base = json!({
            "status": "healthy",
            "app": self.app_name,
            "node": node_identity(&self.cfg),
            "peers": self.cfg.peers,
            "backends": self.backend_pool.all(),
            "backend_strategy": self.cfg.backend_strategy,
            "model_placement": self.model_placement.to_dict(),
        });
        if let (Some(base_obj), Some(extra_obj)) = (base.as_object_mut(), extra.as_object()) {
            for (k, v) in extra_obj {
                base_obj.insert(k.clone(), v.clone());
            }
        }
        base
    }
}
