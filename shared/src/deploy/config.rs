//! Environment configuration for model runtime auto-deploy.

use crate::deploy::engines::{
    default_image, default_model_id, default_port, is_model_free_nim_image, nim_model_path,
    normalize_engine, served_model_name, slug,
};
use crate::{env_bool, env_int, split_csv};
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeployConfig {
    pub auto_deploy: bool,
    pub teardown_on_exit: bool,
    pub shared: bool,
    pub project: String,
    pub engine: String,
    pub app_name: String,
    pub model_id: String,
    pub deploy_mode: String,
    pub replica_count: usize,
    pub tensor_parallel_size: usize,
    pub pipeline_parallel_size: usize,
    pub gpu_per_replica: usize,
    pub gpu_devices: Vec<String>,
    pub port_base: u16,
    pub wait_seconds: u64,
    pub teardown_timeout: u64,
    pub remove_volumes: bool,
    pub compose_file: String,
    pub image: String,
    pub ngc_api_key: String,
    pub hf_token: String,
    pub vllm_max_model_len: usize,
    pub enable_prefix_caching: bool,
    pub deploy_nodes: Vec<String>,
    pub remote_dir: String,
    pub ssh_user: String,
    pub nim_model_path: String,
    pub nim_served_model_name: String,
    pub nim_model_profile: String,
    pub nim_cache_host: String,
    pub trust_remote_code: bool,
    pub tool_call_parser: String,
    pub reasoning_parser: String,
    pub extra_args: Vec<String>,
}

impl DeployConfig {
    pub fn gpus_per_replica(&self) -> usize {
        self.gpu_per_replica
            .max(self.tensor_parallel_size * self.pipeline_parallel_size)
            .max(1)
    }

    pub fn resolved_project(&self) -> String {
        if !self.project.is_empty() {
            return slug(&self.project);
        }
        if self.shared {
            return slug(&format!("model-serving-{}-shared", self.engine));
        }
        slug(&format!("model-serving-{}-{}", self.engine, self.app_name))
    }

    pub fn resolved_replicas(&self) -> usize {
        if self.deploy_mode == "sharded" {
            1
        } else {
            self.replica_count.max(1)
        }
    }

    pub fn is_multi_node(&self) -> bool {
        !self.deploy_nodes.is_empty()
    }

    pub fn model_free_nim(&self) -> bool {
        if self.engine != "nim" {
            return false;
        }
        if !self.nim_model_path.is_empty() {
            return true;
        }
        is_model_free_nim_image(&self.image)
    }
}

pub fn load_deploy_config(
    engine: &str,
    app_name: &str,
    model_id: Option<&str>,
) -> DeployConfig {
    let eng = normalize_engine(Some(engine), "vllm");
    let mut mode = env::var("MODEL_DEPLOY_MODE")
        .or_else(|_| env::var("LLM_DEPLOY_MODE"))
        .unwrap_or_else(|_| "hybrid".to_string())
        .trim()
        .to_ascii_lowercase();
    if !matches!(mode.as_str(), "replica" | "sharded" | "hybrid") {
        mode = "hybrid".to_string();
    }

    let rep_env = match eng.as_str() {
        "nim" => "NIM_REPLICA_COUNT",
        "sglang" => "SGLANG_REPLICA_COUNT",
        _ => "VLLM_REPLICA_COUNT",
    };
    let replica_count = env_int(rep_env, env_int("REPLICA_COUNT", 1)).max(1) as usize;
    let tp = env_int("TENSOR_PARALLEL_SIZE", env_int("TP_SIZE", 1)).max(1) as usize;
    let pp = env_int("PIPELINE_PARALLEL_SIZE", env_int("PP_SIZE", 1)).max(1) as usize;
    let gpr = env_int("GPU_PER_REPLICA", (tp * pp).max(1) as i64).max(1) as usize;
    let resolved_port = default_port(&eng, 8000);
    let resolved_image = default_image(&eng);
    let model = model_id
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| default_model_id(&eng));

    let mut path = env::var("NIM_MODEL_PATH").unwrap_or_default();
    if eng == "nim" && is_model_free_nim_image(&resolved_image) && path.trim().is_empty() {
        path = nim_model_path(&model, "");
    }
    let served = env::var("NIM_SERVED_MODEL_NAME").unwrap_or_default();
    let gpus = split_csv(
        &env::var("GPU_DEVICES")
            .or_else(|_| env::var("CUDA_VISIBLE_DEVICES"))
            .unwrap_or_default(),
    );
    let home = env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let nim_cache = env::var("NIM_CACHE")
        .or_else(|_| env::var("LOCAL_NIM_CACHE"))
        .unwrap_or_else(|_| format!("{home}/.cache/nim"));

    DeployConfig {
        auto_deploy: env_bool("AUTO_DEPLOY_MODEL", true),
        teardown_on_exit: env_bool("AUTO_DEPLOY_TEARDOWN_ON_EXIT", true),
        shared: env_bool("AUTO_DEPLOY_SHARED", false),
        project: env::var("AUTO_DEPLOY_PROJECT").unwrap_or_default(),
        engine: eng,
        app_name: if app_name.is_empty() {
            "app".to_string()
        } else {
            app_name.to_string()
        },
        model_id: model.clone(),
        deploy_mode: mode,
        replica_count,
        tensor_parallel_size: tp,
        pipeline_parallel_size: pp,
        gpu_per_replica: gpr,
        gpu_devices: gpus,
        port_base: resolved_port,
        wait_seconds: env_int("AUTO_DEPLOY_WAIT_SECONDS", 300).max(1) as u64,
        teardown_timeout: env_int("AUTO_DEPLOY_TEARDOWN_TIMEOUT_SECONDS", 60).max(1) as u64,
        remove_volumes: env_bool("AUTO_DEPLOY_REMOVE_VOLUMES", false),
        compose_file: String::new(),
        image: resolved_image,
        ngc_api_key: env::var("NGC_API_KEY").unwrap_or_default(),
        hf_token: env::var("HUGGING_FACE_HUB_TOKEN")
            .or_else(|_| env::var("HF_TOKEN"))
            .unwrap_or_default(),
        vllm_max_model_len: env_int("VLLM_MAX_MODEL_LEN", 8192).max(512) as usize,
        enable_prefix_caching: env_bool("VLLM_ENABLE_PREFIX_CACHING", true),
        deploy_nodes: split_csv(&env::var("DEPLOY_NODES").unwrap_or_default()),
        remote_dir: env::var("DEPLOY_REMOTE_DIR")
            .unwrap_or_else(|_| "/tmp/model-serving-runtime".to_string()),
        ssh_user: env::var("DEPLOY_SSH_USER").unwrap_or_default(),
        nim_model_path: path,
        nim_served_model_name: served_model_name(&model, &served),
        nim_model_profile: env::var("NIM_MODEL_PROFILE").unwrap_or_default(),
        nim_cache_host: nim_cache,
        trust_remote_code: env_bool("TRUST_REMOTE_CODE", false),
        tool_call_parser: env::var("TOOL_CALL_PARSER").unwrap_or_default(),
        reasoning_parser: env::var("REASONING_PARSER").unwrap_or_default(),
        extra_args: split_csv(&env::var("ENGINE_EXTRA_ARGS").unwrap_or_default()),
    }
}
