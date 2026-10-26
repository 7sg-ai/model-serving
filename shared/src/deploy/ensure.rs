//! Ensure NIM/vLLM/SGLang containers are running; tear down on exit when owned.

use crate::backends::openai_chat::normalize_chat_url;
use crate::deploy::compose_gen::generate_compose;
use crate::deploy::config::load_deploy_config;
use crate::deploy::docker_util::{compose_up, docker_available};
use crate::deploy::engines::{normalize_engine, slug};
use crate::deploy::health::{any_healthy, wait_until_healthy};
use crate::deploy::refcount;
use crate::switchyard::config::ModelSpec;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::process;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeRuntime {
    pub target: String,
    pub advertise: String,
    pub compose_file_local: String,
    pub compose_file_remote: String,
    pub urls: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeHandle {
    pub engine: String,
    pub project: String,
    pub urls: Vec<String>,
    pub owned: bool,
    pub teardown_on_exit: bool,
    pub compose_file: String,
    pub shared: bool,
    pub pid: u32,
    pub multi_node: bool,
    pub node_runtimes: Vec<NodeRuntime>,
}

impl RuntimeHandle {
    pub fn to_dict(&self) -> Value {
        json!({
            "engine": self.engine,
            "project": self.project,
            "urls": self.urls,
            "owned": self.owned,
            "teardown_on_exit": self.teardown_on_exit,
            "compose_file": self.compose_file,
            "shared": self.shared,
            "pid": self.pid,
            "multi_node": self.multi_node,
        })
    }
}

pub fn ensure_model_runtime(
    engine: &str,
    app_name: &str,
    model_id: Option<&str>,
    existing_urls: &[String],
    api_key: &str,
) -> RuntimeHandle {
    let cfg = load_deploy_config(engine, app_name, model_id);
    let project = cfg.resolved_project();
    let normalized_existing: Vec<String> = existing_urls
        .iter()
        .map(|u| normalize_chat_url(u))
        .filter(|u| !u.is_empty())
        .collect();

    if !cfg.auto_deploy {
        return RuntimeHandle {
            engine: cfg.engine,
            project,
            urls: normalized_existing,
            owned: false,
            teardown_on_exit: false,
            compose_file: String::new(),
            shared: cfg.shared,
            pid: process::id(),
            multi_node: false,
            node_runtimes: Vec::new(),
        };
    }

    if !normalized_existing.is_empty() && any_healthy(&normalized_existing, api_key) {
        return RuntimeHandle {
            engine: cfg.engine,
            project,
            urls: normalized_existing,
            owned: false,
            teardown_on_exit: false,
            compose_file: String::new(),
            shared: cfg.shared,
            pid: process::id(),
            multi_node: false,
            node_runtimes: Vec::new(),
        };
    }

    let Ok((compose_path, generated_urls)) = generate_compose(&cfg) else {
        return RuntimeHandle {
            engine: cfg.engine,
            project,
            urls: normalized_existing,
            owned: false,
            teardown_on_exit: false,
            compose_file: String::new(),
            shared: cfg.shared,
            pid: process::id(),
            multi_node: false,
            node_runtimes: Vec::new(),
        };
    } ;

    if any_healthy(&generated_urls, api_key) {
        return RuntimeHandle {
            engine: cfg.engine,
            project,
            urls: generated_urls,
            owned: false,
            teardown_on_exit: false,
            compose_file: compose_path.to_string_lossy().into_owned(),
            shared: cfg.shared,
            pid: process::id(),
            multi_node: false,
            node_runtimes: Vec::new(),
        };
    }

    if !docker_available() {
        let fallback = if !normalized_existing.is_empty() {
            normalized_existing
        } else {
            generated_urls
        };
        return RuntimeHandle {
            engine: cfg.engine,
            project,
            urls: fallback,
            owned: false,
            teardown_on_exit: false,
            compose_file: compose_path.to_string_lossy().into_owned(),
            shared: cfg.shared,
            pid: process::id(),
            multi_node: false,
            node_runtimes: Vec::new(),
        };
    }

    if compose_up(&project, &compose_path).is_ok() {
        let _ = wait_until_healthy(&generated_urls, cfg.wait_seconds, 3, api_key);
        let _ = refcount::acquire(&project, process::id(), &compose_path.to_string_lossy());
        return RuntimeHandle {
            engine: cfg.engine,
            project,
            urls: generated_urls,
            owned: true,
            teardown_on_exit: cfg.teardown_on_exit,
            compose_file: compose_path.to_string_lossy().into_owned(),
            shared: cfg.shared,
            pid: process::id(),
            multi_node: false,
            node_runtimes: Vec::new(),
        };
    }

    RuntimeHandle {
        engine: cfg.engine,
        project,
        urls: if !normalized_existing.is_empty() {
            normalized_existing
        } else {
            generated_urls
        },
        owned: false,
        teardown_on_exit: false,
        compose_file: compose_path.to_string_lossy().into_owned(),
        shared: cfg.shared,
        pid: process::id(),
        multi_node: false,
        node_runtimes: Vec::new(),
    }
}

pub fn ensure_models_runtime(
    engine: &str,
    models: &mut [ModelSpec],
    app_name: &str,
    api_key: &str,
    default_backend_urls: &[String],
) -> Vec<RuntimeHandle> {
    let eng = normalize_engine(Some(engine), "vllm");
    if models.is_empty() {
        return vec![ensure_model_runtime(
            &eng,
            app_name,
            None,
            default_backend_urls,
            api_key,
        )];
    }

    let mut handles = Vec::new();
    for spec in models.iter_mut() {
        let mut existing: Vec<String> = spec
            .deployment
            .backend_urls
            .iter()
            .map(|u| normalize_chat_url(u))
            .filter(|u| !u.is_empty())
            .collect();
        if existing.is_empty() && !spec.chat_url().is_empty() {
            existing.push(normalize_chat_url(&spec.chat_url()));
        }
        if existing.is_empty() {
            existing = default_backend_urls
                .iter()
                .map(|u| normalize_chat_url(u))
                .filter(|u| !u.is_empty())
                .collect();
        }
        let sub_app = format!("{app_name}-{}", slug(&spec.id));
        let handle = ensure_model_runtime(&eng, &sub_app, Some(&spec.id), &existing, api_key);
        if !handle.urls.is_empty() {
            spec.deployment.backend_urls = handle.urls.clone();
        } else if !existing.is_empty() {
            spec.deployment.backend_urls = existing;
        }
        handles.push(handle);
    }
    handles
}
