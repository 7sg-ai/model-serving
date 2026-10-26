//! Generate compose YAML for NIM, vLLM, or SGLang model stacks.

use crate::deploy::config::DeployConfig;
use crate::deploy::engines::{container_port, service_prefix, yaml_quote};
use crate::deploy::scheduler::NodePlan;
use anyhow::{anyhow, Result};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

pub fn compose_output_path(cfg: &DeployConfig, suffix: &str) -> PathBuf {
    if !cfg.compose_file.is_empty() && suffix.is_empty() {
        return PathBuf::from(&cfg.compose_file);
    }
    let deploy_dir = PathBuf::from("deploy");
    let _ = fs::create_dir_all(&deploy_dir);
    let mut name = format!("docker-compose.{}.{}", cfg.engine, cfg.resolved_project());
    if !suffix.is_empty() {
        name.push('.');
        name.push_str(suffix);
    }
    deploy_dir.join(format!("{name}.generated.yml"))
}

fn gpu_slices(cfg: &DeployConfig) -> Result<Vec<Vec<String>>> {
    let replicas = cfg.resolved_replicas();
    let gpr = cfg.gpus_per_replica();
    let needed = replicas * gpr;
    let mut gpus = cfg.gpu_devices.clone();
    if gpus.is_empty() {
        gpus = (0..needed).map(|i| i.to_string()).collect();
    }
    if gpus.len() < needed {
        return Err(anyhow!(
            "Not enough GPU_DEVICES ({}) for {replicas} x {gpr} GPUs (need {needed})",
            gpus.len()
        ));
    }
    let mut slices = Vec::new();
    for i in 0..replicas {
        slices.push(gpus[i * gpr..(i + 1) * gpr].to_vec());
    }
    Ok(slices)
}

fn gpu_reservation(devices: &[String]) -> Vec<String> {
    let ids: Vec<String> = devices.iter().map(|d| yaml_quote(d)).collect();
    vec![
        "    deploy:".to_string(),
        "      resources:".to_string(),
        "        reservations:".to_string(),
        "          devices:".to_string(),
        "            - driver: nvidia".to_string(),
        "              capabilities: [gpu]".to_string(),
        format!("              device_ids: [{}]", ids.join(", ")),
    ]
}

fn hf_cache() -> String {
    let home = env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    env::var("HF_CACHE").unwrap_or_else(|_| format!("{home}/.cache/huggingface"))
}

fn service_lines(
    cfg: &DeployConfig,
    name: &str,
    devices: &[String],
    host_port: u16,
) -> Vec<String> {
    let cvd = devices.join(",");
    let cport = container_port(&cfg.engine, &cfg.image);
    let mut lines = vec![format!("  {name}:"), format!("    image: {}", cfg.image)];

    if cfg.engine == "nim" {
        lines.push("    environment:".to_string());
        lines.push(format!("      NGC_API_KEY: {}", yaml_quote(&cfg.ngc_api_key)));
        lines.push(format!("      CUDA_VISIBLE_DEVICES: {}", yaml_quote(&cvd)));
        if cfg.model_free_nim() {
            lines.push(format!(
                "      NIM_MODEL_PATH: {}",
                yaml_quote(&cfg.nim_model_path)
            ));
            lines.push(format!("      HF_TOKEN: {}", yaml_quote(&cfg.hf_token)));
            lines.push(format!(
                "      HUGGING_FACE_HUB_TOKEN: {}",
                yaml_quote(&cfg.hf_token)
            ));
            if !cfg.nim_served_model_name.is_empty() {
                lines.push(format!(
                    "      NIM_SERVED_MODEL_NAME: {}",
                    yaml_quote(&cfg.nim_served_model_name)
                ));
            }
        }
        lines.push("    ports:".to_string());
        lines.push(format!("      - \"{host_port}:8000\""));
        lines.push("    shm_size: \"16gb\"".to_string());
        lines.extend(gpu_reservation(devices));
        return lines;
    }

    lines.push("    command:".to_string());
    if cfg.engine == "sglang" {
        lines.extend([
            "      - \"python3\"".to_string(),
            "      - \"-m\"".to_string(),
            "      - \"sglang.launch_server\"".to_string(),
            "      - \"--model-path\"".to_string(),
            format!("      - {}", yaml_quote(&cfg.model_id)),
            "      - \"--tp-size\"".to_string(),
            format!("      - \"{}\"", cfg.tensor_parallel_size),
            "      - \"--host\"".to_string(),
            "      - \"0.0.0.0\"".to_string(),
            "      - \"--port\"".to_string(),
            format!("      - \"{cport}\""),
        ]);
    } else {
        lines.extend([
            "      - \"--model\"".to_string(),
            format!("      - {}", yaml_quote(&cfg.model_id)),
            "      - \"--tensor-parallel-size\"".to_string(),
            format!("      - \"{}\"", cfg.tensor_parallel_size),
            "      - \"--max-model-len\"".to_string(),
            format!("      - \"{}\"", cfg.vllm_max_model_len),
            "      - \"--host\"".to_string(),
            "      - \"0.0.0.0\"".to_string(),
            "      - \"--port\"".to_string(),
            "      - \"8000\"".to_string(),
        ]);
        if cfg.enable_prefix_caching {
            lines.push("      - \"--enable-prefix-caching\"".to_string());
        }
    }
    lines.push("    environment:".to_string());
    lines.push(format!("      CUDA_VISIBLE_DEVICES: {}", yaml_quote(&cvd)));
    lines.push(format!(
        "      HUGGING_FACE_HUB_TOKEN: {}",
        yaml_quote(&cfg.hf_token)
    ));
    lines.push("    ports:".to_string());
    lines.push(format!("      - \"{host_port}:{cport}\""));
    lines.push("    volumes:".to_string());
    lines.push(format!("      - {}:/root/.cache/huggingface", hf_cache()));
    lines.push("    ipc: host".to_string());
    lines.extend(gpu_reservation(devices));
    lines
}

pub fn generate_compose(cfg: &DeployConfig) -> Result<(PathBuf, Vec<String>)> {
    let slices = gpu_slices(cfg)?;
    let prefix = service_prefix(&cfg.engine);
    let mut lines = vec![
        format!(
            "# AUTO-GENERATED model runtime ({}) project={}",
            cfg.engine,
            cfg.resolved_project()
        ),
        format!(
            "# mode={} replicas={} tp={}",
            cfg.deploy_mode,
            slices.len(),
            cfg.tensor_parallel_size
        ),
        "services:".to_string(),
    ];
    let mut urls = Vec::new();
    for (i, devices) in slices.iter().enumerate() {
        let name = format!("{prefix}-{i}");
        let host_port = cfg.port_base + i as u16;
        lines.extend(service_lines(cfg, &name, devices, host_port));
        urls.push(format!("http://127.0.0.1:{host_port}/v1/chat/completions"));
    }
    let path = compose_output_path(cfg, "");
    fs::write(&path, format!("{}\n", lines.join("\n")))?;
    Ok((path, urls))
}

pub fn generate_compose_for_node_plan(
    cfg: &DeployConfig,
    plan: &NodePlan,
) -> Result<(PathBuf, Vec<String>)> {
    let safe = plan.target.replace(['@', ':'], "_");
    let mut lines = vec![
        format!(
            "# AUTO-GENERATED {} node={} project={}",
            cfg.engine,
            plan.target,
            cfg.resolved_project()
        ),
        "services:".to_string(),
    ];
    for rep in &plan.replicas {
        lines.extend(service_lines(
            cfg,
            &rep.service_name,
            &rep.device_ids,
            rep.host_port,
        ));
    }
    let path = compose_output_path(cfg, &safe);
    fs::write(&path, format!("{}\n", lines.join("\n")))?;
    Ok((path, plan.urls()))
}

pub fn ensure_dir_for_file(p: &Path) {
    if let Some(parent) = p.parent() {
        let _ = fs::create_dir_all(parent);
    }
}
