//! Remote compose up/down over SSH.

use crate::deploy::config::DeployConfig;
use crate::deploy::scheduler::NodePlan;
use crate::deploy::ssh_util::{scp_put, ssh_run};
use anyhow::Result;
use std::env;
use std::path::Path;

pub fn remote_dir(_cfg: &DeployConfig) -> String {
    env::var("DEPLOY_REMOTE_DIR")
        .unwrap_or_else(|_| "/tmp/model-serving-runtime".to_string())
        .trim_end_matches('/')
        .to_string()
}

fn compose_bin_check() -> &'static str {
    "if docker compose version >/dev/null 2>&1; then C=\"docker compose\"; elif command -v docker-compose >/dev/null 2>&1; then C=\"docker-compose\"; else echo NO_COMPOSE; exit 1; fi"
}

pub fn deploy_node_plan(cfg: &DeployConfig, plan: &NodePlan, local_compose: &Path) -> Result<String> {
    let rdir = remote_dir(cfg);
    let safe = plan.target.replace(['@', ':'], "_");
    let remote_path = format!("{rdir}/{}-{safe}.yml", cfg.resolved_project());
    scp_put(&local_compose.to_string_lossy(), &plan.target, &remote_path)?;
    let project = cfg.resolved_project();
    let cmd = format!(
        "{}; cd '{}'; $C -p '{}' -f '{}' up -d --remove-orphans",
        compose_bin_check(),
        rdir,
        project,
        remote_path
    );
    ssh_run(&plan.target, &cmd, true)?;
    Ok(remote_path)
}

pub fn teardown_node_plan(cfg: &DeployConfig, plan: &NodePlan, remote_compose_path: &str) {
    let project = cfg.resolved_project();
    let extra = if cfg.remove_volumes { " -v" } else { "" };
    let down = format!(
        "{}; $C -p '{}' -f '{}' down --remove-orphans{}",
        compose_bin_check(),
        project,
        remote_compose_path,
        extra
    );
    let _ = ssh_run(&plan.target, &down, false);
}
