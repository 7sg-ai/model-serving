//! GPU / Docker inventory on local and SSH nodes.

use crate::deploy::ssh_util::{advertise_host, normalize_ssh_target, ssh_run};
use crate::split_csv;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeInventory {
    pub target: String,
    pub advertise: String,
    pub gpu_ids: Vec<String>,
    pub docker_ok: bool,
    pub error: String,
}

impl NodeInventory {
    pub fn gpu_count(&self) -> usize {
        self.gpu_ids.len()
    }
}

pub fn parse_deploy_nodes(raw: Option<&str>) -> Vec<String> {
    let s = raw
        .map(|r| r.to_string())
        .unwrap_or_else(|| env::var("DEPLOY_NODES").unwrap_or_default());
    split_csv(&s)
        .into_iter()
        .map(|p| normalize_ssh_target(&p))
        .collect()
}

fn probe_one(target: &str) -> NodeInventory {
    let t = normalize_ssh_target(target);
    let mut inv = NodeInventory {
        target: t.clone(),
        advertise: advertise_host(&t),
        gpu_ids: Vec::new(),
        docker_ok: false,
        error: String::new(),
    };

    match ssh_run(&t, "docker info >/dev/null 2>&1 && echo DOCKER_OK", false) {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            inv.docker_ok = stdout.contains("DOCKER_OK");
            if !inv.docker_ok {
                inv.error = format!(
                    "docker not available: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
                return inv;
            }
        }
        Err(e) => {
            inv.error = e.to_string();
            return inv;
        }
    }

    let cmd = env::var("DEPLOY_GPU_INVENTORY_CMD")
        .unwrap_or_else(|_| "nvidia-smi --query-gpu=index --format=csv,noheader".to_string());
    if let Ok(out) = ssh_run(&t, &cmd, false) {
        let stdout = String::from_utf8_lossy(&out.stdout);
        for line in stdout.lines() {
            let tok = line.split(',').next().unwrap_or("").trim();
            if !tok.is_empty() {
                inv.gpu_ids.push(tok.to_string());
            }
        }
        if inv.gpu_ids.is_empty() && !out.status.success() {
            inv.error = format!(
                "GPU inventory failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
    inv
}

pub fn inventory_nodes(targets: Option<Vec<String>>) -> Vec<NodeInventory> {
    let mut list = targets.unwrap_or_else(|| parse_deploy_nodes(None));
    if list.is_empty() {
        list.push("local".to_string());
    }
    list.iter().map(|t| probe_one(t)).collect()
}
