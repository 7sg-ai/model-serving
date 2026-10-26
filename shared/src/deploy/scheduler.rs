//! Pack model replicas onto inventoried nodes by free GPUs.

use crate::deploy::config::DeployConfig;
use crate::deploy::inventory::NodeInventory;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplicaPlacement {
    pub replica_index: usize,
    pub target: String,
    pub advertise: String,
    pub device_ids: Vec<String>,
    pub host_port: u16,
    pub service_name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodePlan {
    pub target: String,
    pub advertise: String,
    pub replicas: Vec<ReplicaPlacement>,
}

impl NodePlan {
    pub fn urls(&self) -> Vec<String> {
        self.replicas
            .iter()
            .map(|r| format!("http://{}:{}/v1/chat/completions", r.advertise, r.host_port))
            .collect()
    }
}

pub fn schedule_replicas(cfg: &DeployConfig, nodes: &[NodeInventory]) -> Result<Vec<NodePlan>> {
    let usable: Vec<&NodeInventory> = nodes
        .iter()
        .filter(|n| n.docker_ok && n.gpu_count() > 0)
        .collect();
    if usable.is_empty() {
        return Err(anyhow!(
            "No usable GPU nodes after inventory. Check DEPLOY_NODES, SSH, docker, and nvidia-smi."
        ));
    }

    let gpr = cfg.gpus_per_replica();
    let replicas = cfg.resolved_replicas();
    let total_gpus: usize = usable.iter().map(|n| n.gpu_count()).sum();
    let need = replicas * gpr;
    if total_gpus < need {
        return Err(anyhow!(
            "Not enough GPUs across nodes: have {total_gpus}, need {need} ({replicas} replicas x {gpr} GPUs)."
        ));
    }

    let mut free: BTreeMap<String, Vec<String>> = usable
        .iter()
        .map(|n| (n.target.clone(), n.gpu_ids.clone()))
        .collect();
    let mut port_cursor: BTreeMap<String, u16> =
        usable.iter().map(|n| (n.target.clone(), cfg.port_base)).collect();
    let mut local_svc_idx: BTreeMap<String, usize> =
        usable.iter().map(|n| (n.target.clone(), 0)).collect();
    let mut plans: BTreeMap<String, NodePlan> = usable
        .iter()
        .map(|n| {
            (
                n.target.clone(),
                NodePlan {
                    target: n.target.clone(),
                    advertise: n.advertise.clone(),
                    replicas: Vec::new(),
                },
            )
        })
        .collect();

    for r in 0..replicas {
        let mut best_target: Option<String> = None;
        let mut best_leftover = usize::MAX;
        for (t, gpus) in &free {
            if gpus.len() >= gpr {
                let leftover = gpus.len() - gpr;
                if leftover < best_leftover {
                    best_leftover = leftover;
                    best_target = Some(t.clone());
                }
            }
        }
        let Some(t) = best_target else {
            return Err(anyhow!("Failed to pack replica {r} (fragmentation)"));
        };
        let gpus = free.get_mut(&t).unwrap();
        let devices: Vec<String> = gpus.drain(0..gpr).collect();
        let port = *port_cursor.get(&t).unwrap();
        port_cursor.insert(t.clone(), port + 1);
        let svc_i = *local_svc_idx.get(&t).unwrap();
        local_svc_idx.insert(t.clone(), svc_i + 1);
        let plan = plans.get_mut(&t).unwrap();
        plan.replicas.push(ReplicaPlacement {
            replica_index: r,
            target: t.clone(),
            advertise: plan.advertise.clone(),
            device_ids: devices,
            host_port: port,
            service_name: format!("{}-{svc_i}", cfg.engine),
        });
    }

    Ok(plans
        .into_values()
        .filter(|p| !p.replicas.is_empty())
        .collect())
}
