//! Passwordless SSH helpers for multi-node model deploy.

use anyhow::{anyhow, Result};
use std::env;
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

pub fn is_local_target(target: &str) -> bool {
    let t = target.trim().to_ascii_lowercase();
    if matches!(t.as_str(), "" | "local" | "localhost" | "127.0.0.1" | "::1") {
        return true;
    }
    if let Some((_, host)) = t.split_once('@') {
        if matches!(host, "localhost" | "127.0.0.1" | "local") {
            return true;
        }
    }
    false
}

pub fn normalize_ssh_target(raw: &str) -> String {
    let r = raw.trim();
    if r.is_empty() || is_local_target(r) {
        return "local".to_string();
    }
    if r.contains('@') {
        return r.to_string();
    }
    if let Ok(user) = env::var("DEPLOY_SSH_USER") {
        if !user.trim().is_empty() {
            return format!("{}@{}", user.trim(), r);
        }
    }
    r.to_string()
}

pub fn advertise_host(target: &str) -> String {
    let t = normalize_ssh_target(target);
    if is_local_target(&t) {
        return env::var("DEPLOY_LOCAL_ADVERTISE_HOST")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "127.0.0.1".to_string());
    }
    let host = t
        .split_once('@')
        .map(|(_, h)| h.to_string())
        .unwrap_or_else(|| t.clone());
    if let Ok(raw_map) = env::var("DEPLOY_NODE_URL_HOSTS") {
        for part in raw_map.split(',') {
            if let Some((k, v)) = part.trim().split_once(':') {
                if k.trim() == host || k.trim() == t {
                    return v.trim().to_string();
                }
            }
        }
    }
    host
}

pub fn ssh_run(target: &str, remote_cmd: &str, check: bool) -> Result<Output> {
    let t = normalize_ssh_target(target);
    let out = if is_local_target(&t) {
        Command::new("bash").args(["-lc", remote_cmd]).output()?
    } else {
        let mut cmd = Command::new("ssh");
        cmd.args([
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=15",
            &t,
            remote_cmd,
        ]);
        cmd.output()?
    };
    if check && !out.status.success() {
        return Err(anyhow!(
            "Command failed on {t}: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(out)
}

pub fn scp_put(local_path: &str, target: &str, remote_path: &str) -> Result<()> {
    let t = normalize_ssh_target(target);
    if is_local_target(&t) {
        if let Some(parent) = Path::new(remote_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::copy(local_path, remote_path)?;
        return Ok(());
    }
    if let Some((remote_dir, _)) = remote_path.rsplit_once('/') {
        let _ = ssh_run(&t, &format!("mkdir -p '{remote_dir}'"), true);
    }
    let out = Command::new("scp")
        .args([
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=15",
            local_path,
            &format!("{t}:{remote_path}"),
        ])
        .output()?;
    if !out.status.success() {
        return Err(anyhow!(
            "scp failed to {t}:{remote_path}: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}
