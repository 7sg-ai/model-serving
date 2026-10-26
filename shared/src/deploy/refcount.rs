//! Simple file-based refcount for shared AUTO_DEPLOY_PROJECT stacks.

use chrono::Utc;
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;

fn state_dir() -> PathBuf {
    let base = env::var("SHARED_DATA_DIR")
        .or_else(|_| env::var("AUTO_DEPLOY_STATE_DIR"))
        .unwrap_or_default();
    let d = if !base.trim().is_empty() {
        PathBuf::from(base.trim()).join(".model-runtime")
    } else {
        PathBuf::from(env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string()))
            .join("model-serving-runtime")
    };
    let _ = fs::create_dir_all(&d);
    d
}

pub fn state_path(project: &str) -> PathBuf {
    let safe: String = project
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    state_dir().join(format!("{safe}.json"))
}

pub fn read_state(project: &str) -> Value {
    let p = state_path(project);
    if let Ok(raw) = fs::read_to_string(&p) {
        if let Ok(v) = serde_json::from_str::<Value>(&raw) {
            return v;
        }
    }
    json!({
        "project": project,
        "refcount": 0,
        "owned_pids": [],
        "compose_file": "",
    })
}

pub fn write_state(project: &str, data: &Value) {
    let p = state_path(project);
    if let Ok(s) = serde_json::to_string_pretty(data) {
        let _ = fs::write(&p, s);
    }
}

pub fn acquire(project: &str, pid: u32, compose_file: &str) -> usize {
    let mut st = read_state(project);
    let mut pids: Vec<u32> = st
        .get("owned_pids")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default();
    if !pids.contains(&pid) {
        pids.push(pid);
    }
    let count = pids.len();
    st["owned_pids"] = json!(pids);
    st["refcount"] = json!(count);
    if !compose_file.is_empty() {
        st["compose_file"] = json!(compose_file);
    }
    st["project"] = json!(project);
    st["updated_at"] = json!(Utc::now().timestamp());
    write_state(project, &st);
    count
}

pub fn release(project: &str, pid: u32) -> usize {
    let mut st = read_state(project);
    let pids: Vec<u32> = st
        .get("owned_pids")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|n| n as u32))
                .filter(|p| *p != pid)
                .collect()
        })
        .unwrap_or_default();
    let count = pids.len();
    st["owned_pids"] = json!(pids);
    st["refcount"] = json!(count);
    st["updated_at"] = json!(Utc::now().timestamp());
    write_state(project, &st);
    count
}
