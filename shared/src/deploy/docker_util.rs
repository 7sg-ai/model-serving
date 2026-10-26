//! Docker / Compose helpers.

use anyhow::{anyhow, Result};
use std::path::Path;
use std::process::Command;

pub fn docker_available() -> bool {
    Command::new("docker")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

pub fn compose_cmd() -> Result<Vec<String>> {
    if !docker_available() {
        return Err(anyhow!(
            "Docker is not installed or not on PATH. Install Docker or set AUTO_DEPLOY_MODEL=false and provide BACKEND_URLS."
        ));
    }
    if let Ok(out) = Command::new("docker").args(["compose", "version"]).output() {
        if out.status.success() {
            return Ok(vec!["docker".to_string(), "compose".to_string()]);
        }
    }
    if Command::new("docker-compose")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return Ok(vec!["docker-compose".to_string()]);
    }
    Err(anyhow!(
        "Docker Compose is not available (tried `docker compose` and `docker-compose`)."
    ))
}

pub fn compose_up(project: &str, compose_file: &Path) -> Result<()> {
    let base = compose_cmd()?;
    let mut cmd = Command::new(&base[0]);
    for arg in &base[1..] {
        cmd.arg(arg);
    }
    let out = cmd
        .args([
            "-p",
            project,
            "-f",
            &compose_file.to_string_lossy(),
            "up",
            "-d",
            "--remove-orphans",
        ])
        .output()?;
    if !out.status.success() {
        return Err(anyhow!(
            "compose up failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

pub fn compose_down(project: &str, compose_file: &Path, remove_volumes: bool) {
    let Ok(base) = compose_cmd() else {
        return;
    };
    let mut cmd = Command::new(&base[0]);
    for arg in &base[1..] {
        cmd.arg(arg);
    }
    cmd.args([
        "-p",
        project,
        "-f",
        &compose_file.to_string_lossy(),
        "down",
        "--remove-orphans",
    ]);
    if remove_volumes {
        cmd.arg("-v");
    }
    let _ = cmd.output();
}

pub fn compose_ps(project: &str, compose_file: &Path) -> String {
    let Ok(base) = compose_cmd() else {
        return "docker compose not available".to_string();
    };
    let mut cmd = Command::new(&base[0]);
    for arg in &base[1..] {
        cmd.arg(arg);
    }
    match cmd
        .args(["-p", project, "-f", &compose_file.to_string_lossy(), "ps"])
        .output()
    {
        Ok(out) => format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
        Err(e) => e.to_string(),
    }
}
