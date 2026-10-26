//! CLI: cargo run -p model-serving-shared --bin deploy -- up|down|status|generate|inventory|plan

use clap::Parser;
use model_serving_shared::deploy::{
    compose_down, compose_ps, docker_available, ensure_model_runtime, generate_compose,
    generate_compose_for_node_plan, inventory_nodes, load_deploy_config, parse_deploy_nodes,
    remote_dir, schedule_replicas, teardown_node_plan,
};
use serde_json::json;
use std::env;

#[derive(Parser, Debug)]
#[command(about = "Deploy/teardown NIM, vLLM, or SGLang model runtimes")]
struct Cli {
    #[arg(value_parser = ["up", "down", "status", "generate", "inventory", "plan"])]
    action: String,

    #[arg(long, default_value = "vllm", value_parser = ["nim", "vllm", "sglang"])]
    engine: String,

    #[arg(long, default_value = "cli")]
    app_name: String,

    #[arg(long, default_value = "")]
    model: String,

    #[arg(long, default_value = "")]
    nodes: String,

    #[arg(long)]
    no_teardown_register: bool,
}

fn main() {
    let args = Cli::parse();
    if !args.nodes.is_empty() {
        env::set_var("DEPLOY_NODES", &args.nodes);
    }

    if args.action == "inventory" {
        let targets = if args.nodes.is_empty() {
            None
        } else {
            Some(parse_deploy_nodes(Some(&args.nodes)))
        };
        for n in inventory_nodes(targets) {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "target": n.target,
                    "advertise": n.advertise,
                    "gpu_ids": n.gpu_ids,
                    "gpu_count": n.gpu_count(),
                    "docker_ok": n.docker_ok,
                    "error": if n.error.is_empty() { serde_json::Value::Null } else { json!(n.error) },
                }))
                .unwrap_or_default()
            );
        }
        return;
    }

    let model_opt = if args.model.is_empty() {
        None
    } else {
        Some(args.model.as_str())
    };
    let cfg = load_deploy_config(&args.engine, &args.app_name, model_opt);

    if args.action == "plan" {
        let nodes = inventory_nodes(if cfg.deploy_nodes.is_empty() {
            None
        } else {
            Some(cfg.deploy_nodes.clone())
        });
        match schedule_replicas(&cfg, &nodes) {
            Ok(plans) => {
                for plan in plans {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "target": plan.target,
                            "advertise": plan.advertise,
                            "replicas": plan.replicas,
                        }))
                        .unwrap_or_default()
                    );
                }
            }
            Err(e) => {
                eprintln!("plan error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    if args.action == "generate" {
        if cfg.is_multi_node() {
            let nodes = inventory_nodes(Some(cfg.deploy_nodes.clone()));
            if let Ok(plans) = schedule_replicas(&cfg, &nodes) {
                let mut urls = Vec::new();
                for plan in plans {
                    if let Ok((path, u)) = generate_compose_for_node_plan(&cfg, &plan) {
                        println!("{}", path.display());
                        urls.extend(u);
                    }
                }
                println!("BACKEND_URLS={}", urls.join(","));
            }
        } else if let Ok((path, urls)) = generate_compose(&cfg) {
            println!("{}", path.display());
            println!("BACKEND_URLS={}", urls.join(","));
        }
        return;
    }

    if args.action == "up" {
        let handle = ensure_model_runtime(&args.engine, &args.app_name, model_opt, &[], "");
        println!(
            "{}",
            serde_json::to_string_pretty(&handle.to_dict()).unwrap_or_default()
        );
        println!("BACKEND_URLS={}", handle.urls.join(","));
        return;
    }

    let project = cfg.resolved_project();
    if args.action == "status" {
        if !docker_available() {
            println!("docker not available");
            std::process::exit(1);
        }
        if let Ok((path, urls)) = generate_compose(&cfg) {
            println!("{}", compose_ps(&project, &path));
            println!("urls: {:?}", urls);
        }
        return;
    }

    if args.action == "down" {
        if cfg.is_multi_node() {
            let nodes = inventory_nodes(Some(cfg.deploy_nodes.clone()));
            if let Ok(plans) = schedule_replicas(&cfg, &nodes) {
                for plan in plans {
                    let safe = plan.target.replace(['@', ':'], "_");
                    let remote_path = format!("{}/{project}-{safe}.yml", remote_dir(&cfg));
                    teardown_node_plan(&cfg, &plan, &remote_path);
                    println!("down {}", plan.target);
                }
            }
            return;
        }
        if let Ok((path, _)) = generate_compose(&cfg) {
            compose_down(&project, &path, cfg.remove_volumes);
            println!("down {project}");
        }
    }
}
