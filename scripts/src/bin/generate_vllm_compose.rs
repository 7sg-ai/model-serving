//! Generate docker-compose.vllm.generated.yml from placement env (Rust).

use clap::Parser;
use model_serving_shared::deploy::{generate_compose, load_deploy_config};
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Generate multi-vLLM compose file")]
struct Args {
    #[arg(short, long, default_value = "deploy/docker-compose.vllm.generated.yml")]
    output: String,

    #[arg(long)]
    include_apps: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut cfg = load_deploy_config("vllm", "vllm", None);
    cfg.compose_file = args.output.clone();
    let (path, urls) = generate_compose(&cfg)?;
    let backend_csv = urls.join(",");

    let env_out = PathBuf::from(&args.output).with_extension("env");
    fs::write(
        &env_out,
        format!(
            "MODEL_DEPLOY_MODE={}\nVLLM_REPLICA_COUNT={}\nTENSOR_PARALLEL_SIZE={}\nPIPELINE_PARALLEL_SIZE={}\nGPU_PER_REPLICA={}\nVLLM_MODEL={}\nHF_MODEL_NAME={}\nBACKEND_URLS={}\nLOAD_MODEL_WEIGHTS=false\n",
            cfg.deploy_mode,
            cfg.resolved_replicas(),
            cfg.tensor_parallel_size,
            cfg.pipeline_parallel_size,
            cfg.gpus_per_replica(),
            cfg.model_id,
            cfg.model_id,
            backend_csv
        ),
    )?;
    println!("Wrote {}", path.display());
    println!("Wrote {}", env_out.display());
    println!("BACKEND_URLS={backend_csv}");
    Ok(())
}
