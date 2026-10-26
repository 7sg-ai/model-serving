//! NIM Model IDE Coding Assistant - OpenAI & Anthropic Compatible API Server (Rust)

use model_serving_shared::ide_server::{run_ide_server, IdeServerConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = IdeServerConfig {
        family: "nim".to_string(),
        app_name: "nim-ide-assistant".to_string(),
        default_port: 8080,
        default_model: "qwen/qwen2.5-coder-32b-instruct".to_string(),
        default_api_key: "nim-coding-assistant-key".to_string(),
    };
    run_ide_server(cfg).await
}
