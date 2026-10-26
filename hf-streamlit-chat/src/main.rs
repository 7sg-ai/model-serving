//! HuggingFace Model Chat Interface (vLLM backend) in Rust

use model_serving_shared::chat_server::{run_chat_server, ChatServerConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = ChatServerConfig {
        family: "hf".to_string(),
        app_name: "hf-streamlit-chat".to_string(),
        default_port: 8501,
        default_model: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
        page_title: "HuggingFace Chat Interface".to_string(),
        page_icon: "🤗".to_string(),
    };
    run_chat_server(cfg).await
}
