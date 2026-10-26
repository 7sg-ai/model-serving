//! NIM Model Chat Interface in Rust

use model_serving_shared::chat_server::{run_chat_server, ChatServerConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = ChatServerConfig {
        family: "nim".to_string(),
        app_name: "nim-streamlit-chat".to_string(),
        default_port: 8501,
        default_model: "meta/llama-3.1-8b-instruct".to_string(),
        page_title: "NIM Chat Interface".to_string(),
        page_icon: "💬".to_string(),
    };
    run_chat_server(cfg).await
}
