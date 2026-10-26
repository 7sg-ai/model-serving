//! HuggingFace Model with WebRTC Voice Interface - Backend Server (vLLM backend) in Rust

use model_serving_shared::voice_server::{run_voice_server, VoiceServerConfig};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let static_dir = if PathBuf::from("hf-webrtc-voice/static").is_dir() {
        PathBuf::from("hf-webrtc-voice/static")
    } else {
        PathBuf::from("static")
    };
    let cfg = VoiceServerConfig {
        family: "hf".to_string(),
        app_name: "hf-webrtc-voice".to_string(),
        default_port: 5001,
        default_model: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
        static_dir,
    };
    run_voice_server(cfg).await
}
