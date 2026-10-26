//! NIM Model with WebRTC Voice Interface - Backend Server in Rust

use model_serving_shared::voice_server::{run_voice_server, VoiceServerConfig};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let static_dir = if PathBuf::from("nim-webrtc-voice/static").is_dir() {
        PathBuf::from("nim-webrtc-voice/static")
    } else {
        PathBuf::from("static")
    };
    let cfg = VoiceServerConfig {
        family: "nim".to_string(),
        app_name: "nim-webrtc-voice".to_string(),
        default_port: 5000,
        default_model: "meta/llama-3.1-8b-instruct".to_string(),
        static_dir,
    };
    run_voice_server(cfg).await
}
