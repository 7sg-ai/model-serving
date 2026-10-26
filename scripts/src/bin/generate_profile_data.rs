//! Generate profile_data/ scenarios for each model-serving application (Rust).

use clap::Parser;
use serde_json::json;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

const APPS: &[&str] = &[
    "nim-streamlit-chat",
    "hf-streamlit-chat",
    "nim-webrtc-voice",
    "hf-webrtc-voice",
    "nim-ide-assistant",
    "hf-ide-assistant",
];

const TOPICS: &[&str] = &[
    "rust", "python", "typescript", "docker", "kubernetes", "linux", "networking",
    "security", "machine learning", "data structures", "algorithms", "apis", "databases",
];

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 1000)]
    count: usize,

    #[arg(long)]
    skip_audio: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    for app in APPS {
        let app_dir = PathBuf::from(app);
        if !app_dir.is_dir() {
            continue;
        }
        let out_dir = app_dir.join("profile_data");
        fs::create_dir_all(&out_dir)?;
        let out_path = out_dir.join("scenarios.jsonl");
        let file = File::create(&out_path)?;
        let mut writer = BufWriter::new(file);
        let is_ide = app.contains("ide-assistant");
        let backend = if app.starts_with("nim-") { "nim" } else { "hf" };

        for i in 0..args.count {
            let topic = TOPICS[i % TOPICS.len()];
            let row = if is_ide {
                json!({
                    "id": format!("{app}-{i:04}"),
                    "app": app,
                    "backend": backend,
                    "topic": topic,
                    "turns": [
                        {"role": "user", "content": format!("Implement a clean {topic} helper #{i} in Rust.")},
                        {"role": "assistant", "content": format!("Here is the Rust implementation for {topic} helper #{i}.")}
                    ]
                })
            } else {
                json!({
                    "id": format!("{app}-{i:04}"),
                    "app": app,
                    "backend": backend,
                    "topic": topic,
                    "question": format!("Explain {topic} concept #{i} clearly."),
                    "answer": format!("Overview of {topic} concept #{i} with practical examples.")
                })
            };
            writeln!(writer, "{}", serde_json::to_string(&row)?)?;
        }
        println!("{app}: wrote {} scenarios", args.count);
    }
    Ok(())
}
