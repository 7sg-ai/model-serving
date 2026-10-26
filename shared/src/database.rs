//! Shared chat history module for all applications.

use chrono::Utc;
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

pub fn default_chat_db_path() -> String {
    if let Ok(explicit) = env::var("CHAT_DB_PATH") {
        let trimmed = explicit.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    let shared = env::var("SHARED_DATA_DIR")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "shared/database".to_string());
    Path::new(&shared)
        .join("chat_history.db")
        .to_string_lossy()
        .into_owned()
}

#[derive(Clone, Debug)]
pub struct ChatHistory {
    pub db_path: String,
}

impl Default for ChatHistory {
    fn default() -> Self {
        Self::new(None)
    }
}

impl ChatHistory {
    pub fn new(db_path: Option<&str>) -> Self {
        let resolved = db_path
            .map(|s| s.to_string())
            .unwrap_or_else(default_chat_db_path);
        let ch = Self { db_path: resolved };
        let _ = ch.init_db();
        ch
    }

    fn connect(&self) -> rusqlite::Result<Connection> {
        let path = PathBuf::from(&self.db_path);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        Connection::open(path)
    }

    fn init_db(&self) -> rusqlite::Result<()> {
        let conn = self.connect()?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                app_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )",
            [],
        )?;
        Ok(())
    }

    pub fn create_conversation(&self, user_id: Option<i64>, app_name: &str) -> i64 {
        let Ok(conn) = self.connect() else {
            return 0;
        };
        if conn
            .execute(
                "INSERT INTO conversations (user_id, app_name) VALUES (?1, ?2)",
                params![user_id, app_name],
            )
            .is_ok()
        {
            conn.last_insert_rowid()
        } else {
            0
        }
    }

    pub fn add_message(
        &self,
        conversation_id: i64,
        role: &str,
        content: &str,
        metadata: Option<&Value>,
    ) {
        let Ok(conn) = self.connect() else {
            return;
        };
        let metadata_json = metadata.and_then(|m| serde_json::to_string(m).ok());
        let _ = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?1, ?2, ?3, ?4)",
            params![conversation_id, role, content, metadata_json],
        );
        let now = Utc::now().to_rfc3339();
        let _ = conn.execute(
            "UPDATE conversations SET updated_at = ?1 WHERE id = ?2",
            params![now, conversation_id],
        );
    }

    pub fn get_conversation_history(&self, conversation_id: i64) -> Vec<Value> {
        let Ok(conn) = self.connect() else {
            return Vec::new();
        };
        let Ok(mut stmt) = conn.prepare(
            "SELECT role, content, metadata, created_at FROM messages WHERE conversation_id = ?1 ORDER BY created_at, id",
        ) else {
            return Vec::new();
        };
        stmt.query_map(params![conversation_id], |row| {
            let role: String = row.get(0)?;
            let content: String = row.get(1)?;
            let metadata_raw: Option<String> = row.get(2)?;
            let created_at: Option<String> = row.get(3)?;
            let metadata = metadata_raw
                .and_then(|s| serde_json::from_str::<Value>(&s).ok())
                .unwrap_or(Value::Null);
            Ok(json!({
                "role": role,
                "content": content,
                "metadata": metadata,
                "created_at": created_at,
            }))
        })
        .map(|it| it.filter_map(Result::ok).collect())
        .unwrap_or_default()
    }

    pub fn get_user_conversations(&self, user_id: i64, app_name: Option<&str>) -> Vec<Value> {
        let Ok(conn) = self.connect() else {
            return Vec::new();
        };
        let map_row = |row: &rusqlite::Row<'_>| -> rusqlite::Result<Value> {
            Ok(json!({
                "id": row.get::<_, i64>(0)?,
                "app_name": row.get::<_, String>(1)?,
                "created_at": row.get::<_, Option<String>>(2)?,
                "updated_at": row.get::<_, Option<String>>(3)?,
            }))
        };
        if let Some(app) = app_name {
            let Ok(mut stmt) = conn.prepare(
                "SELECT id, app_name, created_at, updated_at FROM conversations WHERE user_id = ?1 AND app_name = ?2 ORDER BY updated_at DESC, id DESC",
            ) else {
                return Vec::new();
            };
            stmt.query_map(params![user_id, app], map_row)
                .map(|it| it.filter_map(Result::ok).collect())
                .unwrap_or_default()
        } else {
            let Ok(mut stmt) = conn.prepare(
                "SELECT id, app_name, created_at, updated_at FROM conversations WHERE user_id = ?1 ORDER BY updated_at DESC, id DESC",
            ) else {
                return Vec::new();
            };
            stmt.query_map(params![user_id], map_row)
                .map(|it| it.filter_map(Result::ok).collect())
                .unwrap_or_default()
        }
    }

    pub fn delete_conversation(&self, conversation_id: i64) {
        if let Ok(conn) = self.connect() {
            let _ = conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?1",
                params![conversation_id],
            );
            let _ = conn.execute(
                "DELETE FROM conversations WHERE id = ?1",
                params![conversation_id],
            );
        }
    }
}
