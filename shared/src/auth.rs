//! Shared authentication module for all applications.
//! Supports optional login with session management backed by SQLite.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use chrono::{Duration, Utc};
use rand::RngCore;
use rusqlite::{params, Connection};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

pub fn default_auth_db_path() -> String {
    if let Ok(explicit) = env::var("AUTH_DB_PATH") {
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
        .join("users.db")
        .to_string_lossy()
        .into_owned()
}

#[derive(Clone, Debug)]
pub struct AuthManager {
    pub db_path: String,
}

impl Default for AuthManager {
    fn default() -> Self {
        Self::new(None)
    }
}

impl AuthManager {
    pub fn new(db_path: Option<&str>) -> Self {
        let resolved = db_path
            .map(|s| s.to_string())
            .unwrap_or_else(default_auth_db_path);
        let mgr = Self { db_path: resolved };
        let _ = mgr.init_db();
        mgr
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
            "CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )",
            [],
        )?;
        Ok(())
    }

    pub fn hash_password(password: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(password.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn register_user(&self, username: &str, password: &str) -> bool {
        let Ok(conn) = self.connect() else {
            return false;
        };
        let password_hash = Self::hash_password(password);
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?1, ?2)",
            params![username, password_hash],
        )
        .is_ok()
    }

    pub fn login(&self, username: &str, password: &str) -> Option<String> {
        let conn = self.connect().ok()?;
        let password_hash = Self::hash_password(password);
        let user_id: i64 = conn
            .query_row(
                "SELECT id FROM users WHERE username = ?1 AND password_hash = ?2",
                params![username, password_hash],
                |row| row.get(0),
            )
            .ok()?;

        let mut token_bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut token_bytes);
        let session_token = URL_SAFE_NO_PAD.encode(token_bytes);
        let expires_at = (Utc::now() + Duration::days(7)).to_rfc3339();

        conn.execute(
            "INSERT INTO sessions (user_id, session_token, expires_at) VALUES (?1, ?2, ?3)",
            params![user_id, session_token, expires_at],
        )
        .ok()?;

        Some(session_token)
    }

    pub fn validate_session(&self, session_token: &str) -> Option<i64> {
        let conn = self.connect().ok()?;
        let now = Utc::now().to_rfc3339();
        conn.query_row(
            "SELECT user_id FROM sessions WHERE session_token = ?1 AND expires_at > ?2",
            params![session_token, now],
            |row| row.get(0),
        )
        .ok()
    }

    pub fn logout(&self, session_token: &str) {
        if let Ok(conn) = self.connect() {
            let _ = conn.execute(
                "DELETE FROM sessions WHERE session_token = ?1",
                params![session_token],
            );
        }
    }

    pub fn get_username(&self, user_id: i64) -> Option<String> {
        let conn = self.connect().ok()?;
        conn.query_row(
            "SELECT username FROM users WHERE id = ?1",
            params![user_id],
            |row| row.get(0),
        )
        .ok()
    }
}
