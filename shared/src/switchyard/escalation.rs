//! NeMo Switchyard-compatible Escalation Router.

use crate::backends::openai_tools::response_has_tool_calls;
use crate::switchyard::client::ModelClient;
use crate::switchyard::config::{EscalationSettings, ModelSpec, SwitchyardConfig};
use anyhow::Result;
use chrono::Utc;
use indexmap::IndexMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

pub const DEFAULT_JUDGE_PROMPT: &str = "You are a trajectory judge for an AI coding agent. The weak (cheaper) model just produced a reply. Decide whether the session should ESCALATE to a stronger model because the weak model is stuck, looping, repeatedly failing, drifting, or producing low-quality / incorrect code.\n\nRespond with ONLY a JSON object (no markdown fences) using this schema:\n  {\"verdict\": \"escalate\" | \"decline\", \"reason\": \"<short explanation>\"}\n\nUse escalate when you see sustained difficulty: repeated errors, tool loops, contradictions, inability to make progress, or clearly wrong code after retries. Use decline when the weak model is making reasonable progress.";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EscalationState {
    pub session_id: String,
    pub streak: usize,
    pub latched: bool,
    pub latched_at: Option<i64>,
    pub last_verdict: String,
    pub last_reason: String,
    pub turns: usize,
}

impl EscalationState {
    pub fn new(session_id: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            streak: 0,
            latched: false,
            latched_at: None,
            last_verdict: String::new(),
            last_reason: String::new(),
            turns: 0,
        }
    }

    pub fn to_dict(&self) -> Value {
        json!({
            "session_id": self.session_id,
            "streak": self.streak,
            "latched": self.latched,
            "latched_at": self.latched_at,
            "last_verdict": self.last_verdict,
            "last_reason": self.last_reason,
            "turns": self.turns,
        })
    }
}

#[derive(Clone, Debug)]
pub struct EscalationResult {
    pub response: Value,
    pub assistant_text: String,
    pub served_by: String,
    pub model: ModelSpec,
    pub escalated: bool,
    pub latched: bool,
    pub judge_verdict: String,
    pub judge_reason: String,
    pub streak: usize,
    pub state: Option<EscalationState>,
}

impl EscalationResult {
    pub fn to_meta(&self) -> Value {
        json!({
            "served_by": self.served_by,
            "model_id": self.model.id,
            "model_name": self.model.name,
            "escalated": self.escalated,
            "latched": self.latched,
            "judge_verdict": if self.judge_verdict.is_empty() { Value::Null } else { json!(self.judge_verdict) },
            "judge_reason": if self.judge_reason.is_empty() { Value::Null } else { json!(self.judge_reason) },
            "streak": self.streak,
        })
    }
}

#[derive(Default, Debug)]
struct EscalationStats {
    weak_calls: u64,
    strong_calls: u64,
    judge_calls: u64,
    escalations: u64,
    judge_errors: u64,
}

#[derive(Clone, Debug)]
pub struct EscalationRouter {
    pub weak: ModelSpec,
    pub strong: ModelSpec,
    pub judge: ModelSpec,
    pub settings: EscalationSettings,
    weak_client: ModelClient,
    strong_client: ModelClient,
    judge_client: ModelClient,
    sessions: Arc<RwLock<IndexMap<String, EscalationState>>>,
    stats: Arc<RwLock<EscalationStats>>,
}

impl EscalationRouter {
    pub fn new(
        weak: ModelSpec,
        strong: ModelSpec,
        judge: ModelSpec,
        settings: EscalationSettings,
    ) -> Self {
        let weak_client = ModelClient::new(weak.clone());
        let strong_client = ModelClient::new(strong.clone());
        let judge_client = ModelClient::new(judge.clone());
        Self {
            weak,
            strong,
            judge,
            settings,
            weak_client,
            strong_client,
            judge_client,
            sessions: Arc::new(RwLock::new(IndexMap::new())),
            stats: Arc::new(RwLock::new(EscalationStats::default())),
        }
    }

    pub fn is_session_latched(&self, session_id: &str) -> bool {
        let sid = if session_id.is_empty() {
            "default"
        } else {
            session_id
        };
        let guard = self.sessions.read();
        guard.get(sid).map(|s| s.latched).unwrap_or(false)
    }

    pub fn reset_session(&self, session_id: &str) {
        let mut guard = self.sessions.write();
        guard.shift_remove(session_id);
    }

    pub fn clear_sessions(&self) {
        self.sessions.write().clear();
    }

    pub fn stats(&self) -> Value {
        let s = self.stats.read();
        let sess = self.sessions.read();
        let latched_sessions = sess.values().filter(|v| v.latched).count();
        json!({
            "weak_calls": s.weak_calls,
            "strong_calls": s.strong_calls,
            "judge_calls": s.judge_calls,
            "escalations": s.escalations,
            "judge_errors": s.judge_errors,
            "active_sessions": sess.len(),
            "latched_sessions": latched_sessions,
            "confirmations": self.settings.confirmations,
            "weak_model": self.weak.id,
            "strong_model": self.strong.id,
            "judge_model": self.judge.id,
        })
    }
}

impl EscalationRouter {
    fn get_or_create_state(&self, session_id: &str) -> EscalationState {
        let sid = if session_id.is_empty() {
            "default"
        } else {
            session_id
        };
        let mut guard = self.sessions.write();
        if let Some(st) = guard.shift_remove(sid) {
            let out = st.clone();
            guard.insert(sid.to_string(), st);
            out
        } else {
            let st = EscalationState::new(sid);
            guard.insert(sid.to_string(), st.clone());
            while guard.len() > self.settings.max_sessions.max(1) {
                guard.shift_remove_index(0);
            }
            st
        }
    }

    fn save_state(&self, state: EscalationState) {
        let mut guard = self.sessions.write();
        guard.shift_remove(&state.session_id);
        guard.insert(state.session_id.clone(), state);
    }

    fn run_judge(&self, messages: &[Value], weak_reply: &str) -> (String, String) {
        self.stats.write().judge_calls += 1;
        let prompt = if self.settings.prompt.is_empty() {
            DEFAULT_JUDGE_PROMPT
        } else {
            &self.settings.prompt
        };
        let mut transcript = Vec::new();
        for m in messages.iter().rev().take(self.settings.recent_turn_window).rev() {
            let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let content = m.get("content").and_then(|v| v.as_str()).unwrap_or("");
            transcript.push(format!(
                "[{role}]: {}",
                content
                    .chars()
                    .take(self.settings.window_message_chars)
                    .collect::<String>()
            ));
        }
        transcript.push(format!(
            "[weak_assistant_candidate]: {}",
            weak_reply
                .chars()
                .take(self.settings.window_message_chars)
                .collect::<String>()
        ));
        let judge_msgs = vec![
            json!({"role": "system", "content": prompt}),
            json!({"role": "user", "content": transcript.join("\n\n")}),
        ];
        match self.judge_client.chat(
            &judge_msgs,
            Some(0.0),
            Some(self.settings.max_output_tokens),
            None,
            None,
        ) {
            Ok(resp) => {
                let text = self.judge_client.extract_assistant_text(&resp);
                if let Ok(parsed) = serde_json::from_str::<Value>(text.trim()) {
                    let v = parsed
                        .get("verdict")
                        .and_then(|x| x.as_str())
                        .unwrap_or("decline")
                        .to_ascii_lowercase();
                    let reason = parsed
                        .get("reason")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string();
                    if v == "escalate" {
                        return ("escalate".to_string(), reason);
                    }
                    return ("decline".to_string(), reason);
                }
                if text.to_ascii_lowercase().contains("\"escalate\"") {
                    ("escalate".to_string(), text)
                } else {
                    ("decline".to_string(), text)
                }
            }
            Err(e) => {
                self.stats.write().judge_errors += 1;
                ("decline".to_string(), format!("judge_error: {e}"))
            }
        }
    }
}

impl EscalationRouter {
    pub fn route(
        &self,
        messages: &[Value],
        session_id: &str,
        temperature: Option<f64>,
        max_tokens: Option<usize>,
        extra: Option<&Value>,
    ) -> Result<EscalationResult> {
        let mut state = self.get_or_create_state(session_id);
        state.turns += 1;

        if state.latched {
            self.stats.write().strong_calls += 1;
            let mut resp = self
                .strong_client
                .chat(messages, temperature, max_tokens, extra, None)?;
            let text = self.strong_client.extract_assistant_text(&resp);
            resp["switchyard"] = json!({
                "served_by": "strong",
                "latched": true,
                "escalated": false,
                "streak": state.streak,
                "route": "escalation",
            });
            self.save_state(state.clone());
            return Ok(EscalationResult {
                response: resp,
                assistant_text: text,
                served_by: "strong".to_string(),
                model: self.strong.clone(),
                escalated: false,
                latched: true,
                judge_verdict: String::new(),
                judge_reason: String::new(),
                streak: state.streak,
                state: Some(state),
            });
        }

        self.stats.write().weak_calls += 1;
        let mut weak_resp = self
            .weak_client
            .chat(messages, temperature, max_tokens, extra, None)?;
        let weak_text = self.weak_client.extract_assistant_text(&weak_resp);

        let (verdict, reason) = self.run_judge(messages, &weak_text);
        state.last_verdict = verdict.clone();
        state.last_reason = reason.clone();
        if verdict == "escalate" {
            state.streak += 1;
        } else {
            state.streak = 0;
        }

        if state.streak >= self.settings.confirmations.max(1) {
            state.latched = true;
            state.latched_at = Some(Utc::now().timestamp());
            self.stats.write().escalations += 1;

            if response_has_tool_calls(&weak_resp) {
                weak_resp["switchyard"] = json!({
                    "served_by": "weak",
                    "latched": true,
                    "escalated": true,
                    "judge_verdict": verdict,
                    "judge_reason": reason,
                    "streak": state.streak,
                    "route": "escalation",
                    "deferred_strong": true,
                });
                self.save_state(state.clone());
                return Ok(EscalationResult {
                    response: weak_resp,
                    assistant_text: weak_text,
                    served_by: "weak".to_string(),
                    model: self.weak.clone(),
                    escalated: true,
                    latched: true,
                    judge_verdict: verdict,
                    judge_reason: reason,
                    streak: state.streak,
                    state: Some(state),
                });
            }

            self.stats.write().strong_calls += 1;
            let mut strong_resp = self
                .strong_client
                .chat(messages, temperature, max_tokens, extra, None)?;
            let strong_text = self.strong_client.extract_assistant_text(&strong_resp);
            strong_resp["switchyard"] = json!({
                "served_by": "strong",
                "latched": true,
                "escalated": true,
                "judge_verdict": verdict,
                "judge_reason": reason,
                "streak": state.streak,
                "route": "escalation",
                "discarded_weak": true,
            });
            self.save_state(state.clone());
            return Ok(EscalationResult {
                response: strong_resp,
                assistant_text: strong_text,
                served_by: "strong".to_string(),
                model: self.strong.clone(),
                escalated: true,
                latched: true,
                judge_verdict: verdict,
                judge_reason: reason,
                streak: state.streak,
                state: Some(state),
            });
        }

        weak_resp["switchyard"] = json!({
            "served_by": "weak",
            "latched": false,
            "escalated": false,
            "judge_verdict": verdict,
            "judge_reason": reason,
            "streak": state.streak,
            "route": "escalation",
        });
        self.save_state(state.clone());
        Ok(EscalationResult {
            response: weak_resp,
            assistant_text: weak_text,
            served_by: "weak".to_string(),
            model: self.weak.clone(),
            escalated: false,
            latched: false,
            judge_verdict: verdict,
            judge_reason: reason,
            streak: state.streak,
            state: Some(state),
        })
    }
}

pub fn build_escalation_router(cfg: &SwitchyardConfig) -> Option<EscalationRouter> {
    let weak = cfg.weak()?.clone();
    let strong = cfg.strong()?.clone();
    let judge = cfg.judge().cloned().unwrap_or_else(|| weak.clone());
    Some(EscalationRouter::new(
        weak,
        strong,
        judge,
        cfg.escalation.clone(),
    ))
}
