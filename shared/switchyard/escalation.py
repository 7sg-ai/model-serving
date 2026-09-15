"""
NeMo Switchyard-compatible Escalation Router.

Behavior (matches docs/routing_algorithms/escalation_router_routing.md):

  For each turn in an unlatched session:
    1. Call weak_target, buffer reply
    2. Ask classifier_target (judge) to evaluate the completed turn
    3. escalate verdict → increment streak; decline → reset streak
    4. If streak < confirmations → serve weak reply
    5. If streak >= confirmations → discard weak, call strong, latch session

  Latched sessions skip the judge and go straight to strong.
  Judge failures fail-open (serve weak, keep streak).
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


from .client import ModelClient
from .config import EscalationSettings, ModelSpec, SwitchyardConfig

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_PROMPT = """You are a trajectory judge for an AI coding agent. \
The weak (cheaper) model just produced a reply. Decide whether the session \
should ESCALATE to a stronger model because the weak model is stuck, looping, \
repeatedly failing, drifting, or producing low-quality / incorrect code.

Respond with ONLY a JSON object (no markdown fences) using this schema:
  {"verdict": "escalate" | "decline", "reason": "<short explanation>"}

Use escalate when you see sustained difficulty: repeated errors, tool loops, \
contradictions, inability to make progress, or clearly wrong code after retries. \
Use decline when the weak model is making reasonable progress."""



@dataclass
class EscalationState:
    session_id: str
    streak: int = 0
    latched: bool = False
    latched_at: Optional[float] = None
    last_verdict: str = ""
    last_reason: str = ""
    turns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "streak": self.streak,
            "latched": self.latched,
            "latched_at": self.latched_at,
            "last_verdict": self.last_verdict,
            "last_reason": self.last_reason,
            "turns": self.turns,
        }


@dataclass
class EscalationResult:
    """Outcome of one routed turn."""

    response: Dict[str, Any]
    assistant_text: str
    served_by: str  # weak | strong
    model: ModelSpec
    escalated: bool = False
    latched: bool = False
    judge_verdict: str = ""
    judge_reason: str = ""
    streak: int = 0
    state: Optional[EscalationState] = None

    def to_meta(self) -> Dict[str, Any]:
        return {
            "served_by": self.served_by,
            "model_id": self.model.id,
            "model_name": self.model.name,
            "escalated": self.escalated,
            "latched": self.latched,
            "judge_verdict": self.judge_verdict or None,
            "judge_reason": self.judge_reason or None,
            "streak": self.streak,
        }


class EscalationRouter:
    """
    In-process escalation router with session latch state.
    """

    def __init__(
        self,
        weak: ModelSpec,
        strong: ModelSpec,
        judge: ModelSpec,
        settings: Optional[EscalationSettings] = None,
        *,
        local_generate: Optional[
            Callable[[ModelSpec, List[Dict[str, Any]], float, int], Tuple[str, Dict[str, Any]]]
        ] = None,
    ):
        self.weak = weak
        self.strong = strong
        self.judge = judge
        self.settings = settings or EscalationSettings()
        self._local_generate = local_generate
        self._states: Dict[str, EscalationState] = {}
        self._lock = threading.Lock()
        self._weak_client = ModelClient(weak)
        self._strong_client = ModelClient(strong)
        self._judge_client = ModelClient(judge)

    def get_state(self, session_id: str) -> EscalationState:
        sid = session_id or "default"
        with self._lock:
            if sid not in self._states:
                self._states[sid] = EscalationState(session_id=sid)
            return self._states[sid]

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id or "default", None)

    def is_latched(self, session_id: str) -> bool:
        """True when this session is already latched to the strong tier."""
        return bool(self.get_state(session_id or "default").latched)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            latched = sum(1 for s in self._states.values() if s.latched)
            return {
                "sessions": len(self._states),
                "latched_sessions": latched,
                "confirmations": self.settings.confirmations,
                "weak": self.weak.id,
                "strong": self.strong.id,
                "judge": self.judge.id,
            }

    def _call_model(
        self,
        spec: ModelSpec,
        client: ModelClient,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> Tuple[Dict[str, Any], str]:
        # Prefer local generator when this model is marked local and hook provided
        if self._local_generate and spec.deployment.load_weights_locally:
            text, meta = self._local_generate(spec, messages, temperature, max_tokens)
            result = {
                "id": meta.get("id", "chatcmpl-local"),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": spec.id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": meta.get("usage", {}),
                "switchyard_target": spec.name,
                "switchyard_model_id": spec.id,
            }
            return result, text

        if not client.pool.urls and not spec.chat_url():
            if self._local_generate:
                text, meta = self._local_generate(spec, messages, temperature, max_tokens)
                result = {
                    "id": meta.get("id", "chatcmpl-local"),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": spec.id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": meta.get("usage", {}),
                    "switchyard_target": spec.name,
                    "switchyard_model_id": spec.id,
                }
                return result, text
            raise RuntimeError(
                f"Model '{spec.name}' has no backend URL and no local generator"
            )

        result = client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return result, client.extract_assistant_text(result)

    def _truncate_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        window = self.settings.recent_turn_window
        char_cap = self.settings.window_message_chars
        recent = list(messages[-window:]) if window > 0 else list(messages)
        out: List[Dict[str, str]] = []
        for m in recent:
            content = m.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            if char_cap > 0 and len(content) > char_cap:
                content = content[:char_cap] + "…"
            out.append({"role": str(m.get("role", "user")), "content": content})
        return out

    def _parse_verdict(self, text: str) -> Tuple[str, str]:
        """Return (verdict, reason) where verdict is escalate|decline."""
        text = (text or "").strip()
        if not text:
            return "decline", "empty_judge_response"

        # Strip markdown fences if present
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence:
            text = fence.group(1).strip()

        # Try JSON object
        try:
            # Find first {...}
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                obj = json.loads(text[start : end + 1])
                verdict = str(obj.get("verdict", obj.get("decision", ""))).strip().lower()
                reason = str(obj.get("reason", obj.get("explanation", "")) or "")
                if verdict in ("escalate", "escalation", "strong", "yes", "true"):
                    return "escalate", reason
                if verdict in ("decline", "keep", "weak", "no", "false"):
                    return "decline", reason
        except json.JSONDecodeError:
            pass

        lower = text.lower()
        if re.search(r"\bescalate\b", lower) and not re.search(
            r"\bdo not escalate\b|\bdont escalate\b|\bdon't escalate\b", lower
        ):
            return "escalate", text[:200]
        if re.search(r"\bdecline\b|\bkeep weak\b|\bno escalation\b", lower):
            return "decline", text[:200]

        # Ambiguous → fail-open decline
        return "decline", f"unparseable:{text[:120]}"

    def _judge(
        self,
        messages: List[Dict[str, Any]],
        weak_reply: str,
    ) -> Tuple[str, str]:
        """Call judge; returns (verdict, reason). Failures → decline if fail_open."""
        transcript = self._truncate_messages(messages)
        transcript.append(
            {
                "role": "assistant",
                "content": (weak_reply or "")[: self.settings.window_message_chars],
            }
        )
        prompt = self.settings.prompt or DEFAULT_JUDGE_PROMPT
        judge_messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    "Conversation transcript (recent turns):\n"
                    + json.dumps(transcript, ensure_ascii=False, indent=2)
                    + "\n\nReturn the JSON verdict now."
                ),
            },
        ]
        try:
            result = self._judge_client.chat(
                judge_messages,
                temperature=0.0,
                max_tokens=self.settings.max_output_tokens,
            )
            text = self._judge_client.extract_assistant_text(result)
            return self._parse_verdict(text)
        except Exception as exc:
            logger.warning("Escalation judge failed (fail_open=%s): %s", self.settings.fail_open, exc)
            if self.settings.fail_open:
                return "decline", f"judge_error:{exc}"
            raise

    def route(
        self,
        messages: List[Dict[str, Any]],
        *,
        session_id: str = "default",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> EscalationResult:
        state = self.get_state(session_id)
        state.turns += 1

        # Already latched → strong only
        if state.latched:
            temp = self.strong.temperature if temperature is None else temperature
            mt = self.strong.max_tokens if max_tokens is None else max_tokens
            # Cap to model context
            mt = max(1, min(int(mt), self.strong.context_window - 512))
            resp, text = self._call_model(
                self.strong, self._strong_client, messages, temp, mt
            )
            resp = dict(resp)
            resp["switchyard"] = {
                "served_by": "strong",
                "latched": True,
                "escalated": True,
                "streak": state.streak,
                "route": "escalation",
            }
            return EscalationResult(
                response=resp,
                assistant_text=text,
                served_by="strong",
                model=self.strong,
                escalated=True,
                latched=True,
                streak=state.streak,
                state=state,
            )

        # 1) Weak call
        w_temp = self.weak.temperature if temperature is None else temperature
        w_mt = self.weak.max_tokens if max_tokens is None else max_tokens
        w_mt = max(1, min(int(w_mt), self.weak.context_window - 512))
        weak_resp, weak_text = self._call_model(
            self.weak, self._weak_client, messages, w_temp, w_mt
        )

        # 2) Judge
        verdict, reason = self._judge(messages, weak_text)
        state.last_verdict = verdict
        state.last_reason = reason

        if verdict == "escalate":
            state.streak += 1
        else:
            state.streak = 0

        # 3) Confirmed escalation?
        if state.streak >= max(1, self.settings.confirmations):
            state.latched = True
            state.latched_at = time.time()
            s_temp = self.strong.temperature if temperature is None else temperature
            s_mt = self.strong.max_tokens if max_tokens is None else max_tokens
            s_mt = max(1, min(int(s_mt), self.strong.context_window - 512))
            strong_resp, strong_text = self._call_model(
                self.strong, self._strong_client, messages, s_temp, s_mt
            )
            strong_resp = dict(strong_resp)
            strong_resp["switchyard"] = {
                "served_by": "strong",
                "latched": True,
                "escalated": True,
                "judge_verdict": verdict,
                "judge_reason": reason,
                "streak": state.streak,
                "route": "escalation",
                "discarded_weak": True,
            }
            return EscalationResult(
                response=strong_resp,
                assistant_text=strong_text,
                served_by="strong",
                model=self.strong,
                escalated=True,
                latched=True,
                judge_verdict=verdict,
                judge_reason=reason,
                streak=state.streak,
                state=state,
            )

        # Serve weak
        weak_resp = dict(weak_resp)
        weak_resp["switchyard"] = {
            "served_by": "weak",
            "latched": False,
            "escalated": False,
            "judge_verdict": verdict,
            "judge_reason": reason,
            "streak": state.streak,
            "route": "escalation",
        }
        return EscalationResult(
            response=weak_resp,
            assistant_text=weak_text,
            served_by="weak",
            model=self.weak,
            escalated=False,
            latched=False,
            judge_verdict=verdict,
            judge_reason=reason,
            streak=state.streak,
            state=state,
        )


    def open_strong_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Open a live SSE response from the strong tier (caller proxies it)."""
        s_temp = self.strong.temperature if temperature is None else temperature
        s_mt = self.strong.max_tokens if max_tokens is None else max_tokens
        s_mt = max(1, min(int(s_mt), self.strong.context_window - 512))
        if self._local_generate is not None:
            raise RuntimeError(
                "open_strong_stream requires a remote strong backend "
                "(_local_generate cannot produce SSE)"
            )
        return self._strong_client.chat_stream(
            messages, temperature=s_temp, max_tokens=s_mt
        )

    def route_unlatched_commit(
        self,
        messages: List[Dict[str, Any]],
        *,
        session_id: str = "default",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> EscalationResult:
        """
        Run weak+judge for an unlatched session and commit state.

        On confirmed escalate, does not call strong — caller should open a
        live strong stream via open_strong_stream (discard weak this turn).
        """
        state = self.get_state(session_id)
        state.turns += 1

        if state.latched:
            return self.route(
                messages,
                session_id=session_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        w_temp = self.weak.temperature if temperature is None else temperature
        w_mt = self.weak.max_tokens if max_tokens is None else max_tokens
        w_mt = max(1, min(int(w_mt), self.weak.context_window - 512))
        weak_resp, weak_text = self._call_model(
            self.weak, self._weak_client, messages, w_temp, w_mt
        )

        verdict, reason = self._judge(messages, weak_text)
        state.last_verdict = verdict
        state.last_reason = reason

        if verdict == "escalate":
            state.streak += 1
        else:
            state.streak = 0

        if state.streak >= max(1, self.settings.confirmations):
            state.latched = True
            state.latched_at = time.time()
            placeholder: Dict[str, Any] = {
                "switchyard": {
                    "served_by": "strong",
                    "latched": True,
                    "escalated": True,
                    "judge_verdict": verdict,
                    "judge_reason": reason,
                    "streak": state.streak,
                    "route": "escalation",
                    "discarded_weak": True,
                    "stream_strong": True,
                }
            }
            return EscalationResult(
                response=placeholder,
                assistant_text="",
                served_by="strong",
                model=self.strong,
                escalated=True,
                latched=True,
                judge_verdict=verdict,
                judge_reason=reason,
                streak=state.streak,
                state=state,
            )

        weak_resp = dict(weak_resp)
        weak_resp["switchyard"] = {
            "served_by": "weak",
            "latched": False,
            "escalated": False,
            "judge_verdict": verdict,
            "judge_reason": reason,
            "streak": state.streak,
            "route": "escalation",
        }
        return EscalationResult(
            response=weak_resp,
            assistant_text=weak_text,
            served_by="weak",
            model=self.weak,
            escalated=False,
            latched=False,
            judge_verdict=verdict,
            judge_reason=reason,
            streak=state.streak,
            state=state,
        )



def build_escalation_router(
    cfg: SwitchyardConfig,
    local_generate: Optional[
        Callable[[ModelSpec, List[Dict[str, Any]], float, int], Tuple[str, Dict[str, Any]]]
    ] = None,
) -> Optional[EscalationRouter]:
    weak = cfg.weak()
    strong = cfg.strong()
    judge = cfg.judge()
    if not weak or not strong:
        return None
    if judge is None:
        judge = weak
    return EscalationRouter(
        weak=weak,
        strong=strong,
        judge=judge,
        settings=cfg.escalation,
        local_generate=local_generate,
    )
