"""
Unified Switchyard router used by NIM / HF IDE assistants.

Strategies:
  - escalation  — EscalationRouter (weak → judge → strong latch)
  - capability  — one-shot judge picks weak vs strong before answering
  - random      — weighted random among selectable models
  - passthrough — single default / selected model
  - external    — proxy to switchyard-server
"""
from __future__ import annotations

import logging
import random
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import requests

from .client import ModelClient
from .config import ModelSpec, SwitchyardConfig, get_switchyard_config
from .escalation import EscalationRouter, EscalationResult, build_escalation_router

logger = logging.getLogger(__name__)

# local_generate(spec, messages, temperature, max_tokens) -> (text, meta_dict)
LocalGenerateFn = Callable[
    [ModelSpec, List[Dict[str, Any]], float, int],
    Tuple[str, Dict[str, Any]],
]


@dataclass
class StreamOutcome:
    """Result of a streaming Switchyard turn for IDE servers to proxy."""

    # Exactly one of upstream / completion is set for a successful outcome.
    upstream: Optional[requests.Response] = None
    completion: Optional[Dict[str, Any]] = None
    assistant_text: str = ""
    served: Optional[ModelSpec] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    # When True, IDE should synthesize SSE from completion (buffer path).
    synthesize_sse: bool = False

    @property
    def is_live_stream(self) -> bool:
        return self.upstream is not None


class SwitchyardRouter:
    def __init__(
        self,
        cfg: SwitchyardConfig,
        *,
        local_generate: Optional[LocalGenerateFn] = None,
        fallback_backend: Optional[Callable[[], Optional[str]]] = None,
        fallback_model_id: str = "",
    ):
        self.cfg = cfg
        self.local_generate = local_generate
        self.fallback_backend = fallback_backend
        self.fallback_model_id = fallback_model_id
        self.escalation: Optional[EscalationRouter] = None
        if cfg.enabled and cfg.strategy == "escalation":
            self.escalation = build_escalation_router(cfg, local_generate=local_generate)
        self._clients: Dict[str, ModelClient] = {}

    def _client(self, spec: ModelSpec) -> ModelClient:
        key = spec.name
        if key not in self._clients:
            self._clients[key] = ModelClient(spec)
        return self._clients[key]

    # ----- listing -----

    def list_models_payload(self) -> Dict[str, Any]:
        created = int(datetime.now().timestamp())
        data: List[Dict[str, Any]] = []

        if self.cfg.enabled:
            # Virtual route id (escalation entrypoint)
            if self.cfg.strategy in ("escalation", "capability", "random"):
                data.append(
                    {
                        "id": self.cfg.route_id,
                        "object": "model",
                        "created": created,
                        "owned_by": self.cfg.owned_by,
                        "root": "switchyard-route",
                        "strategy": self.cfg.strategy,
                        "context_window": (
                            (self.cfg.weak() or self.cfg.default_model() or ModelSpec("x", "x")).context_window
                        ),
                    }
                )
            if self.cfg.expose_models:
                for m in self.cfg.selectable_models():
                    entry = m.to_openai_model_entry(owned_by=self.cfg.owned_by)
                    entry["created"] = created
                    data.append(entry)
        elif self.fallback_model_id:
            data.append(
                {
                    "id": self.fallback_model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": self.cfg.owned_by,
                }
            )

        # Deduplicate by id
        seen = set()
        unique = []
        for item in data:
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            unique.append(item)
        return {"object": "list", "data": unique}

    def health_block(self) -> Dict[str, Any]:
        block: Dict[str, Any] = {
            "enabled": self.cfg.enabled,
            "strategy": self.cfg.strategy if self.cfg.enabled else None,
            "route_id": self.cfg.route_id if self.cfg.enabled else None,
            "models": [m.to_dict() for m in self.cfg.models] if self.cfg.enabled else [],
            "external_url": self.cfg.external_url or None,
        }
        if self.escalation:
            block["escalation"] = self.escalation.stats()
        return block

    # ----- resolution -----

    def resolve_spec(self, requested_model: Optional[str]) -> Optional[ModelSpec]:
        """Pick a ModelSpec for direct (non-route) requests."""
        if not self.cfg.enabled:
            return None
        key = (requested_model or "").strip()
        if not key or key == self.cfg.route_id:
            return None  # means: use strategy route
        if not self.cfg.allow_model_select:
            return None
        return self.cfg.model_by_id_or_name(key)

    def is_route_request(self, requested_model: Optional[str]) -> bool:
        if not self.cfg.enabled:
            return False
        key = (requested_model or "").strip()
        if not key:
            return True  # default to route
        if key == self.cfg.route_id:
            return True
        if key.lower() in ("switchyard", "auto", "escalation", "agent"):
            return True
        # Unknown model with allow_model_select → try direct; else route
        if self.cfg.allow_model_select and self.cfg.model_by_id_or_name(key):
            return False
        return True

    # ----- generation helpers -----

    def _call_spec(
        self,
        spec: ModelSpec,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        if self.local_generate and (
            spec.deployment.load_weights_locally or not spec.chat_url()
        ):
            text, meta = self.local_generate(spec, messages, temperature, max_tokens)
            result = {
                "id": meta.get("id", f"chatcmpl-{int(time.time())}"),
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

        client = self._client(spec)
        if not client.pool.urls:
            # Last resort: fallback_backend URL with this model id
            if self.fallback_backend:
                url = self.fallback_backend()
                if url:
                    # Temporarily use a synthetic client
                    from .config import DeploymentParams

                    tmp = ModelSpec(
                        name=spec.name,
                        id=spec.id,
                        role=spec.role,
                        max_tokens=spec.max_tokens,
                        temperature=spec.temperature,
                        context_window=spec.context_window,
                        request_timeout=spec.request_timeout,
                        extra_body=spec.extra_body,
                        api_key=spec.api_key,
                        api_key_env=spec.api_key_env,
                        deployment=DeploymentParams(backend_urls=[url]),
                    )
                    client = ModelClient(tmp)

        result = client.chat(
            messages, temperature=temperature, max_tokens=max_tokens, extra=extra
        )
        return result, client.extract_assistant_text(result)

    def _external_chat_url(self) -> str:
        base = (self.cfg.external_url or "").rstrip("/")
        if not base:
            raise RuntimeError("SWITCHYARD_SERVER_URL / external_url is empty")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    def _external_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.cfg.external_api_key_env:
            import os

            key = os.getenv(self.cfg.external_api_key_env, "")
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    def _proxy_external(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        stream: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Non-stream JSON proxy to switchyard-server."""
        url = self._external_chat_url()
        payload = {
            "model": model or self.cfg.route_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if extra:
            payload.update(extra)
            payload["stream"] = False
        headers = self._external_headers()
        resp = requests.post(url, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            data.setdefault(
                "switchyard", {"route": "external", "url": self.cfg.external_url}
            )
        return data

    def _proxy_external_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """SSE proxy to switchyard-server (open Response; caller must close)."""
        from shared.backends.streaming import post_chat_stream

        url = self._external_chat_url()
        headers = self._external_headers()
        api_key = ""
        auth = headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            api_key = auth[7:].strip()
        return post_chat_stream(
            url,
            messages,
            model=model or self.cfg.route_id,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            timeout=300.0,
            extra=extra,
            headers={k: v for k, v in headers.items() if k.lower() != "authorization"},
        )

    def _capability_route(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> Tuple[Dict[str, Any], str, ModelSpec]:
        weak = self.cfg.weak()
        strong = self.cfg.strong()
        judge = self.cfg.judge() or weak
        if not weak:
            raise RuntimeError("No weak/default model configured for capability routing")
        if not strong:
            strong = weak

        # Quick classifier: ask judge if task needs strong
        preview = []

        for m in messages[-6:]:
            c = m.get("content", "")
            if not isinstance(c, str):
                c = str(c)
            preview.append({"role": m.get("role", "user"), "content": c[:500]})
        judge_msgs = [
            {
                "role": "system",
                "content": (
                    "You classify coding tasks. "
                    'Return ONLY JSON: {"verdict":"strong"|"weak","reason":"..."}. '
                    "Use strong for complex multi-file refactors, architecture, hard bugs; "
                    "weak for simple edits, explanations, small snippets."
                ),
            },
            {
                "role": "user",
                "content": "Task messages:\n" + str(preview),
            },
        ]
        pick = strong
        try:
            jclient = self._client(judge)
            jres = jclient.chat(judge_msgs, temperature=0.0, max_tokens=256)
            jtext = jclient.extract_assistant_text(jres).lower()
            if "weak" in jtext and "strong" not in jtext.split("weak")[0][-20:]:
                # crude: if verdict weak
                if '"verdict": "weak"' in jtext or '"verdict":"weak"' in jtext:
                    pick = weak
                elif "verdict" in jtext and "strong" in jtext:
                    pick = strong
                else:
                    pick = weak if "weak" in jtext else strong
            elif "strong" in jtext:
                pick = strong
            else:
                pick = weak
        except Exception as exc:
            logger.warning("Capability judge failed, defaulting to weak: %s", exc)
            pick = weak

        mt = max(1, min(int(max_tokens), pick.context_window - 512))
        resp, text = self._call_spec(pick, messages, temperature, mt)
        resp = dict(resp)
        resp["switchyard"] = {
            "route": "capability",
            "served_by": pick.role,
            "model_id": pick.id,
        }
        return resp, text, pick

    def _random_route(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> Tuple[Dict[str, Any], str, ModelSpec]:
        models = self.cfg.selectable_models()
        if not models:
            raise RuntimeError("No models configured for random routing")
        pick = random.choice(models)
        mt = max(1, min(int(max_tokens), pick.context_window - 512))
        resp, text = self._call_spec(pick, messages, temperature, mt)
        resp = dict(resp)
        resp["switchyard"] = {
            "route": "random",
            "served_by": pick.name,
            "model_id": pick.id,
        }
        return resp, text, pick


    def _tools_in_extra(self, extra: Optional[Dict[str, Any]]) -> bool:
        if not extra:
            return False
        tools = extra.get("tools")
        return isinstance(tools, list) and len(tools) > 0

    def _direct_with_extra(
        self,
        spec: ModelSpec,
        messages: List[Dict[str, Any]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        extra: Optional[Dict[str, Any]],
        route: str = "direct",
    ) -> Tuple[Dict[str, Any], str, ModelSpec]:
        temp = spec.temperature if temperature is None else float(temperature)
        mt = spec.max_tokens if max_tokens is None else int(max_tokens)
        mt = max(1, min(mt, spec.context_window - 512))
        resp, text = self._call_spec(spec, messages, temp, mt, extra=extra)
        resp = dict(resp)
        resp["switchyard"] = {
            "route": route,
            "served_by": spec.name,
            "model_id": spec.id,
            "tools_passthrough": bool(self._tools_in_extra(extra)),
        }
        return resp, text, spec

    def _stream_with_extra(
        self,
        spec: ModelSpec,
        messages: List[Dict[str, Any]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        extra: Optional[Dict[str, Any]],
        route: str = "direct",
    ) -> StreamOutcome:
        temp = spec.temperature if temperature is None else float(temperature)
        mt = spec.max_tokens if max_tokens is None else int(max_tokens)
        mt = max(1, min(mt, spec.context_window - 512))
        upstream = self._open_spec_stream(spec, messages, temp, mt, extra=extra)
        return StreamOutcome(
            upstream=upstream,
            served=spec,
            meta={
                "route": route,
                "served_by": spec.name,
                "model_id": spec.id,
                "tools_passthrough": bool(self._tools_in_extra(extra)),
            },
        )

    # ----- main entry -----

    def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        *,
        requested_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        session_id: str = "default",
        stream: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str, Optional[ModelSpec]]:
        """
        Route a chat completion request.

        Returns (openai_response_dict, assistant_text, serving_ModelSpec|None).
        """
        if not self.cfg.enabled:
            raise RuntimeError("Switchyard is not enabled")

        # External proxy mode
        if self.cfg.strategy == "external" or (
            self.cfg.external_url and self.cfg.strategy == "external"
        ):
            model = requested_model or self.cfg.route_id
            temp = 0.3 if temperature is None else temperature
            mt = 32768 if max_tokens is None else max_tokens
            data = self._proxy_external(
                messages,
                model=model,
                temperature=temp,
                max_tokens=mt,
                stream=False,
                extra=extra,
            )
            text = ""
            try:
                text = data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                pass
            return data, text, None

        # Direct model selection
        if not self.is_route_request(requested_model):
            spec = self.resolve_spec(requested_model)
            if spec is None:
                raise RuntimeError(f"Unknown model: {requested_model}")
            return self._direct_with_extra(
                spec, messages, temperature, max_tokens, extra, route="direct"
            )

        # IDE tool loops: skip multi-step escalation/capability judge so tools
        # round-trip on a single known upstream model.
        if self._tools_in_extra(extra):
            spec = (
                self.cfg.default_model()
                or self.cfg.weak()
                or (self.cfg.selectable_models()[0] if self.cfg.selectable_models() else None)
            )
            if not spec:
                raise RuntimeError("No models configured for tools passthrough")
            return self._direct_with_extra(
                spec,
                messages,
                temperature,
                max_tokens,
                extra,
                route="tools_passthrough",
            )

        # Strategy routes
        strategy = self.cfg.strategy
        if strategy == "escalation":
            if not self.escalation:
                raise RuntimeError("Escalation router not configured (need weak + strong models)")
            # Use weak defaults for unspecified temp/tokens; router applies per-tier
            temp = temperature
            mt = max_tokens
            result = self.escalation.route(
                messages,
                session_id=session_id or "default",
                temperature=temp,
                max_tokens=mt,
            )
            return result.response, result.assistant_text, result.model

        if strategy == "capability":
            temp = float(temperature) if temperature is not None else (
                self.cfg.weak().temperature if self.cfg.weak() else 0.3
            )
            mt = int(max_tokens) if max_tokens is not None else (
                self.cfg.weak().max_tokens if self.cfg.weak() else 32768
            )
            return self._capability_route(messages, temp, mt)

        if strategy == "random":
            temp = float(temperature) if temperature is not None else 0.3
            mt = int(max_tokens) if max_tokens is not None else 32768
            return self._random_route(messages, temp, mt)

        # passthrough
        spec = self.cfg.default_model()
        if not spec:
            raise RuntimeError("No models configured")
        return self._direct_with_extra(
            spec, messages, temperature, max_tokens, extra, route="passthrough"
        )


    def _capability_pick(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[ModelSpec, Dict[str, Any]]:
        """Run the capability judge and return (chosen_spec, meta)."""
        weak = self.cfg.weak()
        strong = self.cfg.strong()
        judge = self.cfg.judge() or weak
        if not weak:
            raise RuntimeError("No weak/default model configured for capability routing")
        if not strong:
            strong = weak

        preview = []
        for m in messages[-6:]:
            c = m.get("content", "")
            if not isinstance(c, str):
                c = str(c)
            preview.append({"role": m.get("role", "user"), "content": c[:500]})
        judge_msgs = [
            {
                "role": "system",
                "content": (
                    "You classify coding tasks. "
                    'Return ONLY JSON: {"verdict":"strong"|"weak","reason":"..."}. '
                    "Use strong for complex multi-file refactors, architecture, hard bugs; "
                    "weak for simple edits, explanations, small snippets."
                ),
            },
            {
                "role": "user",
                "content": "Task messages:\n" + str(preview),
            },
        ]
        pick = strong
        try:
            jclient = self._client(judge)
            jres = jclient.chat(judge_msgs, temperature=0.0, max_tokens=256)
            jtext = jclient.extract_assistant_text(jres).lower()
            if "weak" in jtext and "strong" not in jtext.split("weak")[0][-20:]:
                if '"verdict": "weak"' in jtext or '"verdict":"weak"' in jtext:
                    pick = weak
                elif "verdict" in jtext and "strong" in jtext:
                    pick = strong
                else:
                    pick = weak if "weak" in jtext else strong
            elif "strong" in jtext:
                pick = strong
            else:
                pick = weak
        except Exception as exc:
            logger.warning("Capability judge failed, defaulting to weak: %s", exc)
            pick = weak

        meta = {
            "route": "capability",
            "served_by": pick.role or pick.name,
            "model_id": pick.id,
        }
        return pick, meta

    def _open_spec_stream(
        self,
        spec: ModelSpec,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        if self.local_generate is not None and not (
            spec.deployment.backend_urls or spec.deployment.coordinator_url or spec.chat_url()
        ):
            raise RuntimeError(
                f"Cannot stream model '{spec.name}': no remote backend URL "
                "(local_generate is non-stream only)"
            )
        return self._client(spec).chat_stream(
            messages, temperature=temperature, max_tokens=max_tokens, extra=extra
        )

    def chat_completions_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        requested_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        session_id: str = "default",
        extra: Optional[Dict[str, Any]] = None,
    ) -> StreamOutcome:
        """
        Route a streaming chat completion.

        - Target known before answer tokens → live upstream SSE
        - Unlatched escalation (need buffer+judge) → completion + synthesize_sse,
          except confirmed escalate which opens a live strong stream
        - External switchyard-server → live SSE proxy
        """
        if not self.cfg.enabled:
            raise RuntimeError("Switchyard is not enabled")

        sid = session_id or "default"

        # External proxy: live SSE
        if self.cfg.strategy == "external":
            model = requested_model or self.cfg.route_id
            temp = 0.3 if temperature is None else float(temperature)
            mt = 32768 if max_tokens is None else int(max_tokens)
            upstream = self._proxy_external_stream(
                messages,
                model=model,
                temperature=temp,
                max_tokens=mt,
                extra=extra,
            )
            return StreamOutcome(
                upstream=upstream,
                meta={"route": "external", "model": model},
            )

        # Direct model selection → live SSE
        if not self.is_route_request(requested_model):
            spec = self.resolve_spec(requested_model)
            if spec is None:
                raise RuntimeError(f"Unknown model: {requested_model}")
            return self._stream_with_extra(
                spec, messages, temperature, max_tokens, extra, route="direct"
            )

        if self._tools_in_extra(extra):
            spec = (
                self.cfg.default_model()
                or self.cfg.weak()
                or (self.cfg.selectable_models()[0] if self.cfg.selectable_models() else None)
            )
            if not spec:
                raise RuntimeError("No models configured for tools passthrough")
            return self._stream_with_extra(
                spec,
                messages,
                temperature,
                max_tokens,
                extra,
                route="tools_passthrough",
            )

        strategy = self.cfg.strategy

        if strategy == "escalation":
            if not self.escalation:
                raise RuntimeError(
                    "Escalation router not configured (need weak + strong models)"
                )
            esc = self.escalation
            # Already latched → live strong stream
            if esc.is_latched(sid):
                temp = temperature
                mt = max_tokens
                s_temp = esc.strong.temperature if temp is None else float(temp)
                s_mt = esc.strong.max_tokens if mt is None else int(mt)
                s_mt = max(1, min(s_mt, esc.strong.context_window - 512))
                # touch turn counter like route() would
                state = esc.get_state(sid)
                state.turns += 1
                upstream = esc.open_strong_stream(
                    messages, temperature=s_temp, max_tokens=s_mt
                )
                return StreamOutcome(
                    upstream=upstream,
                    served=esc.strong,
                    meta={
                        "route": "escalation",
                        "served_by": "strong",
                        "latched": True,
                        "model_id": esc.strong.id,
                    },
                )

            # Unlatched: buffer weak + judge; stream strong only on escalate
            result = esc.route_unlatched_commit(
                messages,
                session_id=sid,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if result.escalated and result.served_by == "strong":
                s_temp = (
                    esc.strong.temperature
                    if temperature is None
                    else float(temperature)
                )
                s_mt = (
                    esc.strong.max_tokens
                    if max_tokens is None
                    else int(max_tokens)
                )
                s_mt = max(1, min(s_mt, esc.strong.context_window - 512))
                upstream = esc.open_strong_stream(
                    messages, temperature=s_temp, max_tokens=s_mt
                )
                meta = dict(result.response.get("switchyard") or {})
                meta.setdefault("model_id", esc.strong.id)
                return StreamOutcome(
                    upstream=upstream,
                    served=esc.strong,
                    meta=meta,
                )

            # Serve buffered weak as synthesized SSE
            return StreamOutcome(
                completion=result.response,
                assistant_text=result.assistant_text,
                served=result.model,
                meta=dict(result.response.get("switchyard") or {}),
                synthesize_sse=True,
            )

        if strategy == "capability":
            chosen, meta = self._capability_pick(messages)
            temp = float(temperature) if temperature is not None else chosen.temperature
            mt = int(max_tokens) if max_tokens is not None else chosen.max_tokens
            mt = max(1, min(mt, chosen.context_window - 512))
            upstream = self._open_spec_stream(chosen, messages, temp, mt, extra=extra)
            return StreamOutcome(
                upstream=upstream,
                served=chosen,
                meta=meta,
            )

        if strategy == "random":
            models = self.cfg.selectable_models()
            if not models:
                raise RuntimeError("No models configured for random routing")
            spec = random.choice(models)
            temp = float(temperature) if temperature is not None else spec.temperature
            mt = int(max_tokens) if max_tokens is not None else spec.max_tokens
            mt = max(1, min(mt, spec.context_window - 512))
            upstream = self._open_spec_stream(spec, messages, temp, mt, extra=extra)
            return StreamOutcome(
                upstream=upstream,
                served=spec,
                meta={
                    "route": "random",
                    "served_by": spec.name,
                    "model_id": spec.id,
                },
            )

        # passthrough
        spec = self.cfg.default_model()
        if not spec:
            raise RuntimeError("No models configured")
        return self._stream_with_extra(
            spec, messages, temperature, max_tokens, extra, route="passthrough"
        )



_ROUTER: Optional[SwitchyardRouter] = None


def get_switchyard_router(
    cfg: Optional[SwitchyardConfig] = None,
    *,
    local_generate: Optional[LocalGenerateFn] = None,
    fallback_backend: Optional[Callable[[], Optional[str]]] = None,
    fallback_model_id: str = "",
    reload: bool = False,
) -> SwitchyardRouter:
    global _ROUTER
    if _ROUTER is not None and not reload and cfg is None:
        return _ROUTER
    cfg = cfg or get_switchyard_config()
    _ROUTER = SwitchyardRouter(
        cfg,
        local_generate=local_generate,
        fallback_backend=fallback_backend,
        fallback_model_id=fallback_model_id,
    )
    return _ROUTER
