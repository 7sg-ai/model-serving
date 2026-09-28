"""
Multi-model Switchyard configuration.

Models can be declared via:
  1. SWITCHYARD_CONFIG=/path/to/models.json (or .toml)
  2. SWITCHYARD_MODELS='[{...}, {...}]'  (inline JSON)
  3. Convenience env vars (SWITCHYARD_WEAK_*, SWITCHYARD_STRONG_*, SWITCHYARD_JUDGE_*)
  4. Legacy single-model env (NIM_MODEL_NAME / HF_MODEL_NAME) when Switchyard is off

Each model carries independent inference + deployment parameters.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _split_csv(raw: str) -> List[str]:
    return [p.strip() for p in (raw or "").split(",") if p.strip()]


@dataclass
class DeploymentParams:
    """
    Per-model deployment / placement knobs (independent of other models).

    Mirrors shared.multinode placement fields so each Switchyard target can
    sit on different GPUs, TP sizes, or remote backends.
    """

    # Where inference runs
    backend_urls: List[str] = field(default_factory=list)
    backend_strategy: str = "round_robin"  # round_robin | random | first
    load_weights_locally: bool = False
    coordinator_url: str = ""

    # Parallelism / GPU
    deploy_mode: str = "replica"  # replica | sharded | hybrid
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    model_parallel_size: int = 1
    shard_rank: int = 0
    shard_world_size: int = 1
    shard_group: str = ""
    cuda_visible_devices: str = ""
    device_map: str = ""
    max_memory: str = ""  # JSON or CSV pairs
    max_memory_per_gpu: str = ""
    max_memory_cpu: str = ""
    torch_dtype: str = ""
    trust_remote_code: Optional[bool] = None
    offload_folder: str = ""

    # NIM / container hints (documentation + orchestration)
    nim_image: str = ""
    nim_port: int = 0
    gpu_count: int = 0
    node_selector: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def primary_backend(self) -> str:
        if self.coordinator_url:
            return self.coordinator_url.rstrip("/")
        if self.backend_urls:
            return self.backend_urls[0].rstrip("/")
        return ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Drop empty noise for health payloads
        return {k: v for k, v in d.items() if v not in (None, "", [], {}, 0, False)}


@dataclass
class ModelSpec:
    """One selectable model / Switchyard target."""

    # Logical name used in config (weak, strong, judge, or custom)
    name: str
    # Upstream model id (sent in OpenAI `model` field)
    id: str
    # Role in routing: weak | strong | judge | general | passthrough
    role: str = "general"

    # Generation defaults (override global server defaults per model)
    max_tokens: int = 32768
    temperature: float = 0.3
    context_window: int = 1048576
    request_timeout: int = 300
    top_p: float = 0.95
    extra_body: Dict[str, Any] = field(default_factory=dict)

    # Auth for this upstream (env var name or literal; empty = none)
    api_key_env: str = ""
    api_key: str = ""

    # Wire format for switchyard-server export
    format: str = "openai_chat"  # openai_chat | openai_responses | anthropic_messages
    base_url: str = ""  # OpenAI base e.g. http://host:8000/v1 (derived from backend if empty)

    deployment: DeploymentParams = field(default_factory=DeploymentParams)

    def resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env, "")
        return ""

    def chat_url(self) -> str:
        """Full chat/completions URL."""
        primary = self.deployment.primary_backend()
        if primary:
            if primary.endswith("/chat/completions"):
                return primary
            if primary.rstrip("/").endswith("/v1"):
                return primary.rstrip("/") + "/chat/completions"
            if "/v1/" in primary or primary.endswith("/v1"):
                return primary.rstrip("/") + (
                    "" if primary.endswith("chat/completions") else "/chat/completions"
                )
            return primary.rstrip("/") + "/v1/chat/completions"
        if self.base_url:
            base = self.base_url.rstrip("/")
            if base.endswith("/chat/completions"):
                return base
            if base.endswith("/v1"):
                return base + "/chat/completions"
            return base + "/v1/chat/completions"
        return ""

    def openai_base_url(self) -> str:
        """Base URL suitable for switchyard llm_clients.base_url (.../v1)."""
        if self.base_url:
            base = self.base_url.rstrip("/")
            if base.endswith("/chat/completions"):
                return base[: -len("/chat/completions")]
            return base
        url = self.chat_url()
        if url.endswith("/chat/completions"):
            return url[: -len("/chat/completions")]
        return url

    def to_openai_model_entry(self, owned_by: str = "switchyard") -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": "model",
            "owned_by": owned_by,
            "root": self.name,
            "role": self.role,
            "context_window": self.context_window,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "deployment": self.deployment.to_dict(),
            "capabilities": {
                "tools": True,
                "reasoning": any(
                    h in (self.id or "").lower()
                    for h in ("deepseek-r1", "qwen3", "qwq", "reasoning", "think", "kimi", "gpt-oss")
                ),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "id": self.id,
            "role": self.role,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "context_window": self.context_window,
            "request_timeout": self.request_timeout,
            "top_p": self.top_p,
            "extra_body": dict(self.extra_body),
            "api_key_env": self.api_key_env or None,
            "format": self.format,
            "base_url": self.openai_base_url() or None,
            "chat_url": self.chat_url() or None,
            "deployment": self.deployment.to_dict(),
        }


@dataclass
class EscalationSettings:
    """NeMo Switchyard escalation block defaults."""

    confirmations: int = 2
    recent_turn_window: int = 28
    window_message_chars: int = 500
    # Optional override of trajectory-judge prompt
    prompt: str = ""
    max_output_tokens: int = 4096
    # Fail-open on judge errors (serve weak) — Switchyard default
    fail_open: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SwitchyardConfig:
    """
    Top-level multi-model + routing configuration.
    """

    enabled: bool = False
    # escalation | capability | random | passthrough | external
    strategy: str = "escalation"
    # Public model id clients send (route id). Empty → first model / default.
    route_id: str = "switchyard/agent"
    # Also expose individual models on /v1/models for direct selection
    expose_models: bool = True
    # Allow client `model` field to pick a specific ModelSpec.id or name
    allow_model_select: bool = True

    models: List[ModelSpec] = field(default_factory=list)
    escalation: EscalationSettings = field(default_factory=EscalationSettings)

    # Proxy all traffic to external switchyard-server (official binary)
    external_url: str = ""  # e.g. http://127.0.0.1:4000
    external_api_key_env: str = ""
    routes_toml_path: str = ""  # optional path to write/read routes.toml

    # Default owned_by label
    owned_by: str = "switchyard"

    def model_by_role(self, role: str) -> Optional[ModelSpec]:
        role = (role or "").lower()
        for m in self.models:
            if m.role.lower() == role:
                return m
        return None

    def model_by_name(self, name: str) -> Optional[ModelSpec]:
        for m in self.models:
            if m.name == name or m.id == name:
                return m
        return None

    def model_by_id_or_name(self, key: str) -> Optional[ModelSpec]:
        if not key:
            return None
        # Prefer exact id, then name, then role
        for m in self.models:
            if m.id == key:
                return m
        for m in self.models:
            if m.name == key:
                return m
        return self.model_by_role(key)

    def weak(self) -> Optional[ModelSpec]:
        return self.model_by_role("weak") or (
            self.models[0] if self.models else None
        )

    def strong(self) -> Optional[ModelSpec]:
        return self.model_by_role("strong") or (
            self.models[-1] if len(self.models) > 1 else self.weak()
        )

    def judge(self) -> Optional[ModelSpec]:
        return self.model_by_role("judge") or self.weak()

    def selectable_models(self) -> List[ModelSpec]:
        """Models advertised to clients (excludes pure judge unless only model)."""
        if not self.models:
            return []
        out = [m for m in self.models if m.role.lower() != "judge"]
        return out or list(self.models)

    def default_model(self) -> Optional[ModelSpec]:
        if self.strategy == "escalation":
            return self.weak()
        if self.strategy == "passthrough":
            return self.models[0] if self.models else None
        return self.weak() or (self.models[0] if self.models else None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "strategy": self.strategy,
            "route_id": self.route_id,
            "expose_models": self.expose_models,
            "allow_model_select": self.allow_model_select,
            "external_url": self.external_url or None,
            "routes_toml_path": self.routes_toml_path or None,
            "escalation": self.escalation.to_dict(),
            "models": [m.to_dict() for m in self.models],
        }


def _deployment_from_dict(raw: Optional[Dict[str, Any]]) -> DeploymentParams:
    if not raw:
        return DeploymentParams()
    urls = raw.get("backend_urls") or raw.get("backends") or []
    if isinstance(urls, str):
        urls = _split_csv(urls)
    single = raw.get("backend_url") or raw.get("url") or ""
    if single and single not in urls:
        urls = list(urls) + [single]

    return DeploymentParams(
        backend_urls=[u.rstrip("/") for u in urls if u],
        backend_strategy=str(raw.get("backend_strategy", "round_robin")),
        load_weights_locally=bool(raw.get("load_weights_locally", raw.get("local", False))),
        coordinator_url=str(raw.get("coordinator_url", "") or ""),
        deploy_mode=str(raw.get("deploy_mode", "replica")),
        tensor_parallel_size=int(raw.get("tensor_parallel_size", raw.get("tp", 1)) or 1),
        pipeline_parallel_size=int(raw.get("pipeline_parallel_size", raw.get("pp", 1)) or 1),
        model_parallel_size=int(raw.get("model_parallel_size", raw.get("mp", 1)) or 1),
        shard_rank=int(raw.get("shard_rank", 0) or 0),
        shard_world_size=int(raw.get("shard_world_size", 1) or 1),
        shard_group=str(raw.get("shard_group", "") or ""),
        cuda_visible_devices=str(raw.get("cuda_visible_devices", raw.get("gpus", "")) or ""),
        device_map=str(raw.get("device_map", raw.get("hf_device_map", "")) or ""),
        max_memory=str(raw.get("max_memory", raw.get("hf_max_memory", "")) or ""),
        max_memory_per_gpu=str(raw.get("max_memory_per_gpu", "") or ""),
        max_memory_cpu=str(raw.get("max_memory_cpu", "") or ""),
        torch_dtype=str(raw.get("torch_dtype", raw.get("hf_torch_dtype", "")) or ""),
        trust_remote_code=raw.get("trust_remote_code"),
        offload_folder=str(raw.get("offload_folder", "") or ""),
        nim_image=str(raw.get("nim_image", "") or ""),
        nim_port=int(raw.get("nim_port", 0) or 0),
        gpu_count=int(raw.get("gpu_count", 0) or 0),
        node_selector=str(raw.get("node_selector", "") or ""),
        extra={
            k: v
            for k, v in raw.items()
            if k.startswith("x_")
            or k == "extra"
            or k
            in (
                "engine",
                "image",
                "sglang_image",
                "nim_model_path",
                "model_path",
                "nim_served_model_name",
                "served_model_name",
                "nim_model_profile",
                "model_profile",
                "tool_call_parser",
                "reasoning_parser",
                "extra_args",
                "engine_args",
                "trust_remote_code",
            )
        },
    )


def model_spec_from_dict(raw: Dict[str, Any], index: int = 0) -> ModelSpec:
    name = str(raw.get("name") or raw.get("role") or f"model_{index}")
    model_id = str(raw.get("id") or raw.get("model") or raw.get("model_id") or name)
    role = str(raw.get("role") or "general").lower()
    dep_raw = raw.get("deployment") or raw.get("deploy") or {}
    if not isinstance(dep_raw, dict):
        dep_raw = {}
    # Hoist common backend keys onto deployment
    for key in (
        "backend_url",
        "backend_urls",
        "url",
        "cuda_visible_devices",
        "tensor_parallel_size",
        "device_map",
        "torch_dtype",
        "load_weights_locally",
        "local",
        "nim_image",
        "gpu_count",
        "engine",
        "image",
        "nim_model_path",
        "model_path",
        "nim_served_model_name",
        "served_model_name",
        "nim_model_profile",
        "tool_call_parser",
        "reasoning_parser",
        "sglang_image",
    ):
        if key in raw and key not in dep_raw:
            dep_raw[key] = raw[key]

    extra_body = raw.get("extra_body") or {}
    if not isinstance(extra_body, dict):
        extra_body = {}

    return ModelSpec(
        name=name,
        id=model_id,
        role=role,
        max_tokens=int(raw.get("max_tokens", _env_int("MAX_TOKENS", 32768))),
        temperature=float(raw.get("temperature", _env_float("TEMPERATURE", 0.3))),
        context_window=int(raw.get("context_window", _env_int("CONTEXT_WINDOW", 1048576))),
        request_timeout=int(raw.get("request_timeout", _env_int("REQUEST_TIMEOUT", 300))),
        top_p=float(raw.get("top_p", 0.95)),
        extra_body=extra_body,
        api_key_env=str(raw.get("api_key_env", "") or ""),
        api_key=str(raw.get("api_key", "") or ""),
        format=str(raw.get("format", "openai_chat")),
        base_url=str(raw.get("base_url", "") or ""),
        deployment=_deployment_from_dict(dep_raw),
    )


def _parse_toml_models(text: str) -> Dict[str, Any]:
    """Minimal TOML subset parser via tomllib/tomli when available; else JSON-only."""
    try:
        import tomllib  # py311+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            raise RuntimeError(
                "TOML Switchyard config requires Python 3.11+ tomllib or the tomli package"
            )
    data = tomllib.loads(text)
    # Accept either full switchyard shape or {models=[...]}
    if "models" in data or "routes" in data or "targets" in data:
        return _normalize_switchyard_toml(data)
    return data


def _normalize_switchyard_toml(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a NeMo Switchyard routes.toml-like structure into our config dict.
    """
    targets = data.get("targets") or {}
    llm_clients = data.get("llm_clients") or {}
    routes = data.get("routes") or {}

    models: List[Dict[str, Any]] = []
    if isinstance(targets, dict):
        for tname, tval in targets.items():
            if not isinstance(tval, dict):
                continue
            client_name = tval.get("llm_client", "")
            client = llm_clients.get(client_name, {}) if isinstance(llm_clients, dict) else {}
            base_url = ""
            api_key_env = ""
            fmt = "openai_chat"
            if isinstance(client, dict):
                base_url = str(client.get("base_url", "") or "")
                api_key_env = str(client.get("api_key_env", "") or "")
                fmt = str(client.get("format", "openai_chat"))
            role = tname if tname in ("weak", "strong", "judge", "classifier") else "general"
            if tname == "classifier":
                role = "judge"
            models.append(
                {
                    "name": tname,
                    "id": tval.get("id", tname),
                    "role": role,
                    "base_url": base_url,
                    "api_key_env": api_key_env,
                    "format": fmt,
                    "extra_body": tval.get("extra_body") or {},
                    "deployment": {
                        "backend_url": (
                            base_url.rstrip("/") + "/chat/completions"
                            if base_url and not base_url.endswith("chat/completions")
                            else base_url
                        )
                    },
                }
            )

    strategy = "escalation"
    route_id = "switchyard/agent"
    escalation: Dict[str, Any] = {}
    if isinstance(routes, dict) and routes:
        # Prefer first llm_classifier escalation route
        chosen = None
        for _rname, rval in routes.items():
            if not isinstance(rval, dict):
                continue
            if rval.get("type") == "llm_classifier" and rval.get("mode") == "escalation":
                chosen = rval
                break
        if chosen is None:
            chosen = next(iter(routes.values()))
        if isinstance(chosen, dict):
            route_id = str(chosen.get("id") or route_id)
            rtype = chosen.get("type", "")
            mode = chosen.get("mode", "")
            if rtype == "llm_classifier" and mode == "escalation":
                strategy = "escalation"
            elif rtype == "llm_classifier":
                strategy = "capability"
            elif rtype == "random":
                strategy = "random"
            elif rtype == "passthrough":
                strategy = "passthrough"
            esc = chosen.get("escalation") or {}
            if isinstance(esc, dict):
                escalation = dict(esc)
            if chosen.get("prompt"):
                escalation["prompt"] = chosen["prompt"]
            if chosen.get("max_output_tokens"):
                escalation["max_output_tokens"] = chosen["max_output_tokens"]
            # Map target roles from route keys
            role_map = {
                str(chosen.get("weak_target", "")): "weak",
                str(chosen.get("strong_target", "")): "strong",
                str(chosen.get("classifier_target", "")): "judge",
            }
            for m in models:
                if m["name"] in role_map and role_map[m["name"]]:
                    m["role"] = role_map[m["name"]]

    out: Dict[str, Any] = {
        "enabled": True,
        "strategy": strategy,
        "route_id": route_id,
        "models": models,
    }
    if escalation:
        out["escalation"] = escalation
    if data.get("models"):
        # Explicit models array wins / extends
        out["models"] = data["models"]
    return out


def _load_config_file(path: str) -> Dict[str, Any]:
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.endswith(".toml"):
        return _parse_toml_models(text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Switchyard config root must be an object: {path}")
    return data


def _models_from_convenience_env() -> List[Dict[str, Any]]:
    """Build models list from SWITCHYARD_WEAK_* / STRONG_* / JUDGE_* env vars."""
    models: List[Dict[str, Any]] = []

    def _one(prefix: str, role: str) -> None:
        mid = os.getenv(f"SWITCHYARD_{prefix}_MODEL", os.getenv(f"SWITCHYARD_{prefix}_ID", "")).strip()
        url = os.getenv(f"SWITCHYARD_{prefix}_URL", os.getenv(f"SWITCHYARD_{prefix}_BACKEND", "")).strip()
        if not mid and not url:
            return
        if not mid:
            mid = f"{role}-model"
        entry: Dict[str, Any] = {
            "name": role,
            "id": mid,
            "role": role,
            "deployment": {},
        }
        if url:
            entry["deployment"]["backend_url"] = url
        # Optional per-role overrides
        for key, env_suffix, cast in (
            ("max_tokens", "MAX_TOKENS", int),
            ("temperature", "TEMPERATURE", float),
            ("context_window", "CONTEXT_WINDOW", int),
            ("request_timeout", "TIMEOUT", int),
        ):
            raw = os.getenv(f"SWITCHYARD_{prefix}_{env_suffix}", "").strip()
            if raw:
                try:
                    entry[key] = cast(raw)
                except ValueError:
                    pass
        # Deployment extras
        cvd = os.getenv(f"SWITCHYARD_{prefix}_CUDA_VISIBLE_DEVICES", "").strip()
        if cvd:
            entry["deployment"]["cuda_visible_devices"] = cvd
        tp = os.getenv(f"SWITCHYARD_{prefix}_TENSOR_PARALLEL_SIZE", "").strip()
        if tp:
            entry["deployment"]["tensor_parallel_size"] = int(tp)
        local = os.getenv(f"SWITCHYARD_{prefix}_LOCAL", "").strip()
        if local:
            entry["deployment"]["load_weights_locally"] = local.lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        dtype = os.getenv(f"SWITCHYARD_{prefix}_TORCH_DTYPE", "").strip()
        if dtype:
            entry["deployment"]["torch_dtype"] = dtype
        dmap = os.getenv(f"SWITCHYARD_{prefix}_DEVICE_MAP", "").strip()
        if dmap:
            entry["deployment"]["device_map"] = dmap
        base = os.getenv(f"SWITCHYARD_{prefix}_BASE_URL", "").strip()
        if base:
            entry["base_url"] = base
        key_env = os.getenv(f"SWITCHYARD_{prefix}_API_KEY_ENV", "").strip()
        if key_env:
            entry["api_key_env"] = key_env
        models.append(entry)

    _one("WEAK", "weak")
    _one("STRONG", "strong")
    _one("JUDGE", "judge")

    # Additional general models: SWITCHYARD_MODEL_1_ID, SWITCHYARD_MODEL_1_URL, ...
    pattern = re.compile(r"^SWITCHYARD_MODEL_(\d+)_ID$")
    indices = set()
    for k in os.environ:
        m = pattern.match(k)
        if m:
            indices.add(int(m.group(1)))
    for i in sorted(indices):
        mid = os.getenv(f"SWITCHYARD_MODEL_{i}_ID", "").strip()
        if not mid:
            continue
        url = os.getenv(f"SWITCHYARD_MODEL_{i}_URL", "").strip()
        name = os.getenv(f"SWITCHYARD_MODEL_{i}_NAME", f"model_{i}").strip()
        role = os.getenv(f"SWITCHYARD_MODEL_{i}_ROLE", "general").strip()
        entry = {
            "name": name,
            "id": mid,
            "role": role,
            "deployment": {"backend_url": url} if url else {},
        }
        cvd = os.getenv(f"SWITCHYARD_MODEL_{i}_CUDA_VISIBLE_DEVICES", "").strip()
        if cvd:
            entry["deployment"]["cuda_visible_devices"] = cvd
        tp = os.getenv(f"SWITCHYARD_MODEL_{i}_TENSOR_PARALLEL_SIZE", "").strip()
        if tp:
            entry["deployment"]["tensor_parallel_size"] = int(tp)
        models.append(entry)

    return models


def _escalation_from_dict(raw: Optional[Dict[str, Any]]) -> EscalationSettings:
    raw = raw or {}
    return EscalationSettings(
        confirmations=int(
            raw.get(
                "confirmations",
                _env_int("SWITCHYARD_ESCALATION_CONFIRMATIONS", 2),
            )
        ),
        recent_turn_window=int(
            raw.get(
                "recent_turn_window",
                _env_int("SWITCHYARD_ESCALATION_TURN_WINDOW", 28),
            )
        ),
        window_message_chars=int(
            raw.get(
                "window_message_chars",
                _env_int("SWITCHYARD_ESCALATION_MSG_CHARS", 500),
            )
        ),
        prompt=str(raw.get("prompt", os.getenv("SWITCHYARD_ESCALATION_PROMPT", "")) or ""),
        max_output_tokens=int(
            raw.get(
                "max_output_tokens",
                _env_int("SWITCHYARD_JUDGE_MAX_TOKENS", 4096),
            )
        ),
        fail_open=bool(raw.get("fail_open", True)),
    )


def load_switchyard_config(
    default_model_id: str = "",
    default_backend_url: str = "",
    owned_by: str = "switchyard",
) -> SwitchyardConfig:
    """
    Load Switchyard multi-model config from environment / file.

    When Switchyard is disabled, returns enabled=False with an empty or
    single default model (for callers that still want a uniform list).
    """
    external = os.getenv("SWITCHYARD_SERVER_URL", os.getenv("SWITCHYARD_EXTERNAL_URL", "")).strip()
    config_path = os.getenv("SWITCHYARD_CONFIG", os.getenv("SWITCHYARD_CONFIG_PATH", "")).strip()
    models_json = os.getenv("SWITCHYARD_MODELS", "").strip()

    file_data: Dict[str, Any] = {}
    if config_path and os.path.isfile(os.path.expanduser(config_path)):
        try:
            file_data = _load_config_file(config_path)
        except Exception as exc:
            logger.warning("Failed to load SWITCHYARD_CONFIG=%s: %s", config_path, exc)

    if models_json:
        try:
            parsed = json.loads(models_json)
            if isinstance(parsed, list):
                file_data = {**file_data, "models": parsed}
            elif isinstance(parsed, dict):
                file_data = {**file_data, **parsed}
        except json.JSONDecodeError as exc:
            logger.warning("Invalid SWITCHYARD_MODELS JSON: %s", exc)

    convenience = _models_from_convenience_env()
    if convenience and "models" not in file_data:
        file_data = {**file_data, "models": convenience}
    elif convenience and not file_data.get("models"):
        file_data["models"] = convenience

    # Enable if explicitly set, or if multi-model config / external server present
    explicit = os.getenv("SWITCHYARD_ENABLED", "").strip()
    if explicit:
        enabled = explicit.lower() in ("1", "true", "yes", "on")
    else:
        enabled = bool(file_data.get("models") or external or file_data.get("enabled"))

    strategy = (
        os.getenv("SWITCHYARD_STRATEGY", "")
        or str(file_data.get("strategy", "escalation"))
    ).strip().lower() or "escalation"
    if strategy in ("escalate", "escalation_router"):
        strategy = "escalation"
    if external and strategy == "escalation" and not file_data.get("models"):
        # Pure external proxy
        strategy = "external"

    route_id = (
        os.getenv("SWITCHYARD_ROUTE_ID", "")
        or str(file_data.get("route_id", "switchyard/agent"))
    ).strip() or "switchyard/agent"

    models_raw = file_data.get("models") or []
    models = [
        model_spec_from_dict(m, i)
        for i, m in enumerate(models_raw)
        if isinstance(m, dict)
    ]

    # If enabled but no models, synthesize from legacy single-model env
    if enabled and not models and default_model_id:
        dep = DeploymentParams()
        if default_backend_url:
            dep.backend_urls = [default_backend_url.rstrip("/")]
        models = [
            ModelSpec(
                name="default",
                id=default_model_id,
                role="weak",
                deployment=dep,
            )
        ]

    esc_raw = file_data.get("escalation") if isinstance(file_data.get("escalation"), dict) else {}
    escalation = _escalation_from_dict(esc_raw)

    cfg = SwitchyardConfig(
        enabled=enabled,
        strategy="external" if external and not models else strategy,
        route_id=route_id,
        expose_models=_env_bool(
            "SWITCHYARD_EXPOSE_MODELS",
            default=bool(file_data.get("expose_models", True)),
        ),
        allow_model_select=_env_bool(
            "SWITCHYARD_ALLOW_MODEL_SELECT",
            default=bool(file_data.get("allow_model_select", True)),
        ),
        models=models,
        escalation=escalation,
        external_url=external,
        external_api_key_env=os.getenv(
            "SWITCHYARD_EXTERNAL_API_KEY_ENV",
            str(file_data.get("external_api_key_env", "") or ""),
        ).strip(),
        routes_toml_path=os.getenv(
            "SWITCHYARD_ROUTES_TOML",
            str(file_data.get("routes_toml_path", "") or ""),
        ).strip(),
        owned_by=owned_by,
    )

    # If strategy is external but we also have models, keep external_url for optional hybrid
    if external:
        cfg.external_url = external
        if not models:
            cfg.strategy = "external"

    return cfg


_CONFIG: Optional[SwitchyardConfig] = None


def get_switchyard_config(
    default_model_id: str = "",
    default_backend_url: str = "",
    owned_by: str = "switchyard",
    reload: bool = False,
) -> SwitchyardConfig:
    global _CONFIG
    if _CONFIG is not None and not reload:
        return _CONFIG
    _CONFIG = load_switchyard_config(
        default_model_id=default_model_id,
        default_backend_url=default_backend_url,
        owned_by=owned_by,
    )
    return _CONFIG
