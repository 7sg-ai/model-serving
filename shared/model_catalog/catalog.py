"""
Multi-model catalog for all applications.

Uses Switchyard-style model lists (SWITCHYARD_CONFIG / SWITCHYARD_MODELS /
SWITCHYARD_WEAK_* etc.). Non-IDE apps ignore escalation and expose models via
dropdown; IDE apps keep Switchyard routing on top of the same catalog.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shared.switchyard.client import ModelClient
from shared.switchyard.config import (
    DeploymentParams,
    ModelSpec,
    SwitchyardConfig,
    get_switchyard_config,
    load_switchyard_config,
)

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def load_app_models(
    *,
    default_model_id: str = "",
    default_backend_url: str = "",
    owned_by: str = "model-serving",
    reload: bool = False,
) -> SwitchyardConfig:
    """
    Load multi-model config.

    When a models list is present, treat the catalog as available even if
    SWITCHYARD_ENABLED is unset so non-IDE apps share the same JSON.
    """
    if reload:
        get_switchyard_config(
            default_model_id=default_model_id,
            default_backend_url=default_backend_url,
            owned_by=owned_by,
            reload=True,
        )
    cfg = load_switchyard_config(
        default_model_id=default_model_id,
        default_backend_url=default_backend_url,
        owned_by=owned_by,
    )

    if not cfg.models and default_model_id:
        dep = DeploymentParams()
        if default_backend_url:
            dep.backend_urls = [default_backend_url.rstrip("/")]
        cfg.models = [
            ModelSpec(
                name="default",
                id=default_model_id,
                role="weak",
                deployment=dep,
            )
        ]

    if cfg.models and not cfg.enabled:
        explicit = os.getenv("SWITCHYARD_ENABLED", "").strip()
        if not explicit:
            cfg.enabled = True
            if cfg.strategy in ("", "escalation") and not (
                cfg.weak() and cfg.strong()
            ):
                cfg.strategy = "passthrough"

    return cfg


def deploy_key(spec: ModelSpec) -> str:
    """Stable key so shared backends (e.g. weak+judge) deploy once."""
    dep = spec.deployment
    urls = tuple(sorted(u.rstrip("/") for u in (dep.backend_urls or []) if u))
    if urls:
        return "urls:" + ",".join(urls)
    cvd = (dep.cuda_visible_devices or "").strip()
    image = (dep.nim_image or str(dep.extra.get("image", "") or "")).strip()
    path = str(dep.extra.get("nim_model_path") or dep.extra.get("model_path") or "").strip()
    stack = str(dep.extra.get("engine") or dep.extra.get("x_engine") or "").strip()
    tp = int(dep.tensor_parallel_size or 1)
    mode = (dep.deploy_mode or "replica").strip().lower()
    return f"id:{spec.id}|eng:{stack}|img:{image}|path:{path}|cvd:{cvd}|tp:{tp}|mode:{mode}"


@dataclass
class ModelCatalog:
    """Selectable models with per-spec clients after deploy."""

    cfg: SwitchyardConfig
    engine: str = "vllm"
    app_name: str = "app"
    api_key: str = ""
    handles: List[Any] = field(default_factory=list)
    _clients: Dict[str, ModelClient] = field(default_factory=dict)

    @property
    def models(self) -> List[ModelSpec]:
        return list(self.cfg.models)

    @property
    def multi(self) -> bool:
        ids = {m.id for m in self.models}
        keys = {deploy_key(m) for m in self.models}
        return len(self.models) > 1 and (len(ids) > 1 or len(keys) > 1)

    def default_spec(self) -> ModelSpec:
        m = self.cfg.default_model()
        if m is None:
            raise RuntimeError("ModelCatalog has no models configured")
        return m

    def default_id(self) -> str:
        return self.default_spec().id

    def resolve(self, key: Optional[str] = None) -> ModelSpec:
        if not key:
            return self.default_spec()
        key = str(key).strip()
        for m in self.models:
            if m.id == key or m.name == key:
                return m
        low = key.lower()
        for m in self.models:
            if m.id.lower() == low or m.name.lower() == low:
                return m
        logger.warning("Unknown model %r; using default %s", key, self.default_id())
        return self.default_spec()

    def choices_for_ui(self) -> List[Dict[str, str]]:
        """Label/value pairs for dropdowns (dedupe by id)."""
        seen = set()
        out: List[Dict[str, str]] = []
        for m in self.models:
            if m.role == "judge" and not _env_bool("EXPOSE_JUDGE_MODELS", False):
                if any(x.id == m.id and x.role != "judge" for x in self.models):
                    continue
            if m.id in seen:
                continue
            seen.add(m.id)
            if m.name and m.name != m.id:
                label = f"{m.name} ({m.id})"
            else:
                label = m.id
            out.append(
                {"id": m.id, "name": m.name, "label": label, "role": m.role}
            )
        if not out:
            d = self.default_spec()
            out.append(
                {"id": d.id, "name": d.name, "label": d.id, "role": d.role}
            )
        return out

    def streamlit_options(self) -> Tuple[List[str], Dict[str, str]]:
        """Returns (labels, label->model_id) for st.selectbox."""
        choices = self.choices_for_ui()
        labels = [c["label"] for c in choices]
        mapping = {c["label"]: c["id"] for c in choices}
        return labels, mapping

    def client_for(self, key: Optional[str] = None) -> ModelClient:
        spec = self.resolve(key)
        ck = spec.name or spec.id
        if ck not in self._clients:
            self._clients[ck] = ModelClient(spec)
        return self._clients[ck]

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        client = self.client_for(model)
        data = client.chat(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        text = client.extract_assistant_text(data)
        return text, data

    def list_payload(self) -> Dict[str, Any]:
        return {
            "object": "list",
            "default": self.default_id(),
            "data": [
                {
                    "id": c["id"],
                    "name": c["name"],
                    "label": c["label"],
                    "role": c["role"],
                }
                for c in self.choices_for_ui()
            ],
        }

    def health_block(self) -> Dict[str, Any]:
        return {
            "multi_model": self.multi,
            "default": self.default_id(),
            "models": [m.to_dict() for m in self.models],
            "deploy": [
                h.to_dict() for h in self.handles if hasattr(h, "to_dict")
            ],
        }

    def all_backend_urls(self) -> List[str]:
        seen = set()
        out: List[str] = []
        for m in self.models:
            for u in m.deployment.backend_urls or []:
                u = (u or "").rstrip("/")
                if u and u not in seen:
                    seen.add(u)
                    out.append(u)
            if m.deployment.coordinator_url:
                u = m.deployment.coordinator_url.rstrip("/")
                if u and u not in seen:
                    seen.add(u)
                    out.append(u)
        return out


def deploy_app_models(
    engine: str,
    cfg: SwitchyardConfig,
    *,
    app_name: str = "app",
    api_key: str = "",
    default_backend_urls: Optional[Sequence[str]] = None,
    register_teardown: bool = True,
) -> Tuple[ModelCatalog, List[Any]]:
    """
    Ensure every unique model stack is running (in parallel), then return catalog.
    Mutates cfg.models[*].deployment.backend_urls when deploy fills them.
    """
    from shared.deploy import ensure_models_runtime

    handles = ensure_models_runtime(
        engine,
        list(cfg.models),
        app_name=app_name,
        api_key=api_key,
        default_backend_urls=default_backend_urls,
        register_teardown=register_teardown,
    )
    catalog = ModelCatalog(
        cfg=cfg,
        engine=engine,
        app_name=app_name,
        api_key=api_key,
        handles=handles,
    )
    return catalog, handles
