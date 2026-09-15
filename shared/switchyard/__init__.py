"""
NeMo Switchyard integration for multi-model IDE assistants.

Supports:
  - Selecting several models with independent deployment parameters
  - Escalation router (weak → judge → strong latch), matching NeMo Switchyard
  - Optional proxy to an external switchyard-server
  - Export of routes.toml for the official Switchyard binary
"""
from .config import (
    DeploymentParams,
    ModelSpec,
    SwitchyardConfig,
    get_switchyard_config,
    load_switchyard_config,
)
from .client import ModelClient, chat_completion
from .escalation import EscalationRouter, EscalationState
from .router import StreamOutcome, SwitchyardRouter, get_switchyard_router
from .toml_export import export_routes_toml, write_routes_toml

__all__ = [
    "DeploymentParams",
    "ModelSpec",
    "SwitchyardConfig",
    "get_switchyard_config",
    "load_switchyard_config",
    "ModelClient",
    "chat_completion",
    "EscalationRouter",
    "EscalationState",
    "StreamOutcome",
    "SwitchyardRouter",
    "get_switchyard_router",
    "export_routes_toml",
    "write_routes_toml",
]
