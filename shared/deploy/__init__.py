"""Auto-deploy NIM / vLLM / SGLang model containers with optional teardown on exit."""
from .ensure import (
    RuntimeHandle,
    ensure_model_runtime,
    ensure_models_runtime,
    register_runtime_teardown,
    teardown_runtime,
)

__all__ = [
    "RuntimeHandle",
    "ensure_model_runtime",
    "ensure_models_runtime",
    "register_runtime_teardown",
    "teardown_runtime",
]
