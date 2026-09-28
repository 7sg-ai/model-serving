"""Resolve which inference engine an app should auto-deploy.

HF apps default to vLLM and may switch to SGLang.
NIM apps default to catalog NIM and may switch to model-free NIM
(vLLM or SGLang backend) for weights NVIDIA does not host.
"""
from __future__ import annotations

import os

from .engines import normalize_engine


def resolve_app_engine(family: str) -> str:
    """family is ``hf`` or ``nim``."""
    family = (family or "hf").strip().lower()
    explicit = os.getenv("INFERENCE_ENGINE", os.getenv("MODEL_ENGINE", "")).strip()
    if explicit:
        return normalize_engine(explicit)
    if family == "nim":
        backend = os.getenv("NIM_BACKEND", "").strip().lower()
        # Model-free NIM is still the nim engine; the image selects vLLM vs SGLang.
        if backend in ("sglang", "vllm", "model-free", "modelfree", "model_free"):
            return "nim"
        return "nim"
    backend = os.getenv("HF_BACKEND", os.getenv("HF_ENGINE", "vllm")).strip().lower()
    if backend in ("sglang", "sg", "sgl"):
        return "sglang"
    return "vllm"
