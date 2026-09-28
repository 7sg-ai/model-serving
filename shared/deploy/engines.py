"""Inference engines shared by HF and NIM apps.

Engines:
  vllm    community vLLM OpenAI server (HF weights)
  sglang  community SGLang OpenAI server (HF weights)
  nim     NVIDIA NIM — catalog image, or model-free vLLM/SGLang NIM for
          weights NVIDIA does not host (HF, S3, local, NGC URI)

NIM model-free images (NGC, 2026):
  nvcr.io/nim/nvidia/vllm-model-free-nim
  nvcr.io/nim/nvidia/sglang-model-free-nim
Point them at a model with NIM_MODEL_PATH (hf://, s3://, ngc://, or a
local directory) and optional NIM_SERVED_MODEL_NAME.
"""
from __future__ import annotations

import os
import re
from typing import Optional

ENGINES = ("nim", "vllm", "sglang")

# Community OpenAI servers. SGLang's published image uses `sglang serve`
# and listens on 30000; we remap the container port to 8000 in compose.
VLLM_IMAGE = "vllm/vllm-openai:latest"
SGLANG_IMAGE = "lmsysorg/sglang:latest-runtime"
SGLANG_CONTAINER_PORT = 30000

# Catalog NIM (weights baked into a model-specific image).
NIM_CATALOG_IMAGE = "nvcr.io/nim/meta/llama-3.1-8b-instruct:latest"
# Generic NIM images that serve a model supplied at runtime.
NIM_VLLM_MODEL_FREE_IMAGE = "nvcr.io/nim/nvidia/vllm-model-free-nim:latest"
NIM_SGLANG_MODEL_FREE_IMAGE = "nvcr.io/nim/nvidia/sglang-model-free-nim:latest"

_MODEL_FREE_MARKERS = (
    "model-free",
    "modelfree",
    "sglang-model-free",
    "vllm-model-free",
)


def normalize_engine(engine: Optional[str], default: str = "vllm") -> str:
    value = (engine or default or "vllm").strip().lower()
    if value in ("sg", "sgl"):
        value = "sglang"
    if value not in ENGINES:
        raise ValueError(
            "engine must be nim|vllm|sglang, got %s" % (engine or "")
        )
    return value


def is_model_free_nim_image(image: str) -> bool:
    """True when the NIM image expects NIM_MODEL_PATH instead of baked weights."""
    low = (image or "").strip().lower()
    if not low:
        return False
    return any(marker in low for marker in _MODEL_FREE_MARKERS)


def nim_model_path(model_id: str, explicit: str = "") -> str:
    """Build NIM_MODEL_PATH. Explicit wins; HF ids get an hf:// prefix."""
    raw = (explicit or "").strip()
    if raw:
        return raw
    model = (model_id or "").strip()
    if not model:
        return ""
    if "://" in model or model.startswith("/"):
        return model
    return "hf://%s" % model


def served_model_name(model_id: str, explicit: str = "") -> str:
    """Name clients send as OpenAI `model`. NIM derives one if left empty."""
    name = (explicit or model_id or "").strip()
    if name.startswith("hf://"):
        name = name[len("hf://") :]
    return name


def default_image(engine: str) -> str:
    engine = normalize_engine(engine)
    if engine == "sglang":
        return os.getenv("SGLANG_IMAGE", SGLANG_IMAGE).strip() or SGLANG_IMAGE
    if engine == "vllm":
        return os.getenv("VLLM_IMAGE", VLLM_IMAGE).strip() or VLLM_IMAGE
    # NIM: catalog image unless the operator asked for a model-free backend.
    explicit = os.getenv("NIM_IMAGE", "").strip()
    if explicit:
        return explicit
    backend = os.getenv("NIM_BACKEND", os.getenv("NIM_ENGINE", "")).strip().lower()
    if backend in ("sglang", "sg", "sgl"):
        return NIM_SGLANG_MODEL_FREE_IMAGE
    if backend in ("vllm", "model-free", "modelfree", "model_free"):
        return NIM_VLLM_MODEL_FREE_IMAGE
    return NIM_CATALOG_IMAGE


def default_model_id(engine: str) -> str:
    engine = normalize_engine(engine)
    if engine == "nim":
        return os.getenv("NIM_MODEL_NAME", "meta/llama-3.1-8b-instruct")
    return os.getenv(
        "SGLANG_MODEL" if engine == "sglang" else "VLLM_MODEL",
        os.getenv("HF_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"),
    )


def default_port(engine: str, fallback: int = 8000) -> int:
    engine = normalize_engine(engine)
    if engine == "nim":
        name = "NIM_PORT"
    elif engine == "sglang":
        name = "SGLANG_PORT"
    else:
        name = "VLLM_PORT"
    raw = os.getenv(name, "")
    if raw.strip() == "":
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


def container_port(engine: str, image: str = "") -> int:
    """Port the process listens on *inside* the container."""
    engine = normalize_engine(engine)
    if engine == "sglang" and not is_model_free_nim_image(image):
        raw = os.getenv("SGLANG_CONTAINER_PORT", "")
        if raw.strip():
            try:
                return int(raw)
            except ValueError:
                pass
        return SGLANG_CONTAINER_PORT
    return 8000


def service_prefix(engine: str) -> str:
    return normalize_engine(engine)


def yaml_quote(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "-", (value or "app").strip().lower())
    return text.strip("-") or "app"
