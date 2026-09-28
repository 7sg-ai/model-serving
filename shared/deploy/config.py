"""Environment configuration for model runtime auto-deploy."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from .engines import (
    default_image,
    default_model_id,
    default_port,
    is_model_free_nim_image,
    nim_model_path as build_nim_model_path,
    normalize_engine,
    served_model_name,
    slug as _slug,
)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def split_csv(raw: str) -> List[str]:
    return [p.strip() for p in (raw or "").split(",") if p.strip()]


@dataclass
class DeployConfig:
    auto_deploy: bool = True
    teardown_on_exit: bool = True
    shared: bool = False
    project: str = ""
    engine: str = "vllm"  # nim | vllm | sglang
    app_name: str = "app"
    model_id: str = ""
    deploy_mode: str = "hybrid"
    replica_count: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_per_replica: int = 1
    gpu_devices: List[str] = None  # type: ignore
    port_base: int = 8000
    wait_seconds: int = 300
    teardown_timeout: int = 60
    remove_volumes: bool = False
    compose_file: str = ""
    image: str = ""
    ngc_api_key: str = ""
    hf_token: str = ""
    vllm_max_model_len: int = 8192
    enable_prefix_caching: bool = True
    deploy_nodes: List[str] = None  # type: ignore
    remote_dir: str = "/tmp/model-serving-runtime"
    ssh_user: str = ""
    # NIM model-free / non-NVIDIA-hosted weights
    nim_model_path: str = ""
    nim_served_model_name: str = ""
    nim_model_profile: str = ""
    nim_cache_host: str = ""
    trust_remote_code: bool = False
    tool_call_parser: str = ""
    reasoning_parser: str = ""
    extra_args: List[str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.gpu_devices is None:
            self.gpu_devices = []
        if self.deploy_nodes is None:
            self.deploy_nodes = []
        if self.extra_args is None:
            self.extra_args = []

    @property
    def gpus_per_replica(self) -> int:
        return max(
            self.gpu_per_replica,
            self.tensor_parallel_size * self.pipeline_parallel_size,
            1,
        )

    def resolved_project(self) -> str:
        if self.project:
            return _slug(self.project)
        if self.shared:
            return _slug(f"model-serving-{self.engine}-shared")
        return _slug(f"model-serving-{self.engine}-{self.app_name}")

    def resolved_replicas(self) -> int:
        if self.deploy_mode == "sharded":
            return 1
        return max(1, self.replica_count)

    def is_multi_node(self) -> bool:
        return bool(self.deploy_nodes)

    def model_free_nim(self) -> bool:
        """Catalog NIM bakes weights in. Model-free NIM serves a supplied path."""
        if self.engine != "nim":
            return False
        if self.nim_model_path:
            return True
        return is_model_free_nim_image(self.image)


def load_deploy_config(
    engine: str,
    *,
    app_name: str = "app",
    model_id: str = "",
    deploy_mode: Optional[str] = None,
    replica_count: Optional[int] = None,
    tensor_parallel_size: Optional[int] = None,
    pipeline_parallel_size: Optional[int] = None,
    gpu_devices: Optional[List[str]] = None,
    gpu_per_replica: Optional[int] = None,
    port_base: Optional[int] = None,
    image: Optional[str] = None,
    project: Optional[str] = None,
    existing_hint_port: int = 8000,
    nim_model_path: Optional[str] = None,
    nim_served_model_name: Optional[str] = None,
    nim_model_profile: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
    tool_call_parser: Optional[str] = None,
    reasoning_parser: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> DeployConfig:
    engine = normalize_engine(engine)

    mode = (
        deploy_mode
        or os.getenv("MODEL_DEPLOY_MODE", os.getenv("LLM_DEPLOY_MODE", "hybrid"))
    ).strip().lower()
    if mode not in ("replica", "sharded", "hybrid"):
        mode = "hybrid"

    tp = tensor_parallel_size
    if tp is None:
        tp = env_int("TENSOR_PARALLEL_SIZE", env_int("TP_SIZE", 1))
    pp = pipeline_parallel_size
    if pp is None:
        pp = env_int("PIPELINE_PARALLEL_SIZE", env_int("PP_SIZE", 1))

    if replica_count is None:
        primary = {
            "nim": "NIM_REPLICA_COUNT",
            "sglang": "SGLANG_REPLICA_COUNT",
        }.get(engine, "VLLM_REPLICA_COUNT")
        replica_count = env_int(
            primary,
            env_int(
                "VLLM_REPLICA_COUNT",
                env_int(
                    "SGLANG_REPLICA_COUNT",
                    env_int("NIM_REPLICA_COUNT", env_int("REPLICA_COUNT", 1)),
                ),
            ),
        )

    model = (model_id or "").strip() or default_model_id(engine)
    resolved_image = (image or "").strip() or default_image(engine)
    resolved_port = (
        int(port_base) if port_base is not None else default_port(engine, existing_hint_port)
    )

    path = nim_model_path if nim_model_path is not None else os.getenv("NIM_MODEL_PATH", "")
    served = (
        nim_served_model_name
        if nim_served_model_name is not None
        else os.getenv("NIM_SERVED_MODEL_NAME", "")
    )
    profile = (
        nim_model_profile
        if nim_model_profile is not None
        else os.getenv("NIM_MODEL_PROFILE", "")
    )
    if engine == "nim" and is_model_free_nim_image(resolved_image) and not (path or "").strip():
        path = build_nim_model_path(model)

    gpus = (
        list(gpu_devices)
        if gpu_devices is not None
        else split_csv(os.getenv("GPU_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", "")))
    )
    gpr = (
        gpu_per_replica
        if gpu_per_replica is not None
        else env_int("GPU_PER_REPLICA", max(int(tp) * int(pp), 1))
    )

    proj = project if project is not None else os.getenv("AUTO_DEPLOY_PROJECT", "")
    trust = (
        trust_remote_code
        if trust_remote_code is not None
        else env_bool("TRUST_REMOTE_CODE", env_bool("NIM_TRUST_CUSTOM_CODE", default=False))
    )
    tool_parser = (
        tool_call_parser
        if tool_call_parser is not None
        else os.getenv("SGLANG_TOOL_CALL_PARSER", os.getenv("TOOL_CALL_PARSER", ""))
    )
    reason_parser = (
        reasoning_parser
        if reasoning_parser is not None
        else os.getenv("SGLANG_REASONING_PARSER", os.getenv("REASONING_PARSER", ""))
    )
    args = list(extra_args) if extra_args is not None else split_csv(
        os.getenv("SGLANG_EXTRA_ARGS", os.getenv("ENGINE_EXTRA_ARGS", ""))
    )
    ctx_env = "SGLANG_CONTEXT_LENGTH" if engine == "sglang" else "VLLM_MAX_MODEL_LEN"

    return DeployConfig(
        auto_deploy=env_bool("AUTO_DEPLOY_MODEL", default=True),
        teardown_on_exit=env_bool("AUTO_DEPLOY_TEARDOWN_ON_EXIT", default=True),
        shared=env_bool("AUTO_DEPLOY_SHARED", default=False),
        project=(proj or "").strip(),
        engine=engine,
        app_name=app_name or "app",
        model_id=model,
        deploy_mode=mode,
        replica_count=max(1, int(replica_count)),
        tensor_parallel_size=max(1, int(tp)),
        pipeline_parallel_size=max(1, int(pp)),
        gpu_per_replica=int(gpr),
        gpu_devices=gpus,
        port_base=resolved_port,
        wait_seconds=env_int("AUTO_DEPLOY_WAIT_SECONDS", 300),
        teardown_timeout=env_int("AUTO_DEPLOY_TEARDOWN_TIMEOUT_SECONDS", 60),
        remove_volumes=env_bool("AUTO_DEPLOY_REMOVE_VOLUMES", default=False),
        image=resolved_image,
        ngc_api_key=os.getenv("NGC_API_KEY", "").strip(),
        hf_token=os.getenv("HUGGING_FACE_HUB_TOKEN", os.getenv("HF_TOKEN", "")).strip(),
        vllm_max_model_len=env_int(ctx_env, env_int("VLLM_MAX_MODEL_LEN", env_int("NIM_MAX_MODEL_LEN", 8192))),
        enable_prefix_caching=env_bool("VLLM_ENABLE_PREFIX_CACHING", default=True),
        deploy_nodes=split_csv(os.getenv("DEPLOY_NODES", "")),
        remote_dir=(
            os.getenv("DEPLOY_REMOTE_DIR", "/tmp/model-serving-runtime").strip()
            or "/tmp/model-serving-runtime"
        ),
        ssh_user=os.getenv("DEPLOY_SSH_USER", "").strip(),
        nim_model_path=(path or "").strip(),
        nim_served_model_name=served_model_name(model, served or ""),
        nim_model_profile=(profile or "").strip(),
        nim_cache_host=os.path.expanduser(
            os.getenv("NIM_CACHE", os.getenv("LOCAL_NIM_CACHE", "~/.cache/nim"))
        ),
        trust_remote_code=bool(trust),
        tool_call_parser=(tool_parser or "").strip(),
        reasoning_parser=(reason_parser or "").strip(),
        extra_args=args,
    )
