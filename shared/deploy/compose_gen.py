"""Generate compose YAML for NIM, vLLM, or SGLang model stacks."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Tuple

from .config import DeployConfig
from .engines import container_port, service_prefix, yaml_quote

try:
    from .scheduler import NodePlan, ReplicaPlacement
except Exception:  # pragma: no cover
    NodePlan = None  # type: ignore
    ReplicaPlacement = None  # type: ignore


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def compose_output_path(cfg: DeployConfig, suffix: str = "") -> Path:
    if cfg.compose_file and not suffix:
        return Path(cfg.compose_file)
    deploy_dir = _repo_root() / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    name = f"docker-compose.{cfg.engine}.{cfg.resolved_project()}"
    if suffix:
        name += f".{suffix}"
    return deploy_dir / f"{name}.generated.yml"


def _gpu_slices(cfg: DeployConfig) -> List[List[str]]:
    replicas = cfg.resolved_replicas()
    gpr = cfg.gpus_per_replica
    needed = replicas * gpr
    gpus = list(cfg.gpu_devices)
    if not gpus:
        gpus = [str(i) for i in range(needed)]
    if len(gpus) < needed:
        raise RuntimeError(
            f"Not enough GPU_DEVICES ({len(gpus)}) for {replicas} x {gpr} GPUs (need {needed})"
        )
    slices: List[List[str]] = []
    idx = 0
    for _ in range(replicas):
        slices.append(gpus[idx : idx + gpr])
        idx += gpr
    return slices


def _join(lines: List[str]) -> str:
    return "\n".join(lines) + "\n"


def _gpu_reservation(devices: Sequence[str]) -> List[str]:
    ids = ", ".join(yaml_quote(d) for d in devices)
    return [
        "    deploy:",
        "      resources:",
        "        reservations:",
        "          devices:",
        "            - driver: nvidia",
        "              capabilities: [gpu]",
        "              device_ids: [%s]" % ids,
    ]


def _hf_cache() -> str:
    return os.path.expanduser(os.getenv("HF_CACHE", "~/.cache/huggingface"))


def _vllm_service_lines(
    cfg: DeployConfig, name: str, devices: Sequence[str], host_port: int
) -> List[str]:
    cvd = ",".join(devices)
    lines = [
        f"  {name}:",
        f"    image: {cfg.image}",
        "    command:",
        '      - "--model"',
        "      - %s" % yaml_quote(cfg.model_id),
        '      - "--tensor-parallel-size"',
        '      - "%s"' % cfg.tensor_parallel_size,
        '      - "--max-model-len"',
        '      - "%s"' % cfg.vllm_max_model_len,
        '      - "--host"',
        '      - "0.0.0.0"',
        '      - "--port"',
        '      - "8000"',
    ]
    if cfg.enable_prefix_caching:
        lines.append('      - "--enable-prefix-caching"')
    if cfg.pipeline_parallel_size > 1:
        lines.extend(
            [
                '      - "--pipeline-parallel-size"',
                '      - "%s"' % cfg.pipeline_parallel_size,
            ]
        )
    if cfg.trust_remote_code:
        lines.append('      - "--trust-remote-code"')
    for arg in cfg.extra_args:
        lines.append("      - %s" % yaml_quote(arg))
    lines.extend(
        [
            "    environment:",
            "      CUDA_VISIBLE_DEVICES: %s" % yaml_quote(cvd),
            "      HUGGING_FACE_HUB_TOKEN: %s" % yaml_quote(cfg.hf_token),
            "    ports:",
            '      - "%s:8000"' % host_port,
            "    volumes:",
            "      - %s:/root/.cache/huggingface" % _hf_cache(),
            "    ipc: host",
        ]
    )
    lines.extend(_gpu_reservation(devices))
    return lines


def _sglang_service_lines(
    cfg: DeployConfig, name: str, devices: Sequence[str], host_port: int
) -> List[str]:
    """Community SGLang OpenAI server (`sglang serve`, container port 30000)."""
    cvd = ",".join(devices)
    inner = container_port("sglang", cfg.image)
    lines = [
        f"  {name}:",
        f"    image: {cfg.image}",
        "    command:",
        '      - "sglang"',
        '      - "serve"',
        '      - "--model-path"',
        "      - %s" % yaml_quote(cfg.model_id),
        '      - "--host"',
        '      - "0.0.0.0"',
        '      - "--port"',
        '      - "%s"' % inner,
        '      - "--tp"',
        '      - "%s"' % cfg.tensor_parallel_size,
    ]
    if cfg.pipeline_parallel_size > 1:
        lines.extend(['      - "--pp"', '      - "%s"' % cfg.pipeline_parallel_size])
    if cfg.vllm_max_model_len:
        lines.extend(
            ['      - "--context-length"', '      - "%s"' % cfg.vllm_max_model_len]
        )
    served = cfg.nim_served_model_name or cfg.model_id
    if served:
        lines.extend(
            ['      - "--served-model-name"', "      - %s" % yaml_quote(served)]
        )
    if cfg.trust_remote_code:
        lines.append('      - "--trust-remote-code"')
    if cfg.tool_call_parser:
        lines.extend(
            ['      - "--tool-call-parser"', "      - %s" % yaml_quote(cfg.tool_call_parser)]
        )
    if cfg.reasoning_parser:
        lines.extend(
            ['      - "--reasoning-parser"', "      - %s" % yaml_quote(cfg.reasoning_parser)]
        )
    for arg in cfg.extra_args:
        lines.append("      - %s" % yaml_quote(arg))
    lines.extend(
        [
            "    environment:",
            "      CUDA_VISIBLE_DEVICES: %s" % yaml_quote(cvd),
            "      HF_TOKEN: %s" % yaml_quote(cfg.hf_token),
            "      HUGGING_FACE_HUB_TOKEN: %s" % yaml_quote(cfg.hf_token),
            "    ports:",
            '      - "%s:%s"' % (host_port, inner),
            "    volumes:",
            "      - %s:/root/.cache/huggingface" % _hf_cache(),
            "    ipc: host",
            '    shm_size: "32gb"',
        ]
    )
    lines.extend(_gpu_reservation(devices))
    return lines


def _nim_service_lines(
    cfg: DeployConfig, name: str, devices: Sequence[str], host_port: int
) -> List[str]:
    """Catalog NIM, or model-free NIM for weights NVIDIA does not host."""
    cvd = ",".join(devices)
    key = cfg.ngc_api_key or "${NGC_API_KEY}"
    lines = [
        f"  {name}:",
        f"    image: {cfg.image}",
        "    runtime: nvidia",
        "    environment:",
        "      NGC_API_KEY: %s" % yaml_quote(key),
        "      CUDA_VISIBLE_DEVICES: %s" % yaml_quote(cvd),
        "      NVIDIA_VISIBLE_DEVICES: %s" % yaml_quote(cvd),
    ]
    if cfg.model_free_nim():
        model_path = cfg.nim_model_path
        if not model_path:
            raise RuntimeError(
                "Model-free NIM image %s requires NIM_MODEL_PATH "
                "(hf://org/model, s3://bucket/path, ngc://..., or a local dir)"
                % cfg.image
            )
        lines.extend(
            [
                "      NIM_MODEL_PATH: %s" % yaml_quote(model_path),
                "      HF_TOKEN: %s" % yaml_quote(cfg.hf_token),
                "      HUGGING_FACE_HUB_TOKEN: %s" % yaml_quote(cfg.hf_token),
            ]
        )
        if cfg.nim_served_model_name:
            lines.append(
                "      NIM_SERVED_MODEL_NAME: %s" % yaml_quote(cfg.nim_served_model_name)
            )
        if cfg.nim_model_profile:
            lines.append("      NIM_MODEL_PROFILE: %s" % yaml_quote(cfg.nim_model_profile))
        if cfg.tensor_parallel_size > 1:
            lines.append('      NIM_TENSOR_PARALLEL_SIZE: "%s"' % cfg.tensor_parallel_size)
        if cfg.pipeline_parallel_size > 1:
            lines.append('      NIM_PIPELINE_PARALLEL_SIZE: "%s"' % cfg.pipeline_parallel_size)
        if cfg.vllm_max_model_len:
            lines.append('      NIM_MAX_MODEL_LEN: "%s"' % cfg.vllm_max_model_len)
        if cfg.trust_remote_code:
            lines.append('      NIM_TRUST_CUSTOM_CODE: "1"')
    lines.extend(
        [
            "    ports:",
            '      - "%s:8000"' % host_port,
            '    shm_size: "16gb"',
        ]
    )
    if cfg.model_free_nim():
        cache = cfg.nim_cache_host or os.path.expanduser("~/.cache/nim")
        lines.extend(["    volumes:", "      - %s:/opt/nim/.cache" % cache])
    lines.extend(_gpu_reservation(devices))
    return lines


def _service_lines(
    cfg: DeployConfig, name: str, devices: Sequence[str], host_port: int
) -> List[str]:
    if cfg.engine == "nim":
        return _nim_service_lines(cfg, name, devices, host_port)
    if cfg.engine == "sglang":
        return _sglang_service_lines(cfg, name, devices, host_port)
    return _vllm_service_lines(cfg, name, devices, host_port)


def generate_engine_compose(cfg: DeployConfig) -> Tuple[Path, List[str]]:
    slices = _gpu_slices(cfg)
    prefix = service_prefix(cfg.engine)
    lines = [
        f"# AUTO-GENERATED model runtime ({cfg.engine}) project={cfg.resolved_project()}",
        f"# mode={cfg.deploy_mode} replicas={len(slices)} tp={cfg.tensor_parallel_size}",
        "services:",
    ]
    urls: List[str] = []
    for i, devices in enumerate(slices):
        name = f"{prefix}-{i}"
        host_port = cfg.port_base + i
        lines.extend(_service_lines(cfg, name, devices, host_port))
        urls.append("http://127.0.0.1:%s/v1/chat/completions" % host_port)
    path = compose_output_path(cfg)
    path.write_text(_join(lines))
    return path, urls


def generate_vllm_compose(cfg: DeployConfig) -> Tuple[Path, List[str]]:
    return generate_engine_compose(cfg)


def generate_nim_compose(cfg: DeployConfig) -> Tuple[Path, List[str]]:
    return generate_engine_compose(cfg)


def generate_sglang_compose(cfg: DeployConfig) -> Tuple[Path, List[str]]:
    return generate_engine_compose(cfg)


def generate_compose(cfg: DeployConfig) -> Tuple[Path, List[str]]:
    return generate_engine_compose(cfg)


def generate_compose_for_node_plan(cfg: DeployConfig, plan: "NodePlan") -> Tuple[Path, List[str]]:
    """Compose file containing only replicas scheduled on one node."""
    safe = plan.target.replace("@", "_").replace(":", "_")
    lines = [
        f"# AUTO-GENERATED {cfg.engine} node={plan.target} project={cfg.resolved_project()}",
        "services:",
    ]
    for rep in plan.replicas:
        lines.extend(_service_lines(cfg, rep.service_name, rep.device_ids, rep.host_port))
    path = compose_output_path(cfg, suffix=safe)
    path.write_text(_join(lines))
    return path, list(plan.urls)
