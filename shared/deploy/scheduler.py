"""Pack model replicas onto inventoried nodes by free GPUs."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from .config import DeployConfig
from .inventory import NodeInventory

logger = logging.getLogger(__name__)


@dataclass
class ReplicaPlacement:
    replica_index: int
    target: str
    advertise: str
    device_ids: List[str]
    host_port: int
    service_name: str


@dataclass
class NodePlan:
    target: str
    advertise: str
    replicas: List[ReplicaPlacement] = field(default_factory=list)

    @property
    def urls(self) -> List[str]:
        return [
            "http://%s:%s/v1/chat/completions" % (r.advertise, r.host_port)
            for r in self.replicas
        ]


def schedule_replicas(
    cfg: DeployConfig,
    nodes: List[NodeInventory],
) -> List[NodePlan]:
    if cfg.deploy_mode == "sharded" and len([n for n in nodes if n.target != "local" or len(nodes) > 1]) > 1:
        # multi-node list with sharded: only allow if single node effectively
        usable = [n for n in nodes if n.docker_ok and not n.error]
        if len(usable) > 1:
            raise RuntimeError(
                "MODEL_DEPLOY_MODE=sharded across multiple DEPLOY_NODES is not supported "
                "in auto-deploy v1 (no cross-node TP). Use a single node with enough GPUs, "
                "or set MODEL_COORDINATOR_URL to an external multi-node engine."
            )

    usable = []
    for n in nodes:
        if n.error and not n.docker_ok:
            logger.warning("Skipping node %s: %s", n.target, n.error)
            continue
        if not n.docker_ok:
            logger.warning("Skipping node %s: docker not ok (%s)", n.target, n.error)
            continue
        if n.gpu_count <= 0:
            logger.warning("Skipping node %s: no GPUs", n.target)
            continue
        usable.append(n)

    if not usable:
        raise RuntimeError(
            "No usable GPU nodes after inventory. Check DEPLOY_NODES, passwordless SSH, "
            "docker, and nvidia-smi on each host."
        )

    gpr = cfg.gpus_per_replica
    replicas = cfg.resolved_replicas()
    total_gpus = sum(n.gpu_count for n in usable)
    need = replicas * gpr
    if total_gpus < need:
        raise RuntimeError(
            "Not enough GPUs across nodes: have %s, need %s (%s replicas x %s GPUs). "
            "Nodes: %s"
            % (
                total_gpus,
                need,
                replicas,
                gpr,
                ", ".join("%s:%s" % (n.target, n.gpu_count) for n in usable),
            )
        )

    # mutable free gpu lists
    free: Dict[str, List[str]] = {n.target: list(n.gpu_ids) for n in usable}
    meta = {n.target: n for n in usable}
    port_cursor: Dict[str, int] = {n.target: cfg.port_base for n in usable}
    local_svc_idx: Dict[str, int] = {n.target: 0 for n in usable}

    plans: Dict[str, NodePlan] = {
        n.target: NodePlan(target=n.target, advertise=n.advertise) for n in usable
    }
    placements: List[ReplicaPlacement] = []

    for r in range(replicas):
        # best-fit: node with enough GPUs and least leftover
        candidates = []
        for t, gpus in free.items():
            if len(gpus) >= gpr:
                leftover = len(gpus) - gpr
                candidates.append((leftover, -len(gpus), t))
        if not candidates:
            raise RuntimeError("Failed to pack replica %s (fragmentation)" % r)
        candidates.sort()
        t = candidates[0][2]
        devices = free[t][:gpr]
        free[t] = free[t][gpr:]
        port = port_cursor[t]
        port_cursor[t] = port + 1
        svc_i = local_svc_idx[t]
        local_svc_idx[t] = svc_i + 1
        prefix = cfg.engine if cfg.engine in ("vllm", "sglang", "nim") else "model"
        place = ReplicaPlacement(
            replica_index=r,
            target=t,
            advertise=meta[t].advertise,
            device_ids=devices,
            host_port=port,
            service_name="%s-%s" % (prefix, svc_i),
        )
        plans[t].replicas.append(place)
        placements.append(place)
        logger.info(
            "Place replica %s on %s gpus=%s port=%s",
            r,
            t,
            devices,
            port,
        )

    # only nodes that received work
    return [plans[t] for t in plans if plans[t].replicas]
