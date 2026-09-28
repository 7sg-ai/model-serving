"""CLI: python -m shared.deploy up|down|status|generate|inventory"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .compose_gen import generate_compose, generate_compose_for_node_plan
from .config import load_deploy_config
from .docker_util import compose_down, compose_ps, docker_available
from .ensure import ensure_model_runtime
from .inventory import inventory_nodes, parse_deploy_nodes
from .remote_exec import teardown_node_plan
from .scheduler import NodePlan, schedule_replicas

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Deploy/teardown NIM, vLLM, or SGLang model runtimes"
    )
    p.add_argument(
        "action",
        choices=["up", "down", "status", "generate", "inventory", "plan"],
    )
    p.add_argument("--engine", choices=["nim", "vllm", "sglang"], default="vllm")
    p.add_argument("--app-name", default="cli")
    p.add_argument("--model", default="")
    p.add_argument("--nodes", default="", help="Override DEPLOY_NODES csv")
    p.add_argument("--no-teardown-register", action="store_true")
    args = p.parse_args(argv)

    if args.nodes:
        import os

        os.environ["DEPLOY_NODES"] = args.nodes

    if args.action == "inventory":
        nodes = inventory_nodes(parse_deploy_nodes(args.nodes) or None)
        for n in nodes:
            print(
                json.dumps(
                    {
                        "target": n.target,
                        "advertise": n.advertise,
                        "gpu_ids": n.gpu_ids,
                        "gpu_count": n.gpu_count,
                        "docker_ok": n.docker_ok,
                        "error": n.error or None,
                    },
                    indent=2,
                )
            )
        return 0

    cfg = load_deploy_config(args.engine, app_name=args.app_name, model_id=args.model)

    if args.action == "plan":
        nodes = inventory_nodes(cfg.deploy_nodes or parse_deploy_nodes() or ["local"])
        plans = schedule_replicas(cfg, nodes)
        for plan in plans:
            print(
                json.dumps(
                    {
                        "target": plan.target,
                        "advertise": plan.advertise,
                        "replicas": [
                            {
                                "service": r.service_name,
                                "gpus": r.device_ids,
                                "port": r.host_port,
                                "url": "http://%s:%s/v1/chat/completions"
                                % (r.advertise, r.host_port),
                            }
                            for r in plan.replicas
                        ],
                    },
                    indent=2,
                )
            )
        return 0

    if args.action == "generate":
        if cfg.is_multi_node():
            nodes = inventory_nodes(cfg.deploy_nodes)
            plans = schedule_replicas(cfg, nodes)
            urls = []
            for plan in plans:
                path, u = generate_compose_for_node_plan(cfg, plan)
                print(path)
                urls.extend(u)
            print("BACKEND_URLS=" + ",".join(urls))
        else:
            path, urls = generate_compose(cfg)
            print(path)
            print("BACKEND_URLS=" + ",".join(urls))
        return 0

    if args.action == "up":
        handle = ensure_model_runtime(
            args.engine,
            app_name=args.app_name,
            model_id=args.model,
            register_teardown=not args.no_teardown_register,
        )
        print(json.dumps(handle.to_dict(), indent=2))
        print("BACKEND_URLS=" + ",".join(handle.urls))
        if not args.no_teardown_register and handle.teardown_on_exit and handle.owned:
            print("Note: process exit will tear down owned containers.", file=sys.stderr)
        return 0

    project = cfg.resolved_project()

    if args.action == "status":
        if cfg.is_multi_node():
            nodes = inventory_nodes(cfg.deploy_nodes)
            for n in nodes:
                print(n.target, "gpus", n.gpu_ids, "docker", n.docker_ok, n.error)
            return 0
        if not docker_available():
            print("docker not available")
            return 1
        path, urls = generate_compose(cfg)
        print(compose_ps(project, path))
        print("urls:", urls)
        return 0

    if args.action == "down":
        if cfg.is_multi_node():
            # best-effort: inventory + generate paths and down each
            nodes = inventory_nodes(cfg.deploy_nodes)
            try:
                plans = schedule_replicas(cfg, nodes)
            except Exception as exc:
                print("plan failed (tearing down by project name only may be incomplete):", exc)
                plans = []
            for plan in plans:
                path, _u = generate_compose_for_node_plan(cfg, plan)
                # remote path convention
                from .remote_exec import remote_dir

                safe = plan.target.replace("@", "_").replace(":", "_")
                remote_path = "%s/%s-%s.yml" % (remote_dir(cfg), project, safe)
                teardown_node_plan(cfg, plan, remote_path)
                print("down", plan.target)
            return 0
        path, urls = generate_compose(cfg)
        compose_down(
            project, path, timeout=cfg.teardown_timeout, remove_volumes=cfg.remove_volumes
        )
        print("down", project)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
