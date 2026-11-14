# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Throughput benchmark across CPU (``impl="cpp"``) and GPU Warp (``impl="warp"``).

Measures total environment steps per second (including ``ExcitationPipelineProcessor``,
physics substeps, ``SyncForwardDevice``, ``FeatureProducer``, rewards, terminations,
and observation assembly) across representative musculoskeletal complexity tiers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from myosuite_reaf import EnvConfig, make_reaf_env
from myosuite_reaf.devices.sync_forward_device import SyncForwardDevice

DEFAULT_TASKS: tuple[str, ...] = (
    "myoElbowPose1D6MRandom-v0",
    "myoHandReachRandom-v0",
    "myoLegWalk-v0",
)
DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 64, 512, 2048, 4096)


def _sync_if_cuda(device_str: str) -> None:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device_str)


def benchmark_single(
    env_id: str,
    *,
    impl: str,
    num_envs: int,
    device: str,
    warmup_steps: int,
    bench_steps: int,
    sync_forward: bool = True,
) -> dict[str, Any]:
    cfg = EnvConfig(
        impl=impl,  # type: ignore[arg-type]
        num_envs=num_envs,
        device=device,
        seed=0,
        sync_forward=sync_forward,
    )
    env = make_reaf_env(env_id, config=cfg)
    action_spec = env.action_spec()["actions"]
    nu = int(action_spec.shape[-1])
    lo = float(action_spec.minimum[0])
    hi = float(action_spec.maximum[0])
    mid = 0.5 * (lo + hi)

    actions = torch.full(
        (num_envs, nu), mid, dtype=torch.float32, device=torch.device(env.xpu_device)
    )
    env.reset()
    for _ in range(warmup_steps):
        env.step({"actions": actions})
    _sync_if_cuda(env.xpu_device)

    t0 = time.perf_counter()
    for _ in range(bench_steps):
        env.step({"actions": actions})
    _sync_if_cuda(env.xpu_device)
    elapsed = time.perf_counter() - t0

    mj_model = env.device_layer.sim.mj_model
    has_sync = any(
        isinstance(d, SyncForwardDevice) and d._enabled
        for d in getattr(env.device_layer, "_devices", ())
    )
    total_env_steps = num_envs * bench_steps
    steps_per_sec = total_env_steps / max(elapsed, 1e-12)
    substeps_per_step = int(round(env.device_layer.control_timestep / mj_model.opt.timestep))
    physics_steps_per_sec = steps_per_sec * substeps_per_step
    env.close()

    return {
        "env_id": env_id,
        "impl": impl,
        "device": device,
        "num_envs": num_envs,
        "nq": int(mj_model.nq),
        "nv": int(mj_model.nv),
        "nu": int(mj_model.nu),
        "na": int(mj_model.na),
        "substeps": substeps_per_step,
        "sync_forward": has_sync,
        "bench_steps": bench_steps,
        "elapsed_s": round(elapsed, 4),
        "env_steps_per_sec": round(steps_per_sec, 1),
        "physics_substeps_per_sec": round(physics_steps_per_sec, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark myosuite-reaf throughput.")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--warmup-steps", type=int, default=15)
    parser.add_argument("--bench-steps", type=int, default=50)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "N/A"
    print(f"# Throughput Benchmark (GPU: {gpu_name})\n")
    print(
        "| Task | DoF (`nv`) | Muscles (`na`) | Backend | `num_envs` | Control Steps/s | Physics Substeps/s |"
    )
    print("|---|---:|---:|---|---:|---:|---:|")

    for task_id in args.tasks:
        row_cpu = benchmark_single(
            task_id,
            impl="cpp",
            num_envs=1,
            device="cpu",
            warmup_steps=min(args.warmup_steps, 5),
            bench_steps=args.bench_steps,
        )
        results.append(row_cpu)
        print(
            f"| `{task_id}` | {row_cpu['nv']} | {row_cpu['na']} | `cpp` (CPU) | 1 | "
            f"{row_cpu['env_steps_per_sec']:,.1f} | {row_cpu['physics_substeps_per_sec']:,.1f} |"
        )
        if has_cuda:
            for bsz in args.batch_sizes:
                row_gpu = benchmark_single(
                    task_id,
                    impl="warp",
                    num_envs=bsz,
                    device="cuda:0",
                    warmup_steps=args.warmup_steps,
                    bench_steps=args.bench_steps,
                )
                results.append(row_gpu)
                print(
                    f"| `{task_id}` | {row_gpu['nv']} | {row_gpu['na']} | `warp` (CUDA) | {bsz} | "
                    f"{row_gpu['env_steps_per_sec']:,.1f} | {row_gpu['physics_substeps_per_sec']:,.1f} |"
                )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
