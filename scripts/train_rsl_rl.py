# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-platform RSL-RL GPU PPO training CLI via ``mujoco_playground`` + ``reaf-sim``.

Example::

    python scripts/train_rsl_rl.py --env myoElbowPose1D6MRandom-v0 --num-envs 1024 --iterations 50 --no-save
"""

from __future__ import annotations

import argparse
import dataclasses
import tempfile
from datetime import datetime
from pathlib import Path

import torch

from myosuite_reaf import EnvConfig, ensure_windows_fcntl_shim, register_with_playground
from myosuite_reaf.tasks.registry import get_registered_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MyoSuite musculoskeletal policies on GPU via mujoco_playground + reaf-sim + RSL-RL."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="myoElbowPose1D6MRandom-v0",
        help="Registered myosuite-reaf environment ID.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override number of parallel MuJoCo Warp worlds.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override number of PPO learning iterations.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="PyTorch / Warp execution device.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "myosuite_reaf_logs",
        help="Root directory for TensorBoard logs and checkpoints.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable checkpoint and TensorBoard file output (recommended for smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_windows_fcntl_shim()
    register_with_playground()

    from mujoco_playground._src.rl import env_registry
    from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper
    from rsl_rl.runners import OnPolicyRunner

    entry = get_registered_task(args.env)
    pg_entry = env_registry.get(args.env)
    runner_cfg = pg_entry.runner_cfg
    nworld = int(args.num_envs or runner_cfg.nworld)
    iterations = int(args.iterations or runner_cfg.max_iterations)

    reaf_env = pg_entry.env_builder(
        nworld=nworld,
        device=args.device,
        impl="warp",
        config=EnvConfig(
            impl="warp",
            num_envs=nworld,
            device=args.device,
            seed=args.seed,
        ),
    )
    vec_env = vecenv_wrapper.RslRlVecEnvWrapper(
        reaf_env,
        cfg=runner_cfg.vecenv_wrapper_cfg,
        max_episode_length=entry.max_episode_steps,
    )
    vec_env.seed(args.seed)

    train_cfg = dataclasses.asdict(runner_cfg)
    train_cfg["seed"] = int(args.seed)
    train_cfg["nworld"] = nworld
    train_cfg["max_iterations"] = iterations

    log_dir: str | None = None
    if not args.no_save:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = args.log_root / runner_cfg.experiment_name / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        log_dir = str(run_dir)

    runner = OnPolicyRunner(
        vec_env,
        train_cfg,
        log_dir=log_dir,
        device=args.device,
    )
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    vec_env.close()


if __name__ == "__main__":
    main()
