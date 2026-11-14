# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end GPU PPO training verification via ``mujoco_playground`` + ``rsl-rl-lib``."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from myosuite_reaf import EnvConfig, ensure_windows_fcntl_shim, register_with_playground
from myosuite_reaf.tasks.registry import get_registered_task


def _evaluate_policy_return(vec_env: object, policy: object, steps: int = 40) -> float:
    with torch.inference_mode():
        obs, _ = vec_env.reset()  # type: ignore[attr-defined]
        total_reward = torch.zeros(vec_env.num_envs, device=vec_env.device)  # type: ignore[attr-defined]
        for _ in range(steps):
            actions = policy(obs)  # type: ignore[operator]
            obs, rew, _, _ = vec_env.step(actions)  # type: ignore[attr-defined]
            total_reward += rew
    return float(total_reward.mean().item())


@pytest.mark.cuda
@pytest.mark.slow
def test_playground_registry_and_rsl_rl_gpu_ppo_improves_return() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for batched GPU PPO training test")

    ensure_windows_fcntl_shim()
    register_with_playground()

    from mujoco_playground._src.rl import env_registry
    from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper
    from rsl_rl.runners import OnPolicyRunner

    env_id = "myoElbowPose1D6MFixed-v0"
    entry = get_registered_task(env_id)
    pg_entry = env_registry.get(env_id)
    runner_cfg = pg_entry.runner_cfg

    nworld = 512
    reaf_env = pg_entry.env_builder(
        nworld=nworld,
        device="cuda:0",
        impl="warp",
        config=EnvConfig(impl="warp", num_envs=nworld, device="cuda:0", seed=42),
    )
    vec_env = vecenv_wrapper.RslRlVecEnvWrapper(
        reaf_env,
        cfg=runner_cfg.vecenv_wrapper_cfg,
        max_episode_length=entry.max_episode_steps,
    )
    vec_env.seed(42)

    train_cfg = dataclasses.asdict(runner_cfg)
    train_cfg["nworld"] = nworld
    train_cfg["max_iterations"] = 20

    # log_dir=None guarantees zero .pt checkpoint or event files are written anywhere
    runner = OnPolicyRunner(vec_env, train_cfg, log_dir=None, device="cuda:0")

    initial_return = _evaluate_policy_return(vec_env, runner.alg.get_policy(), steps=40)
    runner.learn(num_learning_iterations=20, init_at_random_ep_len=True)
    trained_return = _evaluate_policy_return(vec_env, runner.alg.get_policy(), steps=40)

    for p in runner.alg.actor.parameters():
        assert torch.all(torch.isfinite(p))
    for p in runner.alg.critic.parameters():
        assert torch.all(torch.isfinite(p))

    assert trained_return > initial_return + 5.0, (
        f"Expected 20 iterations of GPU PPO to improve return significantly: "
        f"initial={initial_return:.2f}, trained={trained_return:.2f}"
    )
    vec_env.close()
