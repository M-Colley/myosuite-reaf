# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU (``impl="cpp"``) vs GPU MuJoCo Warp (``impl="warp"``) numerical parity tests.

Verifies that across every musculoskeletal model family in ``myosuite-reaf`` —
pure Hill-type muscles, pure motors, hybrid muscle + exoskeleton + carry-weight DR,
39-muscle hand, 63-muscle wrapping arm, 210-muscle passive-exosuit torso with
unnamed freejoints and welds, and 80-muscle bipedal walking with polynomial
joint couplings and floor contacts — CPU MuJoCo 4 (float64) and GPU MuJoCo Warp 4
(float32 batched across ``nworld=8``) produce matching trajectories, observations,
and rewards from synced initial states.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from myosuite_reaf import EnvConfig, make_env, make_reaf_env

PARITY_CASES: tuple[tuple[str, float, float], ...] = (
    # (env_id, qpos_atol, obs_atol)
    ("myoElbowPose1D6MRandom-v0", 5e-5, 1e-4),
    ("myoElbowPose1D6MExoRandom-v0", 5e-5, 1e-4),
    ("myoFingerPoseRandom-v0", 5e-5, 1e-4),
    ("motorFingerReachRandom-v0", 5e-5, 1e-4),
    ("myoHandPoseRandom-v0", 2e-4, 5e-4),
    ("myoHandReachRandom-v0", 2e-4, 5e-4),
    ("myoArmReachFixed-v0", 5e-4, 1e-3),
    ("myoTorsoExoPoseFixed-v0", 1e-3, 2e-3),
    ("myoLegWalk-v0", 2e-3, 1e-2),
)


def _capture_cpu_reset_override(cpu_env: Any, env_id: str) -> dict[str, Any]:
    sim = cpu_env.reaf_env.device_layer.sim
    mj_model = sim.mj_model
    override: dict[str, Any] = {
        "qpos": sim.data.qpos[0].detach().cpu().numpy().copy(),
        "qvel": sim.data.qvel[0].detach().cpu().numpy().copy(),
    }
    leaf = cpu_env._leaf_resetters[0]
    if hasattr(leaf, "_target_state"):
        override["target_angles"] = (
            leaf._target_state.ensure(1, torch.device("cpu"))[0].detach().cpu().numpy().copy()
        )
    if getattr(leaf, "_weight_body_id", None) is not None:
        bid = leaf._weight_body_id
        override["carry_weight"] = float(sim.model.body_mass[0, bid].item())
    if "Reach" in env_id:
        target_sids = [
            s for s in range(mj_model.nsite) if mj_model.site(s).name.endswith("_target")
        ]
        override["target_pos"] = sim.model.site_pos[0, target_sids].detach().cpu().numpy().copy()
    return override


@pytest.mark.cuda
@pytest.mark.parametrize(("env_id", "qpos_atol", "obs_atol"), PARITY_CASES)
def test_cpu_vs_batched_warp_trajectory_parity(
    env_id: str, qpos_atol: float, obs_atol: float
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for CPU vs Warp GPU parity tests")

    nworld = 8
    n_steps = 10
    rng = np.random.default_rng(2026)

    cpu_env = make_env(env_id, impl="cpp", seed=42)
    obs_cpu_0, _ = cpu_env.reset(seed=42)
    override = _capture_cpu_reset_override(cpu_env, env_id)

    warp_env = make_reaf_env(
        env_id,
        config=EnvConfig(impl="warp", num_envs=nworld, device="cuda:0", seed=999),
    )
    warp_leaf = warp_env.environment_reset._resetters[0]
    warp_leaf.set_next_reset_override(**override)
    ts_warp_0 = warp_env.reset()
    obs_warp_0 = ts_warp_0.observation["policy"].detach().cpu().numpy()

    # Initial reset observation parity across all 8 GPU worlds
    for w in range(nworld):
        np.testing.assert_allclose(obs_warp_0[w], obs_cpu_0, atol=obs_atol, rtol=1e-4)

    lo = cpu_env.action_space.low
    hi = cpu_env.action_space.high
    actions = rng.uniform(lo, hi, size=(n_steps, lo.shape[0])).astype(np.float32)

    cpu_sim = cpu_env.reaf_env.device_layer.sim
    warp_sim = warp_env.device_layer.sim

    for step_idx in range(n_steps):
        act_np = actions[step_idx]
        obs_cpu, rwd_cpu, _, _, _ = cpu_env.step(act_np)

        act_gpu = (
            torch.as_tensor(act_np, dtype=torch.float32, device="cuda:0")
            .unsqueeze(0)
            .expand(nworld, -1)
            .contiguous()
        )
        ts_warp = warp_env.step({"actions": act_gpu})

        qpos_cpu = cpu_sim.data.qpos[0].detach().cpu().numpy()
        qpos_warp = warp_sim.data.qpos.detach().cpu().numpy()
        obs_warp = ts_warp.observation["policy"].detach().cpu().numpy()
        rwd_warp = ts_warp.reward["reward"].detach().cpu().numpy()

        # Verify all 8 GPU worlds remain tightly consistent (mujoco_warp atomic contact
        # allocation and tiled Cholesky reductions introduce ~1e-7..1e-5 float32 ULP spread)
        np.testing.assert_allclose(
            qpos_warp,
            qpos_warp[:1].repeat(nworld, axis=0),
            atol=2e-5,
            rtol=1e-4,
        )

        # Verify ALL 8 GPU worlds match CPU float64 MuJoCo 4 within float32 physics tolerances
        for w in range(nworld):
            np.testing.assert_allclose(
                qpos_warp[w],
                qpos_cpu,
                atol=qpos_atol,
                rtol=1e-3,
                err_msg=f"{env_id} world {w} qpos diverged at step {step_idx}",
            )
            np.testing.assert_allclose(
                obs_warp[w],
                obs_cpu,
                atol=obs_atol,
                rtol=2e-3,
                err_msg=f"{env_id} world {w} obs diverged at step {step_idx}",
            )
            assert float(rwd_warp[w]) == pytest.approx(rwd_cpu, abs=max(obs_atol * 10.0, 5e-3))

        if cpu_sim.mj_model.na > 0:
            act_state_cpu = cpu_sim.data.act[0].detach().cpu().numpy()
            act_state_warp = warp_sim.data.act[0].detach().cpu().numpy()
            np.testing.assert_allclose(act_state_warp, act_state_cpu, atol=1e-5, rtol=1e-4)

    cpu_env.close()
    warp_env.close()


@pytest.mark.cuda
def test_negative_parity_control_detects_action_or_target_mismatch() -> None:
    """Negative test confirming our parity tolerances reject perturbed actions or targets."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    cpu_env = make_env("myoElbowPose1D6MRandom-v0", impl="cpp", seed=0)
    cpu_env.reset(seed=0)
    override = _capture_cpu_reset_override(cpu_env, "myoElbowPose1D6MRandom-v0")

    warp_env = make_reaf_env(
        "myoElbowPose1D6MRandom-v0",
        config=EnvConfig(impl="warp", num_envs=2, device="cuda:0", seed=0),
    )
    warp_env.environment_reset._resetters[0].set_next_reset_override(**override)
    warp_env.reset()

    act_cpu = np.zeros(6, dtype=np.float32)
    act_warp = torch.zeros((2, 6), dtype=torch.float32, device="cuda:0")
    act_warp[:, 0] = 0.25  # Perturb BIClong command by +0.25

    obs_cpu, _, _, _, _ = cpu_env.step(act_cpu)
    ts_warp = warp_env.step({"actions": act_warp})
    obs_warp = ts_warp.observation["policy"][0].detach().cpu().numpy()

    assert float(np.max(np.abs(obs_warp - obs_cpu))) > 5e-4
    cpu_env.close()
    warp_env.close()
