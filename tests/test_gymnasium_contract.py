# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests verifying MyoSuite's five cross-backend Gymnasium invariants."""

from __future__ import annotations

from functools import partial

import gymnasium as gym
import numpy as np
import pytest

from myosuite_reaf import (
    DictObservationWrapper,
    ExcitationStageWrapper,
    FatigueWrapper,
    LowPassStage,
    MotorNoiseCfg,
    MotorNoiseWrapper,
    ReafferentationWrapper,
    SarcopeniaWrapper,
    list_envs,
    make_env,
)
from myosuite_reaf.assets import peak_isometric_force

REPRESENTATIVE_TASKS: tuple[str, ...] = (
    "myoElbowPose1D6MRandom-v0",
    "myoElbowPose1D6MExoRandom-v0",
    "motorFingerPoseRandom-v0",
    "motorFingerReachRandom-v0",
    "myoFingerReachRandom-v0",
    "myoHandPoseRandom-v0",
    "myoHandReachRandom-v0",
    "myoArmReachRandom-v0",
    "myoTorsoPoseFixed-v0",
    "myoTorsoExoPoseFixed-v0",
    "myoLegWalk-v0",
)


def test_registry_contains_all_base_and_physiological_variants() -> None:
    all_ids = list_envs()
    base_ids = list_envs(base_only=True)
    assert len(base_ids) == 21
    assert len(all_ids) == 59
    assert "myoSarcLegWalk-v0" in all_ids
    assert "myoFatiArmReachRandom-v0" in all_ids
    assert "myoReafHandReachRandom-v0" in all_ids
    # Motor-actuated finger envs do not generate myoSarc/myoFati variants
    assert "myoSarcmotorFingerPoseFixed-v0" not in all_ids


@pytest.mark.parametrize("env_id", REPRESENTATIVE_TASKS)
def test_five_invariants_and_step_api(env_id: str) -> None:
    env = make_env(env_id, seed=123)

    # Invariant 1: float32 unbounded Box(-inf, inf) observations
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.dtype == np.float32
    assert np.all(np.isneginf(env.observation_space.low))
    assert np.all(np.isposinf(env.observation_space.high))

    # Invariant 2: float32 action bounds ([-1, 1] except LegWalk [0, 1])
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.dtype == np.float32
    if env_id == "myoLegWalk-v0":
        np.testing.assert_allclose(env.action_space.low, 0.0)
        np.testing.assert_allclose(env.action_space.high, 1.0)
    else:
        np.testing.assert_allclose(env.action_space.low, -1.0)
        np.testing.assert_allclose(env.action_space.high, 1.0)

    # Invariant 3: ctrl_dt == frame_skip * sim_dt (50 Hz or 100 Hz integer divisor)
    assert env.dt == pytest.approx(env.frame_skip * env.sim_dt)

    # Reset and initial non-terminal check
    obs0, info0 = env.reset(seed=42)
    assert obs0.dtype == np.float32
    assert obs0.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs0))
    assert "obs_dict" in info0

    # Neutral first step must be finite and not immediately NaN/diverged
    neutral = 0.5 * (env.action_space.low + env.action_space.high)
    obs1, reward, terminated, truncated, info1 = env.step(neutral)
    assert obs1.dtype == np.float32
    assert obs1.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs1))
    assert isinstance(reward, float) and np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert not terminated, f"{env_id} terminated on its very first neutral control step!"
    assert not truncated
    assert "obs_dict" in info1 and "rwd_dict" in info1 and "solved" in info1
    assert info1["rwd_dict"]["dense"] == pytest.approx(reward, abs=1e-6)

    # obs_dict concatenation matches flat obs vector bit-for-bit
    concat_obs = np.concatenate(list(info1["obs_dict"].values()), axis=0)
    np.testing.assert_array_equal(concat_obs, obs1)
    env.close()


def test_deterministic_seeded_resets_and_trajectories() -> None:
    env = MotorNoiseWrapper(
        FatigueWrapper(
            make_env("myoElbowPose1D6MRandom-v0"),
            fatigue_reset_random=True,
        ),
        MotorNoiseCfg.van_beers_2004(),
    )
    actions = np.linspace(-0.7, 0.8, 6 * 5, dtype=np.float32).reshape(5, 6)

    obs_a0, _ = env.reset(seed=777)
    traj_a = [env.step(a)[0] for a in actions]

    obs_b0, _ = env.reset(seed=777)
    traj_b = [env.step(a)[0] for a in actions]

    obs_c0, _ = env.reset(seed=888)

    np.testing.assert_array_equal(obs_a0, obs_b0)
    for step_a, step_b in zip(traj_a, traj_b, strict=True):
        np.testing.assert_array_equal(step_a, step_b)
    assert not np.allclose(obs_a0, obs_c0)
    env.close()


def test_exact_max_episode_steps_truncation() -> None:
    env = make_env("myoElbowPose1D6MFixed-v0", seed=0)
    assert env.max_episode_steps == 100
    env.reset(seed=0)
    zero_act = np.zeros(env.action_space.shape, dtype=np.float32)

    for step_idx in range(1, env.max_episode_steps + 1):
        _, _, terminated, truncated, _ = env.step(zero_act)
        assert not terminated
        if step_idx < env.max_episode_steps:
            assert not truncated
        else:
            assert truncated
    env.close()


def test_wrappers_dict_obs_and_single_installation_guards() -> None:
    base = make_env("myoHandPoseFixed-v0", seed=0)
    f0 = peak_isometric_force(base.mj_model).copy()

    wrapped = DictObservationWrapper(
        ExcitationStageWrapper(
            ReafferentationWrapper(
                FatigueWrapper(
                    MotorNoiseWrapper(
                        SarcopeniaWrapper(base, force_scale=0.5),
                        MotorNoiseCfg.van_beers_2004(),
                    )
                )
            ),
            partial(LowPassStage, 0.3),
        )
    )

    # Public attribute forwarding through 6 nested wrappers
    assert wrapped.dt == pytest.approx(0.02)
    assert wrapped.frame_skip == 10
    np.testing.assert_allclose(peak_isometric_force(wrapped.mj_model), 0.5 * f0, rtol=1e-6)
    assert wrapped.pipeline.stage_names == ("noise", "fatigue", "reroute", "lowpass")

    obs_dict, _ = wrapped.reset(seed=0)
    assert isinstance(obs_dict, dict)
    assert set(obs_dict.keys()) == {"qpos", "qvel", "pose_err", "act"}
    obs_next, rwd, term, trunc, _ = wrapped.step(wrapped.action_space.sample())
    assert set(obs_next.keys()) == {"qpos", "qvel", "pose_err", "act"}
    assert np.isfinite(rwd) and not term and not trunc

    # Duplicate wrapper guards must raise ValueError
    with pytest.raises(ValueError, match="Sarcopenia is already applied"):
        SarcopeniaWrapper(wrapped)
    with pytest.raises(ValueError, match="already installed"):
        FatigueWrapper(wrapped)
    with pytest.raises(ValueError, match="already installed"):
        MotorNoiseWrapper(wrapped, MotorNoiseCfg.van_beers_2004())
    with pytest.raises(ValueError, match="already installed"):
        ReafferentationWrapper(wrapped)
    wrapped.close()

    # Pre-registered condition IDs also guard against double-wrapping
    sarc_id_env = make_env("myoSarcElbowPose1D6MFixed-v0")
    with pytest.raises(ValueError, match="Sarcopenia is already applied"):
        SarcopeniaWrapper(sarc_id_env)
    sarc_id_env.close()
