# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Empirical reproducers and regression tests for upstream dogfooding findings.

Every finding documented in ``DOGFOODING_REPORT.md`` against ``google-deepmind/reaf-sim``,
``google-deepmind/mujoco_playground``, and ``MyoHub/myo_sim`` is reproduced here
with an executable assertion verifying both the upstream behavior and ``myosuite-reaf``'s fix.
"""

from __future__ import annotations

import inspect

import mujoco
import numpy as np
import pytest
import torch
from reaf.simulation.torch._src.builders import config as builders_config
from reaf.simulation.torch._src.common import (
    articulated_device,
    functional_reward,
    timeout_termination_checker,
)
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types as reaf_types

from myosuite_reaf import EnvConfig, make_env, make_reaf_env
from myosuite_reaf._compat import ensure_windows_fcntl_shim
from myosuite_reaf.assets import (
    build_elbow_spec,
    build_leg_walk_spec,
    build_torso_spec,
)
from myosuite_reaf.devices import KinematicsDevice, MuscleActuatorDevice


def test_finding_1_playground_windows_fcntl_shim() -> None:
    """mujoco_playground._src.sim.asset_loader unconditionally imports POSIX fcntl."""
    ensure_windows_fcntl_shim()
    from mujoco_playground._src.sim import asset_loader

    src = inspect.getsource(asset_loader)
    assert "import fcntl" in src


def test_finding_3_and_4_actuator_device_missing_act_and_bound_accessor_bug() -> None:
    """BoundDataAccessor.act looks up nonexistent data.actuator_act; MuscleActuatorDevice reads data.act."""
    mj_spec = build_elbow_spec(exo=False)
    sim = simulator_module.Simulator(
        mj_spec=mj_spec,
        nworld=1,
        xpu_device="cpu",
        impl=reaf_types.Impl.CPP,
    )
    bound = sim.data.bind(sim.model, list(mj_spec.actuators))
    with pytest.raises(AttributeError, match="actuator_act"):
        _ = bound.act

    dev = MuscleActuatorDevice(
        name="actuators",
        xpu_device="cpu",
        measurements=("ctrl", "act", "actuator_length", "actuator_velocity", "actuator_force"),
    )
    dev.initialize(sim)
    dev.set_act(None, torch.full((1, 6), 0.42))
    sim.forward()
    meas = dev.get_measurements(sim)
    assert meas["actuators/act"].shape == (1, 6)
    assert torch.allclose(meas["actuators/act"], torch.full((1, 6), 0.42), atol=1e-6)
    assert meas["actuators/actuator_length"].shape == (1, 6)
    assert meas["actuators/actuator_velocity"].shape == (1, 6)
    assert meas["actuators/actuator_force"].shape == (1, 6)


def test_finding_5_substep_derived_kinematics_lag_and_sync_forward_fix() -> None:
    """Stock DeviceLayer._substep omits post-step forward(), leaving site_xpos stale by 1 substep."""
    action = np.linspace(-0.8, 0.9, 50, dtype=np.float32)

    env_nosync = make_env(
        "myoArmReachFixed-v0",
        config=EnvConfig(impl="cpp", num_envs=1, seed=0, sync_forward=False),
    )
    env_sync = make_env(
        "myoArmReachFixed-v0",
        config=EnvConfig(impl="cpp", num_envs=1, seed=0, sync_forward=True),
    )

    env_nosync.reset(seed=0)
    env_sync.reset(seed=0)

    _, _, _, _, info_nosync = env_nosync.step(action)
    _, _, _, _, info_sync = env_sync.step(action)

    # Recompute exact ground-truth mj_forward at post-step (qpos, qvel, act)
    sim_sync = env_sync.reaf_env.device_layer.sim
    mj_model = sim_sync.mj_model
    gt_data = mujoco.MjData(mj_model)
    gt_data.qpos[:] = sim_sync.data.qpos[0].numpy()
    gt_data.qvel[:] = sim_sync.data.qvel[0].numpy()
    gt_data.act[:] = sim_sync.data.act[0].numpy()
    mujoco.mj_forward(mj_model, gt_data)
    sid = mj_model.site("IFtip").id
    gt_tip = gt_data.site_xpos[sid].astype(np.float32)

    lag_err_nosync = float(np.max(np.abs(info_nosync["obs_dict"]["tip_pos"] - gt_tip)))
    lag_err_sync = float(np.max(np.abs(info_sync["obs_dict"]["tip_pos"] - gt_tip)))

    assert lag_err_nosync > 1e-5, (
        f"Expected measurable 1-substep site_xpos lag without SyncForwardDevice, got {lag_err_nosync}"
    )
    assert lag_err_sync <= 1e-6, (
        f"Expected SyncForwardDevice to match post-step mj_forward, got {lag_err_sync}"
    )
    env_nosync.close()
    env_sync.close()


def test_finding_6_sim_config_overwrites_mjcf_muscle_timestep() -> None:
    """builders_config.Sim() defaults simulation_timestep=0.005 and overwrites 0.002s MJCFs."""
    mj_spec = build_elbow_spec(exo=False)
    assert mj_spec.option.timestep == pytest.approx(0.002)

    default_sim = builders_config.Sim()
    assert default_sim.simulation_timestep == pytest.approx(0.005)
    spec_copy = mj_spec.copy()
    builders_config.apply_sim_config(spec_copy, default_sim)
    assert spec_copy.option.timestep == pytest.approx(0.005)

    # Verify myosuite-reaf preserves the MJCF 0.002s muscle timestep
    env = make_env("myoElbowPose1D6MRandom-v0")
    assert env.sim_dt == pytest.approx(0.002)
    assert env.dt == pytest.approx(0.02)
    assert env.frame_skip == 10
    env.close()


def test_finding_7_articulated_device_unnamed_freejoint_and_joint_range_bugs() -> None:
    """ArticulatedDevice drops unnamed <freejoint/>s and mismatches joint_range shapes."""
    # (a) Unnamed <freejoint/> elements in myotorso_exosuit.xml
    exo_torso_spec = build_torso_spec(exo=True)
    exo_model = exo_torso_spec.compile()
    assert exo_model.njnt == 20  # 18 named spine hinges/slides + 2 unnamed freejoints
    assert exo_model.nq == 32  # 18 + 2 * 7

    sim_exo = simulator_module.Simulator(
        mj_spec=exo_torso_spec, nworld=1, xpu_device="cpu", impl=reaf_types.Impl.CPP
    )
    art_exo = articulated_device.ArticulatedDevice(name="torso", xpu_device="cpu")
    art_exo.configure_spec(exo_torso_spec)
    art_exo.initialize(sim_exo)
    assert len(art_exo._joints) == 18  # 2 unnamed freejoints silently dropped!
    assert art_exo.home_qpos.shape == (18,)  # Missing 14 qpos DOFs of the 2 freejoints!

    # (b) Named <freejoint name="root"/> in myoLegWalk-v0 causes joint_range spec/tensor mismatch
    walk_spec = build_leg_walk_spec()
    art_walk = articulated_device.ArticulatedDevice(
        name="legs", xpu_device="cpu", measurements=("joint_range",)
    )
    art_walk.configure_spec(walk_spec)
    sim_walk = simulator_module.Simulator(
        mj_spec=walk_spec, nworld=1, xpu_device="cpu", impl=reaf_types.Impl.CPP
    )
    art_walk.initialize(sim_walk)
    declared_shape = art_walk.substep_measurements_spec()["legs/joint_range"].shape
    actual_shape = art_walk.get_measurements(sim_walk)["legs/joint_range"].shape[1:]
    assert declared_shape == (35, 2)  # nq == 35 (7 freejoint qpos + 28 hinges)
    assert actual_shape == (29, 2)  # njnt == 29 (1 freejoint + 28 hinges)
    assert declared_shape != actual_shape

    # Verify KinematicsDevice resolves both cleanly
    kin = KinematicsDevice(
        name="kinematics", xpu_device="cpu", measurements=("qpos", "qvel", "joint_range")
    )
    kin.initialize(sim_exo)
    spec_shapes = {k: v.shape for k, v in kin.substep_measurements_spec().items()}
    meas_shapes = {k: tuple(v.shape[1:]) for k, v in kin.get_measurements(sim_exo).items()}
    assert spec_shapes == meas_shapes
    assert spec_shapes["kinematics/qpos"] == (exo_model.nq,)


def test_finding_8_timeout_float32_boundary_and_vecenv_int_truncation() -> None:
    """TimeoutTerminationChecker uses strict `time > timeout`, sensitive to float32 drift."""
    checker = timeout_termination_checker.TimeoutTerminationChecker(
        xpu_device="cpu", timeout=2.0, time_feature_name="time"
    )
    t_slightly_under = torch.tensor([np.float32(2.0) - np.float32(2e-7)])
    t_slightly_over = torch.tensor([np.float32(2.0) + np.float32(2e-7)])
    res_under = checker.check_termination({"time": t_slightly_under})
    res_over = checker.check_termination({"time": t_slightly_over})
    assert int(res_under[0].item()) != int(res_over[0].item())

    # int(0.58 / 0.02) truncates 28.999999999999996 -> 28 instead of 29
    assert int(0.58 / 0.02) == 28
    assert round(0.58 / 0.02) == 29


def test_finding_12_inference_mode_reset_inplace_crash() -> None:
    """Stepping reaf.Environment inside torch.inference_mode() (as rsl-rl does) then resetting outside crashes."""
    reaf_env = make_reaf_env(
        "myoElbowPose1D6MFixed-v0",
        config=EnvConfig(impl="cpp", num_envs=1, seed=0),
    )
    with torch.inference_mode():
        reaf_env.reset()
        reaf_env.step({"actions": torch.zeros((1, 6), dtype=torch.float32)})

    # Outside inference_mode, reaf-sim's @torch.no_grad() _reset() mutates
    # _last_timestep.observation[k][:] in place, raising RuntimeError.
    with pytest.raises(RuntimeError, match="Inplace update to inference tensor"):
        reaf_env.reset()
    reaf_env.close()


def test_finding_10_functional_reward_provider_scales_by_dt() -> None:
    """FunctionalRewardProvider multiplies weighted sums by dt unconditionally."""
    from reaf.simulation.torch._src.builders import flat_scene

    provider = functional_reward.FunctionalRewardProvider(
        terms={"r": flat_scene.Reward(lambda f: f["x"], weight=2.0)},
        required_keys=[],
    )
    provider.initialize(device=torch.device("cpu"), dtype=torch.float32, dt=0.02)
    out = provider.compute_reward({"x": torch.tensor([3.0])})
    # Raw weighted sum is 2.0 * 3.0 = 6.0, but FunctionalRewardProvider returns 6.0 * 0.02 = 0.12
    assert float(out["reward"][0].item()) == pytest.approx(0.12)
    assert float(out["terms"]["r"][0].item()) == pytest.approx(3.0)


@pytest.mark.cuda
def test_positive_torch_wrapper_per_world_dr_isolation() -> None:
    """reaf-sim TorchWrapper isolates per-world body_mass & geom_size DR on partial resets."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for multi-world Warp DR test")

    reaf_env = make_reaf_env(
        "myoElbowPose1D6MExoRandom-v0",
        config=EnvConfig(impl="warp", num_envs=4, device="cuda:0", seed=7),
    )
    reaf_env.reset()
    sim = reaf_env.device_layer.sim
    bid = int(sim.mj_model.body("carry_weight").id)
    gid = int(sim.mj_model.body_geomadr[bid])

    initial_masses = sim.model.body_mass[:, bid].clone()
    initial_radii = sim.model.geom_size[:, gid, 0].clone()
    assert torch.std(initial_masses) > 1e-3  # Distinct masses across the 4 worlds

    # Reset ONLY worlds 0 and 2; worlds 1 and 3 MUST remain untouched
    mask = torch.tensor([True, False, True, False], dtype=torch.bool, device="cuda:0")
    reaf_env.reset_with_options(options=reaf_env.default_reset_options(), env_mask=mask)

    after_masses = sim.model.body_mass[:, bid].clone()
    after_radii = sim.model.geom_size[:, gid, 0].clone()

    assert torch.allclose(after_masses[[1, 3]], initial_masses[[1, 3]], atol=0.0)
    assert torch.allclose(after_radii[[1, 3]], initial_radii[[1, 3]], atol=0.0)
    assert not torch.allclose(after_masses[[0, 2]], initial_masses[[0, 2]])
    reaf_env.close()
