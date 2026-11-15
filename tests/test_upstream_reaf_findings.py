# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Empirical reproducers and forward-compatible regression tests for upstream dogfooding findings.

Every finding documented in ``DOGFOODING_REPORT.md`` against ``google-deepmind/reaf-sim``,
``google-deepmind/mujoco_playground``, and ``MyoHub/myo_sim`` is verified here against
both ``reaf-sim==1.0.1`` / ``playground==1.0.1`` (pre-fix reproducer) and upstream ``main``
with PRs ``reaf-sim`` #11–#15 and ``mujoco_playground`` #390–#391 applied.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from reaf.simulation.torch._src.builders import config as builders_config
from reaf.simulation.torch._src.common import (
    actuator_device,
    articulated_device,
    functional_reward,
    timeout_termination_checker,
)
from reaf.simulation.torch._src.core import device_layer as device_layer_module
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
    """Verify Windows asset_loader compatibility (pre-fix shim or mujoco_playground#391)."""
    ensure_windows_fcntl_shim()
    from etils import epath
    from mujoco_playground._src.sim import asset_loader

    src = inspect.getsource(asset_loader)
    if hasattr(asset_loader, "_file_lock"):
        assert 'sys.platform == "win32"' in src
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = epath.Path(Path(tmp) / "test.lock")
            with asset_loader._file_lock(lock_path):
                assert lock_path.exists()
    else:
        assert "import fcntl" in src


def test_finding_3_and_4_actuator_device_missing_act_and_bound_accessor_bug() -> None:
    """Verify BoundDataAccessor.act (reaf-sim#13), ActuatorDevice (reaf-sim#14), and MuscleActuatorDevice."""
    mj_spec = build_elbow_spec(exo=False)
    sim = simulator_module.Simulator(
        mj_spec=mj_spec,
        nworld=1,
        xpu_device="cpu",
        impl=reaf_types.Impl.CPP,
    )
    bound = sim.data.bind(sim.model, list(mj_spec.actuators))
    if hasattr(bound, "act_id"):
        assert bound.act.shape == (1, 6)
        assert bound.act_dot.shape == (1, 6)
    else:
        with pytest.raises(AttributeError, match="actuator_act"):
            _ = bound.act

    stock_dev = actuator_device.ActuatorDevice(
        name="stock",
        xpu_device="cpu",
        targets=[a.name for a in mj_spec.actuators],
        measurements=["act", "actuator_length", "actuator_velocity"],
    )
    stock_dev.initialize(sim)
    if hasattr(stock_dev, "set_act"):
        stock_specs = stock_dev.substep_measurements_spec()
        assert stock_specs["stock/act"].shape == (6,)
        assert stock_specs["stock/actuator_length"].shape == (6,)
        assert stock_specs["stock/actuator_velocity"].shape == (6,)
        stock_dev.set_act(None, torch.full((1, 6), 0.31))
        assert torch.allclose(
            stock_dev.get_measurements(sim)["stock/act"].float(),
            torch.full((1, 6), 0.31),
            atol=1e-6,
        )
    else:
        with pytest.raises(KeyError, match="act"):
            stock_dev.substep_measurements_spec()

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
    """Verify post-step forward() sync (SyncForwardDevice or reaf-sim#11)."""
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

    substep_src = inspect.getsource(device_layer_module.DeviceLayer._substep)
    if "self.sim.forward()" in substep_src:
        assert lag_err_nosync <= 1e-6, (
            f"Expected upstream reaf-sim#11 post-step forward() to match mj_forward, got {lag_err_nosync}"
        )
    else:
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
    """Verify ArticulatedDevice unnamed freejoints & joint_range (pre-fix or reaf-sim#15) plus KinematicsDevice."""
    exo_torso_spec = build_torso_spec(exo=True)
    exo_model = exo_torso_spec.compile()
    assert exo_model.njnt == 20  # 18 named spine hinges/slides + 2 unnamed freejoints
    assert exo_model.nq == 32  # 18 + 2 * 7

    sim_exo = simulator_module.Simulator(
        mj_spec=exo_torso_spec, nworld=1, xpu_device="cpu", impl=reaf_types.Impl.CPP
    )
    art_exo = articulated_device.ArticulatedDevice(name="torso", xpu_device="cpu")
    if hasattr(art_exo, "configure_spec"):
        art_exo.configure_spec(exo_torso_spec)
    art_exo.initialize(sim_exo)

    walk_spec = build_leg_walk_spec()
    art_walk = articulated_device.ArticulatedDevice(
        name="legs", xpu_device="cpu", measurements=["joint_range"]
    )
    if hasattr(art_walk, "configure_spec"):
        art_walk.configure_spec(walk_spec)
    sim_walk = simulator_module.Simulator(
        mj_spec=walk_spec, nworld=1, xpu_device="cpu", impl=reaf_types.Impl.CPP
    )
    art_walk.initialize(sim_walk)
    declared_shape = art_walk.substep_measurements_spec()["legs/joint_range"].shape
    actual_shape = art_walk.get_measurements(sim_walk)["legs/joint_range"].shape[1:]

    init_src = inspect.getsource(articulated_device.ArticulatedDevice.initialize)
    if "if j.name" in init_src:
        assert len(art_exo._joints) == 18
        assert art_exo.home_qpos.shape == (18,)
        assert declared_shape == (35, 2)
        assert actual_shape == (29, 2)
        assert declared_shape != actual_shape
    else:
        assert len(art_exo._joints) == 20
        assert art_exo.home_qpos.shape == (32,)
        assert declared_shape == (29, 2)
        assert actual_shape == (29, 2)

    # Verify KinematicsDevice resolves both cleanly across all versions
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
    """Verify Environment.reset() after torch.inference_mode() rollout (pre-fix or reaf-sim#12)."""
    reaf_env = make_reaf_env(
        "myoElbowPose1D6MFixed-v0",
        config=EnvConfig(impl="cpp", num_envs=1, seed=0),
    )
    with torch.inference_mode():
        reaf_env.reset()
        ts_inf = reaf_env.step({"actions": torch.zeros((1, 6), dtype=torch.float32)})

    if ts_inf.observation["policy"].is_inference():
        with pytest.raises(RuntimeError, match="Inplace update to inference tensor"):
            reaf_env.reset()
    else:
        ts_out = reaf_env.reset()
        assert not ts_out.observation["policy"].is_inference()
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
    reaf_env.close()
