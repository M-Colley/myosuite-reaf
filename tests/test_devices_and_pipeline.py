# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for biomechanical devices and the ordered muscle excitation pipeline."""

from __future__ import annotations

from functools import partial

import mujoco
import numpy as np
import pytest
import torch

from myosuite_reaf import (
    CustomStageSpec,
    EnvConfig,
    ExcitationStage,
    FatigueSpec,
    LowPassStage,
    MotorNoiseCfg,
    MotorNoiseSpec,
    StageOrderWarning,
    apply_motor_noise,
    make_env,
    make_reaf_env,
    sigmoid_excitation,
)
from myosuite_reaf.assets import build_elbow_spec, peak_isometric_force
from myosuite_reaf.processors import ExcitationPipelineProcessor


class _GainStage(ExcitationStage):
    def __init__(self, scale: float, name: str = "gain", order: float | None = None) -> None:
        self.scale = scale
        self.name = name
        self.order = order

    def __call__(self, u: torch.Tensor, xp: object) -> torch.Tensor:
        return xp.clip(u * self.scale, 0.0, 1.0)  # type: ignore[attr-defined]


def test_sigmoid_excitation_matches_canonical_formula() -> None:
    actions = np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32)
    expected = 1.0 / (1.0 + np.exp(-5.0 * (actions - 0.5)))
    np.testing.assert_allclose(sigmoid_excitation(actions, np), expected, rtol=1e-6)
    t_out = sigmoid_excitation(torch.from_numpy(actions), torch).numpy()
    np.testing.assert_allclose(t_out, expected, rtol=1e-6)
    assert float(expected[2]) == pytest.approx(0.5)


def test_motor_noise_cfg_validation_and_signal_dependent_scaling() -> None:
    cfg = MotorNoiseCfg.van_beers_2004()
    assert cfg.enabled
    assert cfg.signal_dependent_std == pytest.approx(0.103)
    assert cfg.constant_std == pytest.approx(0.185)
    assert not MotorNoiseCfg().enabled

    with pytest.raises(ValueError, match="finite and >= 0"):
        MotorNoiseCfg(signal_dependent_std=-0.1)
    with pytest.raises(ValueError, match="finite and >= 0"):
        MotorNoiseCfg(constant_std=float("nan"))

    rng = torch.Generator().manual_seed(0)
    n1 = torch.randn((20_000, 1), generator=rng)
    n2 = torch.randn((20_000, 1), generator=rng)
    u_low = torch.full((20_000, 1), 0.1)
    u_high = torch.full((20_000, 1), 0.8)

    noisy_low = apply_motor_noise(u_low, n1, n2, 0.3, 0.01, torch)
    noisy_high = apply_motor_noise(u_high, n1, n2, 0.3, 0.01, torch)
    assert float(noisy_low.min()) >= 0.0 and float(noisy_high.max()) <= 1.0
    assert float(noisy_high.std()) > 3.0 * float(noisy_low.std())


def test_pipeline_stage_order_is_canonical_regardless_of_spec_order() -> None:
    mj_model = build_elbow_spec(exo=False).compile()
    # Intentionally pass features in reverse order: custom -> fatigue -> noise
    proc = ExcitationPipelineProcessor(
        mj_model=mj_model,
        control_timestep=0.02,
        features=(
            CustomStageSpec(partial(LowPassStage, 0.4, name="lp")),
            FatigueSpec(),
            MotorNoiseSpec(MotorNoiseCfg.van_beers_2004()),
        ),
    )
    assert proc.stage_names == ("noise", "fatigue", "lp")


def test_pipeline_duplicate_stage_and_priority_clash_warnings() -> None:
    mj_model = build_elbow_spec(exo=False).compile()
    with pytest.raises(ValueError, match="already installed"):
        ExcitationPipelineProcessor(
            mj_model=mj_model,
            control_timestep=0.02,
            features=(MotorNoiseSpec(), MotorNoiseSpec()),
        )

    with pytest.raises(ValueError, match="built-in stage name"):
        ExcitationPipelineProcessor(
            mj_model=mj_model,
            control_timestep=0.02,
            features=(CustomStageSpec(lambda: _GainStage(0.5, name="noise")),),
        )

    with pytest.raises(ValueError, match="strictly in"):
        ExcitationPipelineProcessor(
            mj_model=mj_model,
            control_timestep=0.02,
            features=(CustomStageSpec(lambda: _GainStage(0.5, name="early", order=5.0)),),
        )

    with pytest.warns(StageOrderWarning, match="STAGE ORDER CLASH"):
        ExcitationPipelineProcessor(
            mj_model=mj_model,
            control_timestep=0.02,
            features=(
                CustomStageSpec(lambda: _GainStage(0.9, name="s1", order=55.0)),
                CustomStageSpec(lambda: _GainStage(0.8, name="s2", order=55.0)),
            ),
        )


def test_hybrid_muscle_and_exo_pipeline_leaves_motor_channel_unmapped_by_sigmoid() -> None:
    mj_model = build_elbow_spec(exo=True).compile()
    assert mj_model.nu == 7 and mj_model.na == 6
    muscle_mask = mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
    motor_idx = int(np.flatnonzero(~muscle_mask)[0])

    proc = ExcitationPipelineProcessor(mj_model=mj_model, control_timestep=0.02)
    raw = torch.full((1, 7), -0.5, dtype=torch.float32)
    ctrl = proc.process_commands({"actions": raw})["actuators/ctrl"]

    # Motor channel keeps -0.5; muscle channels pass through sigmoid(-0.5)
    expected_muscle = float(sigmoid_excitation(np.float32(-0.5), np))
    assert float(ctrl[0, motor_idx].item()) == pytest.approx(-0.5)
    for m_idx in np.flatnonzero(muscle_mask):
        assert float(ctrl[0, int(m_idx)].item()) == pytest.approx(expected_muscle, abs=1e-6)


def test_3ccr_fatigue_accumulates_conserves_compartments_and_supports_partial_reset() -> None:
    env = make_env("myoFatiElbowPose1D6MRandom-v0", seed=0)
    env.reset(seed=0)
    fs = env.pipeline.fatigue_state
    assert fs is not None

    max_act = np.ones(env.action_space.shape, dtype=np.float32)
    mf_history: list[float] = []
    for _ in range(15):
        env.step(max_act)
        total = fs.MA + fs.MR + fs.MF
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
        mf_history.append(float(fs.MF.mean().item()))

    assert mf_history[-1] > mf_history[0] > 0.0
    env.reset(seed=0)
    assert float(fs.MF.abs().max().item()) == pytest.approx(0.0, abs=1e-7)
    env.close()

    # Subset muscle fatigue test: only BIClong fatigues
    sub_env = make_env(
        "myoElbowPose1D6MRandom-v0",
        features=(FatigueSpec(muscles=("BIClong",)),),
    )
    sub_env.reset(seed=0)
    for _ in range(10):
        sub_env.step(np.ones(6, dtype=np.float32))
    assert sub_env.pipeline.fatigue_state is not None
    assert sub_env.pipeline.fatigue_state.MF.shape == (1, 1)
    assert float(sub_env.pipeline.fatigue_state.MF[0, 0].item()) > 0.0
    sub_env.close()


def test_reafferentation_reroutes_eip_to_epl_and_silences_eip() -> None:
    env = make_env("myoReafHandPoseFixed-v0", seed=0)
    env.reset(seed=0)
    mj_model = env.mj_model
    eip_id = int(mj_model.actuator("EIP_r").id)
    epl_id = int(mj_model.actuator("EPL_r").id)

    action = np.full(mj_model.nu, -1.0, dtype=np.float32)
    action[eip_id] = 0.9  # Distinct high command on EIP_r
    action[epl_id] = -0.8  # Distinct low command on EPL_r (should be overwritten by EIP_r!)

    env.step(action)
    applied_ctrl = env.reaf_env.device_layer.sim.data.ctrl[0].detach().cpu().numpy()
    expected_epl = float(sigmoid_excitation(np.float32(0.9), np))

    assert float(applied_ctrl[eip_id]) == pytest.approx(0.0, abs=1e-7)
    assert float(applied_ctrl[epl_id]) == pytest.approx(expected_epl, abs=1e-6)
    env.close()


def test_sarcopenia_halves_peak_isometric_force_including_auto_force_muscles() -> None:
    base_elbow = make_env("myoElbowPose1D6MFixed-v0")
    sarc_elbow = make_env("myoSarcElbowPose1D6MFixed-v0")
    f_base = peak_isometric_force(base_elbow.mj_model)
    f_sarc = peak_isometric_force(sarc_elbow.mj_model)
    np.testing.assert_allclose(f_sarc, 0.5 * f_base, rtol=1e-6)
    assert np.all(f_sarc > 0.0)
    base_elbow.close()
    sarc_elbow.close()

    base_walk = make_env("myoLegWalk-v0")
    sarc_walk = make_env("myoSarcLegWalk-v0")
    fw_base = peak_isometric_force(base_walk.mj_model)
    fw_sarc = peak_isometric_force(sarc_walk.mj_model)
    np.testing.assert_allclose(fw_sarc, 0.5 * fw_base, rtol=1e-6)
    assert np.all(fw_sarc > 0.0)
    base_walk.close()
    sarc_walk.close()


def test_lowpass_stage_filters_excitations_and_resets_per_world() -> None:
    lp = LowPassStage(alpha=0.25)
    u0 = torch.zeros((2, 3), dtype=torch.float32)
    u1 = torch.ones((2, 3), dtype=torch.float32)

    out0 = lp(u0, torch)
    assert torch.allclose(out0, u0)

    out1 = lp(u1, torch)
    assert torch.allclose(out1, torch.full((2, 3), 0.25))

    # Reset only world 0: next call should snap world 0 directly to u1 while world 1 continues filtering
    lp.reset(torch.tensor([True, False]))
    out2 = lp(u1, torch)
    assert torch.allclose(out2[0], torch.ones(3))
    assert torch.allclose(out2[1], torch.full((3,), 0.25 + 0.25 * 0.75))


@pytest.mark.cuda
def test_batched_warp_fatigue_masked_reset_preserves_unmasked_worlds() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    reaf_env = make_reaf_env(
        "myoFatiElbowPose1D6MRandom-v0",
        config=EnvConfig(impl="warp", num_envs=4, device="cuda:0", seed=0),
    )
    reaf_env.reset()
    proc = reaf_env.task_layer._commands_processors[0]
    assert isinstance(proc, ExcitationPipelineProcessor) and proc.fatigue_state is not None

    act = torch.ones((4, 6), dtype=torch.float32, device="cuda:0")
    for _ in range(8):
        reaf_env.step({"actions": act})

    mf_before = proc.fatigue_state.MF.clone()
    assert torch.all(mf_before > 0.0)

    mask = torch.tensor([True, False, False, True], dtype=torch.bool, device="cuda:0")
    reaf_env.reset_with_options(options=reaf_env.default_reset_options(), env_mask=mask)
    mf_after = proc.fatigue_state.MF.clone()

    assert torch.allclose(mf_after[[0, 3]], torch.zeros_like(mf_after[[0, 3]]), atol=1e-7)
    assert torch.allclose(mf_after[[1, 2]], mf_before[[1, 2]], atol=1e-7)
    reaf_env.close()
