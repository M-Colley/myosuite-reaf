# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordered muscle excitation pipeline implemented as a ``reaf-sim`` ``CommandsProcessor``.

Implements MyoSuite's canonical control-rate pipeline (``docs/wiki/cross-backend-contract.md``):
``clip -> action-to-excitation map (10) -> motor noise (20) -> 3CC-r fatigue (30)
-> reafferentation reroute (40) -> custom ExcitationStages (50+) -> actuators/ctrl (100)``.

Because this processor lives in ``TaskLayer.command_processors`` (rather than
``DeviceLayer.command_processors``), it runs exactly once per **control step**
(``ctrl_dt = frame_skip * sim_dt``) and holds the resulting ``ctrl`` across all
physics substeps — matching MyoSuite's ``_apply_action()`` + ``frame_skip`` contract.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence

import mujoco
import numpy as np
import torch
from dm_env import specs
from myo_sim.physics.fatigue import TorchFatigueState
from reaf.core import commands_processor

from myosuite_reaf.config import (
    LATE_ORDER,
    MAP_ORDER,
    STAGE_ORDER,
    WRITE_ORDER,
    CustomStageSpec,
    ExcitationStage,
    FatigueSpec,
    FeatureSpec,
    MotorNoiseCfg,
    MotorNoiseSpec,
    ReafferentationSpec,
    SarcopeniaSpec,
    StageOrderWarning,
    apply_motor_noise,
    sigmoid_excitation,
)


class ExcitationPipelineProcessor(commands_processor.CommandsProcessor):
    """Task-layer command processor executing MyoSuite's muscle excitation pipeline."""

    def __init__(
        self,
        name: str = "excitation_pipeline",
        *,
        mj_model: mujoco.MjModel,
        control_timestep: float,
        normalize_act: bool = True,
        muscle_sigmoid: bool = True,
        features: Sequence[FeatureSpec] = (),
        actuator_suffix: str = "",
        seed: int = 0,
        nworld: int = 1,
        xpu_device: str = "cpu",
        input_key: str = "actions",
        output_key: str = "actuators/ctrl",
    ) -> None:
        self._name = name
        self._mj_model = mj_model
        self._actuator_suffix = actuator_suffix
        self._nworld = nworld
        self._xpu_device = xpu_device
        self._device = torch.device(xpu_device)
        self._ctrl_dt = float(control_timestep)
        self._normalize_act = normalize_act
        self._muscle_sigmoid = muscle_sigmoid
        self._input_key = input_key
        self._output_key = output_key
        self._nu = int(mj_model.nu)
        self._na = int(mj_model.na)

        muscle_mask_np = mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self._muscle_mask = torch.as_tensor(muscle_mask_np, dtype=torch.bool, device=self._device)
        self._has_muscles = bool(muscle_mask_np.any())

        ctrlrange = np.asarray(mj_model.actuator_ctrlrange, dtype=np.float32)
        self._ctrl_mid = torch.as_tensor(
            np.mean(ctrlrange, axis=-1), dtype=torch.float32, device=self._device
        )
        self._ctrl_half_span = torch.as_tensor(
            0.5 * (ctrlrange[:, 1] - ctrlrange[:, 0]),
            dtype=torch.float32,
            device=self._device,
        )

        if normalize_act:
            if self._na > 0 and not muscle_sigmoid:
                low_np = np.zeros(self._nu, dtype=np.float32)
                high_np = np.ones(self._nu, dtype=np.float32)
            else:
                low_np = -np.ones(self._nu, dtype=np.float32)
                high_np = np.ones(self._nu, dtype=np.float32)
        else:
            low_np = ctrlrange[:, 0].copy()
            high_np = ctrlrange[:, 1].copy()

        self._action_low_np = low_np
        self._action_high_np = high_np
        self._action_low = torch.as_tensor(low_np, dtype=torch.float32, device=self._device)
        self._action_high = torch.as_tensor(high_np, dtype=torch.float32, device=self._device)

        self._rng = torch.Generator(device=self._device)
        self._rng.manual_seed(int(seed))

        self._motor_noise_cfg = MotorNoiseCfg()
        self._fatigue_state: TorchFatigueState | None = None
        self._fatigue_indices: torch.Tensor | None = None
        self._fatigue_reset_vec: tuple[float, ...] | None = None
        self._fatigue_reset_random: bool = False
        self._reroute_pair: tuple[int, int] | None = None

        self._stages: list[
            tuple[
                float,
                str,
                Callable[[torch.Tensor], torch.Tensor],
                Callable[[torch.Tensor | None], None] | None,
            ]
        ] = []
        self._installed_orders: dict[str, float] = {}

        for feat in features:
            if isinstance(feat, SarcopeniaSpec):
                continue
            if isinstance(feat, MotorNoiseSpec):
                self._install_noise(feat.cfg)
            elif isinstance(feat, FatigueSpec):
                self._install_fatigue(mj_model, feat)
            elif isinstance(feat, ReafferentationSpec):
                self._install_reroute(mj_model, feat, actuator_suffix)
            elif isinstance(feat, CustomStageSpec):
                self._install_custom_stage(feat.make_stage())
            else:
                raise TypeError(f"Unsupported FeatureSpec: {feat!r}")

        self._stages.sort(key=lambda entry: entry[0])

    @property
    def name(self) -> str:
        return self._name

    @property
    def action_low(self) -> np.ndarray:
        """1-D float32 action lower bound array of shape ``(nu,)``."""
        return self._action_low_np

    @property
    def action_high(self) -> np.ndarray:
        """1-D float32 action upper bound array of shape ``(nu,)``."""
        return self._action_high_np

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Installed stage names in execution order."""
        return tuple(name for _, name, _, _ in self._stages)

    @property
    def fatigue_state(self) -> TorchFatigueState | None:
        """Active 3CC-r :class:`TorchFatigueState`, if installed."""
        return self._fatigue_state

    @property
    def motor_noise(self) -> MotorNoiseCfg:
        """Current :class:`MotorNoiseCfg`."""
        return self._motor_noise_cfg

    def set_motor_noise(self, cfg: MotorNoiseCfg | Mapping[str, float] | None) -> None:
        """Update motor noise levels at runtime (installs stage if needed)."""
        parsed = MotorNoiseCfg.from_value(cfg)
        if "noise" not in self._installed_orders:
            self._install_noise(parsed)
            self._stages.sort(key=lambda entry: entry[0])
        else:
            self._motor_noise_cfg = parsed

    def set_fatigue_reset_random(self, enabled: bool) -> None:
        """Toggle random initial fatigue sampling on reset."""
        if self._fatigue_state is None:
            raise AttributeError("No FatigueSpec stage is installed on this pipeline.")
        self._fatigue_reset_random = bool(enabled)

    def reseed(self, seed: int) -> None:
        """Reseed the pipeline's internal PyTorch RNG."""
        self._rng.manual_seed(int(seed))

    def install_motor_noise(self, cfg: MotorNoiseCfg | Mapping[str, float] | None = None) -> None:
        """Install the motor-noise stage once (raises ``ValueError`` if already installed)."""
        self._install_noise(MotorNoiseCfg.from_value(cfg))
        self._stages.sort(key=lambda entry: entry[0])

    def install_fatigue(self, spec: FatigueSpec | None = None) -> None:
        """Install the 3CC-r muscle-fatigue stage once."""
        self._install_fatigue(self._mj_model, spec or FatigueSpec())
        self._stages.sort(key=lambda entry: entry[0])

    def install_reafferentation(self, spec: ReafferentationSpec | None = None) -> None:
        """Install the EIP -> EPL reafferentation reroute stage once."""
        self._install_reroute(self._mj_model, spec or ReafferentationSpec(), self._actuator_suffix)
        self._stages.sort(key=lambda entry: entry[0])

    def install_custom_stage(self, stage: ExcitationStage) -> None:
        """Install a custom :class:`ExcitationStage` once."""
        self._install_custom_stage(stage)
        self._stages.sort(key=lambda entry: entry[0])

    def _register_stage(
        self,
        name: str,
        order: float,
        apply_fn: Callable[[torch.Tensor], torch.Tensor],
        reset_fn: Callable[[torch.Tensor | None], None] | None = None,
    ) -> None:
        if name in self._installed_orders:
            raise ValueError(f"Stage {name!r} is already installed; install it once.")
        clash = [n for n, o in self._installed_orders.items() if o == order]
        if clash:
            warnings.warn(
                f"STAGE ORDER CLASH: stages {[*clash, name]} share order {order}.",
                StageOrderWarning,
                stacklevel=3,
            )
        self._installed_orders[name] = order
        self._stages.append((order, name, apply_fn, reset_fn))

    def _install_noise(self, cfg: MotorNoiseCfg) -> None:
        self._motor_noise_cfg = cfg

        def _apply(ctrl: torch.Tensor) -> torch.Tensor:
            if not self._motor_noise_cfg.enabled or not self._has_muscles:
                return ctrl
            u = ctrl[:, self._muscle_mask]
            n1 = torch.randn(u.shape, generator=self._rng, device=u.device, dtype=u.dtype)
            n2 = torch.randn(u.shape, generator=self._rng, device=u.device, dtype=u.dtype)
            noisy = apply_motor_noise(
                u,
                n1,
                n2,
                self._motor_noise_cfg.signal_dependent_std,
                self._motor_noise_cfg.constant_std,
                torch,
            )
            out = ctrl.clone()
            out[:, self._muscle_mask] = noisy
            return out

        self._register_stage("noise", STAGE_ORDER["noise"], _apply)

    def _install_fatigue(self, mj_model: mujoco.MjModel, spec: FatigueSpec) -> None:
        self._fatigue_reset_vec = spec.fatigue_reset_vec
        self._fatigue_reset_random = spec.fatigue_reset_random
        self._fatigue_state = TorchFatigueState.from_mj_model(
            mj_model,
            num_envs=self._nworld,
            device=self._xpu_device,
            use_uniform_params=True,
            muscles=spec.muscles,
        )
        if spec.muscles is None:
            idx_np = np.flatnonzero(mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
        else:
            idx_np = np.array(
                [
                    int(m) if isinstance(m, int) else int(mj_model.actuator(m).id)
                    for m in spec.muscles
                ],
                dtype=np.int64,
            )
        self._fatigue_indices = torch.as_tensor(idx_np, dtype=torch.long, device=self._device)

        def _apply(ctrl: torch.Tensor) -> torch.Tensor:
            assert self._fatigue_state is not None and self._fatigue_indices is not None
            if self._fatigue_indices.numel() == 0:
                return ctrl
            ma = self._fatigue_state.step(ctrl[:, self._fatigue_indices], dt=self._ctrl_dt)
            out = ctrl.clone()
            out[:, self._fatigue_indices] = ma.to(dtype=ctrl.dtype)
            return out

        def _reset(env_mask: torch.Tensor | None) -> None:
            assert self._fatigue_state is not None
            self._fatigue_state.reset(
                env_ids=env_mask,
                fatigue_reset_vec=self._fatigue_reset_vec,
                fatigue_reset_random=self._fatigue_reset_random,
                generator=self._rng,
            )

        self._register_stage("fatigue", STAGE_ORDER["fatigue"], _apply, _reset)

    def _install_reroute(
        self,
        mj_model: mujoco.MjModel,
        spec: ReafferentationSpec,
        suffix: str,
    ) -> None:
        src_name = f"{spec.source_actuator}{suffix}"
        dst_name = f"{spec.target_actuator}{suffix}"
        try:
            src_id = int(mj_model.actuator(src_name).id)
            dst_id = int(mj_model.actuator(dst_name).id)
        except KeyError:
            src_id = int(mj_model.actuator(spec.source_actuator).id)
            dst_id = int(mj_model.actuator(spec.target_actuator).id)
        self._reroute_pair = (src_id, dst_id)

        def _apply(ctrl: torch.Tensor) -> torch.Tensor:
            assert self._reroute_pair is not None
            eip_id, epl_id = self._reroute_pair
            out = ctrl.clone()
            out[:, epl_id] = ctrl[:, eip_id]
            out[:, eip_id] = 0.0
            return out

        self._register_stage("reroute", STAGE_ORDER["reroute"], _apply)

    def _install_custom_stage(self, stage: ExcitationStage) -> None:
        if stage.name in STAGE_ORDER:
            raise ValueError(f"Custom stage cannot reuse built-in stage name {stage.name!r}.")
        if stage.order is not None and not (MAP_ORDER < stage.order < WRITE_ORDER):
            raise ValueError(
                f"Custom stage order must lie strictly in ({MAP_ORDER}, {WRITE_ORDER}), "
                f"got {stage.order}."
            )
        order = LATE_ORDER if stage.order is None else float(stage.order)

        def _apply(ctrl: torch.Tensor) -> torch.Tensor:
            if not self._has_muscles:
                return stage(ctrl, torch)
            out = ctrl.clone()
            out[:, self._muscle_mask] = stage(ctrl[:, self._muscle_mask], torch)
            return out

        def _reset(env_mask: torch.Tensor | None) -> None:
            stage.reset(env_mask)

        self._register_stage(stage.name, order, _apply, _reset)

    def consumed_commands_spec(self) -> Mapping[str, specs.Array]:
        return {
            self._input_key: specs.BoundedArray(
                shape=(self._nu,),
                dtype=np.float32,
                minimum=self._action_low_np,
                maximum=self._action_high_np,
            )
        }

    def produced_commands_keys(self) -> set[str]:
        return {self._output_key}

    def process_commands(
        self, consumed_commands: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        raw = consumed_commands[self._input_key].to(device=self._device, dtype=torch.float32)
        ctrl = torch.clamp(raw, self._action_low, self._action_high)

        if self._normalize_act:
            if self._na > 0 and self._muscle_sigmoid:
                ctrl = ctrl.clone()
                ctrl[:, self._muscle_mask] = sigmoid_excitation(ctrl[:, self._muscle_mask], torch)
            elif self._na == 0:
                ctrl = self._ctrl_mid + ctrl * self._ctrl_half_span

        for _, _, apply_fn, _ in self._stages:
            ctrl = apply_fn(ctrl)

        return {self._output_key: ctrl}

    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        for _, _, _, reset_fn in self._stages:
            if reset_fn is not None:
                reset_fn(env_mask)
