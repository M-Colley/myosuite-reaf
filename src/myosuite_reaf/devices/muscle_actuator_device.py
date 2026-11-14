# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Biomechanical muscle + motor actuator device for ``reaf-sim``.

Extends ``reaf-sim``'s actuation interface with the internal musculoskeletal
states absent from stock ``ActuatorDevice``:
  - ``act``: Hill-type muscle activation states ``data.act`` (shape ``(nworld, na)``),
    indexed via ``actuator_actadr`` / ``actuator_actnum`` (bypassing
    ``BoundDataAccessor.act``'s ``data.actuator_act`` lookup bug).
  - ``actuator_length``: Muscle-tendon unit lengths ``data.actuator_length`` (``(nworld, nu)``).
  - ``actuator_velocity``: Muscle-tendon unit velocities ``data.actuator_velocity`` (``(nworld, nu)``).
  - ``actuator_force``: Muscle/motor forces ``data.actuator_force`` (``(nworld, nu)``).
  - ``tendon_length`` / ``tendon_velocity``: Spatial/fixed tendon kinematics when present.
  - ``set_act()`` / ``set_ctrl()``: Masked puppet writers for episode resets and parity tests.

Covers all ``nu`` actuators of a model in exact MuJoCo ``ctrl`` order so pure-muscle,
pure-motor, and hybrid muscle + exoskeleton models expose one unified command tensor
compatible with ``mujoco_playground``'s ``RslRlVecEnvWrapper``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import override

import mujoco
import numpy as np
import torch
from dm_env import specs
from reaf.simulation.torch._src.core import device
from reaf.simulation.torch._src.core import simulator as simulator_module

DEFAULT_MUSCLE_MEASUREMENTS: tuple[str, ...] = (
    "ctrl",
    "act",
    "actuator_length",
    "actuator_velocity",
    "actuator_force",
)


class MuscleActuatorDevice(device.Device):
    """Unified musculoskeletal + motor actuator device for ``reaf-sim``."""

    def __init__(
        self,
        name: str = "actuators",
        xpu_device: str = "cpu",
        *,
        targets: Sequence[str] | None = None,
        measurements: Sequence[str] = DEFAULT_MUSCLE_MEASUREMENTS,
        home_ctrl: torch.Tensor | Sequence[float] | None = None,
    ) -> None:
        """Initialize the muscle/motor actuator device.

        Args:
            name: Device name prefix for command and measurement keys.
            xpu_device: PyTorch device string injected by ``flat_scene.build_device_layer``.
            targets: Optional explicit actuator names. Defaults to all model actuators
                in MuJoCo ``ctrl`` order.
            measurements: Measurement names to expose under ``f"{name}/{m}"``.
                Supported: ``"ctrl"``, ``"home_ctrl"``, ``"act"``,
                ``"actuator_length"``, ``"actuator_velocity"``, ``"actuator_force"``,
                ``"tendon_length"``, ``"tendon_velocity"``.
            home_ctrl: Optional default control vector of shape ``(nu,)``.
        """
        self._name = name
        self._xpu_device = xpu_device
        self._target_names = list(targets) if targets is not None else None
        self._measurements = list(measurements)
        self._init_home_ctrl = home_ctrl

        self._ctrl_ids: torch.Tensor | None = None
        self._actuator_ids: torch.Tensor | None = None
        self._act_ids: torch.Tensor | None = None
        self._tendon_ids: torch.Tensor | None = None
        self._muscle_mask: torch.Tensor | None = None
        self._home_ctrl: torch.Tensor | None = None
        self._data: simulator_module.Data | None = None

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    def muscle_mask(self) -> torch.Tensor:
        """Boolean mask of shape ``(num_targets,)`` marking ``mjDYN_MUSCLE`` actuators."""
        if self._muscle_mask is None:
            raise RuntimeError("MuscleActuatorDevice must be initialized first.")
        return self._muscle_mask

    @property
    def num_actuations(self) -> int:
        """Number of activation states ``na`` associated with target actuators."""
        if self._act_ids is None:
            raise RuntimeError("MuscleActuatorDevice must be initialized first.")
        return int(self._act_ids.numel())

    @override
    def initialize(self, simulator: simulator_module.Simulator) -> None:
        self._data = simulator.data
        mj_model = simulator.mj_model

        if self._target_names is None:
            act_ids = list(range(mj_model.nu))
        else:
            act_ids = [int(mj_model.actuator(n).id) for n in self._target_names]

        ctrl_ids: list[int] = []
        state_act_ids: list[int] = []
        tendon_ids: list[int] = []
        muscle_flags: list[bool] = []

        for aid in act_ids:
            ctrl_adr = int(mj_model.actuator_ctrladr[aid])
            ctrl_num = int(mj_model.actuator_ctrlnum[aid])
            ctrl_ids.extend(range(ctrl_adr, ctrl_adr + max(ctrl_num, 1)))

            act_adr = int(mj_model.actuator_actadr[aid])
            act_num = int(mj_model.actuator_actnum[aid])
            if act_adr >= 0 and act_num > 0:
                state_act_ids.extend(range(act_adr, act_adr + act_num))

            trntype = mujoco.mjtTrn(mj_model.actuator_trntype[aid])
            if trntype == mujoco.mjtTrn.mjTRN_TENDON:
                tendon_ids.append(int(mj_model.actuator_trnid[aid, 0]))

            is_muscle = mujoco.mjtDyn(mj_model.actuator_dyntype[aid]) == mujoco.mjtDyn.mjDYN_MUSCLE
            muscle_flags.append(is_muscle)

        self._actuator_ids = torch.tensor(act_ids, dtype=torch.long, device=self._xpu_device)
        self._ctrl_ids = torch.tensor(ctrl_ids, dtype=torch.long, device=self._xpu_device)
        self._act_ids = torch.tensor(state_act_ids, dtype=torch.long, device=self._xpu_device)
        self._tendon_ids = torch.tensor(tendon_ids, dtype=torch.long, device=self._xpu_device)
        self._muscle_mask = torch.tensor(muscle_flags, dtype=torch.bool, device=self._xpu_device)

        if self._init_home_ctrl is None:
            self._home_ctrl = torch.zeros(
                len(ctrl_ids), dtype=torch.float32, device=self._xpu_device
            )
        else:
            hc = torch.as_tensor(self._init_home_ctrl, dtype=torch.float32, device=self._xpu_device)
            if hc.shape != (len(ctrl_ids),):
                raise ValueError(f"home_ctrl shape {tuple(hc.shape)} != ({len(ctrl_ids)},).")
            self._home_ctrl = hc

    @override
    def reset(
        self,
        simulator: simulator_module.Simulator,
        env_mask: torch.Tensor | None = None,
    ) -> None:
        del simulator, env_mask

    @override
    def substep_commands_spec(self) -> Mapping[str, specs.Array]:
        if self._ctrl_ids is None:
            raise RuntimeError("MuscleActuatorDevice must be initialized first.")
        return {
            f"{self._name}/ctrl": specs.Array(
                shape=(int(self._ctrl_ids.numel()),), dtype=np.float32
            )
        }

    @override
    def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
        if (
            self._ctrl_ids is None
            or self._actuator_ids is None
            or self._act_ids is None
            or self._tendon_ids is None
        ):
            raise RuntimeError("MuscleActuatorDevice must be initialized first.")

        result: dict[str, specs.Array] = {}
        for m in self._measurements:
            if m in ("ctrl", "home_ctrl"):
                shape = (int(self._ctrl_ids.numel()),)
            elif m == "act":
                if self._act_ids.numel() == 0:
                    continue
                shape = (int(self._act_ids.numel()),)
            elif m in ("actuator_length", "actuator_velocity", "actuator_force"):
                shape = (int(self._actuator_ids.numel()),)
            elif m in ("tendon_length", "tendon_velocity"):
                shape = (int(self._tendon_ids.numel()),)
            else:
                raise KeyError(f"Unsupported MuscleActuatorDevice measurement: {m!r}")
            result[f"{self._name}/{m}"] = specs.Array(shape=shape, dtype=np.float32)
        return result

    @override
    def set_substep_commands(
        self,
        simulator: simulator_module.Simulator,
        consumed_substep_commands: Mapping[str, torch.Tensor],
    ) -> None:
        assert self._ctrl_ids is not None
        ctrl = consumed_substep_commands[f"{self._name}/ctrl"]
        simulator.data.ctrl[:, self._ctrl_ids] = ctrl.to(dtype=simulator.data.ctrl.dtype)

    @override
    def get_measurements(
        self,
        simulator: simulator_module.Simulator,
    ) -> Mapping[str, torch.Tensor]:
        assert (
            self._ctrl_ids is not None
            and self._actuator_ids is not None
            and self._act_ids is not None
            and self._tendon_ids is not None
            and self._home_ctrl is not None
        )
        data = simulator.data
        out: dict[str, torch.Tensor] = {}
        for m in self._measurements:
            if m == "ctrl":
                val = data.ctrl[:, self._ctrl_ids]
            elif m == "home_ctrl":
                val = self._home_ctrl.unsqueeze(0).expand(data.ctrl.shape[0], -1)
            elif m == "act":
                if self._act_ids.numel() == 0:
                    continue
                val = data.act[:, self._act_ids]
            elif m == "actuator_length":
                val = data.actuator_length[:, self._actuator_ids]
            elif m == "actuator_velocity":
                val = data.actuator_velocity[:, self._actuator_ids]
            elif m == "actuator_force":
                val = data.actuator_force[:, self._actuator_ids]
            elif m == "tendon_length":
                val = data.ten_length[:, self._tendon_ids]
            elif m == "tendon_velocity":
                val = data.ten_velocity[:, self._tendon_ids]
            else:
                raise KeyError(m)
            out[f"{self._name}/{m}"] = val.to(dtype=torch.float32)
        return out

    def set_ctrl(
        self,
        env_mask: torch.Tensor | None,
        ctrl: torch.Tensor,
    ) -> None:
        """Puppet writer for ``data.ctrl`` on selected worlds."""
        assert self._data is not None and self._ctrl_ids is not None
        target = ctrl.to(dtype=self._data.ctrl.dtype, device=self._data.ctrl.device)
        if env_mask is None:
            self._data.ctrl[:, self._ctrl_ids] = target
        else:
            curr = self._data.ctrl[:, self._ctrl_ids]
            self._data.ctrl[:, self._ctrl_ids] = torch.where(env_mask.unsqueeze(1), target, curr)

    def set_act(
        self,
        env_mask: torch.Tensor | None,
        act: torch.Tensor,
    ) -> None:
        """Puppet writer for Hill-type muscle activation state ``data.act``."""
        assert self._data is not None and self._act_ids is not None
        if self._act_ids.numel() == 0:
            return
        target = act.to(dtype=self._data.act.dtype, device=self._data.act.device)
        if env_mask is None:
            self._data.act[:, self._act_ids] = target
        else:
            curr = self._data.act[:, self._act_ids]
            self._data.act[:, self._act_ids] = torch.where(env_mask.unsqueeze(1), target, curr)
