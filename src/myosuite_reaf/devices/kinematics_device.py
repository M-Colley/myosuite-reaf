# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generalized-coordinate, body-kinematics, and site-pose device for ``reaf-sim``.

Addresses two limitations in stock ``reaf-sim`` ``ArticulatedDevice``:
1. ``ArticulatedDevice`` filters ``[j for j in mj_spec.joints if j.name]``, which
   silently drops unnamed ``<freejoint/>`` elements (such as those in
   ``myotorso_exosuit.xml``) and produces a ``joint_range`` spec shape mismatch
   ``(qpos_size, 2)`` vs ``(n_joints, 2)`` whenever a ``<freejoint>`` is present.
2. Musculoskeletal tasks require body world poses (``xpos``, ``xipos``, ``xquat``),
   spatial body velocities (``cvel``, used for centre-of-mass velocity in walking),
   and end-effector/target site coordinates (``site_xpos``) routed cleanly through
   ``DeviceLayer`` rather than reading ``MjData`` inside ``TaskLayer``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import override

import numpy as np
import torch
from dm_env import specs
from reaf.simulation.torch._src.core import device
from reaf.simulation.torch._src.core import simulator as simulator_module

DEFAULT_KINEMATICS_MEASUREMENTS: tuple[str, ...] = (
    "qpos",
    "qvel",
    "site_xpos",
)


class KinematicsDevice(device.Device):
    """Exposes generalized coordinates, body kinematics, and site positions."""

    def __init__(
        self,
        name: str = "kinematics",
        xpu_device: str = "cpu",
        *,
        measurements: Sequence[str] = DEFAULT_KINEMATICS_MEASUREMENTS,
        site_ids: Sequence[int] | None = None,
        body_ids: Sequence[int] | None = None,
    ) -> None:
        """Initialize the kinematics device.

        Args:
            name: Device name prefix.
            xpu_device: PyTorch device string injected by ``flat_scene.build_device_layer``.
            measurements: Subset of ``("qpos", "qvel", "joint_range", "site_xpos",
                "xpos", "xipos", "xquat", "cvel")``.
            site_ids: Optional subset of site indices for ``site_xpos`` (default: all sites).
            body_ids: Optional subset of body indices for body kinematics (default: all bodies).
        """
        self._name = name
        self._xpu_device = xpu_device
        self._measurements = list(measurements)
        self._init_site_ids = list(site_ids) if site_ids is not None else None
        self._init_body_ids = list(body_ids) if body_ids is not None else None

        self._nq: int = 0
        self._nv: int = 0
        self._njnt: int = 0
        self._site_ids: torch.Tensor | None = None
        self._body_ids: torch.Tensor | None = None
        self._jnt_range: torch.Tensor | None = None

    @property
    @override
    def name(self) -> str:
        return self._name

    @override
    def initialize(self, simulator: simulator_module.Simulator) -> None:
        mj_model = simulator.mj_model
        self._nq = int(mj_model.nq)
        self._nv = int(mj_model.nv)
        self._njnt = int(mj_model.njnt)

        sids = (
            self._init_site_ids if self._init_site_ids is not None else list(range(mj_model.nsite))
        )
        bids = (
            self._init_body_ids if self._init_body_ids is not None else list(range(mj_model.nbody))
        )
        self._site_ids = torch.tensor(sids, dtype=torch.long, device=self._xpu_device)
        self._body_ids = torch.tensor(bids, dtype=torch.long, device=self._xpu_device)
        self._jnt_range = torch.as_tensor(
            mj_model.jnt_range, dtype=torch.float32, device=self._xpu_device
        )

    @override
    def reset(
        self,
        simulator: simulator_module.Simulator,
        env_mask: torch.Tensor | None = None,
    ) -> None:
        del simulator, env_mask

    @override
    def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
        assert self._site_ids is not None and self._body_ids is not None
        nsite = int(self._site_ids.numel())
        nbody = int(self._body_ids.numel())
        out: dict[str, specs.Array] = {}
        for m in self._measurements:
            if m == "qpos":
                shape: tuple[int, ...] = (self._nq,)
            elif m == "qvel":
                shape = (self._nv,)
            elif m == "joint_range":
                shape = (self._njnt, 2)
            elif m == "site_xpos":
                shape = (nsite, 3)
            elif m in ("xpos", "xipos"):
                shape = (nbody, 3)
            elif m == "xquat":
                shape = (nbody, 4)
            elif m == "cvel":
                shape = (nbody, 6)
            else:
                raise KeyError(f"Unsupported KinematicsDevice measurement: {m!r}")
            out[f"{self._name}/{m}"] = specs.Array(shape=shape, dtype=np.float32)
        return out

    @override
    def get_measurements(
        self,
        simulator: simulator_module.Simulator,
    ) -> Mapping[str, torch.Tensor]:
        assert (
            self._site_ids is not None
            and self._body_ids is not None
            and self._jnt_range is not None
        )
        data = simulator.data
        nworld = data.qpos.shape[0]
        out: dict[str, torch.Tensor] = {}
        for m in self._measurements:
            if m == "qpos":
                val = data.qpos
            elif m == "qvel":
                val = data.qvel
            elif m == "joint_range":
                val = self._jnt_range.unsqueeze(0).expand(nworld, -1, -1)
            elif m == "site_xpos":
                val = data.site_xpos[:, self._site_ids]
            elif m == "xpos":
                val = data.xpos[:, self._body_ids]
            elif m == "xipos":
                val = data.xipos[:, self._body_ids]
            elif m == "xquat":
                val = data.xquat[:, self._body_ids]
            elif m == "cvel":
                val = data.cvel[:, self._body_ids]
            else:
                raise KeyError(m)
            out[f"{self._name}/{m}"] = val.to(dtype=torch.float32)
        return out
