# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-substep forward-kinematics synchronization device for ``reaf-sim``.

In ``reaf-sim`` v1.0.1, ``DeviceLayer._substep`` executes ``self.sim.step()`` and
immediately reads device measurements without calling ``self.sim.forward()`` on
the final substep. Because MuJoCo's ``mj_step`` / ``mjw.step`` integrates
``(qpos, qvel, act)`` in stage 2 *after* computing derived kinematics, tendon
lengths, actuator forces, and sensor values in stage 1, every derived measurement
lags ``(qpos, qvel, act)`` by one physics substep (``sim_dt``) unless ``forward()``
is called before reading measurements.

Registering :class:`SyncForwardDevice` as the first device in ``DeviceLayer``
runs ``simulator.forward()`` inside ``on_step(simulator)`` (which ``DeviceLayer``
invokes on ``is_last_substep=True`` right before ``get_measurements``), bringing
all downstream device measurements into exact post-step alignment and matching
MyoSuite's ``_step_physics()`` contract (``mj_step`` followed by ``mj_forward``).
"""

from __future__ import annotations

from typing import override

import torch
from reaf.simulation.torch._src.core import device
from reaf.simulation.torch._src.core import simulator as simulator_module


class SyncForwardDevice(device.Device):
    """Calls ``simulator.forward()`` after the final physics substep of each control step."""

    def __init__(
        self,
        name: str = "sync_forward",
        *,
        enabled: bool = True,
        every_substep: bool = False,
    ) -> None:
        """Initialize the forward-sync device.

        Args:
            name: Device instance name.
            enabled: When ``False``, becomes a no-op (used to benchmark or test
                the 1-substep lag present in stock ``reaf-sim``).
            every_substep: When ``True``, runs ``simulator.forward()`` after
                every physics substep (for substep measurement processors that
                consume derived forces/kinematics at ``sim_dt``). When ``False``
                (default), runs once per control step on ``is_last_substep=True``.
        """
        self._name = name
        self._enabled = enabled
        self._every_substep = every_substep

    @property
    @override
    def name(self) -> str:
        return self._name

    @override
    def reset(
        self,
        simulator: simulator_module.Simulator,
        env_mask: torch.Tensor | None = None,
    ) -> None:
        del simulator, env_mask

    @override
    def on_substep(self, simulator: simulator_module.Simulator) -> None:
        if self._enabled and self._every_substep:
            simulator.forward()

    @override
    def on_step(self, simulator: simulator_module.Simulator) -> None:
        if self._enabled and not self._every_substep:
            simulator.forward()
