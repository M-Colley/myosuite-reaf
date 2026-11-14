# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration types, excitation stages, and physiological condition specs.

Designed to mirror ``myosuite.terms.base_action``, ``myosuite.envs.muscle_stages``,
and ``myosuite.envs.myo.backends.mjlab`` so that ``myosuite_reaf`` can later be
integrated directly as ``myosuite/envs/myo/backends/reaf/`` once MyoSuite
migrates to MuJoCo 4.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

SimBackend = Literal["cpp", "warp"]

MAP_ORDER: float = 10.0
WRITE_ORDER: float = 100.0
STAGE_ORDER: dict[str, float] = {"noise": 20.0, "fatigue": 30.0, "reroute": 40.0}
LATE_ORDER: float = 50.0


class StageOrderWarning(UserWarning):
    """Emitted when two muscle excitation stages share the same explicit priority."""


warnings.simplefilter("always", StageOrderWarning)


def sigmoid_excitation(action: Any, xp: Any) -> Any:
    """Apply the canonical MyoSuite sigmoid muscle mapping ``1 / (1 + exp(-5 * (a - 0.5)))``.

    Args:
        action: Policy output in ``[-1, 1]``.
        xp: Array module (``numpy`` or ``torch``).

    Returns:
        Muscle excitation in ``(0, 1)`` with the same shape as *action*.
    """
    return 1.0 / (1.0 + xp.exp(-5.0 * (action - 0.5)))


def apply_motor_noise(
    excitation: Any,
    normals_sd: Any,
    normals_c: Any,
    signal_dependent_std: float,
    constant_std: float,
    xp: Any,
) -> Any:
    """Apply signal-dependent + constant Gaussian motor noise to muscle excitations.

    Args:
        excitation: Muscle excitations ``u`` in ``[0, 1]``.
        normals_sd: Standard normal draws for the multiplicative term.
        normals_c: Standard normal draws for the additive term.
        signal_dependent_std: Multiplicative noise scale (Harris & Wolpert 1998).
        constant_std: Additive noise scale.
        xp: Array module (``numpy`` or ``torch``).

    Returns:
        Noisy excitations clipped to ``[0, 1]``.
    """
    noisy = excitation + signal_dependent_std * excitation * normals_sd + constant_std * normals_c
    return xp.clip(noisy, 0.0, 1.0)


@dataclass(frozen=True)
class MotorNoiseCfg:
    """Gaussian motor noise configuration on muscle excitations (off by default).

    Attributes:
        signal_dependent_std: Multiplicative noise standard deviation.
        constant_std: Additive noise standard deviation.
    """

    signal_dependent_std: float = 0.0
    constant_std: float = 0.0

    def __post_init__(self) -> None:
        for name in ("signal_dependent_std", "constant_std"):
            val = float(getattr(self, name))
            if not math.isfinite(val) or val < 0.0:
                raise ValueError(f"MotorNoiseCfg.{name} must be finite and >= 0, got {val}.")
            object.__setattr__(self, name, val)

    @property
    def enabled(self) -> bool:
        """Whether either noise component is non-zero."""
        return self.signal_dependent_std > 0.0 or self.constant_std > 0.0

    @classmethod
    def van_beers_2004(cls) -> MotorNoiseCfg:
        """Literature defaults (0.103 signal-dependent, 0.185 constant; van Beers et al., 2004)."""
        return cls(signal_dependent_std=0.103, constant_std=0.185)

    @classmethod
    def from_value(cls, value: MotorNoiseCfg | Mapping[str, float] | None) -> MotorNoiseCfg:
        """Coerce ``None``, a mapping, or an existing :class:`MotorNoiseCfg`."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**value)
        raise TypeError(f"Expected MotorNoiseCfg, Mapping, or None, got {type(value)!r}.")


class ExcitationStage:
    """Portable custom stage operating on muscle excitations ``u`` in ``[0, 1]``.

    Subclasses implement ``__call__(u, xp)`` using operations shared by ``numpy``
    and ``torch``, and clear per-env state in :meth:`reset`.
    """

    name: str = "stage"
    order: float | None = None

    def __call__(self, u: Any, xp: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def reset(self, env_mask: Any = None) -> None:
        """Reset stage state for environments selected by *env_mask*."""


class LowPassStage(ExcitationStage):
    """First-order exponential low-pass filter on muscle excitations."""

    def __init__(
        self,
        alpha: float = 0.3,
        name: str = "lowpass",
        order: float | None = None,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must lie in (0, 1], got {alpha}.")
        self.alpha = alpha
        self.name = name
        self.order = order
        self._y: Any = None
        self._fresh: Any = None

    def __call__(self, u: Any, xp: Any) -> Any:
        if self._y is None:
            self._y = xp.zeros_like(u)
            self._fresh = xp.ones_like(u[..., 0], dtype=bool)
        fresh = self._fresh[..., None]
        self._y = xp.where(fresh, u, self._y + self.alpha * (u - self._y))
        self._fresh = xp.zeros_like(self._fresh)
        return self._y.clone() if hasattr(self._y, "clone") else self._y.copy()

    def reset(self, env_mask: Any = None) -> None:
        if self._fresh is None:
            return
        if env_mask is None:
            self._fresh[...] = True
        else:
            self._fresh[env_mask] = True


@dataclass(frozen=True)
class MotorNoiseSpec:
    """Declarative motor-noise stage specification."""

    cfg: MotorNoiseCfg = MotorNoiseCfg.van_beers_2004()


@dataclass(frozen=True)
class FatigueSpec:
    """Declarative 3CC-r muscle-fatigue stage specification."""

    fatigue_reset_vec: tuple[float, ...] | None = None
    fatigue_reset_random: bool = False
    muscles: tuple[str | int, ...] | None = None


@dataclass(frozen=True)
class ReafferentationSpec:
    """Declarative EIP -> EPL tendon-transfer (reafferentation) stage specification."""

    source_actuator: str = "EIP"
    target_actuator: str = "EPL"


@dataclass(frozen=True)
class SarcopeniaSpec:
    """Declarative sarcopenia (peak isometric force reduction) specification."""

    force_scale: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.force_scale <= 1.0:
            raise ValueError(f"force_scale must be in (0, 1], got {self.force_scale}.")


@dataclass(frozen=True)
class CustomStageSpec:
    """Declarative wrapper around a zero-arg :class:`ExcitationStage` factory."""

    make_stage: Callable[[], ExcitationStage]


FeatureSpec = MotorNoiseSpec | FatigueSpec | ReafferentationSpec | SarcopeniaSpec | CustomStageSpec


@dataclass(frozen=True)
class EnvConfig:
    """Runtime execution configuration for building a ``reaf-sim`` MyoSuite environment.

    Attributes:
        impl: Simulation backend: ``"cpp"`` (CPU MuJoCo C++) or ``"warp"`` (MuJoCo Warp).
        num_envs: Number of parallel worlds (``1`` required on ``"cpp"``; arbitrary ``N`` on ``"warp"``).
        device: PyTorch / Warp device string (e.g. ``"cpu"`` or ``"cuda:0"``). When ``None``,
            defaults to ``"cpu"`` for ``impl="cpp"`` and ``"cuda:0"`` (if available) for ``impl="warp"``.
        seed: Initial RNG seed for deterministic resets and motor noise.
        features: Ordered physiological / control pipeline feature specifications.
        sync_forward: Whether :class:`SyncForwardDevice` runs ``simulator.forward()`` after the
            final physics substep so derived kinematics/forces match post-step ``(qpos, qvel, act)``.
        nconmax: Optional per-world maximum contact allocation for Warp.
        njmax: Optional per-world maximum constraint allocation for Warp.
    """

    impl: SimBackend = "cpp"
    num_envs: int = 1
    device: str | None = None
    seed: int = 0
    features: tuple[FeatureSpec, ...] = ()
    sync_forward: bool = True
    nconmax: int | None = None
    njmax: int | None = None

    def __post_init__(self) -> None:
        if self.impl not in ("cpp", "warp"):
            raise ValueError(f"impl must be 'cpp' or 'warp', got {self.impl!r}.")
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs}.")
        if self.impl == "cpp" and self.num_envs != 1:
            raise ValueError(
                f"reaf-sim impl='cpp' supports num_envs=1 only (got {self.num_envs}); "
                "use impl='warp' for batched environments."
            )
        if not isinstance(self.features, tuple):
            object.__setattr__(self, "features", tuple(self.features))

    def resolved_device(self) -> str:
        """Resolve the target XPU device string."""
        if self.device is not None:
            return self.device
        if self.impl == "cpp":
            return "cpu"
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def with_features(self, *extra: FeatureSpec | Sequence[FeatureSpec]) -> EnvConfig:
        """Return a copy with additional :data:`FeatureSpec` entries appended."""
        merged = list(self.features)
        for item in extra:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                merged.extend(item)
            else:
                merged.append(item)  # type: ignore[arg-type]
        return EnvConfig(
            impl=self.impl,
            num_envs=self.num_envs,
            device=self.device,
            seed=self.seed,
            features=tuple(merged),
            sync_forward=self.sync_forward,
            nconmax=self.nconmax,
            njmax=self.njmax,
        )
