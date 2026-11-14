# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gymnasium 5-tuple adapter and physiological wrappers over ``reaf-sim``.

Bridges ``reaf.simulation.torch.Environment`` (batched ``dm_env.TimeStep`` on
CPU C++ or GPU MuJoCo Warp) to MyoSuite's canonical ``gymnasium.Env`` contract
(``docs/wiki/cross-backend-contract.md``) so policies, evaluation harnesses,
and physiological wrappers behave identically across backends.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import dm_env
import gymnasium as gym
import numpy as np
import torch
from gymnasium.utils import RecordConstructorArgs
from reaf.simulation.torch import AutoResetMode
from reaf.simulation.torch import Environment as ReafEnvironment

from myosuite_reaf.assets import apply_sarcopenia_to_model
from myosuite_reaf.config import (
    STAGE_ORDER,
    EnvConfig,
    ExcitationStage,
    FatigueSpec,
    FeatureSpec,
    MotorNoiseCfg,
    ReafferentationSpec,
    SarcopeniaSpec,
    SimBackend,
)
from myosuite_reaf.processors.excitation_pipeline import ExcitationPipelineProcessor
from myosuite_reaf.tasks.registry import get_registered_task, make_reaf_env


class _ForwardPublicAttributes:
    """Forward public attributes missing on a wrapper to the wrapped env."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "env":
            raise AttributeError(name)
        return self.env.get_wrapper_attr(name)


def _find_excitation_pipeline(reaf_env: ReafEnvironment) -> ExcitationPipelineProcessor:
    processors = getattr(reaf_env.task_layer, "_commands_processors", ())
    for proc in processors:
        if isinstance(proc, ExcitationPipelineProcessor):
            return proc
    raise RuntimeError("No ExcitationPipelineProcessor found in reaf_env.task_layer.")


def _iter_leaf_resetters(resetter: Any) -> list[Any]:
    children = getattr(resetter, "_resetters", None)
    if children is not None:
        out: list[Any] = []
        for child in children:
            out.extend(_iter_leaf_resetters(child))
        return out
    return [resetter]


def _extract_obs_slices(reaf_env: ReafEnvironment) -> dict[str, slice]:
    adapter = getattr(reaf_env, "_observation_space_adapter", None)
    obs_groups = getattr(adapter, "_observations", None)
    if not isinstance(obs_groups, Mapping) or "policy" not in obs_groups:
        return {}
    slices: dict[str, slice] = {}
    offset = 0
    for term_name, term_cfg in obs_groups["policy"].terms.items():
        size = int(np.prod(term_cfg.output_shape or (1,)))
        slices[term_name] = slice(offset, offset + size)
        offset += size
    return slices


class GymnasiumEnvAdapter(gym.Env[np.ndarray, np.ndarray]):
    """Single-world ``gymnasium.Env`` adapter over a ``reaf.simulation.torch.Environment``.

    Supports both ``impl="cpp"`` (CPU C++ MuJoCo 4) and ``impl="warp"`` (GPU
    MuJoCo Warp 4 with ``num_envs=1``), enforcing MyoSuite's five cross-backend
    invariants:

    1. Unbounded ``float32`` ``Box(-inf, inf)`` observations with named ``obs_dict`` slices.
    2. Exact ``float32`` action space matching ``ExcitationPipelineProcessor`` bounds.
    3. ``dt == frame_skip * sim_dt``.
    4. Exact step-count ``max_episode_steps`` truncation independent of float32 clock drift.
    5. Deterministic ``reset(seed=...)`` across task resetters and excitation stages.
    """

    metadata: dict[str, Any] = {"render_modes": []}
    supports_ctrl_stages: bool = True

    def __init__(
        self,
        reaf_env: ReafEnvironment,
        *,
        env_id: str | None = None,
        max_episode_steps: int = 100,
        sarcopenia_applied: bool = False,
    ) -> None:
        super().__init__()
        if reaf_env.nworld != 1:
            raise ValueError(
                f"GymnasiumEnvAdapter requires nworld=1, got nworld={reaf_env.nworld}. "
                "Use make_reaf_env(..., EnvConfig(impl='warp', num_envs=N)) with "
                "RslRlVecEnvWrapper for batched GPU environments."
            )
        self._reaf_env = reaf_env
        self._reaf_env.auto_reset_mode = AutoResetMode.DISABLED
        self._env_id = env_id
        self._max_episode_steps = int(max_episode_steps)
        self._xpu_device = torch.device(reaf_env.xpu_device)
        self._pipeline = _find_excitation_pipeline(reaf_env)
        self._leaf_resetters = _iter_leaf_resetters(reaf_env.environment_reset)
        self._obs_slices = _extract_obs_slices(reaf_env)
        self._sarcopenia_applied = bool(sarcopenia_applied)
        self._step_count = 0

        obs_spec = reaf_env.timestep_spec().observation["policy"]
        obs_dim = int(obs_spec.shape[-1])
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=self._pipeline.action_low.copy(),
            high=self._pipeline.action_high.copy(),
            dtype=np.float32,
        )
        self._last_obs: np.ndarray = np.zeros((obs_dim,), dtype=np.float32)

    @property
    def env_id(self) -> str | None:
        """Registered environment identifier, if created via :func:`make_env`."""
        return self._env_id

    @property
    def reaf_env(self) -> ReafEnvironment:
        """Underlying ``reaf.simulation.torch.Environment``."""
        return self._reaf_env

    @property
    def pipeline(self) -> ExcitationPipelineProcessor:
        """Task-layer :class:`ExcitationPipelineProcessor`."""
        return self._pipeline

    @property
    def mj_model(self) -> Any:
        """Compiled CPU ``mujoco.MjModel`` owned by the simulator."""
        return self._reaf_env.device_layer.sim.mj_model

    @property
    def sim_dt(self) -> float:
        """Physics substep timestep in seconds."""
        return float(self.mj_model.opt.timestep)

    @property
    def dt(self) -> float:
        """Control timestep in seconds (``frame_skip * sim_dt``)."""
        return float(self._reaf_env.device_layer.control_timestep)

    @property
    def frame_skip(self) -> int:
        """Number of physics substeps per control step."""
        return int(round(self.dt / self.sim_dt))

    @property
    def max_episode_steps(self) -> int:
        """Episode horizon in control steps."""
        return self._max_episode_steps

    def set_motor_noise(self, motor_noise: MotorNoiseCfg | Mapping[str, float] | None) -> None:
        """Configure or update Gaussian motor noise on muscle excitations."""
        self._pipeline.set_motor_noise(motor_noise)

    def set_fatigue_reset_random(self, fatigue_reset_random: bool) -> None:
        """Enable or disable random initial fatigue sampling on reset."""
        self._pipeline.set_fatigue_reset_random(fatigue_reset_random)

    def apply_sarcopenia(self, force_scale: float = 0.5) -> None:
        """Scale peak isometric muscle forces once on the live simulator model."""
        if self._sarcopenia_applied:
            raise ValueError(
                "Sarcopenia is already applied to this environment's model "
                "(via a myoSarc* ID, SarcopeniaSpec, or SarcopeniaWrapper)."
            )
        sim = self._reaf_env.device_layer.sim
        mj_model = sim.mj_model
        apply_sarcopenia_to_model(mj_model, force_scale=force_scale)
        gain_t = torch.as_tensor(
            mj_model.actuator_gainprm,
            dtype=sim.model.actuator_gainprm.dtype,
            device=self._xpu_device,
        ).unsqueeze(0)
        bias_t = torch.as_tensor(
            mj_model.actuator_biasprm,
            dtype=sim.model.actuator_biasprm.dtype,
            device=self._xpu_device,
        ).unsqueeze(0)
        sim.model.actuator_gainprm[:] = gain_t
        sim.model.actuator_biasprm[:] = bias_t
        sim.set_const()
        sim.model.clear_defaults()
        sim.forward()
        self._sarcopenia_applied = True

    def _extract_obs(self, timestep: dm_env.TimeStep) -> np.ndarray:
        obs_t = timestep.observation["policy"][0]
        return np.asarray(obs_t.detach().cpu().numpy(), dtype=np.float32).copy()

    def _slice_obs_dict(self, flat_obs: np.ndarray) -> dict[str, np.ndarray]:
        if not self._obs_slices:
            return {"policy": flat_obs.copy()}
        return {name: flat_obs[slc].copy() for name, slc in self._obs_slices.items()}

    def get_obs_dict(self) -> dict[str, np.ndarray]:
        """Return structured per-term ``float32`` observations for the latest state."""
        return self._slice_obs_dict(self._last_obs)

    def _eval_solved(self) -> bool:
        metric_provider = getattr(self._reaf_env.task_layer, "_metric_provider", None)
        features = getattr(self._reaf_env, "_latest_features", None)
        if metric_provider is None or not features:
            return False
        for name, func, params in getattr(metric_provider, "_ordered_items", ()):
            if name == "solved":
                val = func(features, **params)
                return bool(float(val[0].item()) > 0.5)
        return False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            for idx, child in enumerate(self._leaf_resetters):
                if hasattr(child, "reseed"):
                    child.reseed(int(seed) + idx * 10_007)
            self._pipeline.reseed(int(seed) + 99_991)
            self.action_space.seed(int(seed))

        if options and "reset_override" in options:
            override = options["reset_override"]
            for child in self._leaf_resetters:
                if hasattr(child, "set_next_reset_override"):
                    child.set_next_reset_override(**override)

        timestep = self._reaf_env.reset()
        self._step_count = 0
        obs = self._extract_obs(timestep)
        self._last_obs = obs
        info: dict[str, Any] = {
            "obs_dict": self._slice_obs_dict(obs),
            "solved": self._eval_solved(),
        }
        return obs, info

    def step(
        self, action: np.ndarray | Sequence[float]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        act_np = np.asarray(action, dtype=np.float32).reshape(1, self.action_space.shape[0])
        act_t = torch.as_tensor(act_np, dtype=torch.float32, device=self._xpu_device)
        timestep = self._reaf_env.step({"actions": act_t})
        self._step_count += 1

        obs = self._extract_obs(timestep)
        self._last_obs = obs

        rew_struct = timestep.reward
        if isinstance(rew_struct, Mapping):
            reward = float(rew_struct["reward"][0].item())
            rwd_dict = {
                k: float(v[0].item())
                for k, v in rew_struct["terms"].items()
                if not k.startswith("weighted/")
            }
            rwd_dict["dense"] = reward
        else:
            reward = float(rew_struct[0].item())
            rwd_dict = {"dense": reward}

        is_last = int(timestep.step_type[0].item()) == int(dm_env.StepType.LAST)
        discount = float(timestep.discount[0].item())
        terminated = bool(is_last and discount == 0.0)
        truncated = bool((self._step_count >= self._max_episode_steps) and not terminated)

        info: dict[str, Any] = {
            "obs_dict": self._slice_obs_dict(obs),
            "rwd_dict": rwd_dict,
            "solved": self._eval_solved(),
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._reaf_env.close()


def _stage_host(env: gym.Env[Any, Any], wrapper_name: str) -> GymnasiumEnvAdapter:
    host = env.unwrapped
    if not isinstance(host, GymnasiumEnvAdapter) or not getattr(
        host, "supports_ctrl_stages", False
    ):
        raise TypeError(
            f"{wrapper_name} requires a GymnasiumEnvAdapter supporting muscle stages, "
            f"got {type(host).__name__}."
        )
    return host


class DictObservationWrapper(
    _ForwardPublicAttributes,
    RecordConstructorArgs,
    gym.ObservationWrapper[dict[str, np.ndarray], np.ndarray, np.ndarray],
):
    """Expose structured per-term observations as a ``gymnasium.spaces.Dict``."""

    def __init__(self, env: gym.Env[np.ndarray, np.ndarray]) -> None:
        RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        host = _stage_host(env, "DictObservationWrapper")
        sample = host.get_obs_dict()
        self.observation_space = gym.spaces.Dict(
            {
                k: gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=np.atleast_1d(v).shape,
                    dtype=np.float32,
                )
                for k, v in sample.items()
            }
        )

    @staticmethod
    def _coerce(obs_dict: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return {k: np.atleast_1d(np.asarray(v, dtype=np.float32)) for k, v in obs_dict.items()}

    def observation(self, observation: np.ndarray) -> dict[str, np.ndarray]:
        del observation
        return self._coerce(self.env.unwrapped.get_obs_dict())  # type: ignore[attr-defined]

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        _, info = self.env.reset(**kwargs)
        obs = (
            self._coerce(info["obs_dict"])
            if "obs_dict" in info
            else self.observation(np.empty(0, dtype=np.float32))
        )
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        _, reward, terminated, truncated, info = self.env.step(action)
        obs = (
            self._coerce(info["obs_dict"])
            if "obs_dict" in info
            else self.observation(np.empty(0, dtype=np.float32))
        )
        return obs, float(reward), bool(terminated), bool(truncated), info


class MotorNoiseWrapper(
    _ForwardPublicAttributes, RecordConstructorArgs, gym.Wrapper[Any, np.ndarray, Any, np.ndarray]
):
    """Install signal-dependent + constant Gaussian motor noise on muscle excitations."""

    def __init__(
        self,
        env: gym.Env[Any, np.ndarray],
        motor_noise: MotorNoiseCfg | Mapping[str, float] | None = None,
    ) -> None:
        RecordConstructorArgs.__init__(self, motor_noise=motor_noise)
        gym.Wrapper.__init__(self, env)
        self.motor_noise = MotorNoiseCfg.from_value(motor_noise)
        host = _stage_host(env, "MotorNoiseWrapper")
        host.pipeline.install_motor_noise(self.motor_noise)

    def set_motor_noise(self, motor_noise: MotorNoiseCfg | Mapping[str, float] | None) -> None:
        self.motor_noise = MotorNoiseCfg.from_value(motor_noise)
        self.env.unwrapped.set_motor_noise(self.motor_noise)  # type: ignore[attr-defined]


class FatigueWrapper(
    _ForwardPublicAttributes, RecordConstructorArgs, gym.Wrapper[Any, np.ndarray, Any, np.ndarray]
):
    """Install 3CC-r cumulative muscle fatigue on muscle excitations."""

    def __init__(
        self,
        env: gym.Env[Any, np.ndarray],
        fatigue_reset_vec: Sequence[float] | None = None,
        fatigue_reset_random: bool = False,
        muscles: Sequence[str | int] | None = None,
    ) -> None:
        RecordConstructorArgs.__init__(
            self,
            fatigue_reset_vec=fatigue_reset_vec,
            fatigue_reset_random=fatigue_reset_random,
            muscles=muscles,
        )
        gym.Wrapper.__init__(self, env)
        host = _stage_host(env, "FatigueWrapper")
        spec = FatigueSpec(
            fatigue_reset_vec=(
                tuple(float(x) for x in fatigue_reset_vec)
                if fatigue_reset_vec is not None
                else None
            ),
            fatigue_reset_random=bool(fatigue_reset_random),
            muscles=tuple(muscles) if muscles is not None else None,
        )
        host.pipeline.install_fatigue(spec)

    def set_fatigue_reset_random(self, fatigue_reset_random: bool) -> None:
        self.env.unwrapped.set_fatigue_reset_random(fatigue_reset_random)  # type: ignore[attr-defined]


class ReafferentationWrapper(
    _ForwardPublicAttributes, RecordConstructorArgs, gym.Wrapper[Any, np.ndarray, Any, np.ndarray]
):
    """Install EIP -> EPL tendon-transfer (reafferentation) excitation rerouting."""

    def __init__(
        self,
        env: gym.Env[Any, np.ndarray],
        source_actuator: str = "EIP",
        target_actuator: str = "EPL",
    ) -> None:
        RecordConstructorArgs.__init__(
            self,
            source_actuator=source_actuator,
            target_actuator=target_actuator,
        )
        gym.Wrapper.__init__(self, env)
        host = _stage_host(env, "ReafferentationWrapper")
        host.pipeline.install_reafferentation(
            ReafferentationSpec(
                source_actuator=source_actuator,
                target_actuator=target_actuator,
            )
        )


class SarcopeniaWrapper(
    _ForwardPublicAttributes, RecordConstructorArgs, gym.Wrapper[Any, np.ndarray, Any, np.ndarray]
):
    """Apply peak isometric muscle force reduction (sarcopenia) once to the model."""

    def __init__(self, env: gym.Env[Any, np.ndarray], force_scale: float = 0.5) -> None:
        RecordConstructorArgs.__init__(self, force_scale=force_scale)
        gym.Wrapper.__init__(self, env)
        SarcopeniaSpec(force_scale=force_scale)
        host = _stage_host(env, "SarcopeniaWrapper")
        host.apply_sarcopenia(force_scale=force_scale)
        self.force_scale = force_scale


class ExcitationStageWrapper(
    _ForwardPublicAttributes, RecordConstructorArgs, gym.Wrapper[Any, np.ndarray, Any, np.ndarray]
):
    """Install a portable custom :class:`ExcitationStage` into the excitation pipeline."""

    def __init__(
        self,
        env: gym.Env[Any, np.ndarray],
        make_stage: Callable[[], ExcitationStage],
    ) -> None:
        RecordConstructorArgs.__init__(self, make_stage=make_stage)
        gym.Wrapper.__init__(self, env)
        stage = make_stage()
        if stage.name in STAGE_ORDER:
            raise ValueError(
                f"{stage.name!r} is a built-in stage; use its dedicated wrapper instead."
            )
        host = _stage_host(env, "ExcitationStageWrapper")
        host.pipeline.install_custom_stage(stage)
        self.stage = stage
        self._make_stage = make_stage


def make_env(
    env_id: str,
    *,
    impl: SimBackend = "cpp",
    device: str | None = None,
    seed: int = 0,
    features: Sequence[FeatureSpec] = (),
    config: EnvConfig | None = None,
) -> GymnasiumEnvAdapter:
    """Build a single-world MyoSuite :class:`GymnasiumEnvAdapter` backed by ``reaf-sim``.

    Args:
        env_id: Registered ``myosuite-reaf`` task ID (e.g. ``"myoElbowPose1D6MRandom-v0"``).
        impl: ``"cpp"`` (CPU MuJoCo 4) or ``"warp"`` (GPU MuJoCo Warp 4, ``num_envs=1``).
        device: Optional PyTorch/Warp device override (e.g. ``"cuda:0"`` or ``"cpu"``).
        seed: Initial RNG seed.
        features: Optional physiological / custom stage specs appended to ``config``.
        config: Optional explicit :class:`EnvConfig` (overrides *impl*, *device*, *seed*).

    Returns:
        A ready-to-use :class:`GymnasiumEnvAdapter`.
    """
    entry = get_registered_task(env_id)
    if config is not None:
        eff_cfg = config.with_features(features) if features else config
    else:
        eff_cfg = EnvConfig(
            impl=impl,
            num_envs=1,
            device=device,
            seed=seed,
            features=tuple(features),
        )
    has_sarc = any(
        isinstance(f, SarcopeniaSpec) for f in (*entry.default_features, *eff_cfg.features)
    )
    reaf_env = make_reaf_env(env_id, config=eff_cfg)
    return GymnasiumEnvAdapter(
        reaf_env,
        env_id=env_id,
        max_episode_steps=entry.max_episode_steps,
        sarcopenia_applied=has_sarc,
    )
