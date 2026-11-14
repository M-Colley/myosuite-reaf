# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-site spatial reaching tasks on ``reaf-sim`` (``ReachEnvV0``).

Covers:
  - ``motorFingerReachFixed-v0``, ``motorFingerReachRandom-v0``
  - ``myoFingerReachFixed-v0``, ``myoFingerReachRandom-v0``
  - ``myoHandReachFixed-v0``, ``myoHandReachRandom-v0``
  - ``myoArmReachFixed-v0``, ``myoArmReachRandom-v0``
  - Plus ``myoSarc*``, ``myoFati*``, and ``myoReaf*`` physiological condition variants.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import torch
from dm_env import specs
from reaf.simulation.torch import (
    EnvironmentReset,
    FeatureProducer,
    TerminationResult,
    flat_scene,
)
from reaf.simulation.torch._src.builders import config as builders_config

from myosuite_reaf.assets import (
    apply_sarcopenia_to_spec,
    compute_init_qpos,
    reachable_target_points,
)
from myosuite_reaf.config import EnvConfig, SarcopeniaSpec
from myosuite_reaf.devices import (
    KinematicsDevice,
    MuscleActuatorDevice,
    SyncForwardDevice,
)
from myosuite_reaf.processors import ExcitationPipelineProcessor


class ReachFeatureProducer(FeatureProducer):
    """Computes flattened end-effector tip positions, target positions, and reach error.

    Because ``ReachResetter`` writes sampled target positions into ``sim.model.site_pos``
    and ``Environment.reset()`` propagates them via ``sim.forward()`` into
    ``data.site_xpos``, ``TaskLayer`` reads both ``tip_pos`` and ``target_pos`` directly
    from ``kinematics/site_xpos`` without any privileged ``MjData`` side-channel.
    """

    def __init__(
        self,
        name: str = "reach_features",
        *,
        tip_sids: Sequence[int],
        target_sids: Sequence[int],
        ctrl_dt: float,
        far_th: float = 0.35,
        nworld: int = 1,
        xpu_device: str = "cpu",
    ) -> None:
        self._name = name
        self._nworld = nworld
        self._device = torch.device(xpu_device)
        self._tip_sids = torch.tensor(list(tip_sids), dtype=torch.long, device=self._device)
        self._target_sids = torch.tensor(list(target_sids), dtype=torch.long, device=self._device)
        self._n_sites = len(tip_sids)
        self._ctrl_dt = float(ctrl_dt)
        self._near_th = 0.0125 * self._n_sites
        self._far_limit = float(far_th) * self._n_sites

    @property
    def name(self) -> str:
        return self._name

    def required_features_keys(self) -> set[str]:
        return {"time", "kinematics/site_xpos"}

    def produced_features_spec(self) -> Mapping[str, specs.Array]:
        dim = 3 * self._n_sites
        return {
            "reach/tip_pos": specs.Array(shape=(dim,), dtype=np.float32),
            "reach/target_pos": specs.Array(shape=(dim,), dtype=np.float32),
            "reach/reach_err": specs.Array(shape=(dim,), dtype=np.float32),
            "reach/dist": specs.Array(shape=(), dtype=np.float32),
            "reach/far_exceeded": specs.Array(shape=(), dtype=np.float32),
        }

    def produce_features(
        self, required_features: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        site_xpos = required_features["kinematics/site_xpos"].float()
        time_s = required_features["time"].float()
        nworld = site_xpos.shape[0]
        tip_pos = site_xpos[:, self._tip_sids].reshape(nworld, 3 * self._n_sites)
        target_pos = site_xpos[:, self._target_sids].reshape(nworld, 3 * self._n_sites)
        reach_err = target_pos - tip_pos
        dist = torch.linalg.norm(reach_err, dim=-1)
        penalty_active = torch.round(time_s / self._ctrl_dt) >= 2.0
        far_exceeded = (penalty_active & (dist > self._far_limit)).float()
        return {
            "reach/tip_pos": tip_pos,
            "reach/target_pos": target_pos,
            "reach/reach_err": reach_err,
            "reach/dist": dist,
            "reach/far_exceeded": far_exceeded,
        }


class ReachResetter(EnvironmentReset):
    """Samples reachable or box-uniform 3-D site targets and resets joint coordinates."""

    def __init__(
        self,
        *,
        init_qpos: np.ndarray,
        init_qvel: np.ndarray,
        target_sids: Sequence[int],
        box_low: np.ndarray,
        box_high: np.ndarray,
        workspace_points: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        self._init_qpos_np = np.asarray(init_qpos, dtype=np.float32)
        self._init_qvel_np = np.asarray(init_qvel, dtype=np.float32)
        self._target_sids = list(target_sids)
        self._box_low_np = np.asarray(box_low, dtype=np.float32)
        self._box_high_np = np.asarray(box_high, dtype=np.float32)
        self._workspace_np = (
            np.asarray(workspace_points, dtype=np.float32) if workspace_points is not None else None
        )
        self._seed = int(seed)
        self._rng: torch.Generator | None = None
        self._workspace_tensor: torch.Tensor | None = None
        self._override: dict[str, Any] | None = None

    def reseed(self, seed: int) -> None:
        self._seed = int(seed)
        if self._rng is not None:
            self._rng.manual_seed(self._seed)

    def set_next_reset_override(
        self,
        *,
        qpos: np.ndarray | torch.Tensor | None = None,
        qvel: np.ndarray | torch.Tensor | None = None,
        target_pos: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        """Override sampled targets/poses on the next reset (used by parity tests)."""
        self._override = {"qpos": qpos, "qvel": qvel, "target_pos": target_pos}

    def _generator(self, device: torch.device) -> torch.Generator:
        if self._rng is None or self._rng.device != device:
            self._rng = torch.Generator(device=device)
            self._rng.manual_seed(self._seed)
        return self._rng

    def on_reset(self, config: Any, env_mask: torch.Tensor) -> None:
        del config
        sim = self._env.device_layer.sim
        data = sim.data
        model = sim.model
        device = data.qpos.device
        nworld = data.qpos.shape[0]
        gen = self._generator(device)
        override = self._override
        self._override = None

        n_sites = len(self._target_sids)
        if override is not None and override["target_pos"] is not None:
            sampled_targets = (
                torch.as_tensor(override["target_pos"], dtype=model.site_pos.dtype, device=device)
                .reshape(-1, n_sites, 3)
                .broadcast_to((nworld, n_sites, 3))
            )
        elif self._workspace_np is not None:
            if self._workspace_tensor is None or self._workspace_tensor.device != device:
                self._workspace_tensor = torch.as_tensor(
                    self._workspace_np, dtype=model.site_pos.dtype, device=device
                )
            indices = torch.randint(
                0,
                self._workspace_tensor.shape[0],
                (nworld,),
                generator=gen,
                device=device,
            )
            sampled_targets = self._workspace_tensor[indices]
        else:
            lo = torch.as_tensor(self._box_low_np, dtype=torch.float32, device=device)
            hi = torch.as_tensor(self._box_high_np, dtype=torch.float32, device=device)
            u = torch.rand((nworld, n_sites, 3), generator=gen, device=device, dtype=torch.float32)
            sampled_targets = (lo + (hi - lo) * u).to(dtype=model.site_pos.dtype)

        for idx, tsid in enumerate(self._target_sids):
            model.site_pos[:, tsid] = torch.where(
                env_mask.unsqueeze(1), sampled_targets[:, idx], model.site_pos[:, tsid]
            )

        if override is not None and override["qpos"] is not None:
            new_qpos = torch.as_tensor(
                override["qpos"], dtype=data.qpos.dtype, device=device
            ).broadcast_to(data.qpos.shape)
        else:
            new_qpos = (
                torch.as_tensor(self._init_qpos_np, dtype=data.qpos.dtype, device=device)
                .unsqueeze(0)
                .expand(nworld, -1)
            )

        if override is not None and override["qvel"] is not None:
            new_qvel = torch.as_tensor(
                override["qvel"], dtype=data.qvel.dtype, device=device
            ).broadcast_to(data.qvel.shape)
        else:
            new_qvel = (
                torch.as_tensor(self._init_qvel_np, dtype=data.qvel.dtype, device=device)
                .unsqueeze(0)
                .expand(nworld, -1)
            )

        data.qpos[:] = torch.where(env_mask.unsqueeze(1), new_qpos, data.qpos)
        data.qvel[:] = torch.where(env_mask.unsqueeze(1), new_qvel, data.qvel)
        if data.act.numel() > 0:
            data.act[:] = torch.where(env_mask.unsqueeze(1), torch.zeros_like(data.act), data.act)


def _obs_qpos(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return features["kinematics/qpos"].float()


def _obs_qvel_scaled(features: Mapping[str, torch.Tensor], *, ctrl_dt: float) -> torch.Tensor:
    return features["kinematics/qvel"].float() * ctrl_dt


def _obs_tip_pos(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return features["reach/tip_pos"].float()


def _obs_reach_err(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return features["reach/reach_err"].float()


def _obs_act(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return features["actuators/act"].float()


def _rwd_reach(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return -features["reach/dist"].float()


def _rwd_bonus(features: Mapping[str, torch.Tensor], *, near_th: float) -> torch.Tensor:
    dist = features["reach/dist"].float()
    return (dist < 2.0 * near_th).float() + (dist < near_th).float()


def _rwd_penalty(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return -features["reach/far_exceeded"].float()


def _rwd_act_reg(features: Mapping[str, torch.Tensor], *, na: int) -> torch.Tensor:
    if na == 0 or "actuators/act" not in features:
        return torch.zeros_like(features["reach/dist"].float())
    act = features["actuators/act"].float()
    return -(torch.linalg.norm(act, dim=-1) / float(na))


def _metric_solved(features: Mapping[str, torch.Tensor], *, near_th: float) -> torch.Tensor:
    return (features["reach/dist"].float() < near_th).float()


def _term_far(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    exceeded = features["reach/far_exceeded"].float() > 0.5
    return torch.where(
        exceeded,
        int(TerminationResult.TERMINATE),
        int(TerminationResult.DO_NOT_TERMINATE),
    )


@dataclass(frozen=True)
class ReachTaskSpec:
    """Declarative specification for a MyoSuite spatial reaching environment."""

    spec_builder: Callable[[], mujoco.MjSpec]
    target_reach_range: Mapping[str, tuple[tuple[float, float, float], tuple[float, float, float]]]
    target_sampling: str = "box"
    far_th: float = 0.35
    max_episode_steps: int = 100
    frame_skip: int = 10
    normalize_act: bool = True
    actuator_suffix: str = ""
    weighted_reward_keys: Mapping[str, float] = field(
        default_factory=lambda: {
            "reach": 1.0,
            "bonus": 4.0,
            "penalty": 50.0,
        }
    )


def build_reach_env_config(
    task: ReachTaskSpec,
    env_cfg: EnvConfig | None = None,
) -> flat_scene.Environment:
    """Build a ``reaf.simulation.torch.flat_scene.Environment`` config for a reach task."""
    cfg = env_cfg or EnvConfig()
    mj_spec = task.spec_builder()
    for feat in cfg.features:
        if isinstance(feat, SarcopeniaSpec):
            apply_sarcopenia_to_spec(mj_spec, force_scale=feat.force_scale)

    sim_dt = float(mj_spec.option.timestep)
    ctrl_dt = sim_dt * task.frame_skip
    mj_model = mj_spec.compile()

    tip_sids = [int(mj_model.site(name).id) for name in task.target_reach_range]
    target_sids = [int(mj_model.site(f"{name}_target").id) for name in task.target_reach_range]
    spans = list(task.target_reach_range.values())
    box_low = np.asarray([s[0] for s in spans], dtype=np.float32)
    box_high = np.asarray([s[1] for s in spans], dtype=np.float32)

    workspace_pts: np.ndarray | None = None
    if task.target_sampling == "workspace":
        workspace_pts = reachable_target_points(mj_model, tip_sids, box_low, box_high)
    elif task.target_sampling != "box":
        raise ValueError(
            f"target_sampling must be 'box' or 'workspace', got {task.target_sampling!r}."
        )

    init_qpos = compute_init_qpos(mj_model, normalize_act=task.normalize_act)
    init_qvel = np.zeros(mj_model.nv, dtype=np.float32)
    n_sites = len(tip_sids)
    near_th = 0.0125 * n_sites

    obs_terms: dict[str, flat_scene.Obs] = {
        "qpos": flat_scene.Obs(_obs_qpos).configure(output_shape=(int(mj_model.nq),)),
        "qvel": flat_scene.Obs(_obs_qvel_scaled, ctrl_dt=ctrl_dt).configure(
            output_shape=(int(mj_model.nv),)
        ),
        "tip_pos": flat_scene.Obs(_obs_tip_pos).configure(output_shape=(3 * n_sites,)),
        "reach_err": flat_scene.Obs(_obs_reach_err).configure(output_shape=(3 * n_sites,)),
    }
    if mj_model.na > 0:
        obs_terms["act"] = flat_scene.Obs(_obs_act).configure(output_shape=(int(mj_model.na),))

    # Pre-divide weights by ctrl_dt to counteract reaf-sim's unconditional * dt scaling.
    reward_defs: dict[str, flat_scene.Reward] = {}
    for key, wt in task.weighted_reward_keys.items():
        if wt == 0.0:
            continue
        eff_wt = float(wt) / ctrl_dt
        if key == "reach":
            reward_defs["reach"] = flat_scene.Reward(_rwd_reach, weight=eff_wt)
        elif key == "bonus":
            reward_defs["bonus"] = flat_scene.Reward(_rwd_bonus, weight=eff_wt, near_th=near_th)
        elif key == "penalty":
            reward_defs["penalty"] = flat_scene.Reward(_rwd_penalty, weight=eff_wt)
        elif key == "act_reg":
            reward_defs["act_reg"] = flat_scene.Reward(
                _rwd_act_reg, weight=eff_wt, na=int(mj_model.na)
            )

    devices: dict[str, flat_scene.Component[Any]] = {
        "sync_forward": flat_scene.Component(SyncForwardDevice, enabled=cfg.sync_forward),
        "actuators": flat_scene.Component(
            MuscleActuatorDevice,
            measurements=("ctrl", "act") if mj_model.na > 0 else ("ctrl",),
        ),
        "kinematics": flat_scene.Component(
            KinematicsDevice,
            measurements=("qpos", "qvel", "site_xpos"),
        ),
    }

    task_layer = flat_scene.TaskLayer(
        command_processors={
            "excitation_pipeline": flat_scene.Component(
                ExcitationPipelineProcessor,
                mj_model=mj_model,
                control_timestep=ctrl_dt,
                normalize_act=task.normalize_act,
                muscle_sigmoid=True,
                features=cfg.features,
                actuator_suffix=task.actuator_suffix,
                seed=cfg.seed,
            )
        },
        feature_producers={
            "reach_features": flat_scene.Component(
                ReachFeatureProducer,
                tip_sids=tip_sids,
                target_sids=target_sids,
                ctrl_dt=ctrl_dt,
                far_th=task.far_th,
            )
        },
        rewards=reward_defs,
        metrics={
            "solved": flat_scene.Metric(_metric_solved, near_th=near_th),
            "dist": flat_scene.Metric("reach/dist"),
        },
        terminations={
            "far": flat_scene.Term(_term_far),
        },
    )

    resetter = flat_scene.Component(
        ReachResetter,
        init_qpos=init_qpos,
        init_qvel=init_qvel,
        target_sids=target_sids,
        box_low=box_low,
        box_high=box_high,
        workspace_points=workspace_pts,
        seed=cfg.seed,
    )

    sim_cfg = builders_config.Sim(
        impl=cfg.impl,
        xpu_device=cfg.resolved_device(),
        nworld=cfg.num_envs,
        simulation_timestep=sim_dt,
        control_timestep=ctrl_dt,
        nconmax=cfg.nconmax,
        njmax=cfg.njmax,
    )

    return flat_scene.Environment(
        base_scene=mj_spec,
        device_layer=flat_scene.DeviceLayer(devices=devices),
        task_layer=task_layer,
        observation_adapter={
            "policy": flat_scene.ObsGroup(terms=obs_terms),
        },
        resetters={"reach_reset": resetter},
        sim=sim_cfg,
        episode_length_secs=float(task.max_episode_steps * ctrl_dt),
    )
