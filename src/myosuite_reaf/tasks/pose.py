# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Musculoskeletal joint-pose tracking tasks on ``reaf-sim`` (``PoseEnvV0`` / ``TorsoEnvV0``).

Covers:
  - ``myoElbowPose1D6MFixed-v0``, ``myoElbowPose1D6MRandom-v0``
  - ``myoElbowPose1D6MExoFixed-v0``, ``myoElbowPose1D6MExoRandom-v0`` (hybrid muscle + exo + carry_weight DR)
  - ``myoFingerPoseFixed-v0``, ``myoFingerPoseRandom-v0``
  - ``motorFingerPoseFixed-v0``, ``motorFingerPoseRandom-v0``
  - ``myoHandPoseFixed-v0``, ``myoHandPoseRandom-v0``
  - ``myoTorsoPoseFixed-v0``, ``myoTorsoExoPoseFixed-v0``
  - Plus ``myoSarc*``, ``myoFati*``, and ``myoReaf*`` physiological condition variants.
"""

from __future__ import annotations

import math
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

from myosuite_reaf.assets import apply_sarcopenia_to_spec, compute_init_qpos
from myosuite_reaf.config import EnvConfig, SarcopeniaSpec
from myosuite_reaf.devices import (
    KinematicsDevice,
    MuscleActuatorDevice,
    SyncForwardDevice,
)
from myosuite_reaf.processors import ExcitationPipelineProcessor

_POSE_BONUS_FAR_MULTIPLIER: float = 1.5
_POSE_FAR_THRESHOLD: float = 2.0 * math.pi


@dataclass
class PoseTargetState:
    """Shared batched target buffer linking ``PoseResetter`` and ``PoseFeatureProducer``."""

    n_pose_jnts: int
    default_target: np.ndarray
    _target_angles: torch.Tensor | None = field(default=None, init=False, repr=False)

    def ensure(self, nworld: int, device: torch.device) -> torch.Tensor:
        if (
            self._target_angles is None
            or self._target_angles.shape[0] != nworld
            or self._target_angles.device != device
        ):
            base = torch.as_tensor(self.default_target, dtype=torch.float32, device=device)
            self._target_angles = base.unsqueeze(0).expand(nworld, -1).clone()
        return self._target_angles

    def update(self, env_mask: torch.Tensor, values: torch.Tensor) -> None:
        buf = self.ensure(env_mask.shape[0], env_mask.device)
        buf[:] = torch.where(env_mask.unsqueeze(1), values.to(buf.dtype), buf)


class PoseFeatureProducer(FeatureProducer):
    """Exposes current target joint angles and pose error inside ``TaskLayer``."""

    def __init__(
        self,
        name: str = "pose_features",
        *,
        target_state: PoseTargetState,
        nworld: int = 1,
        xpu_device: str = "cpu",
    ) -> None:
        self._name = name
        self._target_state = target_state
        self._nworld = nworld
        self._device = torch.device(xpu_device)

    @property
    def name(self) -> str:
        return self._name

    def required_features_keys(self) -> set[str]:
        return {"kinematics/qpos"}

    def produced_features_spec(self) -> Mapping[str, specs.Array]:
        n = self._target_state.n_pose_jnts
        return {
            "pose/target_angles": specs.Array(shape=(n,), dtype=np.float32),
            "pose/pose_err": specs.Array(shape=(n,), dtype=np.float32),
            "pose/dist": specs.Array(shape=(), dtype=np.float32),
        }

    def produce_features(
        self, required_features: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        qpos = required_features["kinematics/qpos"].float()
        n = self._target_state.n_pose_jnts
        targets = self._target_state.ensure(qpos.shape[0], qpos.device)
        err = targets - qpos[:, :n]
        dist = torch.linalg.norm(err, dim=-1)
        return {
            "pose/target_angles": targets,
            "pose/pose_err": err,
            "pose/dist": dist,
        }


class PoseResetter(EnvironmentReset):
    """Initializes joint poses, episode targets, visualization sites, and carry-weight DR."""

    def __init__(
        self,
        *,
        mj_model: mujoco.MjModel,
        target_state: PoseTargetState,
        init_qpos: np.ndarray,
        init_qvel: np.ndarray,
        reset_type: str = "init",
        target_type: str = "generate",
        target_jnt_ids: Sequence[int] | None = None,
        target_jnt_range: np.ndarray | None = None,
        fixed_target_angles: np.ndarray | None = None,
        tip_sids: Sequence[int] = (),
        target_sids: Sequence[int] = (),
        weight_body_id: int | None = None,
        weight_geom_id: int | None = None,
        weight_range: tuple[float, float] | None = None,
        seed: int = 0,
    ) -> None:
        self._mj_model = mj_model
        self._fk_data = mujoco.MjData(mj_model)
        self._target_state = target_state
        self._init_qpos_np = np.asarray(init_qpos, dtype=np.float32)
        self._init_qvel_np = np.asarray(init_qvel, dtype=np.float32)
        self._jnt_low_np = np.asarray(mj_model.jnt_range[:, 0], dtype=np.float32)
        self._jnt_high_np = np.asarray(mj_model.jnt_range[:, 1], dtype=np.float32)
        self._jnt_qpos_adr_np = np.asarray(mj_model.jnt_qposadr, dtype=np.int64)
        self._reset_type = reset_type
        self._target_type = target_type
        self._target_jnt_ids = list(target_jnt_ids) if target_jnt_ids is not None else None
        self._target_jnt_range_np = (
            np.asarray(target_jnt_range, dtype=np.float32) if target_jnt_range is not None else None
        )
        self._fixed_target_np = (
            np.asarray(fixed_target_angles, dtype=np.float32)
            if fixed_target_angles is not None
            else np.asarray(target_state.default_target, dtype=np.float32)
        )
        self._tip_sids = list(tip_sids)
        self._target_sids = list(target_sids)
        self._weight_body_id = weight_body_id
        self._weight_geom_id = weight_geom_id
        self._weight_range = weight_range
        self._seed = int(seed)
        self._rng: torch.Generator | None = None
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
        target_angles: np.ndarray | torch.Tensor | None = None,
        carry_weight: float | np.ndarray | torch.Tensor | None = None,
    ) -> None:
        """Override sampled values on the next ``on_reset`` call (used by parity tests)."""
        self._override = {
            "qpos": qpos,
            "qvel": qvel,
            "target_angles": target_angles,
            "carry_weight": carry_weight,
        }

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
        qpos_dtype = data.qpos.dtype
        qvel_dtype = data.qvel.dtype
        gen = self._generator(device)
        override = self._override
        self._override = None

        if self._weight_body_id is not None and self._weight_range is not None:
            lo_w, hi_w = self._weight_range
            if override is not None and override["carry_weight"] is not None:
                weights = torch.as_tensor(
                    override["carry_weight"], dtype=model.body_mass.dtype, device=device
                ).broadcast_to((nworld,))
            else:
                u = torch.rand((nworld,), generator=gen, device=device, dtype=torch.float32)
                weights = (lo_w + (hi_w - lo_w) * u).to(dtype=model.body_mass.dtype)
            bid = self._weight_body_id
            model.body_mass[:, bid] = torch.where(env_mask, weights, model.body_mass[:, bid])
            if self._weight_geom_id is not None:
                gid = self._weight_geom_id
                radii = (0.01 + 2.5 * weights / 100.0).to(dtype=model.geom_size.dtype)
                model.geom_size[:, gid, 0] = torch.where(
                    env_mask, radii, model.geom_size[:, gid, 0]
                )

        n_pose = self._target_state.n_pose_jnts
        if override is not None and override["target_angles"] is not None:
            tgt = torch.as_tensor(
                override["target_angles"], dtype=torch.float32, device=device
            ).broadcast_to((nworld, n_pose))
        elif self._target_type == "generate" and self._target_jnt_range_np is not None:
            lo_t = torch.as_tensor(
                self._target_jnt_range_np[:, 0], dtype=torch.float32, device=device
            )
            hi_t = torch.as_tensor(
                self._target_jnt_range_np[:, 1], dtype=torch.float32, device=device
            )
            u_t = torch.rand(
                (nworld, lo_t.numel()), generator=gen, device=device, dtype=torch.float32
            )
            tgt = lo_t + (hi_t - lo_t) * u_t
        else:
            tgt = (
                torch.as_tensor(self._fixed_target_np, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .expand(nworld, -1)
            )
        self._target_state.update(env_mask, tgt)

        if self._tip_sids and self._target_sids and nworld <= 64:
            tgt_cpu = self._target_state.ensure(nworld, device).detach().cpu().numpy()
            mask_cpu = env_mask.detach().cpu().numpy()
            site_updates = np.zeros((nworld, len(self._target_sids), 3), dtype=np.float32)
            for w in np.flatnonzero(mask_cpu):
                self._fk_data.qpos[:] = self._init_qpos_np
                self._fk_data.qpos[:n_pose] = tgt_cpu[w]
                mujoco.mj_kinematics(self._mj_model, self._fk_data)
                site_updates[w] = self._fk_data.site_xpos[self._tip_sids]
            site_tensor = torch.as_tensor(site_updates, dtype=model.site_pos.dtype, device=device)
            for idx, tsid in enumerate(self._target_sids):
                model.site_pos[:, tsid] = torch.where(
                    env_mask.unsqueeze(1), site_tensor[:, idx], model.site_pos[:, tsid]
                )

        if override is not None and override["qpos"] is not None:
            new_qpos = torch.as_tensor(
                override["qpos"], dtype=qpos_dtype, device=device
            ).broadcast_to(data.qpos.shape)
        elif self._reset_type == "random":
            base_qpos = (
                torch.as_tensor(self._init_qpos_np, dtype=qpos_dtype, device=device)
                .unsqueeze(0)
                .expand(nworld, -1)
                .clone()
            )
            lo_j = torch.as_tensor(self._jnt_low_np, dtype=torch.float32, device=device)
            hi_j = torch.as_tensor(self._jnt_high_np, dtype=torch.float32, device=device)
            u_j = torch.rand(
                (nworld, lo_j.numel()), generator=gen, device=device, dtype=torch.float32
            )
            sampled = (lo_j + (hi_j - lo_j) * u_j).to(dtype=qpos_dtype)
            if sampled.shape[1] == base_qpos.shape[1]:
                new_qpos = sampled
            else:
                qadr = torch.as_tensor(self._jnt_qpos_adr_np, dtype=torch.long, device=device)
                base_qpos[:, qadr] = sampled
                new_qpos = base_qpos
        else:
            new_qpos = (
                torch.as_tensor(self._init_qpos_np, dtype=qpos_dtype, device=device)
                .unsqueeze(0)
                .expand(nworld, -1)
            )

        if override is not None and override["qvel"] is not None:
            new_qvel = torch.as_tensor(
                override["qvel"], dtype=qvel_dtype, device=device
            ).broadcast_to(data.qvel.shape)
        else:
            new_qvel = (
                torch.as_tensor(self._init_qvel_np, dtype=qvel_dtype, device=device)
                .unsqueeze(0)
                .expand(nworld, -1)
            )

        data.qpos[:] = torch.where(env_mask.unsqueeze(1), new_qpos, data.qpos)
        data.qvel[:] = torch.where(env_mask.unsqueeze(1), new_qvel, data.qvel)
        if data.act.numel() > 0:
            zeros_act = torch.zeros_like(data.act)
            data.act[:] = torch.where(env_mask.unsqueeze(1), zeros_act, data.act)


def _obs_qpos(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return features["kinematics/qpos"].float()


def _obs_qvel_scaled(features: Mapping[str, torch.Tensor], *, ctrl_dt: float) -> torch.Tensor:
    return features["kinematics/qvel"].float() * ctrl_dt


def _obs_pose_err(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return features["pose/pose_err"].float()


def _obs_act(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return features["actuators/act"].float()


def _rwd_pose(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return -features["pose/dist"].float()


def _rwd_bonus(features: Mapping[str, torch.Tensor], *, pose_thd: float) -> torch.Tensor:
    dist = features["pose/dist"].float()
    return (dist < pose_thd).float() + (dist < _POSE_BONUS_FAR_MULTIPLIER * pose_thd).float()


def _rwd_penalty(features: Mapping[str, torch.Tensor], *, far_thd: float) -> torch.Tensor:
    dist = features["pose/dist"].float()
    return -(dist > far_thd).float()


def _rwd_act_reg(features: Mapping[str, torch.Tensor], *, na: int) -> torch.Tensor:
    if na == 0 or "actuators/act" not in features:
        return torch.zeros_like(features["pose/dist"].float())
    act = features["actuators/act"].float()
    return -(torch.linalg.norm(act, dim=-1) / float(na))


def _metric_solved(features: Mapping[str, torch.Tensor], *, pose_thd: float) -> torch.Tensor:
    return (features["pose/dist"].float() < pose_thd).float()


def _term_far(features: Mapping[str, torch.Tensor], *, far_thd: float) -> torch.Tensor:
    dist = features["pose/dist"].float()
    return torch.where(
        dist > far_thd,
        int(TerminationResult.TERMINATE),
        int(TerminationResult.DO_NOT_TERMINATE),
    )


@dataclass(frozen=True)
class PoseTaskSpec:
    """Declarative specification for a MyoSuite joint-pose tracking environment."""

    spec_builder: Callable[[], mujoco.MjSpec]
    max_episode_steps: int = 100
    frame_skip: int = 10
    pose_thd: float = 0.35
    far_thd: float = _POSE_FAR_THRESHOLD
    reset_type: str = "init"
    target_type: str = "generate"
    target_jnt_range: Mapping[str, tuple[float, float]] | None = None
    target_jnt_value: tuple[float, ...] | None = None
    n_pose_jnts: int | None = None
    viz_site_targets: tuple[str, ...] = ()
    normalize_act: bool = True
    actuator_suffix: str = ""
    weight_bodyname: str | None = None
    weight_range: tuple[float, float] | None = None
    weighted_reward_keys: Mapping[str, float] = field(
        default_factory=lambda: {
            "pose": 1.0,
            "bonus": 4.0,
            "act_reg": 1.0,
            "penalty": 50.0,
        }
    )


def build_pose_env_config(
    task: PoseTaskSpec,
    env_cfg: EnvConfig | None = None,
) -> flat_scene.Environment:
    """Build a ``reaf.simulation.torch.flat_scene.Environment`` config for a pose task."""
    cfg = env_cfg or EnvConfig()
    mj_spec = task.spec_builder()
    for feat in cfg.features:
        if isinstance(feat, SarcopeniaSpec):
            apply_sarcopenia_to_spec(mj_spec, force_scale=feat.force_scale)

    sim_dt = float(mj_spec.option.timestep)
    ctrl_dt = sim_dt * task.frame_skip
    mj_model = mj_spec.compile()

    if task.target_jnt_range is not None:
        jnt_ids = [int(mj_model.joint(name).id) for name in task.target_jnt_range]
        jnt_ranges = np.asarray(list(task.target_jnt_range.values()), dtype=np.float32)
        default_target = np.mean(jnt_ranges, axis=1).astype(np.float32)
        n_pose = task.n_pose_jnts or len(jnt_ids)
    elif task.target_jnt_value is not None:
        jnt_ids = None
        jnt_ranges = None
        default_target = np.asarray(task.target_jnt_value, dtype=np.float32)
        n_pose = task.n_pose_jnts or int(default_target.size)
    else:
        raise ValueError("PoseTaskSpec requires target_jnt_range or target_jnt_value.")

    tip_sids = [int(mj_model.site(s).id) for s in task.viz_site_targets]
    target_sids = [int(mj_model.site(f"{s}_target").id) for s in task.viz_site_targets]

    weight_bid: int | None = None
    weight_gid: int | None = None
    if task.weight_bodyname is not None:
        weight_bid = int(mj_model.body(task.weight_bodyname).id)
        weight_gid = int(mj_model.body_geomadr[weight_bid])

    init_qpos = compute_init_qpos(mj_model, normalize_act=task.normalize_act)
    init_qvel = np.zeros(mj_model.nv, dtype=np.float32)
    target_state = PoseTargetState(n_pose_jnts=n_pose, default_target=default_target)

    obs_terms: dict[str, flat_scene.Obs] = {
        "qpos": flat_scene.Obs(_obs_qpos).configure(output_shape=(int(mj_model.nq),)),
        "qvel": flat_scene.Obs(_obs_qvel_scaled, ctrl_dt=ctrl_dt).configure(
            output_shape=(int(mj_model.nv),)
        ),
        "pose_err": flat_scene.Obs(_obs_pose_err).configure(output_shape=(n_pose,)),
    }
    if mj_model.na > 0:
        obs_terms["act"] = flat_scene.Obs(_obs_act).configure(output_shape=(int(mj_model.na),))

    # reaf-sim's FunctionalRewardProvider multiplies (weights @ terms) by ctrl_dt
    # unconditionally (no scale_rewards_by_dt=False option yet); pre-dividing by
    # ctrl_dt preserves exact per-step reward parity with MyoSuite CPU & mjlab.
    reward_defs: dict[str, flat_scene.Reward] = {}
    for key, wt in task.weighted_reward_keys.items():
        if wt == 0.0:
            continue
        eff_wt = float(wt) / ctrl_dt
        if key == "pose":
            reward_defs["pose"] = flat_scene.Reward(_rwd_pose, weight=eff_wt)
        elif key == "bonus":
            reward_defs["bonus"] = flat_scene.Reward(
                _rwd_bonus, weight=eff_wt, pose_thd=task.pose_thd
            )
        elif key == "penalty":
            reward_defs["penalty"] = flat_scene.Reward(
                _rwd_penalty, weight=eff_wt, far_thd=task.far_thd
            )
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
            "pose_features": flat_scene.Component(PoseFeatureProducer, target_state=target_state)
        },
        rewards=reward_defs,
        metrics={
            "solved": flat_scene.Metric(_metric_solved, pose_thd=task.pose_thd),
            "dist": flat_scene.Metric("pose/dist"),
        },
        terminations={
            "far": flat_scene.Term(_term_far, far_thd=task.far_thd),
        },
    )

    resetter = flat_scene.Component(
        PoseResetter,
        mj_model=mj_model,
        target_state=target_state,
        init_qpos=init_qpos,
        init_qvel=init_qvel,
        reset_type=task.reset_type,
        target_type=task.target_type,
        target_jnt_ids=jnt_ids,
        target_jnt_range=jnt_ranges,
        fixed_target_angles=default_target,
        tip_sids=tip_sids,
        target_sids=target_sids,
        weight_body_id=weight_bid,
        weight_geom_id=weight_gid,
        weight_range=task.weight_range,
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
        resetters={"pose_reset": resetter},
        sim=sim_cfg,
        episode_length_secs=float(task.max_episode_steps * ctrl_dt),
    )
