# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bipedal 80-muscle locomotion tasks on ``reaf-sim`` (``LegWalkEnvV0``).

Covers:
  - ``myoLegWalk-v0``
  - ``myoSarcLegWalk-v0`` (50% peak isometric muscle force)
  - ``myoFatiLegWalk-v0`` (3CC-r cumulative muscle fatigue)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import torch
from dm_env import specs
from myo_sim.physics.joint_coupling import JointCouplings, project_joint_couplings
from reaf.simulation.torch import (
    EnvironmentReset,
    FeatureProducer,
    TerminationResult,
    flat_scene,
)
from reaf.simulation.torch._src.builders import config as builders_config

from myosuite_reaf.assets import apply_sarcopenia_to_spec
from myosuite_reaf.config import EnvConfig, SarcopeniaSpec
from myosuite_reaf.devices import (
    KinematicsDevice,
    MuscleActuatorDevice,
    SyncForwardDevice,
)
from myosuite_reaf.processors import ExcitationPipelineProcessor

_WALK_HIP_AMPLITUDE: float = 0.8
_WALK_REF_ROT_SCALE: float = 5.0
_WALK_JOINT_ANGLE_SCALE: float = 5.0
_LOCOMOTION_VEL_TOL: float = 0.5


class WalkFeatureProducer(FeatureProducer):
    """Computes biomechanical locomotion observations and reward components."""

    def __init__(
        self,
        name: str = "walk_features",
        *,
        body_mass: np.ndarray,
        torso_id: int,
        pelvis_id: int,
        talus_l_id: int,
        talus_r_id: int,
        hip_flex_indices: tuple[int, int],
        hip_angle_indices: tuple[int, int, int, int],
        target_rot: np.ndarray,
        target_vel: tuple[float, float] = (0.0, 1.2),
        min_height: float = 0.8,
        max_rot: float = 0.8,
        hip_period: int = 100,
        ctrl_dt: float = 0.02,
        nworld: int = 1,
        xpu_device: str = "cpu",
    ) -> None:
        self._name = name
        self._nworld = nworld
        self._device = torch.device(xpu_device)
        self._mass = torch.as_tensor(body_mass, dtype=torch.float32, device=self._device)
        self._total_mass = self._mass.sum()
        self._torso_id = int(torso_id)
        self._pelvis_id = int(pelvis_id)
        self._left_id = int(talus_l_id)
        self._right_id = int(talus_r_id)
        self._feet_ids = torch.tensor(
            [self._left_id, self._right_id], dtype=torch.long, device=self._device
        )
        self._hip_flex_l, self._hip_flex_r = hip_flex_indices
        self._hip_angle_idx = torch.tensor(
            list(hip_angle_indices), dtype=torch.long, device=self._device
        )
        self._target_rot = torch.as_tensor(target_rot, dtype=torch.float32, device=self._device)
        self._target_x_vel, self._target_y_vel = float(target_vel[0]), float(target_vel[1])
        self._min_height = float(min_height)
        self._max_rot = float(max_rot)
        self._hip_period = float(hip_period)
        self._ctrl_dt = float(ctrl_dt)

    @property
    def name(self) -> str:
        return self._name

    def required_features_keys(self) -> set[str]:
        return {
            "time",
            "kinematics/qpos",
            "kinematics/xpos",
            "kinematics/xipos",
            "kinematics/xquat",
            "kinematics/cvel",
        }

    def produced_features_spec(self) -> Mapping[str, specs.Array]:
        return {
            "walk/com_vel": specs.Array(shape=(2,), dtype=np.float32),
            "walk/torso_angle": specs.Array(shape=(4,), dtype=np.float32),
            "walk/feet_heights": specs.Array(shape=(2,), dtype=np.float32),
            "walk/height": specs.Array(shape=(1,), dtype=np.float32),
            "walk/feet_rel_positions": specs.Array(shape=(6,), dtype=np.float32),
            "walk/phase_var": specs.Array(shape=(1,), dtype=np.float32),
            "walk/vel_reward": specs.Array(shape=(), dtype=np.float32),
            "walk/cyclic_hip": specs.Array(shape=(), dtype=np.float32),
            "walk/ref_rot": specs.Array(shape=(), dtype=np.float32),
            "walk/joint_angle_rew": specs.Array(shape=(), dtype=np.float32),
            "walk/fallen": specs.Array(shape=(), dtype=np.float32),
            "walk/solved": specs.Array(shape=(), dtype=np.float32),
        }

    def produce_features(
        self, required_features: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        qpos = required_features["kinematics/qpos"].float()
        xpos = required_features["kinematics/xpos"].float()
        xipos = required_features["kinematics/xipos"].float()
        xquat = required_features["kinematics/xquat"].float()
        cvel = required_features["kinematics/cvel"].float()
        time_s = required_features["time"].float()

        com = (self._mass[None, :, None] * xipos).sum(dim=1) / self._total_mass
        height_vec = com[:, 2:3]
        height = com[:, 2]

        com_cvel = (self._mass[None, :, None] * (-cvel)).sum(dim=1) / self._total_mass
        com_vel = com_cvel[:, 3:5]
        vx, vy = com_vel[:, 0], com_vel[:, 1]

        torso_angle = xquat[:, self._torso_id]
        feet_heights = xpos[:, self._feet_ids, 2]
        pelvis_pos = xpos[:, self._pelvis_id]
        feet_rel = torch.cat(
            [
                xpos[:, self._left_id] - pelvis_pos,
                xpos[:, self._right_id] - pelvis_pos,
            ],
            dim=-1,
        )

        steps = torch.round(time_s / self._ctrl_dt)
        phase = torch.remainder(steps / self._hip_period, 1.0)
        phase_var = phase.unsqueeze(-1)

        vel_reward = torch.exp(-torch.square(self._target_y_vel - vy)) + torch.exp(
            -torch.square(self._target_x_vel - vx)
        )

        quat = qpos[:, 3:7]
        qy, qz = quat[:, 2], quat[:, 3]
        r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        fallen_bool = (height < self._min_height) | (torch.abs(r00) > self._max_rot)
        fallen = fallen_bool.float()

        des_l = _WALK_HIP_AMPLITUDE * torch.cos(phase * 2.0 * math.pi + math.pi)
        des_r = _WALK_HIP_AMPLITUDE * torch.cos(phase * 2.0 * math.pi)
        cyclic_hip = torch.sqrt(
            torch.square(des_l - qpos[:, self._hip_flex_l])
            + torch.square(des_r - qpos[:, self._hip_flex_r])
        )

        ref_rot = torch.exp(
            -torch.linalg.norm(_WALK_REF_ROT_SCALE * (quat - self._target_rot), dim=-1)
        )

        hip_angle_mag = torch.mean(torch.abs(qpos[:, self._hip_angle_idx]), dim=-1)
        joint_angle_rew = torch.exp(-_WALK_JOINT_ANGLE_SCALE * hip_angle_mag)

        vel_err = torch.hypot(self._target_x_vel - vx, self._target_y_vel - vy)
        solved = ((vel_err < _LOCOMOTION_VEL_TOL) & (~fallen_bool)).float()

        return {
            "walk/com_vel": com_vel,
            "walk/torso_angle": torso_angle,
            "walk/feet_heights": feet_heights,
            "walk/height": height_vec,
            "walk/feet_rel_positions": feet_rel,
            "walk/phase_var": phase_var,
            "walk/vel_reward": vel_reward,
            "walk/cyclic_hip": cyclic_hip,
            "walk/ref_rot": ref_rot,
            "walk/joint_angle_rew": joint_angle_rew,
            "walk/fallen": fallen,
            "walk/solved": solved,
        }


class WalkResetter(EnvironmentReset):
    """Initializes walking keyframes and projects coupled knee/patella/ankle equalities."""

    def __init__(
        self,
        *,
        mj_model: mujoco.MjModel,
        reset_type: str = "init",
        seed: int = 0,
    ) -> None:
        self._couplings_np = JointCouplings.from_model(mj_model)
        self._key0_qpos_np = np.asarray(mj_model.key_qpos[0], dtype=np.float32)
        self._key2_qpos_np = np.asarray(mj_model.key_qpos[2], dtype=np.float32)
        self._key2_qvel_np = np.asarray(mj_model.key_qvel[2], dtype=np.float32)
        self._key3_qpos_np = np.asarray(mj_model.key_qpos[3], dtype=np.float32)
        self._key3_qvel_np = np.asarray(mj_model.key_qvel[3], dtype=np.float32)
        self._reset_type = reset_type
        self._seed = int(seed)
        self._rng: torch.Generator | None = None
        self._couplings_torch: JointCouplings | None = None
        self._couplings_key: tuple[torch.device, torch.dtype] | None = None
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
    ) -> None:
        self._override = {"qpos": qpos, "qvel": qvel}

    def _generator(self, device: torch.device) -> torch.Generator:
        if self._rng is None or self._rng.device != device:
            self._rng = torch.Generator(device=device)
            self._rng.manual_seed(self._seed)
        return self._rng

    def _couplings(self, device: torch.device, dtype: torch.dtype) -> JointCouplings:
        if self._couplings_torch is None or self._couplings_key != (device, dtype):
            self._couplings_torch = self._couplings_np.as_torch(device=device, dtype=dtype)
            self._couplings_key = (device, dtype)
        return self._couplings_torch

    def on_reset(self, config: Any, env_mask: torch.Tensor) -> None:
        del config
        sim = self._env.device_layer.sim
        data = sim.data
        device = data.qpos.device
        nworld = data.qpos.shape[0]
        qpos_dtype = data.qpos.dtype
        qvel_dtype = data.qvel.dtype
        gen = self._generator(device)
        override = self._override
        self._override = None

        if override is not None and override["qpos"] is not None:
            new_qpos = (
                torch.as_tensor(override["qpos"], dtype=qpos_dtype, device=device)
                .broadcast_to(data.qpos.shape)
                .clone()
            )
            new_qvel = (
                torch.as_tensor(override["qvel"], dtype=qvel_dtype, device=device)
                .broadcast_to(data.qvel.shape)
                .clone()
                if override["qvel"] is not None
                else torch.zeros_like(data.qvel)
            )
        elif self._reset_type == "random":
            coin = torch.rand((nworld, 1), generator=gen, device=device) < 0.5
            k2_q = torch.as_tensor(self._key2_qpos_np, dtype=qpos_dtype, device=device)
            k3_q = torch.as_tensor(self._key3_qpos_np, dtype=qpos_dtype, device=device)
            k2_v = torch.as_tensor(self._key2_qvel_np, dtype=qvel_dtype, device=device)
            k3_v = torch.as_tensor(self._key3_qvel_np, dtype=qvel_dtype, device=device)
            base_q = torch.where(coin, k2_q.unsqueeze(0), k3_q.unsqueeze(0))
            new_qvel = torch.where(coin, k2_v.unsqueeze(0), k3_v.unsqueeze(0)).clone()
            noise = 0.02 * torch.randn(
                base_q.shape, generator=gen, device=device, dtype=torch.float32
            ).to(qpos_dtype)
            new_qpos = base_q + noise
            new_qpos[:, 2] = base_q[:, 2]
            new_qpos[:, 3:7] = base_q[:, 3:7]
        elif self._reset_type == "init":
            new_qpos = (
                torch.as_tensor(self._key2_qpos_np, dtype=qpos_dtype, device=device)
                .unsqueeze(0)
                .expand(nworld, -1)
                .clone()
            )
            new_qvel = (
                torch.as_tensor(self._key2_qvel_np, dtype=qvel_dtype, device=device)
                .unsqueeze(0)
                .expand(nworld, -1)
                .clone()
            )
        else:
            new_qpos = (
                torch.as_tensor(self._key0_qpos_np, dtype=qpos_dtype, device=device)
                .unsqueeze(0)
                .expand(nworld, -1)
                .clone()
            )
            new_qvel = torch.zeros_like(data.qvel)

        project_joint_couplings(self._couplings(device, qpos_dtype), new_qpos, new_qvel)

        data.qpos[:] = torch.where(env_mask.unsqueeze(1), new_qpos, data.qpos)
        data.qvel[:] = torch.where(env_mask.unsqueeze(1), new_qvel, data.qvel)
        if data.act.numel() > 0:
            data.act[:] = torch.where(env_mask.unsqueeze(1), torch.zeros_like(data.act), data.act)


def _obs_qpos_without_xy(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return features["kinematics/qpos"].float()[:, 2:]


def _obs_qvel_scaled(features: Mapping[str, torch.Tensor], *, ctrl_dt: float) -> torch.Tensor:
    return features["kinematics/qvel"].float() * ctrl_dt


def _obs_muscle_length(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return features["actuators/actuator_length"].float()


def _obs_muscle_velocity(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return features["actuators/actuator_velocity"].float().clamp(-100.0, 100.0)


def _obs_muscle_force(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return (features["actuators/actuator_force"].float() / 1000.0).clamp(-100.0, 100.0)


def _obs_act(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return features["actuators/act"].float()


def _rwd_key(features: Mapping[str, torch.Tensor], *, key: str) -> torch.Tensor:
    return features[key].float()


def _term_fallen(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    fallen = features["walk/fallen"].float() > 0.5
    return torch.where(
        fallen,
        int(TerminationResult.TERMINATE),
        int(TerminationResult.DO_NOT_TERMINATE),
    )


@dataclass(frozen=True)
class WalkTaskSpec:
    """Declarative specification for a MyoSuite bipedal locomotion environment."""

    spec_builder: Callable[[], mujoco.MjSpec]
    max_episode_steps: int = 1000
    frame_skip: int = 10
    min_height: float = 0.8
    max_rot: float = 0.8
    hip_period: int = 100
    reset_type: str = "init"
    target_x_vel: float = 0.0
    target_y_vel: float = 1.2
    normalize_act: bool = True
    weighted_reward_keys: Mapping[str, float] = field(
        default_factory=lambda: {
            "vel_reward": 5.0,
            "done": -100.0,
            "cyclic_hip": -10.0,
            "ref_rot": 10.0,
            "joint_angle_rew": 5.0,
        }
    )


def build_walk_env_config(
    task: WalkTaskSpec,
    env_cfg: EnvConfig | None = None,
) -> flat_scene.Environment:
    """Build a ``reaf.simulation.torch.flat_scene.Environment`` config for walking."""
    cfg = env_cfg or EnvConfig()
    mj_spec = task.spec_builder()
    for feat in cfg.features:
        if isinstance(feat, SarcopeniaSpec):
            apply_sarcopenia_to_spec(mj_spec, force_scale=feat.force_scale)

    sim_dt = float(mj_spec.option.timestep)
    ctrl_dt = sim_dt * task.frame_skip
    mj_model = mj_spec.compile()

    def _qadr(joint_name: str) -> int:
        return int(mj_model.jnt_qposadr[mj_model.joint(joint_name).id])

    hip_flex_indices = (_qadr("hip_flexion_l"), _qadr("hip_flexion_r"))
    hip_angle_indices = (
        _qadr("hip_adduction_l"),
        _qadr("hip_adduction_r"),
        _qadr("hip_rotation_l"),
        _qadr("hip_rotation_r"),
    )
    target_rot = np.asarray(mj_model.key_qpos[0, 3:7], dtype=np.float32)

    nu = int(mj_model.nu)
    na = int(mj_model.na)
    obs_terms: dict[str, flat_scene.Obs] = {
        "qpos_without_xy": flat_scene.Obs(_obs_qpos_without_xy).configure(
            output_shape=(int(mj_model.nq) - 2,)
        ),
        "qvel": flat_scene.Obs(_obs_qvel_scaled, ctrl_dt=ctrl_dt).configure(
            output_shape=(int(mj_model.nv),)
        ),
        "com_vel": flat_scene.Obs("walk/com_vel").configure(output_shape=(2,)),
        "torso_angle": flat_scene.Obs("walk/torso_angle").configure(output_shape=(4,)),
        "feet_heights": flat_scene.Obs("walk/feet_heights").configure(output_shape=(2,)),
        "height": flat_scene.Obs("walk/height").configure(output_shape=(1,)),
        "feet_rel_positions": flat_scene.Obs("walk/feet_rel_positions").configure(
            output_shape=(6,)
        ),
        "phase_var": flat_scene.Obs("walk/phase_var").configure(output_shape=(1,)),
        "muscle_length": flat_scene.Obs(_obs_muscle_length).configure(output_shape=(nu,)),
        "muscle_velocity": flat_scene.Obs(_obs_muscle_velocity).configure(output_shape=(nu,)),
        "muscle_force": flat_scene.Obs(_obs_muscle_force).configure(output_shape=(nu,)),
    }
    if na > 0:
        obs_terms["act"] = flat_scene.Obs(_obs_act).configure(output_shape=(na,))

    key_to_feature = {
        "vel_reward": "walk/vel_reward",
        "done": "walk/fallen",
        "cyclic_hip": "walk/cyclic_hip",
        "ref_rot": "walk/ref_rot",
        "joint_angle_rew": "walk/joint_angle_rew",
    }
    # Pre-divide weights by ctrl_dt to counteract reaf-sim's unconditional * dt scaling.
    reward_defs = {
        rkey: flat_scene.Reward(_rwd_key, weight=float(wt) / ctrl_dt, key=key_to_feature[rkey])
        for rkey, wt in task.weighted_reward_keys.items()
        if wt != 0.0 and rkey in key_to_feature
    }

    devices: dict[str, flat_scene.Component[Any]] = {
        "sync_forward": flat_scene.Component(SyncForwardDevice, enabled=cfg.sync_forward),
        "actuators": flat_scene.Component(
            MuscleActuatorDevice,
            measurements=(
                "ctrl",
                "act",
                "actuator_length",
                "actuator_velocity",
                "actuator_force",
            ),
        ),
        "kinematics": flat_scene.Component(
            KinematicsDevice,
            measurements=("qpos", "qvel", "xpos", "xipos", "xquat", "cvel"),
        ),
    }

    task_layer = flat_scene.TaskLayer(
        command_processors={
            "excitation_pipeline": flat_scene.Component(
                ExcitationPipelineProcessor,
                mj_model=mj_model,
                control_timestep=ctrl_dt,
                normalize_act=task.normalize_act,
                muscle_sigmoid=False,
                features=cfg.features,
                seed=cfg.seed,
            )
        },
        feature_producers={
            "walk_features": flat_scene.Component(
                WalkFeatureProducer,
                body_mass=np.asarray(mj_model.body_mass, dtype=np.float32),
                torso_id=int(mj_model.body("torso").id),
                pelvis_id=int(mj_model.body("pelvis").id),
                talus_l_id=int(mj_model.body("talus_l").id),
                talus_r_id=int(mj_model.body("talus_r").id),
                hip_flex_indices=hip_flex_indices,
                hip_angle_indices=hip_angle_indices,
                target_rot=target_rot,
                target_vel=(task.target_x_vel, task.target_y_vel),
                min_height=task.min_height,
                max_rot=task.max_rot,
                hip_period=task.hip_period,
                ctrl_dt=ctrl_dt,
            )
        },
        rewards=reward_defs,
        metrics={
            "solved": flat_scene.Metric("walk/solved"),
        },
        terminations={
            "fallen": flat_scene.Term(_term_fallen),
        },
    )

    resetter = flat_scene.Component(
        WalkResetter,
        mj_model=mj_model,
        reset_type=task.reset_type,
        seed=cfg.seed,
    )

    sim_cfg = builders_config.Sim(
        impl=cfg.impl,
        xpu_device=cfg.resolved_device(),
        nworld=cfg.num_envs,
        simulation_timestep=sim_dt,
        control_timestep=ctrl_dt,
        nconmax=cfg.nconmax if cfg.nconmax is not None else 64,
        njmax=cfg.njmax if cfg.njmax is not None else 256,
    )

    return flat_scene.Environment(
        base_scene=mj_spec,
        device_layer=flat_scene.DeviceLayer(devices=devices),
        task_layer=task_layer,
        observation_adapter={
            "policy": flat_scene.ObsGroup(terms=obs_terms),
        },
        resetters={"walk_reset": resetter},
        sim=sim_cfg,
        episode_length_secs=float(task.max_episode_steps * ctrl_dt),
    )
