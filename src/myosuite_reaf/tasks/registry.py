# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task registry for ``myosuite-reaf`` and ``mujoco_playground`` integration."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from reaf.simulation.torch import Environment as ReafEnvironment
from reaf.simulation.torch import flat_scene

from myosuite_reaf._compat import ensure_windows_fcntl_shim
from myosuite_reaf.assets import (
    build_arm_reach_spec,
    build_elbow_spec,
    build_finger_spec,
    build_hand_pose_spec,
    build_leg_walk_spec,
    build_torso_spec,
)
from myosuite_reaf.config import (
    EnvConfig,
    FatigueSpec,
    FeatureSpec,
    ReafferentationSpec,
    SarcopeniaSpec,
)
from myosuite_reaf.rl.configs import RLPresets
from myosuite_reaf.tasks.pose import PoseTaskSpec, build_pose_env_config
from myosuite_reaf.tasks.reach import ReachTaskSpec, build_reach_env_config
from myosuite_reaf.tasks.walk import WalkTaskSpec, build_walk_env_config

TaskSpec = PoseTaskSpec | ReachTaskSpec | WalkTaskSpec


@dataclass(frozen=True)
class RegisteredTask:
    """Entry in the ``myosuite-reaf`` task registry."""

    env_id: str
    spec: TaskSpec
    default_features: tuple[FeatureSpec, ...]
    runner_cfg_fn: Callable[[], Any]

    @property
    def max_episode_steps(self) -> int:
        return self.spec.max_episode_steps

    @property
    def frame_skip(self) -> int:
        return self.spec.frame_skip


_HAND_FIXED_POSE: tuple[float, ...] = (
    0.0,
    0.0,
    0.0,
    0.0824475,
    -0.0904,
    -0.681555,
    -0.514888,
    0.0,
    -0.013964,
    -0.0458132,
    0.0,
    0.67553,
    -0.020944,
    0.76979,
    0.65982,
    0.0,
    0.0,
    0.0,
    0.0,
    0.479155,
    -0.099484,
    0.95831,
    0.0,
)

_HAND_JOINT_NAMES: tuple[str, ...] = (
    "pro_sup_r",
    "deviation_r",
    "flexion_r",
    "cmc_flexion_r",
    "cmc_abduction_r",
    "mp_flexion_r",
    "ip_flexion_r",
    "mcp2_flexion_r",
    "mcp2_abduction_r",
    "pm2_flexion_r",
    "md2_flexion_r",
    "mcp3_flexion_r",
    "mcp3_abduction_r",
    "pm3_flexion_r",
    "md3_flexion_r",
    "mcp4_flexion_r",
    "mcp4_abduction_r",
    "pm4_flexion_r",
    "md4_flexion_r",
    "mcp5_flexion_r",
    "mcp5_abduction_r",
    "pm5_flexion_r",
    "md5_flexion_r",
)

_ASL_NUMERALS: tuple[tuple[float, ...], ...] = (
    (
        0,
        0,
        0,
        0.28272,
        0.5624,
        -0.75573,
        -1.309,
        1.30045,
        -0.006982,
        1.45492,
        0.998897,
        1.26466,
        0,
        1.40604,
        0.227795,
        1.07614,
        -0.020944,
        1.46103,
        0.06284,
        0.83263,
        -0.14399,
        1.571,
        1.38248,
    ),
    (
        0,
        0,
        0,
        0.04536,
        0.0248,
        -0.7854,
        -1.309,
        0.366605,
        0.010473,
        0.269258,
        0.111722,
        1.48459,
        0,
        1.45318,
        1.44532,
        1.44532,
        -0.204204,
        1.46103,
        1.44532,
        1.48459,
        -0.2618,
        1.47674,
        1.48459,
    ),
    (
        0,
        0,
        0,
        0.04536,
        0.0248,
        -0.7854,
        -1.13447,
        0.514973,
        0.010473,
        0.128305,
        0.111722,
        0.510575,
        0,
        0.37704,
        0.117825,
        1.44532,
        -0.204204,
        1.46103,
        1.44532,
        1.48459,
        -0.2618,
        1.47674,
        1.48459,
    ),
    (
        0,
        0,
        0,
        0.25305,
        0.3384,
        0.01569,
        -0.0262045,
        0.645885,
        0.010473,
        0.128305,
        0.111722,
        0.510575,
        0,
        0.37704,
        0.117825,
        1.571,
        -0.036652,
        1.52387,
        1.45318,
        1.40604,
        -0.068068,
        1.39033,
        1.571,
    ),
    (
        0,
        0,
        0,
        -0.147495,
        0.6392,
        -0.7854,
        -1.309,
        0.637158,
        0.010473,
        0.128305,
        0.111722,
        0.510575,
        0,
        0.37704,
        0.117825,
        0.306345,
        -0.010472,
        0.400605,
        0.133535,
        0.21994,
        -0.068068,
        0.274925,
        0.01571,
    ),
    (
        0,
        0,
        0,
        0.25305,
        0.3384,
        0.01569,
        -0.0262045,
        0.645885,
        0.010473,
        0.128305,
        0.111722,
        0.510575,
        0,
        0.37704,
        0.117825,
        0.306345,
        -0.010472,
        0.400605,
        0.133535,
        0.21994,
        -0.068068,
        0.274925,
        0.01571,
    ),
    (
        0,
        0,
        0,
        -0.147495,
        0.6392,
        -0.7854,
        -1.309,
        0.637158,
        0.010473,
        0.128305,
        0.111722,
        0.510575,
        0,
        0.37704,
        0.117825,
        0.306345,
        -0.010472,
        0.400605,
        0.133535,
        1.1861,
        -0.2618,
        1.35891,
        1.48459,
    ),
    (
        0,
        0,
        0,
        0.01569,
        0.524,
        -0.7854,
        -1.309,
        0.645885,
        -0.006982,
        0.128305,
        0.111722,
        0.510575,
        0,
        0.37704,
        0.117825,
        1.28036,
        -0.115192,
        1.52387,
        1.45318,
        0.432025,
        -0.068068,
        0.18852,
        0.149245,
    ),
    (
        0,
        0,
        0,
        0.22338,
        0.428,
        -0.7854,
        -1.309,
        0.645885,
        -0.006982,
        0.128305,
        0.194636,
        1.39033,
        0,
        1.08399,
        0.573415,
        0.667675,
        -0.020944,
        0,
        0.06284,
        0.432025,
        -0.068068,
        0.18852,
        0.149245,
    ),
    (
        0,
        0,
        0,
        0.28272,
        0.5624,
        -0.75573,
        -1.309,
        1.30045,
        -0.006982,
        1.45492,
        0.998897,
        0.39275,
        0,
        0.18852,
        0.227795,
        0.667675,
        -0.020944,
        0,
        0.06284,
        0.432025,
        -0.068068,
        0.18852,
        0.149245,
    ),
)

_ASL_ARR = np.asarray(_ASL_NUMERALS, dtype=np.float64)
_HAND_POSE_RANDOM_RANGES: dict[str, tuple[float, float]] = {
    name: (float(np.min(_ASL_ARR[:, i])), float(np.max(_ASL_ARR[:, i])))
    for i, name in enumerate(_HAND_JOINT_NAMES)
}

_TORSO_TARGET_RANGES: dict[str, tuple[float, float]] = {
    "flex_extension": (0.0, 0.0),
    "lat_bending": (-0.1, 0.1),
    "axial_rotation": (0.0, 0.0),
    "Abs_t1": (0.0, 0.0),
    "Abs_t2": (0.0, 0.0),
    "Abs_r3": (0.0, 0.0),
    "L4_L5_FE": (0.0, 0.0),
    "L4_L5_LB": (0.0, 0.0),
    "L4_L5_AR": (0.0, 0.0),
    "L3_L4_FE": (0.0, 0.0),
    "L3_L4_LB": (0.0, 0.0),
    "L3_L4_AR": (0.0, 0.0),
    "L2_L3_FE": (0.0, 0.0),
    "L2_L3_LB": (0.0, 0.0),
    "L2_L3_AR": (0.0, 0.0),
    "L1_L2_FE": (0.0, 0.0),
    "L1_L2_LB": (0.0, 0.0),
    "L1_L2_AR": (0.0, 0.0),
}

_TASKS: dict[str, RegisteredTask] = {}
_PLAYGROUND_REGISTERED: bool = False


def _register_family(
    env_id: str,
    spec: TaskSpec,
    runner_cfg_fn: Callable[[], Any],
) -> None:
    _TASKS[env_id] = RegisteredTask(
        env_id=env_id,
        spec=spec,
        default_features=(),
        runner_cfg_fn=runner_cfg_fn,
    )
    if env_id.startswith("myo"):
        sarc_id = "myoSarc" + env_id[3:]
        _TASKS[sarc_id] = RegisteredTask(
            env_id=sarc_id,
            spec=spec,
            default_features=(SarcopeniaSpec(force_scale=0.5),),
            runner_cfg_fn=runner_cfg_fn,
        )
        fati_id = "myoFati" + env_id[3:]
        _TASKS[fati_id] = RegisteredTask(
            env_id=fati_id,
            spec=spec,
            default_features=(FatigueSpec(),),
            runner_cfg_fn=runner_cfg_fn,
        )
    if env_id.startswith("myoHand"):
        reaf_id = "myoReaf" + env_id[3:]
        _TASKS[reaf_id] = RegisteredTask(
            env_id=reaf_id,
            spec=spec,
            default_features=(ReafferentationSpec(),),
            runner_cfg_fn=runner_cfg_fn,
        )


def _populate_registry() -> None:
    if _TASKS:
        return

    # Elbow pose (pure 6-muscle & hybrid 6-muscle + 1-motor exo)
    _register_family(
        "myoElbowPose1D6MFixed-v0",
        PoseTaskSpec(
            spec_builder=partial(build_elbow_spec, exo=False),
            target_jnt_range={"r_elbow_flex": (2.0, 2.0)},
            viz_site_targets=("wrist",),
            pose_thd=0.175,
            reset_type="random",
        ),
        RLPresets.elbow_pose,
    )
    _register_family(
        "myoElbowPose1D6MRandom-v0",
        PoseTaskSpec(
            spec_builder=partial(build_elbow_spec, exo=False),
            target_jnt_range={"r_elbow_flex": (0.0, 2.27)},
            viz_site_targets=("wrist",),
            pose_thd=0.175,
            reset_type="random",
        ),
        RLPresets.elbow_pose,
    )
    _register_family(
        "myoElbowPose1D6MExoFixed-v0",
        PoseTaskSpec(
            spec_builder=partial(build_elbow_spec, exo=True),
            target_jnt_range={"r_elbow_flex": (2.0, 2.0)},
            viz_site_targets=("wrist",),
            pose_thd=0.175,
            reset_type="random",
            weighted_reward_keys={"pose": 1.0, "bonus": 4.0, "act_reg": 5.0, "penalty": 50.0},
        ),
        RLPresets.elbow_pose,
    )
    _register_family(
        "myoElbowPose1D6MExoRandom-v0",
        PoseTaskSpec(
            spec_builder=partial(build_elbow_spec, exo=True),
            target_jnt_range={"r_elbow_flex": (0.0, 2.27)},
            viz_site_targets=("wrist",),
            pose_thd=0.175,
            reset_type="random",
            weight_bodyname="carry_weight",
            weight_range=(0.1, 2.0),
            weighted_reward_keys={"pose": 1.0, "bonus": 4.0, "act_reg": 5.0, "penalty": 50.0},
        ),
        RLPresets.elbow_pose,
    )

    # Finger pose (motor & muscle)
    _register_family(
        "motorFingerPoseFixed-v0",
        PoseTaskSpec(
            spec_builder=partial(build_finger_spec, motor=True),
            max_episode_steps=200,
            frame_skip=5,
            target_jnt_range={
                "IFadb": (0.0, 0.0),
                "IFmcp": (0.0, 0.0),
                "IFpip": (0.75, 0.75),
                "IFdip": (0.75, 0.75),
            },
            viz_site_targets=("IFtip",),
        ),
        RLPresets.finger_reach,
    )
    _register_family(
        "motorFingerPoseRandom-v0",
        PoseTaskSpec(
            spec_builder=partial(build_finger_spec, motor=True),
            max_episode_steps=200,
            frame_skip=5,
            target_jnt_range={
                "IFadb": (-0.2, 0.2),
                "IFmcp": (-0.4, 1.0),
                "IFpip": (0.1, 1.0),
                "IFdip": (0.1, 1.0),
            },
            viz_site_targets=("IFtip",),
        ),
        RLPresets.finger_reach,
    )
    _register_family(
        "myoFingerPoseFixed-v0",
        PoseTaskSpec(
            spec_builder=partial(build_finger_spec, motor=False),
            target_jnt_range={
                "IFadb": (0.0, 0.0),
                "IFmcp": (0.0, 0.0),
                "IFpip": (0.75, 0.75),
                "IFdip": (0.75, 0.75),
            },
            viz_site_targets=("IFtip",),
        ),
        RLPresets.finger_reach,
    )
    _register_family(
        "myoFingerPoseRandom-v0",
        PoseTaskSpec(
            spec_builder=partial(build_finger_spec, motor=False),
            target_jnt_range={
                "IFadb": (-0.2, 0.2),
                "IFmcp": (-0.4, 1.0),
                "IFpip": (0.1, 1.0),
                "IFdip": (0.1, 1.0),
            },
            viz_site_targets=("IFtip",),
        ),
        RLPresets.finger_reach,
    )

    # Hand pose (23-DoF 39-muscle)
    _register_family(
        "myoHandPoseFixed-v0",
        PoseTaskSpec(
            spec_builder=build_hand_pose_spec,
            target_jnt_value=_HAND_FIXED_POSE,
            viz_site_targets=("THtip_r", "IFtip_r", "MFtip_r", "RFtip_r", "LFtip_r"),
            pose_thd=0.7,
            reset_type="init",
            target_type="fixed",
            actuator_suffix="_r",
        ),
        RLPresets.hand_manipulation,
    )
    _register_family(
        "myoHandPoseRandom-v0",
        PoseTaskSpec(
            spec_builder=build_hand_pose_spec,
            target_jnt_range=_HAND_POSE_RANDOM_RANGES,
            viz_site_targets=("THtip_r", "IFtip_r", "MFtip_r", "RFtip_r", "LFtip_r"),
            pose_thd=0.7,
            reset_type="random",
            target_type="generate",
            actuator_suffix="_r",
        ),
        RLPresets.hand_manipulation,
    )

    # Torso pose (210-muscle & passive-exosuit variant with unnamed freejoints)
    _register_family(
        "myoTorsoPoseFixed-v0",
        PoseTaskSpec(
            spec_builder=partial(build_torso_spec, exo=False),
            max_episode_steps=200,
            frame_skip=5,
            pose_thd=0.25,
            far_thd=math.pi,
            target_type="fixed",
            n_pose_jnts=18,
            target_jnt_range=_TORSO_TARGET_RANGES,
        ),
        RLPresets.torso_pose,
    )
    _register_family(
        "myoTorsoExoPoseFixed-v0",
        PoseTaskSpec(
            spec_builder=partial(build_torso_spec, exo=True),
            max_episode_steps=200,
            frame_skip=5,
            pose_thd=0.25,
            far_thd=math.pi,
            target_type="fixed",
            n_pose_jnts=18,
            target_jnt_range=_TORSO_TARGET_RANGES,
        ),
        RLPresets.torso_pose,
    )

    # Finger reach (motor & muscle)
    _register_family(
        "motorFingerReachFixed-v0",
        ReachTaskSpec(
            spec_builder=partial(build_finger_spec, motor=True),
            max_episode_steps=200,
            frame_skip=5,
            target_reach_range={"IFtip": ((0.2, 0.05, 0.20), (0.2, 0.05, 0.20))},
        ),
        RLPresets.finger_reach,
    )
    _register_family(
        "motorFingerReachRandom-v0",
        ReachTaskSpec(
            spec_builder=partial(build_finger_spec, motor=True),
            max_episode_steps=200,
            frame_skip=5,
            target_sampling="workspace",
            target_reach_range={"IFtip": ((0.1, -0.1, 0.1), (0.27, 0.1, 0.3))},
        ),
        RLPresets.finger_reach,
    )
    _register_family(
        "myoFingerReachFixed-v0",
        ReachTaskSpec(
            spec_builder=partial(build_finger_spec, motor=False),
            target_reach_range={"IFtip": ((0.2, 0.05, 0.20), (0.2, 0.05, 0.20))},
        ),
        RLPresets.finger_reach,
    )
    _register_family(
        "myoFingerReachRandom-v0",
        ReachTaskSpec(
            spec_builder=partial(build_finger_spec, motor=False),
            target_sampling="workspace",
            target_reach_range={"IFtip": ((0.1, -0.1, 0.1), (0.27, 0.1, 0.3))},
        ),
        RLPresets.finger_reach,
    )

    # Hand 5-tip reach
    _register_family(
        "myoHandReachFixed-v0",
        ReachTaskSpec(
            spec_builder=build_hand_pose_spec,
            target_reach_range={
                "THtip_r": ((-0.165, -0.537, 1.495), (-0.165, -0.537, 1.495)),
                "IFtip_r": ((-0.151, -0.547, 1.455), (-0.151, -0.547, 1.455)),
                "MFtip_r": ((-0.146, -0.547, 1.447), (-0.146, -0.547, 1.447)),
                "RFtip_r": ((-0.148, -0.543, 1.445), (-0.148, -0.543, 1.445)),
                "LFtip_r": ((-0.148, -0.528, 1.434), (-0.148, -0.528, 1.434)),
            },
            far_th=0.044,
            actuator_suffix="_r",
        ),
        RLPresets.hand_manipulation,
    )
    _register_family(
        "myoHandReachRandom-v0",
        ReachTaskSpec(
            spec_builder=build_hand_pose_spec,
            target_reach_range={
                "THtip_r": ((-0.185, -0.577, 1.455), (-0.125, -0.517, 1.535)),
                "IFtip_r": ((-0.191, -0.567, 1.445), (-0.111, -0.527, 1.465)),
                "MFtip_r": ((-0.186, -0.567, 1.437), (-0.106, -0.527, 1.457)),
                "RFtip_r": ((-0.188, -0.563, 1.435), (-0.108, -0.523, 1.455)),
                "LFtip_r": ((-0.188, -0.548, 1.424), (-0.108, -0.508, 1.444)),
            },
            far_th=0.075,
            actuator_suffix="_r",
        ),
        RLPresets.hand_manipulation,
    )

    # Full 50-muscle arm reach (immobilized digits, IFtip fingertip target)
    _register_family(
        "myoArmReachFixed-v0",
        ReachTaskSpec(
            spec_builder=build_arm_reach_spec,
            max_episode_steps=150,
            target_reach_range={"IFtip": ((-0.175, -0.245, 1.405), (-0.175, -0.245, 1.405))},
            far_th=1.0,
            actuator_suffix="_r",
        ),
        RLPresets.arm_reach,
    )
    _register_family(
        "myoArmReachRandom-v0",
        ReachTaskSpec(
            spec_builder=build_arm_reach_spec,
            max_episode_steps=150,
            target_reach_range={
                "IFtip": (
                    (-0.175 - 0.175, -0.245 - 0.175, 1.405 - 0.425),
                    (-0.175 + 0.175, -0.245 + 0.175, 1.405 + 0.425),
                )
            },
            far_th=1.3,
            actuator_suffix="_r",
        ),
        RLPresets.arm_reach,
    )

    # Bipedal 80-muscle walking
    _register_family(
        "myoLegWalk-v0",
        WalkTaskSpec(
            spec_builder=build_leg_walk_spec,
            max_episode_steps=1000,
            frame_skip=10,
        ),
        RLPresets.leg_walk,
    )


def list_envs(*, base_only: bool = False) -> list[str]:
    """Return sorted list of registered ``myosuite-reaf`` environment IDs."""
    _populate_registry()
    if not base_only:
        return sorted(_TASKS.keys())
    return sorted(eid for eid, entry in _TASKS.items() if not entry.default_features)


def get_registered_task(env_id: str) -> RegisteredTask:
    """Look up a :class:`RegisteredTask` by environment ID."""
    _populate_registry()
    if env_id not in _TASKS:
        raise KeyError(f"Unknown environment {env_id!r}. Available: {sorted(_TASKS.keys())}")
    return _TASKS[env_id]


def make_env_config(
    env_id: str,
    config: EnvConfig | None = None,
) -> flat_scene.Environment:
    """Create a declarative ``reaf.simulation.torch.flat_scene.Environment`` config."""
    entry = get_registered_task(env_id)
    base_cfg = config or EnvConfig()
    if entry.default_features:
        merged = (*entry.default_features, *base_cfg.features)
        eff_cfg = EnvConfig(
            impl=base_cfg.impl,
            num_envs=base_cfg.num_envs,
            device=base_cfg.device,
            seed=base_cfg.seed,
            features=merged,
            sync_forward=base_cfg.sync_forward,
            nconmax=base_cfg.nconmax,
            njmax=base_cfg.njmax,
        )
    else:
        eff_cfg = base_cfg

    if isinstance(entry.spec, PoseTaskSpec):
        return build_pose_env_config(entry.spec, eff_cfg)
    if isinstance(entry.spec, ReachTaskSpec):
        return build_reach_env_config(entry.spec, eff_cfg)
    if isinstance(entry.spec, WalkTaskSpec):
        return build_walk_env_config(entry.spec, eff_cfg)
    raise TypeError(f"Unsupported TaskSpec {type(entry.spec)!r}")


def make_reaf_env(
    env_id: str,
    config: EnvConfig | None = None,
) -> ReafEnvironment:
    """Instantiate a compiled ``reaf.simulation.torch.Environment``."""
    reaf_cfg = make_env_config(env_id, config=config)
    return flat_scene.build_env(reaf_cfg)


def register_with_playground() -> list[str]:
    """Register all ``myosuite-reaf`` tasks into ``mujoco_playground``'s ``env_registry``."""
    global _PLAYGROUND_REGISTERED
    ensure_windows_fcntl_shim()
    from mujoco_playground._src.rl import env_registry

    _populate_registry()
    newly_registered: list[str] = []
    for env_id, entry in _TASKS.items():
        if env_id in env_registry._REGISTRY:
            continue
        env_registry.register(
            env_id,
            entry.runner_cfg_fn(),
            env_cfg_fn=partial(make_env_config, env_id),
        )
        newly_registered.append(env_id)
    _PLAYGROUND_REGISTERED = True
    return newly_registered
