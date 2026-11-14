# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MyoSuite musculoskeletal environments built on ``reaf-sim`` and ``mujoco_playground``.

Provides unified CPU (MuJoCo 4 C++) and GPU-batched (MuJoCo Warp 4) execution
through Google DeepMind's Robotics Environment Authoring Framework (``reaf-sim``),
designed as a prototype backend for future MuJoCo 4 integration in MyoSuite.
"""

from myosuite_reaf._compat import ensure_windows_fcntl_shim
from myosuite_reaf.config import (
    CustomStageSpec,
    EnvConfig,
    ExcitationStage,
    FatigueSpec,
    FeatureSpec,
    LowPassStage,
    MotorNoiseCfg,
    MotorNoiseSpec,
    ReafferentationSpec,
    SarcopeniaSpec,
    SimBackend,
    StageOrderWarning,
    apply_motor_noise,
    sigmoid_excitation,
)
from myosuite_reaf.devices import (
    KinematicsDevice,
    MuscleActuatorDevice,
    SyncForwardDevice,
)
from myosuite_reaf.processors import ExcitationPipelineProcessor
from myosuite_reaf.tasks import (
    RegisteredTask,
    get_registered_task,
    list_envs,
    make_env_config,
    make_reaf_env,
    register_with_playground,
)
from myosuite_reaf.wrappers import (
    DictObservationWrapper,
    ExcitationStageWrapper,
    FatigueWrapper,
    GymnasiumEnvAdapter,
    MotorNoiseWrapper,
    ReafferentationWrapper,
    SarcopeniaWrapper,
    make_env,
)

__version__ = "0.1.0"

__all__ = [
    "CustomStageSpec",
    "DictObservationWrapper",
    "EnvConfig",
    "ExcitationPipelineProcessor",
    "ExcitationStage",
    "ExcitationStageWrapper",
    "FatigueSpec",
    "FatigueWrapper",
    "FeatureSpec",
    "GymnasiumEnvAdapter",
    "KinematicsDevice",
    "LowPassStage",
    "MotorNoiseCfg",
    "MotorNoiseSpec",
    "MotorNoiseWrapper",
    "MuscleActuatorDevice",
    "ReafferentationSpec",
    "ReafferentationWrapper",
    "RegisteredTask",
    "SarcopeniaSpec",
    "SarcopeniaWrapper",
    "SimBackend",
    "StageOrderWarning",
    "SyncForwardDevice",
    "__version__",
    "apply_motor_noise",
    "ensure_windows_fcntl_shim",
    "get_registered_task",
    "list_envs",
    "make_env",
    "make_env_config",
    "make_reaf_env",
    "register_with_playground",
    "sigmoid_excitation",
]
