# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Musculoskeletal task definitions and registry for ``myosuite-reaf``."""

from myosuite_reaf.tasks.pose import PoseTaskSpec, build_pose_env_config
from myosuite_reaf.tasks.reach import ReachTaskSpec, build_reach_env_config
from myosuite_reaf.tasks.registry import (
    RegisteredTask,
    get_registered_task,
    list_envs,
    make_env_config,
    make_reaf_env,
    register_with_playground,
)
from myosuite_reaf.tasks.walk import WalkTaskSpec, build_walk_env_config

__all__ = [
    "PoseTaskSpec",
    "ReachTaskSpec",
    "RegisteredTask",
    "WalkTaskSpec",
    "build_pose_env_config",
    "build_reach_env_config",
    "build_walk_env_config",
    "get_registered_task",
    "list_envs",
    "make_env_config",
    "make_reaf_env",
    "register_with_playground",
]
