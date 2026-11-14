# Copyright 2026 Mark Colley. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RSL-RL PPO runner configurations for musculoskeletal ``myosuite-reaf`` tasks."""

from __future__ import annotations

import dataclasses

from myosuite_reaf._compat import ensure_windows_fcntl_shim

ensure_windows_fcntl_shim()

from mujoco_playground._src.rl.rsl_rl import rsl_config  # noqa: E402


def make_ppo_runner_cfg(
    *,
    experiment_name: str,
    nworld: int = 2048,
    max_iterations: int = 300,
    num_steps_per_env: int = 24,
    hidden_dims: tuple[int, ...] = (256, 256, 128),
    learning_rate: float = 3e-4,
    entropy_coef: float = 0.005,
    init_std: float = 0.5,
    obs_normalization: bool = True,
) -> rsl_config.RslRlOnPolicyRunnerCfg:
    """Create an ``RslRlOnPolicyRunnerCfg`` tailored for Hill-type muscle control."""
    return rsl_config.RslRlOnPolicyRunnerCfg(
        seed=42,
        nworld=nworld,
        max_iterations=max_iterations,
        num_steps_per_env=num_steps_per_env,
        obs_groups={
            "actor": ["policy"],
            "critic": ["policy"],
        },
        save_interval=50,
        experiment_name=experiment_name,
        actor=rsl_config.RslRlMLPModelCfg(
            hidden_dims=hidden_dims,
            activation="elu",
            obs_normalization=obs_normalization,
            distribution_cfg=rsl_config.RslRlDistributionCfg(
                init_std=init_std,
                std_type="scalar",
            ),
        ),
        critic=rsl_config.RslRlMLPModelCfg(
            hidden_dims=hidden_dims,
            activation="elu",
            obs_normalization=obs_normalization,
            distribution_cfg=None,
        ),
        algorithm=rsl_config.RslRlPpoAlgorithmCfg(
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=learning_rate,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            entropy_coef=entropy_coef,
            desired_kl=0.01,
            max_grad_norm=1.0,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
        ),
    )


@dataclasses.dataclass
class RLPresets:
    """Standard RSL-RL configurations by task family."""

    @staticmethod
    def elbow_pose() -> rsl_config.RslRlOnPolicyRunnerCfg:
        return make_ppo_runner_cfg(
            experiment_name="myo_elbow_pose",
            nworld=1024,
            max_iterations=150,
            num_steps_per_env=24,
            hidden_dims=(128, 128),
            learning_rate=5e-4,
        )

    @staticmethod
    def finger_reach() -> rsl_config.RslRlOnPolicyRunnerCfg:
        return make_ppo_runner_cfg(
            experiment_name="myo_finger_reach",
            nworld=1024,
            max_iterations=200,
            num_steps_per_env=24,
            hidden_dims=(256, 128),
            learning_rate=3e-4,
        )

    @staticmethod
    def hand_manipulation() -> rsl_config.RslRlOnPolicyRunnerCfg:
        return make_ppo_runner_cfg(
            experiment_name="myo_hand",
            nworld=2048,
            max_iterations=400,
            num_steps_per_env=24,
            hidden_dims=(256, 256, 128),
            learning_rate=3e-4,
        )

    @staticmethod
    def arm_reach() -> rsl_config.RslRlOnPolicyRunnerCfg:
        return make_ppo_runner_cfg(
            experiment_name="myo_arm_reach",
            nworld=2048,
            max_iterations=400,
            num_steps_per_env=24,
            hidden_dims=(256, 256, 128),
            learning_rate=3e-4,
        )

    @staticmethod
    def torso_pose() -> rsl_config.RslRlOnPolicyRunnerCfg:
        return make_ppo_runner_cfg(
            experiment_name="myo_torso_pose",
            nworld=1024,
            max_iterations=300,
            num_steps_per_env=24,
            hidden_dims=(512, 256, 128),
            learning_rate=3e-4,
        )

    @staticmethod
    def leg_walk() -> rsl_config.RslRlOnPolicyRunnerCfg:
        return make_ppo_runner_cfg(
            experiment_name="myo_leg_walk",
            nworld=4096,
            max_iterations=1000,
            num_steps_per_env=32,
            hidden_dims=(512, 256, 128),
            learning_rate=3e-4,
            entropy_coef=0.002,
        )
