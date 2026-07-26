# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

import ext_loco.tasks.loco_manipulation.EE_pose.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_random(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum that randomlay set the robot at any terrain level.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    # robots that walked far enough progress to harder terrains
    move_up = torch.ones_like(terrain.terrain_levels[env_ids]) * (terrain.max_terrain_level + 1)
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = torch.zeros_like(move_up)
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.ones(1, dtype=torch.float)


def pos_commands_ranges_level(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    max_range: dict[str, tuple[float, float]],
    update_interval: int = 50 * 24,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "EE_pose",
) -> torch.Tensor:
    """Curriculum that randomlay set the robot at any terrain level.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    command_cfg: mdp.UniformPoseCommandCfg = env.command_manager.get_term(command_name).cfg
    x = command_cfg.ranges.pos_x[1]
    if env.common_step_counter % update_interval == 0:
        x = command_cfg.ranges.pos_x[1] + 0.04
        y = command_cfg.ranges.pos_y[1] + 0.04
        z_low, z_high = (
            command_cfg.ranges.pos_z[0] - 0.004,
            command_cfg.ranges.pos_z[1] + 0.004,
        )
        x = min(x, max_range["pos_x"][1])
        y = min(y, max_range["pos_y"][1])
        z_low = max(z_low, max_range["pos_z"][0])
        z_high = min(z_high, max_range["pos_z"][1])
        command_cfg.ranges.pos_x = (-x, x)
        command_cfg.ranges.pos_y = (-y, y)
        command_cfg.ranges.pos_z = (z_low, z_high)

    # return the mean terrain level
    return torch.ones(1, dtype=torch.float) * x


def orient_commands_ranges_level(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    update_interval: int = 50 * 24,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "EE_pose",
) -> torch.Tensor:
    """Curriculum that randomlay set the robot at any terrain level.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    command_cfg: mdp.UniformPoseCommandCfg = env.command_manager.get_term(command_name).cfg
    x = command_cfg.ranges.roll[1]
    if env.common_step_counter % update_interval == 0:
        x = command_cfg.ranges.roll[1] + 0.04
        y = command_cfg.ranges.pitch[1] + 0.04
        z = command_cfg.ranges.yaw[1] + 0.04
        x = min(x, 1.5)
        y = min(y, 0.8)
        z = min(z, 1.5)
        command_cfg.ranges.roll = (-x, x)
        command_cfg.ranges.pitch = (-y, y)
        command_cfg.ranges.yaw = (-z, z)

    # return the mean terrain level
    return torch.ones(1, dtype=torch.float) * x


def tighten_ee_reward_sigma(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    update_interval: int = 80 * 24,
    decay: float = 0.9,
    pos_min: float = 0.05,
    orn_min: float = 0.1,
    reward_name: str = "pose_product",
) -> torch.Tensor:
    """Curriculum that progressively tightens EE position/orientation reward tolerance.

    This mimics UMI's sigma curriculum: every ``update_interval`` steps, the ``std`` used in EE
    tracking rewards is decayed (down to a floor). It directly mutates the reward term params.

    Supports both:
    - pose_product: uses pos_sigma and orn_sigma parameters
    - track_EE_position_exp / track_EE_orientation_exp: uses std parameter

    Args:
        env: The environment
        env_ids: Environment IDs to update
        update_interval: Number of steps between updates
        decay: Decay factor per update
        pos_min: Minimum position sigma
        orn_min: Minimum orientation sigma
        reward_name: Name of the reward term to modify (default: "pose_product")

    Returns:
        Current std values for logging (pos_std, orn_std).
    """
    # Check which reward term exists
    has_pose_product = False
    has_track_ee = False

    try:
        pose_cfg = env.reward_manager.get_term_cfg(reward_name)
        has_pose_product = True
    except ValueError:
        pass

    if not has_pose_product:
        try:
            pos_cfg = env.reward_manager.get_term_cfg("track_EE_position_exp")
            orn_cfg = env.reward_manager.get_term_cfg("track_EE_orientation_exp")
            has_track_ee = True
        except ValueError:
            raise ValueError(
                f"Neither '{reward_name}' nor 'track_EE_position_exp'/'track_EE_orientation_exp' "
                "reward terms found. Please enable one of them in the reward configuration."
            )

    if env.common_step_counter % update_interval != 0:
        # return single scalar for curriculum manager logging
        if has_pose_product:
            return torch.tensor(pose_cfg.params["pos_sigma"], device=env.device, dtype=torch.float)
        else:
            return torch.tensor(pos_cfg.params["std"], device=env.device, dtype=torch.float)

    # Update sigma based on which reward term is active
    if has_pose_product:
        # Update pose_product (pos_sigma and orn_sigma)
        pos_sigma = max(pos_min, pose_cfg.params["pos_sigma"] * decay)
        orn_sigma = max(orn_min, pose_cfg.params["orn_sigma"] * decay)
        pose_cfg.params["pos_sigma"] = pos_sigma
        pose_cfg.params["orn_sigma"] = orn_sigma
        env.reward_manager.set_term_cfg(reward_name, pose_cfg)
        return torch.tensor(pos_sigma, device=env.device, dtype=torch.float)

    elif has_track_ee:
        # Update track_EE_position_exp and track_EE_orientation_exp (std)
        pos_std = max(pos_min, pos_cfg.params["std"] * decay)
        pos_cfg.params["std"] = pos_std
        env.reward_manager.set_term_cfg("track_EE_position_exp", pos_cfg)

        orn_std = max(orn_min, orn_cfg.params["std"] * decay)
        orn_cfg.params["std"] = orn_std
        env.reward_manager.set_term_cfg("track_EE_orientation_exp", orn_cfg)

        return torch.tensor(pos_std, device=env.device, dtype=torch.float)
