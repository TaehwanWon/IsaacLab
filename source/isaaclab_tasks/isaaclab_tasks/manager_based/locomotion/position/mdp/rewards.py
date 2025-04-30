# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import math

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def reward_climb_up(
    env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for approaching the top of the ladder (z-axis).

    Dense reward based on closeness to the ladder top height.
    """
    rung_count = 15
    margin = 0.3
    rung_spacing = 0.35
    ladder_theta = 0.5
    ladder_height = (rung_count - 1) * rung_spacing + 2 * margin
    ladder_top_z = ladder_height * math.cos(ladder_theta)
    asset = env.scene[asset_cfg.name]
    z_pos = asset.data.root_pos_w[..., 2]

    # Calculate absolute error
    error = torch.abs(z_pos - ladder_top_z)

    # Convert error to reward: higher reward as error decreases
    reward = torch.exp(-error / std**2)  # std는 보상의 민감도를 조절하는 hyperparameter

    return reward


def hand_contact_time_reward(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward is proportional to the total contact time of both hands with the ladder.
    The reward is clamped to a maximum of `threshold` per environment.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 손 별 contact time (e.g. left_hand, right_hand)
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]  # shape: (num_envs, 2)

    # 두 손의 contact 시간 합산
    reward = torch.sum(contact_time, dim=-1)  # shape: (num_envs,)

    # clamp to threshold
    reward = torch.clamp(reward, max=threshold)

    return reward

def sparse_pos_traget_reward(env, command_name: str, Tr: float, c_task: float = 1.0) -> torch.Tensor:
    t_curr = env.episode_length_buf * env.step_dt
    T = env.max_episode_length_s

    # 현재 torso 위치
    pos_curr_z = env.scene["robot"].data.root_pos_w[:, 2]  # z only

    # 목표 torso z 위치 (command manager에서 추출)
    target_pose = env.command_manager.get_command(command_name)  # shape: (num_envs, 7)
    # print("target_pose_in_reward : ", target_pose)
    target_z = target_pose[:, 2]

    # 거리 계산
    dist = torch.abs(pos_curr_z - target_z)
    # print("distance : ", dist )

    reward = torch.where(
        t_curr > (T - Tr),
        c_task / (Tr * (1.0 + dist)),
        torch.zeros_like(dist)
    )
    print("t_curr:", t_curr)
    print("T - Tr:", T - Tr)
    print("조건:", t_curr > (T - Tr))
    print("pos_tracking_reward : ", reward)
    return reward

def penalty_stationary(env, command_name: str = "", threshold: float = 0.2, penalty_val: float = -1.0) -> torch.Tensor:
    """
    Penalize when the full linear velocity (x, y, z) norm is below threshold.
    """
    vel = env.scene["robot"].data.root_lin_vel_w  # shape: (num_envs, 3)
    vel_norm = torch.norm(vel, dim=-1)  # (num_envs,)

    penalty = (vel_norm < threshold).float() * penalty_val
    return penalty

def torso_rotation_penalty(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # torso body index
    torso_id = env.scene[sensor_cfg.name].data.body_names.index("torso_link")

    # z축 각속도 (world frame)
    ang_vel = env.scene[sensor_cfg.name].data.body_ang_vel_w[:, torso_id, 2]

    # 절댓값이 클수록 penalty
    penalty = torch.abs(ang_vel)
    return penalty