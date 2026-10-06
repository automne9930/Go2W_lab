# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.envs import mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =========================================
# feet_stand_well：奖励无指令的时候四腿不抬起
# =========================================
def feet_stand_well(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward all feet being in contact when linear/angular command is near zero."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 1. 判断足端垂直力是否大于支撑阈值 (例如 5.0 N)
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    in_contact = (forces_z > 5.0).float()
    
    # 2. 统计接触地面脚的数量（双足期望为 2，四足期望为 4）
    reward = torch.sum(in_contact, dim=1)
    
    # 3. 仅在无行进指令时生效
    cmd_norm = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    reward *= (cmd_norm < 0.1)
    
    # 4. 倒地衰减保护
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

# =========================================
# joint_power：计算力矩与角速度乘积绝对值，惩罚机械功耗
# =========================================
def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # 1. 提取机器人铰接资产对象 (Articulation)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 2. 计算各关节瞬时机械功率绝对值 |P| = |tau * omega| 并沿关节维度求和
    # shape: (num_envs,)
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward

# =========================================
# stand_still：静止状态下惩罚关节偏离默认姿态
# =========================================
def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # 1. 计算关节角度偏离默认初始姿态的 L1 误差: sum(|q - q_default|)
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    
    # 2. 指令门控：仅在指令模长接近零时激活惩罚 (||cmd|| < threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    
    # 3. 倒地衰减保护：防止翻车后产生无效梯度干扰
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

# =========================================
# joint_not_default：动静态双重惩罚关节偏离默认姿态
# =========================================
def joint_not_default(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # 1. 提取机器人铰接资产对象
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 2. 提取当前指令模长与机身底盘实际水平线速度模长
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    
    # 3. 计算选定关节相对默认初始姿态的 L2 偏离范数: ||q - q_default||_2
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    
    # 4. 动静状态分支加权：
    # 只要有行进指令 (cmd > cmd_th) 或 机身仍在移动 (body_vel > vel_th)，处于动态，施加基础惩罚；
    # 仅当既无指令且速度几乎为零时，视为真正静止，将惩罚放大 stand_still_scale 倍
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    
    # 5. 倒地姿态衰减保护
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

# =========================================
# joint_mirror：惩罚左右对称关节的角度偏差，引导步态对称性
# =========================================
def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # 1. 提取机器人铰接资产对象 (Articulation)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 2. 关节索引查找与缓存（首次调用时解析，后续直接复用，避免频繁字符串搜索）
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # mirror_joints 结构为形如 [["left_joint", "right_joint"], ...] 的成对配对列表
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
        
    reward = torch.zeros(env.num_envs, device=env.device)
    
    # 3. 遍历所有镜像对称关节对，累加两两对应关节位置的差值平方
    for joint_pair in env.joint_mirror_joints_cache:
        # joint_pair[0][0] 为左侧关节索引，joint_pair[1][0] 为对应右侧关节索引
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
        
    # 4. 关节对数量归一化
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    
    # 5. 倒地姿态衰减保护
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward