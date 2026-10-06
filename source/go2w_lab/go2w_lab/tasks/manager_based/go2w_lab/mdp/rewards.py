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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =========================================
# base_height_l2：惩罚机身偏离目标高度
# =========================================
def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel."""
    # 1. 提取机器人资产对象
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 2. 地形自适应高度计算（核心分支）
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # 获取光线击中点的世界坐标系 z 值，shape: (num_envs, num_rays)
        ray_hits = sensor.data.ray_hits_w[..., 2]
        
        # 异常数据防护（NaN, Inf 或超出物理范围的射线未命中值）
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            # 异常降级：直接以当前实际高度作为目标高度，使单步误差为 0，防止极端梯度炸网
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            # 正常情况：相对机身下方的局部平均地形表面高度 + 设定的离地间隙
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # 3. 平地假定分支（无高度传感设备）
        adjusted_target_height = target_height

    # 4. 计算 L2 误差惩罚
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return reward
