# Go2W-速度目标-复杂地形训练环境

# Scene（terrain修改）所需
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
# Curriculum所需
from isaaclab.managers import CurriculumTermCfg as CurrTerm

import math
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from . import mdp

# 导入已有的平坦环境配置
from .go2w_velocity_flat_cfg import Go2W_VelocityFlat_ManagerBasedEnv
from .go2w_velocity_flat_cfg import Go2W_VelocityFlatSceneCfg   # Scene也作一些微调

#=================================
# Scene设置
#=================================
@configclass
class Go2W_VelocityRoughSceneCfg(Go2W_VelocityFlatSceneCfg):
    """Go2W 速度跟踪任务的复杂地面场景（只把继承来的 terrain 换成程序化生成器）"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",              # USD 挂载路径
        terrain_type="generator",               # 程序化生成模式
        terrain_generator=ROUGH_TERRAINS_CFG,   # 使用官方预设的崎岖地形生成器
        max_init_terrain_level=5,               # 初始只在 0~5 级低难度地块出生
        collision_group=-1,                     # 全局共享碰撞体，节约显存
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,                # 静摩擦力
            dynamic_friction=1.0,               # 动摩擦力
            restitution=0.0,                    # 恢复系数 (0=完全吸震无弹力)
        ),
        debug_vis=False,
    )

#=================================
# Curriculum设置（暂时不启用）
#=================================
@configclass
class Go2W_VelocityRoughCurriculumCfg:
    """课程学习配置类：管理难度进阶"""

    # 1. 地形难度课程：根据移动位移动态升降级地块难度
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

#=================================
# 总环境设置（继承平坦地形的基础上进行修改）
#=================================
@configclass
class Go2W_VelocityRough_ManagerBasedEnv(Go2W_VelocityFlat_ManagerBasedEnv):
    """复杂地形运动速度跟踪强化学习环境总配置"""
    scene: Go2W_VelocityRoughSceneCfg = Go2W_VelocityRoughSceneCfg(num_envs=4096, env_spacing=8.0)
    curriculum: Go2W_VelocityRoughCurriculumCfg = Go2W_VelocityRoughCurriculumCfg()       # 课程学习 (地形与指令难度动态递进)

    def __post_init__(self):
        # 运行父类的post_init
        super().__post_init__()