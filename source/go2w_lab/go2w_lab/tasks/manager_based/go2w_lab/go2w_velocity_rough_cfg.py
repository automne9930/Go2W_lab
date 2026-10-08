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
        max_init_terrain_level=5,               # 初始只在 0~5 级低难度地块出生（一共0-9级）
        # max_init_terrain_level=9,              # 如果开启play模式，可以直接全部接受以测试用
        collision_group=-1,                     # 全局共享碰撞体，节约显存
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,                # 静摩擦力
            dynamic_friction=1.0,               # 动摩擦力
            restitution=0.0,                    # 恢复系数 (0=完全吸震无弹力)
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
        # 0.25~0.35 之间（中暗灰，既不会死黑也不会晃眼）
        diffuse_color=(0.3, 0.3, 0.3),
        roughness=0.9,   # 几乎全哑光，不反射多余高光
        metallic=0.0,
        ),
        debug_vis=False,
    )

#=================================
# Curriculum设置
#=================================
@configclass
class Go2W_VelocityRoughCurriculumCfg:
    """课程学习配置类：管理难度进阶"""

    # 1. 地形难度课程：根据移动位移动态升降级地块难度
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    # 2. 指令课程
    # 注册线速度课程项
    lin_vel_cmd = CurrTerm(
        func=mdp.command_levels_lin_vel,
        params={
            # 必须与 RewardsCfg 中配置的速度跟踪奖励项名称一致
            "reward_term_name": "track_lin_vel_xy_exp",
            # 初始速度范围占最终速度范围的比例 [min_ratio, max_ratio]
            # 例如最终范围是 [-2.0, 2.0]，则初始范围被压缩至 [-0.2, 0.2]
            "range_multiplier": (0.1, 1.0),
        },
    )

    # 偏航角速度跟踪课程
    ang_vel_cmd = CurrTerm(
        func=mdp.command_levels_ang_vel,
        params={
            # 必须与 RewardsCfg 中角速度跟踪奖励项名称一致
            "reward_term_name": "track_ang_vel_z_exp",
            "range_multiplier": (0.1, 1.0),
        },
    )

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