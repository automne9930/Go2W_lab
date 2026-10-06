# Go2W-速度目标-复杂地形训练环境

# configclass配置
from isaaclab.utils import configclass
# 导入已有的平坦地形配置
from .go2w_velocity_flat_cfg.py import Go2W_VelocityFlat_ManagerBasedEnv


#=================================
# 总环境设置（继承平坦地形的基础上进行修改）
#=================================
@configclass
class Go2W_VelocityRough_ManagerBasedEnv(Go2W_VelocityFlat_ManagerBasedEnv):
    """复杂地形运动速度跟踪强化学习环境总配置"""

    def __post_init__(self):
        # 运行父类的post_init
        super().__post_init__()