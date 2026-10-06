# Go2W-速度目标-平坦地面训练环境（作为base训练存在）

# import导入
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

# 资产导入
from go2w_lab.assets import UNITREE_GO2W_CFG


#=================================
# Scene设置
#=================================
@configclass
class Go2W_VelocityFlatSceneCfg(InteractiveSceneCfg):
    """Go2W 速度跟踪任务的平坦地面场景"""

    # 地形设置
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # 机器人设置
    robot: ArticulationCfg = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 传感器设置
    # 1、周围地形-高度雷达
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # 2、机身高度雷达
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # 3、力传感器
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", # 传感器挂载在robot的各个link中
                                      history_length=3,            # 保存多少历史帧的接触数据
                                      track_air_time=True)         # 是否记录脚离开地面的时间和接触持续时间,在下面的reward的步态约束有使用到

    # 灯光
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

#=================================
# Observation设置
#=================================
@configclass
class Go2W_ObservationsCfg:
    """Go2W 轮足机器人观测空间配置"""

    # -------------------------------------------------------------------------
    # 1. Policy 观测组：真机本体感知（仅包含实机可获得的物理量）
    # -------------------------------------------------------------------------
    @configclass
    class PolicyCfg(ObsGroup):
        """供策略网络Actor使用-输出决定动作"""

        # (1) 机身角速度 (来自 IMU 陀螺仪，3维)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,    # mdp计算函数
            scale=0.2,    # 量纲缩放系数
            noise=Unoise(n_min=-0.2, n_max=0.2),    # 噪声注入（缩放前在角速度上叠加 [-0.2, 0.2] rad/s 的均匀白噪声）
        )

        # (2) 重力投影向量 (来自 IMU 姿态解算，3维)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # (3) 速度指令 (手柄/上位机目标速度: vx, vy, wz，3维)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # (4) 腿部关节相对位置 (12维: 髋、大腿、小腿)
        joint_pos_legs = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!.*_foot_joint).*"])},
            scale=1.0,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # (5) 腿部关节角速度 (12维)
        joint_vel_legs = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!.*_foot_joint).*"])},
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

        # (6) 轮子转速 (4维: 针对轮足的核心特征，轮子不能读位置，只读速度)
        wheel_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_joint"])},
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

        # (7) 动作历史缓存 (上一控制周期的动作输出，16维: 12腿 + 4轮)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True   # 开启噪声注入（Sim2Real 域随机化）
            self.concatenate_terms = True   # 将上述所有特征自动拼接为一个 1D Tensor

    # -------------------------------------------------------------------------
    # 2. Critic 观测组：特权观测（享受仿真“上帝视角”，训练后丢弃）
    # -------------------------------------------------------------------------
    @configclass
    class CriticCfg(ObsGroup):
        """供价值评估网络Critic使用-提供准确的状态价值评估"""

        # 包含 Policy 组的所有感知量（Critic 也需要掌握策略可见的状态）
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_legs = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!.*_foot_joint).*"])},
            scale=1.0,
        )
        joint_vel_legs = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!.*_foot_joint).*"])},
            scale=0.05,
        )
        wheel_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_joint"])},
            scale=0.05,
        )
        last_action = ObsTerm(func=mdp.last_action)

        # 【特权项 A】：机身真实线速度 (真机无高频精准传感器，仿真中直接取物理引擎真值，3维)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        # 【特权项 B】：前方地形高程扫描图 (来自之前配置的 height_scanner 传感器，187维)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        
        # sensors里还有一个雷达+一个力传感器，不过不在Observation里加入，而是作为后续Rewarsd打分使用

        def __post_init__(self) -> None:
            self.enable_corruption = False  # Critic 评估要求纯净真值，不注入噪声
            self.concatenate_terms = True

    # -------------------------------------------------------------------------
    # 3. 实例化挂载（必须由 ObservationManager 检索）
    # -------------------------------------------------------------------------
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()