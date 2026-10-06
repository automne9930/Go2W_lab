# Go2W-速度目标-平坦地面训练环境（作为base训练存在）

# import导入
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

# Command所需
from isaaclab.envs.mdp import commands  # 导入官方预置的 Command 生成项实现
from isaaclab.managers import CommandTermCfg as CmdTerm
# Observation所需
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# Action所需
from isaaclab.envs.mdp import actions  # 导入官方预置的 Action 项实现
from isaaclab.managers import ActionTermCfg as ActTerm
# Reward所需
from isaaclab.envs.mdp import rewards  # 官方预置的 Reward 计算函数库
from isaaclab.managers import RewardTermCfg as RewTerm
# Termination所需
from isaaclab.envs.mdp import terminations  # 导入官方预置的终止判定函数
from isaaclab.managers import TerminationTermCfg as DoneTerm
# Event所需
from isaaclab.envs.mdp import events  # 导入官方预置的事件函数库
from isaaclab.managers import EventTermCfg as EventTerm
# Curriculum所需
from isaaclab.envs.mdp import curriculum  # 导入官方预置的课程学习逻辑
from isaaclab.managers import CurriculumTermCfg as CurrTerm
# ManageBased环境配置所需
from isaaclab.envs import ManagerBasedRLEnvCfg


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
# Command设置
#=================================
@configclass
class Go2W_CommandsCfg:
    """指令空间总配置"""

    # -------------------------------------------------------------------------
    # 机身底盘速度跟踪指令：模拟手柄遥控机器人的 (vx, vy, wz)
    # -------------------------------------------------------------------------
    base_velocity = CmdTerm(
        class_type=commands.UniformVelocityCommandCfg,
        # 绑定的目标资产实体（指示该速度指令以哪个刚体的局部坐标系为基准）
        asset_name="robot",
        # 重采样周期范围：每隔 10.0 到 10.0 秒（即固定 10 秒）为机器人更换一次新指令
        resampling_time_range=(10.0, 10.0),
        # 航向角度控制设置
        heading_command = True,     # True表示机器人随机下发的是yaw角度指令，False表示下发yaw角速度
        heading_control_stiffness=0.5,    # yaw角度到yaw角速度的P控制器，角度的底层还是通过速度控制
        # 具体的采样物理范围字典
        ranges=commands.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),  # 期望前进/后退速度范围 (m/s)
            lin_vel_y=(-0.5, 0.5),  # 期望侧向平移速度范围 (m/s，轮足通常横向受限，范围略小)
            ang_vel_z=(-1.0, 1.0),  # 期望原地自转角速度范围 (rad/s)，如果heading_command = True，则本项作为P控制器输出后的限幅存在
            heading=(-3.14, 3.14),  # 期望航向朝向角范围 (rad)
        ),
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
            params={"command_name": "base_velocity"},   # 通过CmdTerm的名称相联系
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

#=================================
# Action设置
#=================================
@configclass
class Go2W_ActionsCfg:
    """动作空间总配置"""

    # -------------------------------------------------------------------------
    # 1. 腿部关节动作项：关节位置控制 
    # -------------------------------------------------------------------------
    joint_pos = ActTerm(
        class_type=actions.JointPositionActionCfg,
        asset_name="robot",
        # 通过正则过滤出 12 个腿部关节（排除 4 个轮子关节）
        joint_names=["^(?!.*_foot_joint).*"],
        # 动作缩放系数：28.6 度幅度，挑战动作更大一些
        scale=0.5,
        # 是否基于机器人默认姿态进行增量叠加（True 表示输出的是相对默认姿态的偏差）
        use_default_offset=True,
    )

    # -------------------------------------------------------------------------
    # 2. 轮子关节动作项：轮速控制 
    # -------------------------------------------------------------------------
    joint_vel = ActTerm(
        class_type=actions.JointVelocityActionCfg,
        asset_name="robot",
        # 通过正则精准锁定 4 个轮子连续旋转关节
        joint_names=[".*_foot_joint"],
        # 动作缩放系数：满速20.0 rad/s，给大一些速度
        scale=20.0,
        # 轮子没有默认“默认角度”，直接控制目标转速，故不叠加默认偏置
        use_default_offset=False,
    )


#=================================
# Reward设置
#=================================
@configclass
class Go2W_VelocityFlatRewardsCfg:
    """Go2W平地速度追踪-奖励空间总配置"""

    # -------------------------------------------------------------------------
    # 1. 核心任务奖励 (Task Tracking Rewards) - 正权重
    # -------------------------------------------------------------------------

    # (1) 平面水平线速度跟踪奖励 (XY 平面 vx, vy 跟踪)
    track_lin_vel_xy_exp = RewTerm(
        func=rewards.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # (2) 航向角速度跟踪奖励 (绕 Z 轴 wz 跟踪)
    track_ang_vel_z_exp = RewTerm(
        func=rewards.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # -------------------------------------------------------------------------
    # 2. 姿态与稳定性约束 (Base Stability) - 负权重惩罚
    # -------------------------------------------------------------------------

    # (3) 抑制竖直方向颠簸：惩罚机身沿 Z 轴的垂直线速度 (防止机器人乱蹦乱跳)
    lin_vel_z_l2 = RewTerm(
        func=rewards.lin_vel_z_l2,
        weight=-2.0,
    )

    # (4) 抑制翻滚与俯仰震荡：惩罚机身沿 XY 轴的角速度 (防止狗剧烈摇头晃脑、左右摇摆)
    ang_vel_xy_l2 = RewTerm(
        func=rewards.ang_vel_xy_l2,
        weight=-0.05,
    )

    # (5) 机身姿态调平：惩罚非垂直方向的重力投影误差 (保持机身水平)
    flat_orientation_l2 = RewTerm(
        func=rewards.flat_orientation_l2,
        weight=-1.0,
    )

    # (6) 违规触地惩罚：机身底盘、大腿磕碰地面时扣大分 (利用传感器实现防摔)
    undesired_contacts = RewTerm(
        func=rewards.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_thigh"]),    # 专门筛选出大腿和base基座（禁止触地）
            "threshold": 1.0,
        },
    )

    # -------------------------------------------------------------------------
    # 3. 硬件寿命与平滑度正则化 (Regularization & Safety) - 负权重惩罚
    # -------------------------------------------------------------------------

    # (7) 动作平滑惩罚：惩罚相邻控制步之间的动作跳变 (抑制电机高频抖动，防止真机炸机)
    action_rate_l2 = RewTerm(
        func=rewards.action_rate_l2,
        weight=-0.01,
    )

    # (8) 电机扭矩惩罚：惩罚过大的关节输出力矩 (降低能耗与电机发热)
    dof_torques_l2 = RewTerm(
        func=rewards.joint_torques_l2,
        weight=-1.0e-5,
    )

    # (9) 关节限位惩罚：惩罚关节角度接近机械极限 (防止打到硬件硬限位)
    dof_pos_limits = RewTerm(
        func=rewards.joint_pos_limits,
        weight=-1.0,
    )

#=================================
# Termination设置
#=================================
@configclass
class Go2W_TerminationsCfg:
    """终止条件总配置"""

    # -------------------------------------------------------------------------
    # 1. 超时终止 (Truncation): 回合步数到达上限
    # -------------------------------------------------------------------------
    time_out = DoneTerm(
        func=terminations.time_out,
        time_out=True,  # 【关键】：标记为 Truncation，算法底层据此开启价值自举 (Bootstrapping)
    )

    # -------------------------------------------------------------------------
    # 2. 违规接触/摔倒终止 (Termination): 底盘或躯干砸地
    # -------------------------------------------------------------------------
    illegal_contact = DoneTerm(
        func=terminations.illegal_contact,
        params={
            # 关联场景中之前配置的接触力传感器，监听 base(机身) 和 thigh(大腿)
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_thigh"]),
            # 撞击力阈值 (牛顿)：只要检测到受力大于 1.0 N，立即判死重置
            "threshold": 1.0,
        },
    )

    # -------------------------------------------------------------------------
    # 3. 姿态倾覆终止 (Termination): 机身严重侧翻或底朝天
    # -------------------------------------------------------------------------
    bad_orientation = DoneTerm(
        func=terminations.bad_orientation,
        params={
            # 重力投影 Z 轴阈值：标准水平站立时重力在机身 Z 轴投影约为 -1.0
            # 当该值大于 -0.2 时，说明机身仰角/侧倾角已超过约 78°，判定为翻车
            "limit_angle": 1.36,  # 弧度 (约 78 度)
        },
    )

    # -------------------------------------------------------------------------
    # 4. 跌落深渊/走出边界终止 (Termination)
    # -------------------------------------------------------------------------
    root_height_below_minimum = DoneTerm(
        func=terminations.root_height_below_minimum,
        params={
            # 当机器狗掉落高度低于 -0.5 米（从高台或崎岖悬崖跌落）时直接重置
            "minimum_height": -0.5,
        },
    )

#=================================
# Event设置
#=================================
@configclass
class Go2W_EventCfg:
    """事件与域随机化总配置"""

    # -------------------------------------------------------------------------
    # 1. 启动期随机化 (Startup Mode) - 训练前只执行一次
    # -------------------------------------------------------------------------

    # (1) 机身附加质量随机化：模拟不同传感器挂载重量 (例如加装 0 ~ 3 kg 载荷)
    add_base_mass = EventTerm(
        func=events.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),  # 允许机身质量浮动 -1kg 到 +3kg
            "operation": "add",
        },
    )

    # -------------------------------------------------------------------------
    # 2. 回合重置事件 (Reset Mode) - 每次环境重置时触发
    # -------------------------------------------------------------------------

    # (2) 重置机器人根节点状态：放置回地面，并叠加小范围的随机位置与偏航朝向
    reset_base = EventTerm(
        func=events.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),  # 出生时随机朝向
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # (3) 重置关节状态：在机器人标称默认姿态基础上，随机偏移小角度
    reset_robot_joints = EventTerm(
        func=events.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),  # 关节角度在标称姿态的 0.5 ~ 1.5 倍之间波动
            "velocity_range": (0.0, 0.0),  # 重置时初始关节速度清零
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # (4) 物理摩擦力随机化：每次重置时随机分配地表静态摩擦力 (0.4 ~ 1.25)
    physics_material = EventTerm(
        func=events.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.4, 1.25),   # 模拟瓷砖地到粗糙水泥地
            "dynamic_friction_range": (0.4, 1.25),
            "restitution_range": (0.0, 0.0),        # 弹性恢复系数
            "num_buckets": 64,                      # 划分 64 个材质桶提升并行性能
        },
    )

    # -------------------------------------------------------------------------
    # 3. 运行中扰动事件 (Interval Mode) - 训练过程中周期性突发
    # -------------------------------------------------------------------------

    # (5) 突发外力冲击 (踢狗机制)：每隔 8 秒随机给机身线速度注入突变，检验抗跌倒能力
    push_robot = EventTerm(
        func=events.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 10.0),  # 每隔 6 ~ 10 秒随机挑一个时刻
        params={
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},  # 突加 +/-1.0 m/s 的横向/纵向冲量
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


#=================================
# Curriculum设置
#=================================
@configclass
class Go2W_VelocityFlatCurriculumCfg:
    """Go2W-速度平地-课程学习总配置"""

    # -------------------------------------------------------------------------
    # 1. 地形难度课程：根据速度跟踪表现动态迁移地形难度等级 (0 ~ max_level)
    # -------------------------------------------------------------------------
    # 平地训练无地形难度配置
    # terrain_levels = CurrTerm(
    #     func=curriculum.terrain_levels_vel,
    # )

    # -------------------------------------------------------------------------
    # 2. 线速度范围扩容课程：当水平速度跟踪打分优秀时，按比例放大线速度采样区间
    # -------------------------------------------------------------------------
    command_levels_lin_vel = CurrTerm(
        func=curriculum.command_levels_lin_vel,
        params={
            # 考核的奖励项名称：直接关联 RewardsCfg 里的 track_lin_vel_xy_exp
            "reward_term_name": "track_lin_vel_xy_exp",
            # 指令范围缩放倍率：从原始 Ranges 的 0.1 倍逐步渐进扩容到 1.0 倍
            "range_multiplier": (0.1, 1.0),
        },
    )

    # -------------------------------------------------------------------------
    # 3. 角速度范围扩容课程：当转向角速度跟踪打分达标时，逐步放宽自转速度指令上限
    # -------------------------------------------------------------------------
    command_levels_ang_vel = CurrTerm(
        func=curriculum.command_levels_ang_vel,
        params={
            "reward_term_name": "track_ang_vel_z_exp",
            "range_multiplier": (0.1, 1.0),
        },
    )
    
#=================================
# 总环境设置（采用robotlab同款配置）
#=================================
@configclass
class Go2W_VelocityFlat_ManagerBasedEnv(ManagerBasedRLEnvCfg):
    """平地地形运动速度跟踪强化学习环境总配置"""

    # --- 场景与资产配置 ---
    # 实例化场景：GPU 并行 4096 个子环境，环境间距 2.5 米
    scene: Go2W_VelocityFlatSceneCfg = Go2W_VelocityFlatSceneCfg(num_envs=4096, env_spacing=2.5)

    # --- 基础交互接口配置 ---
    observations: Go2W_ObservationsCfg = Go2W_ObservationsCfg()                 # 观测空间 (Policy/Critic 输入)
    actions: Go2W_ActionsCfg = Go2W_ActionsCfg()                                # 动作空间 (网络输出到执行器的映射)
    commands: Go2W_CommandsCfg = Go2W_CommandsCfg()                             # 指令生成器 (目标速度/航向等任务输入)

    # --- MDP 强化学习核心规则 ---
    rewards: Go2W_VelocityFlatRewardsCfg = Go2W_VelocityFlatRewardsCfg()        # 奖励函数 (跟踪打分与正则化惩罚)
    terminations: Go2W_TerminationsCfg = Go2W_TerminationsCfg()                 # 回合终止条件 (摔倒判死与超时截断)
    events: Go2W_EventCfg = Go2W_EventCfg()                                     # 事件与域随机化 (初始位姿重置、推力扰动等)
    curriculum: Go2W_VelocityFlatCurriculumCfg = Go2W_VelocityFlatCurriculumCfg()        # 课程学习 (地形与指令难度动态递进)

    def __post_init__(self):
        """配置实例化后的派生参数校准与跨模块联锁"""
        # 通用控制时序设置
        self.decimation = 4           # 控制降频比: 物理步走 4 次，策略网络推理 1 次 (50 Hz 决策)
        self.episode_length_s = 20.0  # 单回合最大物理时长 20 秒 (20s / 0.02s = 1000 步截断)

        # 物理引擎底层参数
        self.sim.dt = 0.005           # PhysX 物理仿真步长 (200 Hz 计算刚体与接触)
        self.sim.render_interval = self.decimation  # 渲染频率与策略决策周期对齐 (50 Hz 渲染)
        self.sim.physics_material = self.scene.terrain.physics_material  # 全局绑定地形物理材质
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15            # 扩容 GPU 接触补丁显存池，防止复杂碰撞溢出

        # 传感器采样频率分流更新
        if self.scene.height_scanner is not None:
            # 几何高程射线开销大，按策略决策步降频更新 (50 Hz)
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            # 碰撞冲击属瞬态信号，紧随物理引擎高频步步监听 (200 Hz)
            self.scene.contact_forces.update_period = self.sim.dt

        # 课程学习与地形生成器开关互锁
        # 若配置了地形课程项，自动激活地形生成器的等级递进模式；反之则随机分布
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    def disable_zero_weight_rewards(self):
        """自剪枝方法：自动将权重为 0 的奖励项置为 None，避免浪费 GPU 计算资源"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                # 过滤掉非函数项且权重严格为 0 的奖励项
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)