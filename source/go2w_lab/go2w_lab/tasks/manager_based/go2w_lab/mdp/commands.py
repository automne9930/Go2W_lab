"""
===========================================================================
自定义指令项：修正IsaacLab官方在速度跟踪 metrics 上的问题

官方 ``UniformVelocityCommand._update_metrics`` 会把每步误差除以
``resampling_time_range[1] / step_dt`` 后累加，但 metrics 只在 episode reset
时才会被 flush，命令在回合中途重采样并不会清空累加器。于是：

    metric = Σ_step error / 一个重采样窗口的步数
           = 回合平均误差 × (episode_length_s / resampling_time_max)

指标会随重采样周期变短而成比例放大（20s -> 10s 时翻倍），
也会因为提前摔倒 reset 而偏小。

这里改为累加未归一化的原始误差，并在 reset 时除以本回合实际累计的
控制步数，得到与 episode 长度、重采样周期都无关的真实平均跟踪误差。
===========================================================================
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class Go2W_UniformVelocityCommand(UniformVelocityCommand):
    """与官方实现完全一致，只把 metrics 的分母改成“本回合实际累计控制步数”。"""

    def __init__(self, cfg: UniformVelocityCommandCfg, env):
        super().__init__(cfg, env)
        # 每个环境自 episode reset 以来累计的 metric 采样步数
        self._metric_steps = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        # 累加原始误差（不除以任何时间窗口）
        self.metrics["error_vel_xy"] += torch.norm(
            self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1
        )
        self.metrics["error_vel_yaw"] += torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        self._metric_steps += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """重置时用实际累计步数求平均，然后清零并重采样。"""
        if env_ids is None:
            env_ids = slice(None)

        # 至少除 1，避免刚初始化/刚 reset 就 flush 时除零
        counts = torch.clamp(self._metric_steps[env_ids], min=1.0)

        # 注意：这里要自己算 mean，不能直接调用 super().reset()，
        # 因为父类会把累加和直接当作均值。
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            extras[metric_name] = torch.mean(metric_value[env_ids] / counts).item()
            metric_value[env_ids] = 0.0
        self._metric_steps[env_ids] = 0.0

        # 以下与父类 reset 保持一致
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return extras


@configclass
class Go2W_UniformVelocityCommandCfg(UniformVelocityCommandCfg):
    """使用修正版 metrics 的速度指令配置。"""

    class_type: type = Go2W_UniformVelocityCommand
