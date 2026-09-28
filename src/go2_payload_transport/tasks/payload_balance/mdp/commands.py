"""Commands for the S-curve velocity pattern."""

from __future__ import annotations

from collections.abc import Sequence
import isaaclab.utils.math as math_utils

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils.configclass import configclass


class S_CurveVelocityCommand(UniformVelocityCommand):
    """ 
    Flipping the velocity command between two phases to create a S-curve pattern.
    """
    cfg: S_CurveVelocityCommandCfg

    def __init__(self, cfg: S_CurveVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.turn_sign = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.asset_name = "robot"

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # turn the sign of the turn speed every time step
        phase = self.command_counter[env_ids] % 2
        self.vel_command_b[env_ids, 0] = 1.0
        self.vel_command_b[env_ids, 1] = 0.0
        self.vel_command_b[env_ids, 2] = self.cfg.turn_speed * (1 - 2 * phase)


class ZigZagVelocityCommand(UniformVelocityCommand):
    """ 
    Flipping the velocity command between two phases to create a zig-zag pattern.
    """
    cfg: ZigZagVelocityCommandCfg

    def __init__(self, cfg: ZigZagVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.turn_sign = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.asset_name = "robot"

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        phase = self.command_counter[env_ids] % 2
        if self.cfg.heading_command:
            self.vel_command_b[env_ids, 0] = 1.0
            self.vel_command_b[env_ids, 1] = 0.0
            self.vel_command_b[env_ids, 2] = 0.0
            self.heading_target[env_ids] = (1 - 2 * phase) * self.cfg.heading_amplitude

    def _update_command(self):
        super()._update_command()
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.robot.data.heading_w.torch[env_ids]
            )
            turning = heading_error.abs() > 0.2
            self.vel_command_b[:, 0] = torch.where(turning, 0.3, 1.0)
        

@configclass
class S_CurveVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the S-curve velocity command."""

    class_type: type[S_CurveVelocityCommand] = S_CurveVelocityCommand

    heading_amplitude: float = 1
    """Amplitude of the heading command in rad."""

    turn_speed: float = 1
    """Speed of the turn in rad/s."""


@configclass
class ZigZagVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the zig-zag velocity command."""

    class_type: type[ZigZagVelocityCommand] = ZigZagVelocityCommand

    heading_amplitude: float = 0.5
    """Amplitude of the heading command in rad."""

    turn_speed: float = 1
    """Speed of the turn in rad/s."""

    resampling_time_range: tuple[float, float] = (3.0, 3.0)
    """Range of the resampling time in seconds."""