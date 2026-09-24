# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from isaaclab.utils.configclass import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg


@configclass
class UnitreeGo2FlatOmniEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- velocity ---
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # --- actions ---
        self.actions.joint_pos.scale = 0.25
        # --- rewards ---
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        # self.rewards.lin_vel_z_l2.weight = -0.0002
        # self.rewards.ang_vel_xy_l2.weight = -0.0002
        # self.rewards.dof_torques_l2.weight = -0.0002
        # self.rewards.dof_acc_l2.weight = -2.5e-7
        # self.rewards.action_rate_l2.weight = -0.0002
        # self.rewards.feet_air_time.weight = 0.01
        # self.rewards.undesired_contacts = None


@configclass
class UnitreeGo2FlatPayloadEnvCfg(UnitreeGo2FlatOmniEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- replace base mass with payload mass ---
        # self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 6.0)
        # self.events.add_base_mass.params["operation"] = "add"
        # self.events.add_base_mass.params["distribution"] = "uniform"

        # --- add payload mass ---
        self.events.add_payload_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": (0.0, 6.0),
                "operation": "add",
                "distribution": "uniform",
            },
        )


@configclass
class UnitreeGo2CircleEnvCfg(UnitreeGo2FlatPayloadEnvCfg):
    """Evaluation only: one fixed circle, so runs are comparable."""

    def __post_init__(self):
        super().__post_init__()
        self._set_circle_command()

    def play_mode(self):
        super().play_mode()      # the parent may reopen the command range
        self._set_circle_command()   # so pin it again, last

    def _set_circle_command(self):
        speed, radius = 1.0, 1.0
        self.commands.base_velocity.ranges.lin_vel_x = (speed, speed)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (speed / radius, speed / radius)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)  # never resample


@configclass
class UnitreeGo2ZigZagEnvCfg(UnitreeGo2FlatPayloadEnvCfg):
    """Evaluation only: one fixed zig-zag path, so runs are comparable."""

    def __post_init__(self):
        super().__post_init__()
        self._set_zigzag_command()

    def play_mode(self):
        super().play_mode()      # the parent may reopen the command range
        self._set_zigzag_command()   # so pin it again, last

    def _set_zigzag_command(self):
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_heading_envs = 1.0                          # every robot uses heading_command
        self.commands.base_velocity.rel_standing_envs = 0.0                         # nothing standing
        self.commands.base_velocity.heading_control_stiffness = 0.5                 # wz = 0.5 * heading_error
        self.commands.base_velocity.ranges.heading = (-0.5, 0.5)                    # weave +- 0.5 radians
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)                   # constant forward velocity
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)                   # no sideway motion
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)                  # clip limits applied to yaw rate
        self.commands.base_velocity.resampling_time_range = (2.0, 2.0)              # resample every 2 seconds
        self.episode_length_s = 10.0
        self.commands.base_velocity.debug_vis = True