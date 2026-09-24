# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2FlatPPORunnerCfg

@configclass
class UnitreeGo2FlatOmniPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_flat_omni"      # own log folder
        self.max_iterations = 1500                   # omni is harder than plain flat


@configclass
class UnitreeGo2FlatPayloadPPORunnerCfg(UnitreeGo2FlatOmniPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_flat_payload"


