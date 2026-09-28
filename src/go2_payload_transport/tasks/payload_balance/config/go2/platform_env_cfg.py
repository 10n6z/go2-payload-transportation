from isaaclab.sim.schemas import RigidBodyBaseCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas import MassPropertiesCfg
from isaaclab.sim.schemas import CollisionBaseCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.core.velocity.velocity_env_cfg import MySceneCfg
from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import EventsCfg

import isaaclab_tasks.core.velocity.mdp as mdp


PLATFORM_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/object",
    spawn=sim_utils.CuboidCfg(
        size=(0.23, 0.2, 0.03),
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.03,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
            rest_offset=0.0,
        ),
        visual_material=PreviewSurfaceCfg(
            diffuse_color=(1.0, 1.0, 1.0),
            emissive_color=(0.0, 0.0, 0.0),
            roughness=0.0,
            metallic=0.0,
            opacity=1.0,
        ),
        physics_material = RigidBodyMaterialBaseCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
            density=None,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.475), rot=(0.0, 0.0, 0.0, 1.0)),
)




@configclass
class PlatformRandomizeEventCfg(EventsCfg):
    "Configuration for randomizing the platform mass"
    platform_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("platform"),
            "static_friction_range": (1.0, 1.2),
            "dynamic_friction_range": (1.0, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 50,
        }
    )

    # platform_abs_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("platform"),
    #         "mass_distribution_params": (0.5, 0.5),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #         "recompute_inertia": True,
    #     }
    # )

    # platform_abs_inertia = EventTerm(
    #     func=mdp.randomize_rigid_body_inertia,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("plaform"),
    #         "inertia_distribution_params": (0.5, 0.5),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     }
    # )

    # platform_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("plaform"),
    #         "com_range": (0.0, 0.0),
    #     }
    # )    
    

    reset_platform = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("platform"),
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {},
        },
    )



@configclass
class PlatformSceneEntityCfg(MySceneCfg):
    platform: RigidObjectCfg = PLATFORM_CFG


@configclass
class UnitreeGo2RandomPlatformMassEnvCfg(UnitreeGo2FlatEnvCfg):
    scene: PlatformSceneEntityCfg = PlatformSceneEntityCfg(num_envs=4096, env_spacing=2.5)
    events: PlatformRandomizeEventCfg = PlatformRandomizeEventCfg()

    def __post_init__(self):
        super().__post_init__()

        self.events.reset_base.params["pose_range"] = {"yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {}



