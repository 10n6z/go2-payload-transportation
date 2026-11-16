from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

import torch

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import sample_uniform, quat_apply
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.core.velocity.velocity_env_cfg import MySceneCfg
from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import EventsCfg

import isaaclab_tasks.core.velocity.mdp as mdp

PLATFORM_SIZE = (0.23, 0.2, 0.02)
PLATFORM_MASS = 0.005
PLATFORM_POSITION = (-0.015, 0.0, 0.065)
PLATFORM_ROTATION = (0.0, 0.0, 0.0, 1.0)


def _spawn_fixed_platform(
    # Reference: isaaclab_tasks.contrib.ur10_particle_push.ur10_particle_push_env_cfg.py
    prim_path: str,
    cfg: sim_utils.CuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a fixed platform welded to its parent rigid body (base part of the Go2 robot)."""
    # Keep both the USD imports and clone-wrapper resolution behind SimulationApp startup.
    # Importing either at config-parse time loads the kit-less ``usd-core`` bindings and
    # corrupts Kit's own USD binding registration.
    from pxr import Gf, Sdf, UsdPhysics  # noqa: PLC0415

    @sim_utils.clone
    def _spawn_one(
        prim_path: str,
        cfg: sim_utils.CuboidCfg,
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs,
    ) -> Usd.Prim:
        # The camera pole reuses this welding spawner with a cylinder.
        spawn_shape = sim_utils.spawn_cylinder if isinstance(cfg, sim_utils.CylinderCfg) else sim_utils.spawn_cuboid
        platform_prim = spawn_shape(
            prim_path,
            cfg,
            translation=translation,
            orientation=orientation,
            **kwargs,
        )
        stage = sim_utils.get_current_stage()
        parent_prim = platform_prim.GetParent()
        if not parent_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Platform parent '{parent_prim.GetPath()}' is not a rigid body.")

        local_position = translation if translation is not None else (0.0, 0.0, 0.0)
        local_rotation = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
        joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/FixedJoint")
        joint.CreateBody0Rel().SetTargets([parent_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(prim_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_position))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(local_rotation[3], Gf.Vec3f(*local_rotation[:3])))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        return platform_prim

    return _spawn_one(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)


def _spawn_randomized_payload(
    # reference: isaaclab_tasks.core.lift.mdp.events.py
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    position_range: dict[str, tuple[float, float]],
    spawn_height: float = 0.155,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
) -> None:
    """Spawn a payload and its position relative to the robot's base"""
    robot: Articulation = env.scene[robot_cfg.name]
    payload: RigidObject = env.scene[payload_cfg.name]
    # reset root state
    ranges = torch.tensor([position_range.get(key, (0.0, 0.0))
                          for key in ("x", "y", "z")], device=robot.device)
    offset = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3),
                            device=robot.device)
    offset[:, 2] = spawn_height

    root_pose = robot.data.root_link_pose_w.torch[env_ids].clone()
    root_pose[:, :3] += quat_apply(robot.data.root_quat_w.torch[env_ids], offset)

    payload.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    payload.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros_like(
            payload.data.default_root_vel.torch[env_ids]), env_ids=env_ids
    )


PLATFORM_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/Platform",
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=PLATFORM_POSITION, rot=PLATFORM_ROTATION),
    spawn=sim_utils.CuboidCfg(
        func=_spawn_fixed_platform,
        size=PLATFORM_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=PLATFORM_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
            rest_offset=0.0,
        ),
        physics_material=RigidBodyMaterialBaseCfg(
            static_friction=0.8,
            dynamic_friction=0.7,
            restitution=0.0,
            density=None,
        ),
        visual_material=PreviewSurfaceCfg(
            diffuse_color=(1.0, 1.0, 1.0),
            emissive_color=(0.0, 0.0, 0.0),
            roughness=0.0,
            metallic=0.0,
            opacity=1.0,
        ),
    ),
)


PAYLOAD_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Payload",
    spawn=sim_utils.CylinderCfg(
        radius=0.04,
        height=0.04,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.07,
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
        physics_material=RigidBodyMaterialBaseCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
            density=None,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.455), rot=(0.0, 0.0, 0.0, 1.0)),
)


@configclass
class PayloadRandomizeEventCfg(EventsCfg):
    "Configuration for randomizing the payload"
    payload_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("payload"),
            "static_friction_range": (1.0, 1.2),
            "dynamic_friction_range": (1.0, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 50,
        }
    )

    payload_abs_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("payload"),
            "mass_distribution_params": (0.7, 0.7),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        }
    )

    reset_payload = EventTerm(
        func=_spawn_randomized_payload,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "payload_cfg": SceneEntityCfg("payload"),
            "position_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        },
    )


@configclass
class SceneEntityCfg(MySceneCfg):
    platform: AssetBaseCfg = PLATFORM_CFG
    payload: RigidObjectCfg = PAYLOAD_CFG


@configclass
class UnitreeGo2RandomPayloadMassEnvCfg(UnitreeGo2FlatEnvCfg):
    scene: SceneEntityCfg = SceneEntityCfg(num_envs=4096, env_spacing=2.5)
    events: PayloadRandomizeEventCfg = PayloadRandomizeEventCfg()

    def __post_init__(self):
        super().__post_init__()

        # self.events.reset_base.params["pose_range"] = {"yaw": (0.0, 0.0)}
        # self.events.reset_base.params["velocity_range"] = {}
