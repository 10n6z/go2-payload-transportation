"""Camera pole + RealSense D435i for the simulated Go2 (Isaac Lab release/3.0.0).

Drop these pieces into go2_payload_transport next to platform_env_cfg.py.
Use ONLY in the test/play env config, never in the 4096-env training config.

Placeholders (replace with real measurements, all in the robot "base" frame,
x forward, y left, z up, metres):
    POLE_X      where the pole stands on the back (behind the platform)
    POLE_HEIGHT pole length above the base top
    CAM_TILT    0 = camera looks straight down; >0 tilts the lens toward the front
"""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

from .platform_env_cfg import SceneEntityCfg as PayloadSceneCfg
from .platform_env_cfg import (
    PayloadRandomizeEventCfg,
    UnitreeGo2RandomPayloadMassEnvCfg,
    _spawn_fixed_platform,
)

# ---- placeholders -------------------------------------------------------------
POLE_X = 0.10          # [m] rear end of the back
POLE_Y = 0.0            # [m]
BASE_TOP_Z = 0.055      # [m] platform underside sits ~here (platform center z = 0.065)
POLE_HEIGHT = 0.40      # [m] memory note: camera ~40 cm above the platform
POLE_RADIUS = 0.01      # [m]
POLE_MASS = 0.15        # [kg] pole + screws
CAM_MASS = 0.072        # [kg] D435i datasheet mass
CAM_FORWARD = -0.02      # [m] horizontal arm so the lens sits over the platform center (x = -0.015)
CAM_DOWN = -0.05         # [m] cam attached to the top of the pole not on top
CAM_TILT_DEG = 0.0      # [deg]


def _quat_xyzw_from_rotmat(r):
    """Rotation matrix (3x3 nested lists) -> quaternion (x, y, z, w)."""
    tr = r[0][0] + r[1][1] + r[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s, 0.25 * s)
    i = max(range(3), key=lambda k: r[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + r[i][i] - r[j][j] - r[k][k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (r[j][i] + r[i][j]) / s
    q[k] = (r[k][i] + r[i][k]) / s
    q[3] = (r[k][j] - r[j][k]) / s
    return tuple(q)


def camera_quat_ros(tilt_deg: float) -> tuple[float, float, float, float]:
    """Camera orientation in ROS convention (z = viewing direction, y = image down).

    tilt 0: looks straight down, top of the image = robot forward.
    """
    t = math.radians(tilt_deg)
    z = (math.sin(t), 0.0, -math.cos(t))      # viewing direction in base frame
    y = (-math.cos(t), 0.0, -math.sin(t))     # image "down" points backward
    x = (y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0])
    r = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
    return _quat_xyzw_from_rotmat(r)


# ---- 1) the pole: reuse your platform's fixed-joint spawner ------------------
# Same pattern as the platform in platform_env_cfg.py: a rigid body welded to
# "{ENV_REGEX_NS}/Robot/base" by UsdPhysics.FixedJoint. Only geometry/mass change.
POLE_SPAWN = sim_utils.CylinderCfg(
    radius=POLE_RADIUS,
    height=POLE_HEIGHT,
    axis="Z",
    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    mass_props=sim_utils.MassPropertiesCfg(mass=POLE_MASS + CAM_MASS),  # camera mass lumped in
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
)
POLE_POS_IN_BASE = (POLE_X, POLE_Y, BASE_TOP_Z + POLE_HEIGHT / 2)  # cylinder origin = its center
# -> pass POLE_SPAWN / POLE_POS_IN_BASE to your welding spawner exactly like the platform.

# ---- 2) the camera: a sensor attached to the base with an offset -------------
# Pole is welded, so "pole top" and "base + offset" move identically; attaching
# to the base avoids an extra prim lookup.
CAM_POS_IN_BASE = (POLE_X + CAM_FORWARD, POLE_Y, BASE_TOP_Z + POLE_HEIGHT + CAM_DOWN)

D435I_CAMERA = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/d435i",
    update_period=1 / 30,                       # 30 fps like the real camera
    height=480,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],  # = RealSense z-depth
    # Kitless Newton renderer: the default (Isaac RTX) needs Isaac Sim / omni.usd.
    renderer_cfg=NewtonWarpRendererCfg(),
    # D435i color @640x480: fx = fy ~ 615 px, cx ~ 320, cy ~ 240 (check your rs intrinsics)
    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=[615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0],
        width=640,
        height=480,
        clipping_range=(0.1, 10.0),             # D435i min depth ~0.1-0.28 m
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=CAM_POS_IN_BASE,
        rot=camera_quat_ros(CAM_TILT_DEG),      # (x, y, z, w) in Isaac Lab 3.0
        convention="ros",
    ),
)


# ---- 3) a play-only env config -------------------------------------------------
POLE_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/Pole",
    init_state=AssetBaseCfg.InitialStateCfg(pos=POLE_POS_IN_BASE),
    spawn=POLE_SPAWN.replace(func=_spawn_fixed_platform),
)

# Visible stand-in for the D435i housing (90 x 25 x 25 mm); the sensor prim itself has no geometry.
# Sits just above the lens point so the camera never sees its own body. Mass is already lumped in the pole.
# ponytail: box stays level, ignores CAM_TILT_DEG; rotate it too if the tilt becomes non-zero.
CAM_BODY_SIZE = (0.025, 0.09, 0.025)
CAM_BODY_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/CameraBody",
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(CAM_POS_IN_BASE[0], CAM_POS_IN_BASE[1], CAM_POS_IN_BASE[2] + CAM_BODY_SIZE[2] / 2 + 0.002)
    ),
    spawn=sim_utils.CuboidCfg(
        func=_spawn_fixed_platform,
        size=CAM_BODY_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
    ),
)


@configclass
class PayloadCameraSceneCfg(PayloadSceneCfg):
    pole: AssetBaseCfg = POLE_CFG
    camera_body: AssetBaseCfg = CAM_BODY_CFG
    d435i: TiledCameraCfg = D435I_CAMERA


DEMO_SPEED = 0.5  # [m/s] forward walking speed commanded in the play env
# Viewer camera for the recorded clip, framed for a 5 s walk. Move the eye closer or raise the focal length to zoom in.
VIDEO_EYE = (1.25, -2.2, 0.8)     # [m]
VIDEO_LOOKAT = (1.25, 0.0, 0.3)   # [m]
VIDEO_FOCAL_MM = 14.0
FRAME_DIR = Path("outputs/d435i")
DEPTH_VIS_MAX = 1.0  # [m] depth mapped to white in depth.png


def _save_camera_frame(env, env_ids, sensor_name: str = "d435i") -> None:
    """Overwrite outputs/d435i/{rgb,depth}.png with env 0's latest frame (debug view)."""
    from PIL import Image  # noqa: PLC0415

    out = env.scene[sensor_name].data.output
    rgb, depth = (getattr(out[k], "torch", out[k])[0] for k in ("rgb", "distance_to_image_plane"))
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb[..., :3].cpu().numpy()).save(FRAME_DIR / "rgb.png")
    depth_u8 = (depth[..., 0].clamp(0.0, DEPTH_VIS_MAX) / DEPTH_VIS_MAX * 255).byte()
    Image.fromarray(depth_u8.cpu().numpy()).save(FRAME_DIR / "depth.png")


@configclass
class PayloadCameraEventCfg(PayloadRandomizeEventCfg):
    save_camera_frame = EventTerm(func=_save_camera_frame, mode="interval", interval_range_s=(1.0, 1.0))


@configclass
class UnitreeGo2PayloadCameraEnvCfg_PLAY(UnitreeGo2RandomPayloadMassEnvCfg):
    """Platform + payload + pole + D435i. Play/test only: run with --enable_cameras.

    Read frames: env.unwrapped.scene["d435i"].data.output["rgb"] and
    ["distance_to_image_plane"] (metres, shape [N, 480, 640, 1]).
    """

    scene: PayloadCameraSceneCfg = PayloadCameraSceneCfg(num_envs=4, env_spacing=2.5)
    events: PayloadCameraEventCfg = PayloadCameraEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Demo conditions: payload dead centre on the platform, steady straight walk.
        self.events.reset_payload.params["position_range"] = {}
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (DEMO_SPEED, DEMO_SPEED)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.debug_vis = False  # no velocity arrows above the robot
        # Start at the env origin facing +x and at rest, so the walk stays inside the fixed video frame.
        self.events.reset_base.params["pose_range"] = {}
        self.events.reset_base.params["velocity_range"] = {}
        # Side view (world frame, env 0). The robot covers x = 0 .. DEMO_SPEED * clip length.
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=VIDEO_EYE, lookat=VIDEO_LOOKAT, focal_length=VIDEO_FOCAL_MM)
