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

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

# ---- placeholders -------------------------------------------------------------
POLE_X = -0.16          # [m] rear end of the back
POLE_Y = 0.0            # [m]
BASE_TOP_Z = 0.055      # [m] platform underside sits ~here (platform center z = 0.065)
POLE_HEIGHT = 0.40      # [m] memory note: camera ~40 cm above the platform
POLE_RADIUS = 0.01      # [m]
POLE_MASS = 0.15        # [kg] pole + screws
CAM_MASS = 0.072        # [kg] D435i datasheet mass
CAM_FORWARD = 0.145     # [m] horizontal arm so the lens sits over the platform center (x = -0.015)
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
CAM_POS_IN_BASE = (POLE_X + CAM_FORWARD, POLE_Y, BASE_TOP_Z + POLE_HEIGHT)

D435I_CAMERA = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/d435i",
    update_period=1 / 30,                       # 30 fps like the real camera
    height=480,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],  # = RealSense z-depth
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
# from .platform_env_cfg import UnitreeGo2FlatPayloadEnvCfg_PLAY   # your existing class
#
# @configclass
# class UnitreeGo2PayloadCameraEnvCfg_PLAY(UnitreeGo2FlatPayloadEnvCfg_PLAY):
#     def __post_init__(self):
#         super().__post_init__()
#         self.scene.num_envs = 4
#         self.scene.d435i = D435I_CAMERA
#
# Run with --enable_cameras. Read frames: env.unwrapped.scene["d435i"].data.output["rgb"]
# and ["distance_to_image_plane"] (metres, shape [N, 480, 640, 1]).
