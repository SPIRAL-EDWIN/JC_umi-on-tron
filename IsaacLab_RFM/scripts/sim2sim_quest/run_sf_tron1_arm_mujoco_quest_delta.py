#!/usr/bin/env python3
"""Run the SF_TRON1A + ARXR5 arm policy with a Quest controller in MuJoCo.

The inference chain matches the IsaacLab/RSL-RL export:

    10 x 55 contact observations -> contactNet -> GRU -> 67-D latent
    65-D policy observation + 67-D latent -> actor -> 14 joint targets

Hold the right-controller Grip button to control end-effector position:

    target_position = ee_position_at_grip_press + transformed_controller_delta

Hold the left-controller Grip button to control end-effector orientation:

    target_rotation = controller_rotation_delta @ ee_rotation_at_grip_press

The current URDF represents the UMI gripper as fixed geometry. The right
controller front Trigger is therefore accepted only for input compatibility;
it does not add untrained gripper degrees of freedom to the model.

The original sim2sim script is intentionally left unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort


SCRIPT_PATH = Path(__file__).resolve()
ISAACLAB_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = ISAACLAB_ROOT.parent

DEFAULT_MJCF = (
    ISAACLAB_ROOT
    / "source/ext_loco/ext_loco/assets/SF_TRON1A_ARXR5ARM/assembly.xml"
)
DEPLOYED_MODEL_DIR = (
    REPO_ROOT
    / "tron1_ws/src/tron1-rl-deploy-arm/src/robot_controllers/config/"
    "pointfoot/SF_TRON1A_ARX5ARM/policy"
)
WBC_LOG_ROOT = Path(os.environ.get("WBC_LOG_ROOT", "/media/edwin/ChenJing26/WBC_logs"))
DEFAULT_TRAJECTORY = Path("/home/phi5090ii/UMI-ON-TRON/data/pushing.pkl")
DEFAULT_QUEST_BRIDGE = Path(
    "/home/phi5090ii/steamvr/build/quest_controller_bridge"
)
# Training tracks the URDF's controller-camera eef_link frame directly, so
# trajectory playback needs no additional link6 -> EEF transform.
TIP_OFFSET_POS = np.zeros(3, dtype=np.float64)
TIP_OFFSET_RPY = (0.0, 0.0, 0.0)

# OpenVR is X-right, Y-up, Z-backward. The robot base convention used here is
# X-forward, Y-left, Z-up. This is a proper rotation (determinant +1).
ROBOT_FROM_OPENVR = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# Isaac Lab articulation/action order measured from the training environment.
JOINT_NAMES = (
    "abad_L_Joint",
    "abad_R_Joint",
    "hip_L_Joint",
    "hip_R_Joint",
    "knee_L_Joint",
    "knee_R_Joint",
    "J1",
    "ankle_L_Joint",
    "ankle_R_Joint",
    "J2",
    "J3",
    "J4",
    "J5",
    "J6",
)
LEG_NAMES = (
    "abad_L_Joint",
    "hip_L_Joint",
    "knee_L_Joint",
    "ankle_L_Joint",
    "abad_R_Joint",
    "hip_R_Joint",
    "knee_R_Joint",
    "ankle_R_Joint",
)
ARM_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6")
LEG_IDS = np.array([JOINT_NAMES.index(name) for name in LEG_NAMES], dtype=int)
ARM_IDS = np.array([JOINT_NAMES.index(name) for name in ARM_NAMES], dtype=int)
DEFAULT_JOINT_POS = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)

# IsaacLab actuator gains used by LIMX_SF_TRON1A_ARM.
KP = np.array(
    [40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 18.0, 45.0, 45.0, 18.0, 18.0, 4.0, 4.0, 4.0],
    dtype=np.float64,
)
KD = np.array(
    [1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.0, 0.8, 0.8, 1.0, 1.0, 0.5, 0.5, 0.5],
    dtype=np.float64,
)
TORQUE_LIMIT = np.array(
    [80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 18.0, 40.0, 40.0, 18.0, 18.0, 3.0, 3.0, 3.0],
    dtype=np.float64,
)

# MuJoCo needs a finer contact step than the source PhysX simulation to keep
# the detailed ankle meshes from visibly tunnelling into the floor. The policy
# frequency remains identical to training: 0.001 * 20 = 0.02 s (50 Hz).
PHYSICS_DT = 0.001
POLICY_DECIMATION = 20
POLICY_DT = PHYSICS_DT * POLICY_DECIMATION
HISTORY_LENGTH = 10
OBS_DIM = 65
CONTACT_OBS_DIM = 55
ACTION_DIM = 14
# MuJoCo friction is (sliding, torsional, rolling), not static/dynamic/rolling.
# Keep deterministic sim2sim at the IsaacLab terrain's nominal coefficient;
# IsaacLab's training-side material randomization remains untouched.
MUJOCO_CONTACT_FRICTION = "1.0 0.005 0.0001"


def format_named(names: tuple[str, ...], values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return " ".join(
        f"{name}={values[index]: .4f}"
        for index, name in enumerate(names[: values.size])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quest-delta MuJoCo sim2sim for the SF_TRON1A ARXR5Arm "
            "three-ONNX policy."
        )
    )
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directory containing actor.onnx, contactNet.onnx and gru.onnx. "
        "Default: newest exported run under WBC_LOG_ROOT, then the legacy local log.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Simulation duration in seconds; 0 runs until the viewer closes "
        "(default: 30 s).",
    )
    parser.add_argument("--base-height", type=float, default=0.84)
    parser.add_argument(
        "--sample-latent",
        action="store_true",
        help="Sample the predicted latent distribution instead of using its mean.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument(
        "--render-fps",
        type=float,
        default=60.0,
        help="Viewer synchronization rate; independent of the 1 kHz physics rate.",
    )
    parser.add_argument(
        "--track-camera",
        action="store_true",
        help="Follow base_Link instead of using the default stationary camera.",
    )
    parser.add_argument(
        "--quest-bridge",
        type=Path,
        default=DEFAULT_QUEST_BRIDGE,
        help=f"Quest OpenVR bridge executable (default: {DEFAULT_QUEST_BRIDGE}).",
    )
    parser.add_argument(
        "--quest-rate",
        type=float,
        default=50.0,
        help="Quest bridge sample rate in Hz (default: 50).",
    )
    parser.add_argument(
        "--quest-scale",
        type=float,
        default=1.0,
        help="Controller-delta to EE-delta scale (default: 1.0).",
    )
    parser.add_argument(
        "--quest-max-delta",
        type=float,
        default=0.40,
        help="Maximum norm of commanded EE delta in metres (default: 0.40).",
    )
    parser.add_argument(
        "--quest-timeout",
        type=float,
        default=0.15,
        help="Stop Quest control after this many seconds without data (default: 0.15).",
    )
    parser.add_argument("--log-interval", type=float, default=1.0)
    parser.add_argument(
        "--leg-debug",
        action="store_true",
        help="Print leg q, raw/effective actions, desired/cmd positions, command step, and tracking error.",
    )
    parser.add_argument(
        "--arm-max-step",
        type=float,
        default=0.0,
        help="Maximum arm target-position change per 50 Hz policy update in radians; 0 disables it.",
    )
    parser.add_argument(
        "--max-leg-step",
        type=float,
        default=0.0,
        help="Maximum leg target-position change per 50 Hz policy update in radians; 0 disables it.",
    )
    return parser.parse_args()


def newest_exported_model_dir() -> Path:
    log_roots = (
        WBC_LOG_ROOT,
        ISAACLAB_ROOT / "logs/rsl_rl/ImplicitOneStageARXR5Arm",
    )
    candidates = [
        path
        for log_root in log_roots
        for path in log_root.glob("*/exported")
        if all((path / name).is_file() for name in ("actor.onnx", "contactNet.onnx", "gru.onnx"))
    ]
    if candidates:
        return max(candidates, key=lambda path: (path.parent.stat().st_mtime_ns, path.parent.name))
    return DEPLOYED_MODEL_DIR


def read_command(command: list[float] | None) -> np.ndarray:
    if command is not None:
        return np.asarray(command, dtype=np.float64)
    default = "0.15 0.0 1.0 0.0 0.0 0.0"
    print("输入末端最终点：x y z roll pitch yaw")
    print(f"单位：位置 m，姿态 rad。直接回车使用默认值：{default}")
    while True:
        text = input("> ").strip() or default
        try:
            values = np.asarray([float(item) for item in text.split()], dtype=np.float64)
        except ValueError:
            print("输入包含非数字，请重新输入。")
            continue
        if values.shape == (6,) and np.isfinite(values).all():
            return values
        print("必须输入 6 个有限数值。")


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def rpy_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Return intrinsic XYZ roll/pitch/yaw for R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1.0e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def rotation_from_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Convert an OpenVR/Rerun-order quaternion [x, y, z, w] to a matrix."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("Quaternion must contain four finite xyzw values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-12:
        raise ValueError("Quaternion norm is zero")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(float(cosine)))


def rotation_from_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1.0e-10:
        return np.eye(3)
    x, y, z = axis_angle / angle
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


class PickleTrajectory:
    """Playback compatible with IsaacLab's PicklePoseSequenceCommand."""

    def __init__(
        self,
        path: Path,
        episode_index: int,
        start_delay: float,
        loop: bool,
        planar_center: bool,
        playback_speed: float,
    ):
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Trajectory not found: {path}")
        with path.open("rb") as file:
            episodes = pickle.load(file)
        if isinstance(episodes, dict) and "episodes" in episodes:
            episodes = episodes["episodes"]
        if not isinstance(episodes, (list, tuple)) or not episodes:
            raise ValueError("Trajectory pickle must contain a non-empty episode list")
        if not -len(episodes) <= episode_index < len(episodes):
            raise IndexError(f"trajectory index {episode_index} outside [0, {len(episodes) - 1}]")

        self.path = path
        self.episode_index = episode_index % len(episodes)
        episode = episodes[self.episode_index]
        self.positions = np.asarray(episode["ee_pos"], dtype=np.float64).copy()
        axis_angles = np.asarray(episode["ee_axis_angle"], dtype=np.float64)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(f"Unexpected ee_pos shape: {self.positions.shape}")
        if axis_angles.shape != self.positions.shape:
            raise ValueError(
                f"ee_axis_angle shape {axis_angles.shape} does not match ee_pos {self.positions.shape}"
            )
        if planar_center:
            if len(self.positions) < 4:
                raise ValueError("planar_center requires at least four trajectory frames")
            self.positions[:, :2] -= self.positions[1:4, :2].mean(axis=0)

        self.rotations = np.stack([rotation_from_axis_angle(value) for value in axis_angles])
        time_samples = np.asarray(episode.get("t", []), dtype=np.float64)
        if len(time_samples) >= 2:
            self.sample_dt = float(np.median(np.diff(time_samples)))
        else:
            self.sample_dt = 0.005
        if self.sample_dt <= 0:
            raise ValueError(f"Invalid trajectory sample dt: {self.sample_dt}")
        if not math.isfinite(playback_speed) or playback_speed <= 0:
            raise ValueError("--trajectory-speed must be a finite number greater than zero")

        tip_rotation = rotation_from_rpy(*TIP_OFFSET_RPY)
        self.tip_rotation_inverse = tip_rotation.T
        self.tip_position_inverse = -self.tip_rotation_inverse @ TIP_OFFSET_POS
        self.start_delay = max(0.0, float(start_delay))
        self.loop = loop
        self.playback_speed = float(playback_speed)
        self.source_duration = len(self.positions) * self.sample_dt
        self.duration = self.source_duration / self.playback_speed
        self.world_offset = np.zeros(3, dtype=np.float64)
        self.command_origin: np.ndarray | None = None
        self.current_cycle = -1
        self.finished_message_printed = False

    def translate_offset(self, delta: np.ndarray) -> None:
        self.world_offset += np.asarray(delta, dtype=np.float64)

    def update(self, simulation: "Sim2Sim") -> None:
        elapsed = simulation.data.time - self.start_delay
        if elapsed < 0:
            return

        if self.loop:
            cycle = int(elapsed // self.duration)
            playback_time = elapsed - cycle * self.duration
        else:
            cycle = 0
            playback_time = min(elapsed, self.duration)

        new_cycle = self.command_origin is None or cycle != self.current_cycle
        if new_cycle:
            self.command_origin = simulation.data.xpos[simulation.ee_body_id].copy()
            self.current_cycle = cycle
            self.finished_message_printed = False
            print(
                f"[trajectory] episode={self.episode_index}, cycle={cycle}, "
                f"origin={self.command_origin.round(4).tolist()}"
            )

        source_time = min(
            playback_time * self.playback_speed,
            self.source_duration - self.sample_dt,
        )
        frame = min(int(source_time / self.sample_dt), len(self.positions) - 1)
        tip_position = self.positions[frame]
        tip_rotation = self.rotations[frame]
        link_position = tip_position + tip_rotation @ self.tip_position_inverse
        link_rotation = tip_rotation @ self.tip_rotation_inverse
        world_position = self.command_origin + link_position + self.world_offset
        simulation.set_target(world_position, link_rotation, reset_reference=new_cycle)

        if not self.loop and elapsed >= self.duration and not self.finished_message_printed:
            print("[trajectory] playback complete; holding the final pose.")
            self.finished_message_printed = True


def load_sim_model(mjcf_path: Path) -> mujoco.MjModel:
    """Add simulation-only helpers without changing the robot MJCF on disk."""
    mjcf_path = mjcf_path.expanduser().resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"MJCF not found: {mjcf_path}")

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    meshdir = compiler.get("meshdir", ".")
    compiler.set("meshdir", str((mjcf_path.parent / meshdir).resolve()))

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(PHYSICS_DT))

    # The MuJoCo defaults (solref=0.02 1) are visibly too soft for these
    # high-resolution foot collision meshes. Keep the contact stable and firm
    # without making it perfectly rigid.
    collision_default = root.find("./default/default[@class='collision']/geom")
    if collision_default is not None:
        collision_default.set("solref", "0.005 1")
        collision_default.set("solimp", "0.95 0.99 0.001")

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    if visual.find("headlight") is None:
        ET.SubElement(
            visual,
            "headlight",
            {"diffuse": "0.7 0.7 0.7", "ambient": "0.25 0.25 0.25", "specular": "0.2 0.2 0.2"},
        )

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        worldbody_index = next(
            (index for index, element in enumerate(root) if element.tag == "worldbody"),
            len(root),
        )
        root.insert(worldbody_index, asset)
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "sim2sim_checker_texture",
            "type": "2d",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.12 0.25 0.38",
            "rgb2": "0.24 0.42 0.58",
            "markrgb": "0.55 0.68 0.78",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "sim2sim_checker_material",
            "texture": "sim2sim_checker_texture",
            "texrepeat": "5 5",
            "texuniform": "true",
            "reflectance": "0.15",
        },
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    worldbody.insert(
        0,
        ET.Element(
            "geom",
            {
                "name": "sim2sim_floor",
                "type": "plane",
                "size": "0 0 0.1",
                "material": "sim2sim_checker_material",
                "friction": MUJOCO_CONTACT_FRICTION,
                "condim": "3",
                "solref": "0.005 1",
                "solimp": "0.95 0.99 0.001",
            },
        ),
    )
    worldbody.insert(
        1,
        ET.Element(
            "light",
            {"name": "sim2sim_light", "pos": "0 -1 3", "dir": "0 0 -1", "directional": "true"},
        ),
    )

    # Fixed frame plus a physical door panel that swings about a vertical
    # hinge. It sits in front of the robot and can be pushed or grasped.
    contact_parameters = {
        "friction": MUJOCO_CONTACT_FRICTION,
        "condim": "3",
        "solref": "0.005 1",
        "solimp": "0.95 0.99 0.001",
    }
    for name, pos, size in (
        ("door_frame_left", "0.75 0.93 0.73", "0.06 0.45 0.73"),
        ("door_frame_right", "0.75 -0.93 0.73", "0.06 0.45 0.73"),
        ("door_frame_header", "0.75 0 1.56", "0.06 0.48 0.10"),
    ):
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": name,
                "type": "box",
                "pos": pos,
                "size": size,
                "rgba": "0.45 0.46 0.46 1",
                **contact_parameters,
            },
        )

    door = ET.SubElement(
        worldbody,
        "body",
        {"name": "sim2sim_door", "pos": "0.75 0.45 0.02"},
    )
    ET.SubElement(
        door,
        "inertial",
        {
            "pos": "0 -0.445 0.70",
            "mass": "8.25",
            "diaginertia": "1.78 1.28 0.53",
        },
    )
    ET.SubElement(
        door,
        "joint",
        {
            "name": "sim2sim_door_hinge",
            "type": "hinge",
            "axis": "0 0 1",
            "limited": "true",
            "range": "-1.75 1.75",
            "damping": "1.0",
            "frictionloss": "0.2",
            "armature": "0.01",
        },
    )
    ET.SubElement(
        door,
        "geom",
        {
            "name": "sim2sim_door_panel",
            "type": "box",
            "pos": "0 -0.44 0.69",
            "size": "0.025 0.44 0.69",
            "rgba": "0.45 0.18 0.055 1",
            **contact_parameters,
        },
    )
    ET.SubElement(
        door,
        "geom",
        {
            "name": "sim2sim_door_handle_shaft",
            "type": "cylinder",
            "pos": "-0.075 -0.75 1.02",
            "quat": "0.7071068 0 0.7071068 0",
            "size": "0.022 0.05",
            "rgba": "0.12 0.12 0.12 1",
            **contact_parameters,
        },
    )
    ET.SubElement(
        door,
        "geom",
        {
            "name": "sim2sim_door_handle_knob",
            "type": "sphere",
            "pos": "-0.135 -0.75 1.02",
            "size": "0.032",
            "rgba": "0.12 0.12 0.12 1",
            **contact_parameters,
        },
    )

    # A collision-free mocap body renders the full target pose instead of only
    # a position sphere. Local +X/+Y/+Z are red/green/blue respectively.
    target_frame = ET.Element(
        "body",
        {
            "name": "command_target_frame",
            "mocap": "true",
            "pos": "0.15 0 1",
        },
    )
    ET.SubElement(
        target_frame,
        "site",
        {
            "name": "command_target",
            "type": "sphere",
            "size": "0.012",
            "rgba": "1 1 1 0.9",
            "group": "0",
        },
    )
    for name, endpoint, color in (
        ("command_target_x", "0.16 0 0", "1 0.12 0.05 1"),
        ("command_target_y", "0 0.16 0", "0.1 0.9 0.2 1"),
        ("command_target_z", "0 0 0.16", "0.1 0.35 1 1"),
    ):
        ET.SubElement(
            target_frame,
            "site",
            {
                "name": name,
                "type": "capsule",
                "fromto": f"0 0 0 {endpoint}",
                "size": "0.007",
                "rgba": color,
                "group": "0",
            },
        )
        ET.SubElement(
            target_frame,
            "site",
            {
                "name": f"{name}_tip",
                "type": "sphere",
                "pos": endpoint,
                "size": "0.014",
                "rgba": color,
                "group": "0",
            },
        )
    worldbody.insert(2, target_frame)

    # Show the measured end-effector pose in exactly the same coordinate-frame
    # form as the command target.  These sites are attached to eef_link, so
    # MuJoCo updates them automatically with the robot state and they remain
    # purely visual (no added mass or collision geometry).
    eef_frame = worldbody.find(".//body[@name='eef_link']")
    if eef_frame is None:
        raise ValueError("eef_link body is missing from MJCF")
    ET.SubElement(
        eef_frame,
        "site",
        {
            "name": "current_eef",
            "type": "sphere",
            "size": "0.012",
            "rgba": "1 1 1 0.9",
            "group": "0",
        },
    )
    for name, endpoint, color in (
        ("current_eef_x", "0.16 0 0", "1 0.12 0.05 1"),
        ("current_eef_y", "0 0.16 0", "0.1 0.9 0.2 1"),
        ("current_eef_z", "0 0 0.16", "0.1 0.35 1 1"),
    ):
        ET.SubElement(
            eef_frame,
            "site",
            {
                "name": name,
                "type": "capsule",
                "fromto": f"0 0 0 {endpoint}",
                "size": "0.007",
                "rgba": color,
                "group": "0",
            },
        )
        ET.SubElement(
            eef_frame,
            "site",
            {
                "name": f"{name}_tip",
                "type": "sphere",
                "pos": endpoint,
                "size": "0.014",
                "rgba": color,
                "group": "0",
            },
        )

    # The current assembly.urdf defines the UMI gripper as fixed link6
    # geometry.  Do not recreate the obsolete six-DoF DAS gripper at runtime,
    # otherwise the simulated robot would no longer match eef_link training.

    xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml)


class ThreeOnnxPolicy:
    def __init__(self, model_dir: Path, sample_latent: bool, rng: np.random.Generator):
        model_dir = model_dir.expanduser().resolve()
        missing = [
            name for name in ("actor.onnx", "contactNet.onnx", "gru.onnx") if not (model_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing {missing} in {model_dir}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.actor = ort.InferenceSession(str(model_dir / "actor.onnx"), options, providers=providers)
        self.contact_net = ort.InferenceSession(
            str(model_dir / "contactNet.onnx"), options, providers=providers
        )
        self.gru = ort.InferenceSession(str(model_dir / "gru.onnx"), options, providers=providers)
        self.sample_latent = sample_latent
        self.rng = rng

        actor_inputs = self.actor.get_inputs()
        contact_input = self.contact_net.get_inputs()[0]
        gru_inputs = self.gru.get_inputs()
        if actor_inputs[0].shape[-1] != OBS_DIM or actor_inputs[1].shape[-1] != 67:
            raise ValueError(f"Unexpected actor inputs: {[item.shape for item in actor_inputs]}")
        if contact_input.shape[-1] != CONTACT_OBS_DIM:
            raise ValueError(f"Unexpected contactNet input: {contact_input.shape}")
        if gru_inputs[0].shape[-1] != 131 or gru_inputs[1].shape[-1] != 131:
            raise ValueError(f"Unexpected GRU inputs: {[item.shape for item in gru_inputs]}")

        self.hidden = np.zeros((1, 1, 131), dtype=np.float32)

    def reset(self) -> None:
        self.hidden.fill(0.0)

    def __call__(self, observation: np.ndarray, history: np.ndarray) -> np.ndarray:
        contact_input = history[np.newaxis, :, :].astype(np.float32, copy=False)
        contact_name = self.contact_net.get_inputs()[0].name
        contact_output = self.contact_net.run(None, {contact_name: contact_input})[0]
        contact_output = np.asarray(contact_output[-1:], dtype=np.float32)

        gru_inputs = self.gru.get_inputs()
        gru_output, new_hidden = self.gru.run(
            None,
            {
                gru_inputs[0].name: contact_output,
                gru_inputs[1].name: self.hidden,
            },
        )
        self.hidden = np.asarray(new_hidden, dtype=np.float32)
        gru_output = np.asarray(gru_output, dtype=np.float32)

        # GRU output = [base_lin_vel(3), mu(64), logvar(64)].
        mean = gru_output[:, 3:67]
        if self.sample_latent:
            log_variance = gru_output[:, 67:131]
            std = np.sqrt(np.exp(log_variance) + 1.0e-4)
            predicted = mean + std * self.rng.standard_normal(mean.shape).astype(np.float32)
        else:
            predicted = mean
        actor_latent = np.concatenate((gru_output[:, :3], predicted), axis=1).astype(np.float32)

        actor_inputs = self.actor.get_inputs()
        action = self.actor.run(
            None,
            {
                actor_inputs[0].name: observation[np.newaxis, :].astype(np.float32),
                actor_inputs[1].name: actor_latent,
            },
        )[0]
        action = np.asarray(action[0], dtype=np.float64)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid actor output: shape={action.shape}, values={action}")
        return np.clip(action, -100.0, 100.0)


class QuestHandState:
    """Mutable tracking state for one Grip-clutched Quest controller."""

    def __init__(self) -> None:
        self.controller_origin: np.ndarray | None = None
        self.controller_rotation_origin: np.ndarray | None = None
        self.delta = np.zeros(3, dtype=np.float64)
        self.rotation_delta = np.eye(3, dtype=np.float64)
        self.trigger = 0.0
        self.active = False
        self.ready_to_arm = True
        self.tracking_valid = False
        self.last_update: float | None = None


class QuestDeltaController:
    """Read both Quest controllers and expose independent Grip-clutched deltas."""

    def __init__(
        self,
        bridge: Path,
        rate: float,
        scale: float,
        max_delta: float,
        timeout: float,
    ):
        self.bridge = bridge.expanduser().resolve()
        if not self.bridge.is_file():
            raise FileNotFoundError(f"Quest bridge not found: {self.bridge}")
        if not math.isfinite(rate) or not 1.0 <= rate <= 500.0:
            raise ValueError("--quest-rate must be between 1 and 500 Hz")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("--quest-scale must be finite and greater than zero")
        if not math.isfinite(max_delta) or max_delta <= 0.0:
            raise ValueError("--quest-max-delta must be finite and greater than zero")
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("--quest-timeout must be finite and greater than zero")

        self.rate = float(rate)
        self.scale = float(scale)
        self.max_delta = float(max_delta)
        self.timeout = float(timeout)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._hands = {
            "right": QuestHandState(),
            "left": QuestHandState(),
        }
        self._error: str | None = None

    def start(self) -> None:
        command = [
            str(self.bridge),
            "--format",
            "json",
            "--rate",
            str(self.rate),
            "--hand",
            "both",
        ]
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="quest-dual-controller-reader",
            daemon=True,
        )
        self._thread.start()

    def _reader_loop(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        try:
            for line in self._process.stdout:
                if self._stop.is_set():
                    break
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue

                hand = sample.get("hand")
                if hand not in self._hands:
                    continue
                valid = (
                    bool(sample.get("connected", False))
                    and bool(sample.get("valid", False))
                    and sample.get("tracking_result") == "running_ok"
                )
                grip = bool(sample.get("input", {}).get("grip", False))
                input_state = sample.get("input", {})
                axes = input_state.get("axes", [])
                trigger_axis = 0.0
                if (
                    isinstance(axes, list)
                    and len(axes) > 1
                    and isinstance(axes[1], list)
                    and axes[1]
                ):
                    trigger_axis = float(axes[1][0])
                trigger_button = bool(input_state.get("trigger_button", False))
                trigger = float(
                    np.clip(max(trigger_axis, 1.0 if trigger_button else 0.0), 0.0, 1.0)
                )
                position_raw = np.asarray(
                    sample.get("position_m", [0.0, 0.0, 0.0]),
                    dtype=np.float64,
                )
                quaternion_raw = np.asarray(
                    sample.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0]),
                    dtype=np.float64,
                )
                if (
                    position_raw.shape != (3,)
                    or not np.isfinite(position_raw).all()
                    or quaternion_raw.shape != (4,)
                    or not np.isfinite(quaternion_raw).all()
                    or np.linalg.norm(quaternion_raw) < 1.0e-12
                ):
                    valid = False
                position_base = ROBOT_FROM_OPENVR @ position_raw
                if valid:
                    rotation_openvr = rotation_from_quaternion_xyzw(quaternion_raw)
                    rotation_base = (
                        ROBOT_FROM_OPENVR
                        @ rotation_openvr
                        @ ROBOT_FROM_OPENVR.T
                    )
                else:
                    rotation_base = np.eye(3, dtype=np.float64)

                with self._lock:
                    state = self._hands[hand]
                    state.last_update = time.perf_counter()
                    state.tracking_valid = valid
                    if valid:
                        # Trigger is independent of Grip. Keep the last value if
                        # tracking becomes invalid instead of dropping an object.
                        state.trigger = trigger
                    if not valid:
                        state.active = False
                        state.controller_origin = None
                        state.controller_rotation_origin = None
                        state.ready_to_arm = False
                        state.delta.fill(0.0)
                        state.rotation_delta[:] = np.eye(3)
                    elif not grip:
                        state.active = False
                        state.controller_origin = None
                        state.controller_rotation_origin = None
                        state.ready_to_arm = True
                        state.delta.fill(0.0)
                        state.rotation_delta[:] = np.eye(3)
                    elif state.ready_to_arm:
                        # Grip rising edge: this controller pose is the zero delta.
                        state.controller_origin = position_base.copy()
                        state.controller_rotation_origin = rotation_base.copy()
                        state.delta.fill(0.0)
                        state.rotation_delta[:] = np.eye(3)
                        state.active = True
                        state.ready_to_arm = False
                    elif (
                        state.controller_origin is not None
                        and state.controller_rotation_origin is not None
                    ):
                        delta = self.scale * (
                            position_base - state.controller_origin
                        )
                        delta_norm = float(np.linalg.norm(delta))
                        if delta_norm > self.max_delta:
                            delta *= self.max_delta / delta_norm
                        state.delta = delta
                        # Relative rotation expressed in robot-base axes.
                        state.rotation_delta = (
                            rotation_base @ state.controller_rotation_origin.T
                        )
                        state.active = True
                    else:
                        # Tracking recovered while Grip was still held. Require a
                        # release and a new press before motion can resume.
                        state.active = False
                        state.delta.fill(0.0)
                        state.rotation_delta[:] = np.eye(3)
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
                for state in self._hands.values():
                    state.active = False
                    state.tracking_valid = False
        finally:
            if not self._stop.is_set():
                return_code = self._process.poll()
                with self._lock:
                    if self._error is None:
                        self._error = (
                            "Quest bridge stopped unexpectedly"
                            if return_code is None
                            else f"Quest bridge exited with code {return_code}"
                        )
                    for state in self._hands.values():
                        state.active = False
                        state.tracking_valid = False

    def snapshot(
        self,
        hand: str,
    ) -> tuple[bool, np.ndarray, np.ndarray, float, float]:
        """Return one hand's active flag, deltas, trigger, and sample age."""
        if hand not in self._hands:
            raise ValueError("hand must be 'right' or 'left'")
        with self._lock:
            if self._error is not None:
                raise RuntimeError(self._error)
            state = self._hands[hand]
            now = time.perf_counter()
            age = (
                math.inf
                if state.last_update is None
                else max(0.0, now - state.last_update)
            )
            fresh = age <= self.timeout and state.tracking_valid
            if not fresh:
                # A stale stream may not resume motion until Grip is released.
                if state.active:
                    state.active = False
                    state.controller_origin = None
                    state.controller_rotation_origin = None
                    state.ready_to_arm = False
                    state.delta.fill(0.0)
                    state.rotation_delta[:] = np.eye(3)
                return (
                    False,
                    np.zeros(3, dtype=np.float64),
                    np.eye(3, dtype=np.float64),
                    state.trigger,
                    age,
                )
            return (
                state.active,
                state.delta.copy(),
                state.rotation_delta.copy(),
                state.trigger,
                age,
            )

    def close(self) -> None:
        self._stop.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class Sim2Sim:
    def __init__(
        self,
        model: mujoco.MjModel,
        policy: ThreeOnnxPolicy,
        command: np.ndarray,
        command_frame: str,
        base_height: float,
        arm_max_step: float,
        max_leg_step: float,
    ):
        self.model = model
        self.data = mujoco.MjData(model)
        self.policy = policy
        self.command_frame = command_frame
        self.target_position = command[:3].copy()
        self.target_rotation = rotation_from_rpy(*command[3:])
        if not math.isfinite(arm_max_step) or arm_max_step < 0.0:
            raise ValueError("--arm-max-step must be finite and non-negative")
        if not math.isfinite(max_leg_step) or max_leg_step < 0.0:
            raise ValueError("--max-leg-step must be finite and non-negative")
        self.arm_max_step = float(arm_max_step)
        self.max_leg_step = float(max_leg_step)

        self.joint_qpos_adr = np.array([model.joint(name).qposadr[0] for name in JOINT_NAMES], dtype=int)
        self.joint_dof_adr = np.array([model.joint(name).dofadr[0] for name in JOINT_NAMES], dtype=int)
        self.motor_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor") for name in JOINT_NAMES],
            dtype=int,
        )
        if np.any(self.motor_ids < 0):
            raise ValueError("One or more joint motors are missing from the MJCF")
        self.gripper_trigger = 0.0

        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "eef_link")
        self.target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "command_target")
        target_frame_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "command_target_frame"
        )
        self.target_mocap_id = (
            int(model.body_mocapid[target_frame_body_id]) if target_frame_body_id >= 0 else -1
        )
        self.imu_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        if min(
            self.base_body_id,
            self.ee_body_id,
            self.target_site_id,
            self.target_mocap_id,
            self.imu_sensor_id,
        ) < 0:
            raise ValueError("Required base/EE/IMU/target elements are missing")

        mujoco.mj_resetData(model, self.data)
        root_adr = model.joint("root").qposadr[0]
        self.data.qpos[root_adr : root_adr + 7] = [0.0, 0.0, base_height, 1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.joint_qpos_adr] = DEFAULT_JOINT_POS
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(model, self.data)

        self.raw_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.effective_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.last_torque = np.zeros(ACTION_DIM, dtype=np.float64)
        self.raw_desired_position = DEFAULT_JOINT_POS.copy()
        self.policy_desired_position = DEFAULT_JOINT_POS.copy()
        self.desired_position = DEFAULT_JOINT_POS.copy()
        self.policy_command_step = np.zeros(ACTION_DIM, dtype=np.float64)
        self._previous_policy_command = DEFAULT_JOINT_POS.copy()
        self.history: deque[np.ndarray] = deque(maxlen=HISTORY_LENGTH)
        self.se3_distance_reference = self._initial_se3_distance()
        first_contact_obs = self.contact_observation()
        for _ in range(HISTORY_LENGTH):
            self.history.append(first_contact_obs.copy())
        self.policy.reset()
        self._update_target_marker()

    def base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position = self.data.xpos[self.base_body_id].copy()
        rotation = self.data.xmat[self.base_body_id].reshape(3, 3).copy()
        return position, rotation

    def ee_pose_base(self) -> tuple[np.ndarray, np.ndarray]:
        base_position, base_rotation = self.base_pose()
        ee_position_world = self.data.xpos[self.ee_body_id]
        ee_rotation_world = self.data.xmat[self.ee_body_id].reshape(3, 3)
        position = base_rotation.T @ (ee_position_world - base_position)
        rotation = base_rotation.T @ ee_rotation_world
        return position, rotation

    def target_pose_base(self) -> tuple[np.ndarray, np.ndarray]:
        if self.command_frame == "base":
            return self.target_position, self.target_rotation
        base_position, base_rotation = self.base_pose()
        position = base_rotation.T @ (self.target_position - base_position)
        rotation = base_rotation.T @ self.target_rotation
        return position, rotation

    def target_pose_world(self) -> tuple[np.ndarray, np.ndarray]:
        if self.command_frame == "world":
            return self.target_position, self.target_rotation
        base_position, base_rotation = self.base_pose()
        return (
            base_position + base_rotation @ self.target_position,
            base_rotation @ self.target_rotation,
        )

    @staticmethod
    def pose_6d(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        return np.concatenate((position, rotation[:, 0], rotation[:, 1]))

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.data.qpos[self.joint_qpos_adr].copy(),
            self.data.qvel[self.joint_dof_adr].copy(),
        )

    def base_angular_velocity(self) -> np.ndarray:
        sensor = self.model.sensor(self.imu_sensor_id)
        start = sensor.adr[0]
        return self.data.sensordata[start : start + 3].copy()

    def projected_gravity(self) -> np.ndarray:
        _, base_rotation = self.base_pose()
        return base_rotation.T @ np.array([0.0, 0.0, -1.0])

    def contact_observation(self) -> np.ndarray:
        position, velocity = self.joint_state()
        ee_position, ee_rotation = self.ee_pose_base()
        no_ankle = np.array(["ankle" not in name for name in JOINT_NAMES])
        observation = np.concatenate(
            (
                self.base_angular_velocity(),
                self.projected_gravity(),
                (position - DEFAULT_JOINT_POS)[no_ankle],
                velocity,
                self.last_torque,
                self.pose_6d(ee_position, ee_rotation),
            )
        )
        if observation.shape != (CONTACT_OBS_DIM,):
            raise RuntimeError(f"Contact observation has shape {observation.shape}")
        return observation

    def policy_observation(self) -> np.ndarray:
        position, velocity = self.joint_state()
        ee_position, ee_rotation = self.ee_pose_base()
        target_position, target_rotation = self.target_pose_base()
        no_ankle = np.array(["ankle" not in name for name in JOINT_NAMES])
        observation = np.concatenate(
            (
                self.base_angular_velocity(),
                self.projected_gravity(),
                self.pose_6d(target_position, target_rotation),
                (position - DEFAULT_JOINT_POS)[no_ankle],
                velocity,
                self.last_action,
                self.pose_6d(ee_position, ee_rotation),
                np.array([self.se3_distance_reference]),
            )
        )
        if observation.shape != (OBS_DIM,):
            raise RuntimeError(f"Policy observation has shape {observation.shape}")
        return np.clip(observation, -100.0, 100.0)

    def _initial_se3_distance(self) -> float:
        ee_position, ee_rotation = self.ee_pose_base()
        target_position, target_rotation = self.target_pose_base()
        return float(
            2.0 * np.linalg.norm(target_position - ee_position)
            + rotation_angle(target_rotation @ ee_rotation.T)
        )

    def _update_target_marker(self) -> None:
        target_position, target_rotation = self.target_pose_world()
        target_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(target_quaternion, target_rotation.reshape(-1))
        self.data.mocap_pos[self.target_mocap_id] = target_position
        self.data.mocap_quat[self.target_mocap_id] = target_quaternion

    def translate_target(self, delta: np.ndarray) -> None:
        """Translate the target in the selected command frame."""
        self.target_position += np.asarray(delta, dtype=np.float64)
        self.se3_distance_reference = self._initial_se3_distance()
        self._update_target_marker()

    def set_target(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        *,
        reset_reference: bool = False,
    ) -> None:
        self.target_position = np.asarray(position, dtype=np.float64).copy()
        self.target_rotation = np.asarray(rotation, dtype=np.float64).copy()
        if reset_reference:
            self.se3_distance_reference = self._initial_se3_distance()
        self._update_target_marker()

    def set_gripper_trigger(self, trigger: float) -> None:
        """Keep Quest input compatibility; the current URDF gripper is fixed."""
        self.gripper_trigger = float(np.clip(trigger, 0.0, 1.0))

    def gripper_position(self) -> float:
        """The current URDF has no gripper DoF."""
        return 0.0

    def infer(self) -> None:
        contact_obs = self.contact_observation()
        self.history.append(contact_obs)
        history = np.stack(self.history, axis=0)
        self.raw_action = self.policy(self.policy_observation(), history)
        self.se3_distance_reference = max(0.0, self.se3_distance_reference - POLICY_DT)

    def apply_pd(self, *, policy_updated: bool = False) -> None:
        position, velocity = self.joint_state()
        if policy_updated:
            # Same torque-aware action clamp used by SolefootController.cpp.
            action_min = position - DEFAULT_JOINT_POS + (KD * velocity - TORQUE_LIMIT) / KP
            action_max = position - DEFAULT_JOINT_POS + (KD * velocity + TORQUE_LIMIT) / KP
            self.effective_action = np.clip(self.raw_action, action_min, action_max)
            self.raw_desired_position = DEFAULT_JOINT_POS + self.raw_action
            self.policy_desired_position = DEFAULT_JOINT_POS + self.effective_action

            command = self.policy_desired_position.copy()
            if self.arm_max_step > 0.0:
                command[ARM_IDS] = np.clip(
                    command[ARM_IDS],
                    self._previous_policy_command[ARM_IDS] - self.arm_max_step,
                    self._previous_policy_command[ARM_IDS] + self.arm_max_step,
                )
            if self.max_leg_step > 0.0:
                command[LEG_IDS] = np.clip(
                    command[LEG_IDS],
                    self._previous_policy_command[LEG_IDS] - self.max_leg_step,
                    self._previous_policy_command[LEG_IDS] + self.max_leg_step,
                )
            self.desired_position = command
            self.policy_command_step = command - self._previous_policy_command
            self._previous_policy_command = command.copy()
            # The policy observes the command that was actually applied, just
            # like record_applied_targets() in the real deployment script.
            self.last_action = command - DEFAULT_JOINT_POS

        torque = KP * (self.desired_position - position) - KD * velocity
        torque = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)

        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.motor_ids] = torque
        self.last_torque = torque

    def step(self, physics_step: int) -> None:
        policy_updated = physics_step % POLICY_DECIMATION == 0
        if policy_updated:
            self.infer()
        self.apply_pd(policy_updated=policy_updated)
        mujoco.mj_step(self.model, self.data)
        self._update_target_marker()
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Simulation state became non-finite")

    def error(self) -> tuple[float, float]:
        ee_position, ee_rotation = self.ee_pose_base()
        target_position, target_rotation = self.target_pose_base()
        return (
            float(np.linalg.norm(target_position - ee_position)),
            rotation_angle(target_rotation @ ee_rotation.T),
        )


def configure_viewer(viewer_handle, base_body_id: int, track_robot: bool) -> None:
    if track_robot:
        viewer_handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer_handle.cam.trackbodyid = base_body_id
        viewer_handle.cam.fixedcamid = -1
    else:
        viewer_handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer_handle.cam.trackbodyid = -1
    viewer_handle.cam.lookat[:] = [0.3, 0.0, 0.7]
    viewer_handle.cam.distance = 3.0
    viewer_handle.cam.azimuth = 45.0
    viewer_handle.cam.elevation = -18.0
    viewer_handle.opt.geomgroup[2] = 1
    viewer_handle.opt.geomgroup[3] = 0


def run(args: argparse.Namespace) -> None:
    trajectory = None
    # Sim2Sim needs an initial command during construction. It is replaced with
    # the measured EE pose immediately after mj_forward, before the first step.
    command = np.zeros(6, dtype=np.float64)
    command_frame = "base"

    model_dir = args.model_dir or newest_exported_model_dir()
    rng = np.random.default_rng(args.seed)

    print(f"[sim2sim] MJCF: {args.mjcf.expanduser().resolve()}")
    print(f"[sim2sim] ONNX: {model_dir.expanduser().resolve()}")
    print("[quest] Right Grip: XYZ position control")
    print("[quest] Left Grip: roll/pitch/yaw orientation control")
    print("[quest] Right front Trigger is ignored: the current URDF gripper is fixed.")
    print(
        "[quest] OpenVR -> base: "
        "dx=-dVR_z (forward), dy=-dVR_x (left), dz=dVR_y (up)"
    )
    print(
        f"[quest] bridge={args.quest_bridge.expanduser().resolve()}, "
        f"rate={args.quest_rate:g}Hz, scale={args.quest_scale:g}, "
        f"max_delta={args.quest_max_delta:g}m, timeout={args.quest_timeout:g}s"
    )
    print(
        "[sim2sim] 50Hz目标限幅："
        f"arm_max_step={args.arm_max_step:g} rad，"
        f"max_leg_step={args.max_leg_step:g} rad（0=关闭）"
    )
    model = load_sim_model(args.mjcf)
    policy = ThreeOnnxPolicy(model_dir, args.sample_latent, rng)
    simulation = Sim2Sim(
        model,
        policy,
        command,
        command_frame,
        args.base_height,
        args.arm_max_step,
        args.max_leg_step,
    )
    initial_ee_position, initial_ee_rotation = simulation.ee_pose_base()
    simulation.set_target(
        initial_ee_position,
        initial_ee_rotation,
        reset_reference=True,
    )
    print(
        "[quest] Initial hold target EE(base)="
        f"{initial_ee_position.round(4).tolist()}"
    )

    quest = QuestDeltaController(
        args.quest_bridge,
        args.quest_rate,
        args.quest_scale,
        args.quest_max_delta,
        args.quest_timeout,
    )
    quest.start()

    if args.duration is None:
        if trajectory is None:
            run_duration = 30.0
        elif trajectory.loop:
            run_duration = 0.0
        else:
            # Include one more policy tick so the final recorded frame is
            # applied before an automatic non-looping run exits.
            run_duration = trajectory.start_delay + trajectory.duration + POLICY_DT
    else:
        run_duration = args.duration
    max_steps = math.inf if run_duration <= 0 else math.ceil(run_duration / PHYSICS_DT)
    log_steps = max(1, round(args.log_interval / PHYSICS_DT))
    render_steps = max(1, round(1.0 / (max(args.render_fps, 1.0) * PHYSICS_DT)))

    def loop(viewer_handle=None) -> None:
        physics_step = 0
        right_active_previous = False
        left_active_previous = False
        right_ee_position_origin: np.ndarray | None = None
        left_ee_rotation_origin: np.ndarray | None = None
        # Establish the wall-clock epoch only after the viewer has finished
        # opening. Otherwise its startup cost makes the simulation briefly
        # race ahead in an attempt to "catch up".
        wall_epoch = time.perf_counter() - simulation.data.time
        last_log_wall = time.perf_counter()
        last_log_sim = simulation.data.time
        viewer_sync_count = 0
        while physics_step < max_steps and (viewer_handle is None or viewer_handle.is_running()):
            if physics_step % POLICY_DECIMATION == 0:
                (
                    right_active,
                    right_position_delta,
                    _,
                    right_trigger,
                    right_age,
                ) = quest.snapshot("right")
                (
                    left_active,
                    _,
                    left_rotation_delta,
                    _,
                    left_age,
                ) = quest.snapshot("left")
                simulation.set_gripper_trigger(right_trigger)

                next_position = simulation.target_position.copy()
                next_rotation = simulation.target_rotation.copy()
                target_changed = False
                reset_reference = False

                if right_active and not right_active_previous:
                    right_ee_position_origin, _ = simulation.ee_pose_base()
                    print(
                        "[quest] Right Grip ON: position origin(base)="
                        f"{right_ee_position_origin.round(4).tolist()}"
                    )
                    reset_reference = True
                if right_active:
                    assert right_ee_position_origin is not None
                    next_position = (
                        right_ee_position_origin + right_position_delta
                    )
                    target_changed = True
                elif right_active_previous:
                    hold_position, _ = simulation.ee_pose_base()
                    next_position = hold_position
                    target_changed = True
                    reset_reference = True
                    age_text = (
                        "unknown"
                        if not math.isfinite(right_age)
                        else f"{right_age:.3f}s"
                    )
                    print(
                        "[quest] Right Grip OFF/stale: holding position(base)="
                        f"{hold_position.round(4).tolist()}, sample_age={age_text}"
                    )
                    right_ee_position_origin = None

                if left_active and not left_active_previous:
                    _, left_ee_rotation_origin = simulation.ee_pose_base()
                    print(
                        "[quest] Left Grip ON: orientation origin rpy="
                        f"{rpy_from_rotation(left_ee_rotation_origin).round(4).tolist()}"
                    )
                    reset_reference = True
                if left_active:
                    assert left_ee_rotation_origin is not None
                    next_rotation = (
                        left_rotation_delta @ left_ee_rotation_origin
                    )
                    target_changed = True
                elif left_active_previous:
                    _, hold_rotation = simulation.ee_pose_base()
                    next_rotation = hold_rotation
                    target_changed = True
                    reset_reference = True
                    age_text = (
                        "unknown"
                        if not math.isfinite(left_age)
                        else f"{left_age:.3f}s"
                    )
                    print(
                        "[quest] Left Grip OFF/stale: holding orientation rpy="
                        f"{rpy_from_rotation(hold_rotation).round(4).tolist()}, "
                        f"sample_age={age_text}"
                    )
                    left_ee_rotation_origin = None

                if target_changed:
                    simulation.set_target(
                        next_position,
                        next_rotation,
                        reset_reference=reset_reference,
                    )
                right_active_previous = right_active
                left_active_previous = left_active

            simulation.step(physics_step)
            physics_step += 1
            # Physics runs at 1 kHz, but synchronizing the GUI at 1 kHz makes
            # wall-clock time lag badly and looks like slow motion.
            if viewer_handle is not None and physics_step % render_steps == 0:
                viewer_handle.sync()
                viewer_sync_count += 1
            if not args.no_realtime:
                deadline = wall_epoch + simulation.data.time
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
            if physics_step % log_steps == 0:
                now = time.perf_counter()
                wall_delta = max(now - last_log_wall, 1.0e-9)
                sim_delta = simulation.data.time - last_log_sim
                rtf = sim_delta / wall_delta
                viewer_fps = viewer_sync_count / wall_delta if viewer_handle is not None else 0.0
                pos_error, rot_error = simulation.error()
                target_position_base, target_rotation_base = simulation.target_pose_base()
                target_rpy = rpy_from_rotation(target_rotation_base)
                base_z = simulation.data.xpos[simulation.base_body_id, 2]
                viewer_text = f"  view={viewer_fps:5.1f}fps" if viewer_handle is not None else ""
                print(
                    f"t={simulation.data.time:7.2f}s  RTF={rtf:5.3f}{viewer_text}  "
                    f"base_z={base_z:6.3f}  "
                    f"EE误差={pos_error:6.3f}m/{rot_error:6.3f}rad  "
                    f"|action|max={np.max(np.abs(simulation.last_action)):6.3f}  "
                    f"target_xyz={target_position_base.round(3).tolist()}  "
                    f"target_rpy={target_rpy.round(3).tolist()}  "
                    "gripper=fixed"
                )
                if args.leg_debug:
                    joint_position, _ = simulation.joint_state()
                    leg_q = joint_position[LEG_IDS]
                    leg_actor_raw = simulation.raw_action[LEG_IDS]
                    leg_action = simulation.effective_action[LEG_IDS]
                    leg_desired = simulation.policy_desired_position[LEG_IDS]
                    leg_cmd = simulation.desired_position[LEG_IDS]
                    print(f"leg_q={format_named(LEG_NAMES, leg_q)}")
                    print(
                        "leg_actor_raw="
                        f"{format_named(LEG_NAMES, leg_actor_raw)}"
                    )
                    print(f"leg_action={format_named(LEG_NAMES, leg_action)}")
                    print(f"leg_desired={format_named(LEG_NAMES, leg_desired)}")
                    print(f"leg_cmd={format_named(LEG_NAMES, leg_cmd)}")
                    print(
                        "leg_cmd_step="
                        f"{format_named(LEG_NAMES, simulation.policy_command_step[LEG_IDS])}"
                    )
                    print(
                        "leg_track_error="
                        f"{format_named(LEG_NAMES, leg_cmd - leg_q)}"
                    )
                last_log_wall = now
                last_log_sim = simulation.data.time
                viewer_sync_count = 0

    try:
        if args.headless:
            if run_duration <= 0:
                raise ValueError("--headless requires --duration greater than zero")
            loop()
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(model, simulation.data) as viewer_handle:
                configure_viewer(
                    viewer_handle,
                    simulation.base_body_id,
                    track_robot=args.track_camera,
                )
                loop(viewer_handle)
    except KeyboardInterrupt:
        print("\n[sim2sim] 用户停止。")
    finally:
        quest.close()

    pos_error, rot_error = simulation.error()
    ee_position, _ = simulation.ee_pose_base()
    print(
        f"[sim2sim] 结束：EE(base)={ee_position.tolist()}, "
        f"最终误差={pos_error:.4f} m / {rot_error:.4f} rad"
    )


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"[sim2sim] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
