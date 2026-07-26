#!/usr/bin/env python3
"""
Publish EE target poses to /EEPose_target_world from a UMI pkl trajectory,
replicating PicklePoseSequenceCommand used in LimxEEposeRoughEnvCfg_PLAY.

Key operations (matching IsaacLab play mode exactly):
  1. planar_center: subtract mean of frames [1,2,3] x,y from all frames
  2. EEF frame:     use the controller middle-camera optical frame directly
  3. command_origin anchoring: anchor trajectory to actual EE world pos at start
  4. step rate:     advance int(CONTROL_DT / SIM_DT) = 4 pkl steps per 50Hz publish

Usage:
  rosrun robot_controllers publish_eepose_target_world_from_pkl.py \
    _pickle_path:=.../pushing.pkl \
    _traj_idx:=0 \
    _start_delay_s:=3.0 \
    _loop:=true

SolefootController must have:
  /PointfootCfg/ee_target/use_world_frame: true
"""

from __future__ import annotations

import math
import pickle
from typing import Optional, Tuple

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

# ---------------------------------------------------------------------------
# Parameters matching CommandsCfgPlay in sf_tron1_arm_env_cfg.py
# ---------------------------------------------------------------------------
DEFAULT_PKL_PATH = "/home/phi5090ii/NYX/umi-on-tron-lab/IsaacLab_RFM/data/pushing.pkl"
DEFAULT_PLANAR_CENTER = True
# The URDF's eef_link is already the controller middle-camera optical frame.
TIP_OFFSET_POS = np.zeros(3)
TIP_OFFSET_RPY = (0.0, 0.0, 0.0)
SIM_DT = 0.005    # pkl recording dt = IsaacLab sim dt (200 Hz)
CONTROL_DT = 0.02 # policy / publish rate (50 Hz)
# Fallback EE position in base frame when TF2 and ground truth are unavailable
EE_INIT_POS_BASE = np.array([0.247147, 0.000040, 0.247502])  # camera-center EEF at the configured initial joints
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Math helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def euler_xyz_to_matrix(rpy: Tuple[float, float, float]) -> np.ndarray:
    """Rotation matrix from intrinsic XYZ Euler angles (roll, pitch, yaw).
    Equivalent to scipy.spatial.transform.Rotation.from_euler('xyz', rpy).as_matrix()
    """
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle vector (axis * angle) to 3x3 rotation matrix (Rodrigues)."""
    aa = np.asarray(aa, dtype=float).reshape(3)
    angle = float(np.linalg.norm(aa))
    if angle < 1e-8:
        return np.eye(3)
    axis = aa / angle
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c),     x * y * (1 - c) - z * s,  x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c),      y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s,  c + z * z * (1 - c)    ],
    ])


def matrix_to_quat_wxyz(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(w), float(x), float(y), float(z)


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)    ],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)    ],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------------------
# Main publisher class
# ---------------------------------------------------------------------------

class PklTrajectoryPublisher:

    def __init__(self) -> None:
        rospy.init_node("publish_eepose_target_world_from_pkl", anonymous=False)

        # ROS params
        pkl_path       = rospy.get_param("~pickle_path", DEFAULT_PKL_PATH)
        self.traj_idx  = int(rospy.get_param("~traj_idx", 0))
        self.topic     = rospy.get_param("~topic", "/EEPose_target_world")
        self.frame_id  = rospy.get_param("~frame_id", "world")
        self.loop      = bool(rospy.get_param("~loop", True))
        self.delay_s   = float(rospy.get_param("~start_delay_s", 3.0))
        planar_center  = bool(rospy.get_param("~planar_center", DEFAULT_PLANAR_CENTER))

        # ---- Load & preprocess pkl ----------------------------------------
        with open(pkl_path, "rb") as f:
            episodes = pickle.load(f)
        if isinstance(episodes, dict) and "episodes" in episodes:
            episodes = episodes["episodes"]

        self.ee_pos_all = np.stack(
            [np.asarray(ep["ee_pos"], dtype=float) for ep in episodes], axis=0
        )  # (N_ep, T, 3)
        ee_aa_all = np.stack(
            [np.asarray(ep["ee_axis_angle"], dtype=float) for ep in episodes], axis=0
        )  # (N_ep, T, 3)

        n_ep, n_t, _ = ee_aa_all.shape

        # Convert all axis-angles -> rotation matrices  (N_ep, T, 3, 3)
        rospy.loginfo("Converting %d x %d axis-angle to rotation matrices...", n_ep, n_t)
        self.ee_rot_all = np.stack([
            np.stack([axis_angle_to_matrix(ee_aa_all[i, t]) for t in range(n_t)])
            for i in range(n_ep)
        ])  # (N_ep, T, 3, 3)

        # Planar center: subtract mean of frames [1,2,3] x,y (matches IsaacLab implementation)
        if planar_center:
            start_means = self.ee_pos_all[:, 1:4, :2].mean(axis=1, keepdims=True)  # (N_ep, 1, 2)
            self.ee_pos_all[..., :2] -= start_means
            rospy.loginfo("planar_center applied (mean xy subtracted from frames [1,2,3]).")

        # Kept as an identity transform because the trajectory and eef_link
        # both use the controller middle-camera optical frame.
        tip_rot = euler_xyz_to_matrix(TIP_OFFSET_RPY)
        self.tip_rot_inv = tip_rot.T
        self.tip_pos_inv = -self.tip_rot_inv @ TIP_OFFSET_POS

        self.n_episodes = n_ep
        self.n_steps = n_t
        rospy.loginfo(
            "Loaded pkl: %d episodes, %d steps/ep (%.3fs each, total %.1fs/ep). Loop=%s",
            n_ep, n_t, SIM_DT, n_t * SIM_DT, self.loop
        )

        # ---- ROS I/O -------------------------------------------------------
        self.pub = rospy.Publisher(self.topic, PoseStamped, queue_size=10)
        self._base_pos_w: Optional[np.ndarray] = None
        self._base_quat_wxyz: Optional[np.ndarray] = None
        rospy.Subscriber("/ground_truth/state", Odometry, self._odom_cb, queue_size=1)

        self._tf_buffer = tf2_ros.Buffer(rospy.Duration(5.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        # ---- Playback state ------------------------------------------------
        self.command_origin: Optional[np.ndarray] = None
        self.current_time = 0.0
        self.episode_idx = self.traj_idx % self.n_episodes

    # ------------------------------------------------------------------
    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._base_pos_w = np.array([p.x, p.y, p.z])
        self._base_quat_wxyz = np.array([q.w, q.x, q.y, q.z])

    def _get_ee_world_pos(self) -> Optional[np.ndarray]:
        """Return the camera-center EEF world position.
        Priority: TF2 lookup -> ground truth base + default FK offset.
        """
        # Option 1: TF2
        try:
            tf = self._tf_buffer.lookup_transform(
                "world", "eef_link", rospy.Time(0), rospy.Duration(0.5)
            )
            p = tf.transform.translation
            return np.array([p.x, p.y, p.z])
        except Exception:
            pass

        # Option 2: ground truth base pose + default EE init position in base frame
        if self._base_pos_w is not None and self._base_quat_wxyz is not None:
            R_wb = quat_wxyz_to_matrix(self._base_quat_wxyz)
            return self._base_pos_w + R_wb @ EE_INIT_POS_BASE

        return None

    def _get_target_at_time(self, t_sec: float):
        """Return (world_pos, quat_wxyz, trajectory_done) for the given trajectory time."""
        # step_indices = current_start_time / sim_dt  (matches IsaacLab _update_command_w)
        step_idx = int(t_sec / SIM_DT)
        step_idx = max(0, min(step_idx, self.n_steps - 1))
        ep = self.episode_idx

        target_pos = self.ee_pos_all[ep, step_idx].copy()   # (3,) camera EEF frame, centered
        target_rot = self.ee_rot_all[ep, step_idx]           # (3,3) camera EEF frame

        # Identity trajectory-frame -> eef_link transform
        eef_pos = target_pos + target_rot @ self.tip_pos_inv
        eef_rot = target_rot @ self.tip_rot_inv

        # Anchor to EE world start position
        world_pos = eef_pos + self.command_origin  # type: ignore[operator]
        quat = matrix_to_quat_wxyz(eef_rot)

        done = step_idx >= self.n_steps - 1
        return world_pos, quat, done

    # ------------------------------------------------------------------
    def run(self) -> None:
        rate = rospy.Rate(1.0 / CONTROL_DT)  # 50 Hz

        # Wait for robot to stand up before activating trajectory
        rospy.loginfo("[pkl_traj] Waiting %.1f s for robot to stabilize...", self.delay_s)
        rospy.sleep(self.delay_s)

        # Anchor command_origin to current EE world position
        rospy.loginfo("[pkl_traj] Anchoring command_origin to current EE world pos...")
        for _ in range(30):
            self.command_origin = self._get_ee_world_pos()
            if self.command_origin is not None:
                break
            rospy.sleep(0.1)

        if self.command_origin is None:
            rospy.logwarn("[pkl_traj] Could not get EE world pos, using fallback [0, 0, 1].")
            self.command_origin = np.array([0.0, 0.0, 1.0])

        rospy.loginfo(
            "[pkl_traj] command_origin = [%.3f, %.3f, %.3f]. Starting episode %d.",
            *self.command_origin, self.episode_idx
        )

        self.current_time = 0.0

        while not rospy.is_shutdown():
            world_pos, (w, x, y, z), done = self._get_target_at_time(self.current_time)

            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.frame_id
            msg.pose.position.x = float(world_pos[0])
            msg.pose.position.y = float(world_pos[1])
            msg.pose.position.z = float(world_pos[2])
            msg.pose.orientation.w = float(w)
            msg.pose.orientation.x = float(x)
            msg.pose.orientation.y = float(y)
            msg.pose.orientation.z = float(z)
            self.pub.publish(msg)

            # Advance time: each 50Hz step = 4 pkl steps at 200Hz (matches IsaacLab decimation)
            self.current_time += CONTROL_DT

            if done:
                if self.loop:
                    next_ep = (self.episode_idx + 1) % self.n_episodes
                    rospy.loginfo(
                        "[pkl_traj] Episode %d done. Looping to episode %d.",
                        self.episode_idx, next_ep
                    )
                    self.episode_idx = next_ep
                    self.current_time = 0.0
                    # Re-anchor to current EE position for next episode
                    new_origin = self._get_ee_world_pos()
                    if new_origin is not None:
                        self.command_origin = new_origin
                        rospy.loginfo(
                            "[pkl_traj] Re-anchored origin = [%.3f, %.3f, %.3f].",
                            *self.command_origin
                        )
                else:
                    rospy.loginfo_once("[pkl_traj] Trajectory done. Holding last pose.")

            rate.sleep()


if __name__ == "__main__":
    try:
        PklTrajectoryPublisher().run()
    except rospy.ROSInterruptException:
        pass
