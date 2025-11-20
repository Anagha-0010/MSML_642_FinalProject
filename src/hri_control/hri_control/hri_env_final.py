# hri_env_final.py

import rclpy
from rclpy.node import Node

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration

from hri_control.fk_helper import FKWrapper


# Order of joints for UR5e
UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

MAX_JOINT_VELOCITY = 0.5  # rad/s max from action scaling


class HriEnv(Node, gym.Env):
    """
    Final version of RL environment for Human-Robot Imitation.
    Includes:
      - FK-based end effector position
      - Marker-based moving target
      - UR5e joint control
      - Clean observation + reward
    """

    def __init__(self):
        super().__init__("hri_env_node")
        gym.Env.__init__(self)

        self.get_logger().info("HRI Environment FINAL starting...")

        # obs = joint_pos(6) + joint_vel(6) + ee_xyz(3) + target_xyz(3)
        obs_dim = 6 + 6 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # actions = desired joint velocity commands (6)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # subscribers & publishers
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_cb, 10
        )

        # **CORRECT FIX** — Subscribe to Marker topic published by hand_simulator
        self.target_sub = self.create_subscription(
            Marker, "/target_hand_marker", self.target_cb, 10
        )

        self.trajectory_pub = self.create_publisher(
            JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10
        )

        # Buffers
        self.last_joint_positions = None
        self.last_joint_velocities = None
        self.last_target_pos = None
        self.state_received = False
        self.target_received = False

        # FK wrapper
        try:
            self.fk = FKWrapper(self, UR5E_JOINT_NAMES)
            self.get_logger().info(
                f"FK initialized. Base: {self.fk.base_link}, EE: {self.fk.ee_link}"
            )
        except Exception as e:
            self.get_logger().error(f"FK initialization failed: {e}")
            raise

        # Step timing
        self.timer_period = 0.1  # 10 Hz control
        self.current_step = 0
        self.episode_reward = 0.0

    # --------------------------
    #  CALLBACKS
    # --------------------------

    def joint_state_cb(self, msg: JointState):
        """Map joint states into consistent UR5e order."""
        msg_pos = dict(zip(msg.name, msg.position))
        msg_vel = dict(zip(msg.name, msg.velocity if msg.velocity else [0]*len(msg_pos)))

        pos = [msg_pos.get(name, 0.0) for name in UR5E_JOINT_NAMES]
        vel = [msg_vel.get(name, 0.0) for name in UR5E_JOINT_NAMES]

        self.last_joint_positions = np.array(pos, dtype=np.float32)
        self.last_joint_velocities = np.array(vel, dtype=np.float32)
        self.state_received = True

    def target_cb(self, msg: Marker):
        """Extract XYZ from Marker.pose."""
        self.last_target_pos = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float32,
        )
        self.target_received = True

    # --------------------------
    #  ACTION APPLICATION
    # --------------------------

    def _publish_action(self, action_vel):
        """Convert action into JointTrajectory position targets."""
        if self.last_joint_positions is None:
            return

        scaled = np.clip(action_vel, -1, 1) * MAX_JOINT_VELOCITY
        new_positions = self.last_joint_positions + scaled * self.timer_period

        traj = JointTrajectory()
        traj.joint_names = UR5E_JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = new_positions.tolist()
        point.time_from_start = Duration(sec=0, nanosec=int(self.timer_period * 1e9))

        traj.points.append(point)
        self.trajectory_pub.publish(traj)

    # --------------------------
    #  STEP FUNCTION
    # --------------------------

    def step(self, action):
        """Main RL loop step."""
        # Ensure we have sensor data
        if not self.state_received or not self.target_received:
            rclpy.spin_once(self, timeout_sec=0.05)
            if not self.state_received or not self.target_received:
                return (
                    np.zeros(self.observation_space.shape, dtype=np.float32),
                    -1.0,
                    True,
                    False,
                    {"error": "missing data"},
                )

        # Apply joint action
        self._publish_action(action)

        # Wait one control period
        t_start = self.get_clock().now().nanoseconds
        while self.get_clock().now().nanoseconds < t_start + int(self.timer_period * 1e9):
            rclpy.spin_once(self, timeout_sec=0.01)

        # Compute FK end effector
        ee_pos = self.fk.compute_fk(self.last_joint_positions)

        # Build observation
        obs = np.concatenate(
            [
                self.last_joint_positions,
                self.last_joint_velocities,
                ee_pos,
                self.last_target_pos,
            ]
        ).astype(np.float32)

        # Reward = negative distance + small penalty
        distance = np.linalg.norm(ee_pos - self.last_target_pos)
        vel_penalty = 0.01 * np.sum((action * MAX_JOINT_VELOCITY) ** 2)

        reward = -distance - vel_penalty

        self.episode_reward += reward
        self.current_step += 1

        # Episode termination
        terminated = False
        truncated = False

        if self.current_step >= 200:
            truncated = True
            self.get_logger().info(
                f"Episode finished. Total reward: {self.episode_reward:.3f}"
            )

        return obs, float(reward), terminated, truncated, {}

    # --------------------------
    #  RESET FUNCTION
    # --------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)

        self.current_step = 0
        self.episode_reward = 0.0

        # Wait for data
        t0 = self.get_clock().now().nanoseconds
        timeout = 5.0
        while (not self.state_received or not self.target_received) and \
              ((self.get_clock().now().nanoseconds - t0) * 1e-9 < timeout):
            rclpy.spin_once(self, timeout_sec=0.1)

        if not self.state_received or not self.target_received:
            raise RuntimeError("Timeout waiting for joint_state or target marker")

        # Compute initial FK
        ee_pos = self.fk.compute_fk(self.last_joint_positions)

        obs = np.concatenate(
            [
                self.last_joint_positions,
                self.last_joint_velocities if self.last_joint_velocities is not None else np.zeros(6),
                ee_pos,
                self.last_target_pos,
            ]
        ).astype(np.float32)

        return obs, {}

