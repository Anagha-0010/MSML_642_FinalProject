# hri_env_final.py

import rclpy
from rclpy.node import Node
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import csv, os, time

from hri_control.fk_helper import FKWrapper

UR5_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

MAX_VEL = 0.4

class HriEnv(Node, gym.Env):
    def __init__(self):
        super().__init__("hri_env_node")
        gym.Env.__init__(self)

        self.get_logger().info("HRI Environment FINAL starting...")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_cb, 10
        )
        self.target_sub = self.create_subscription(
            PoseStamped, "/target_pose", self.target_cb, 10
        )
        self.cmd_pub = self.create_publisher(
            JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10
        )

        self.last_pos = None
        self.last_vel = None
        self.last_target = None
        self.state_received = False

        self.fk = FKWrapper(self, UR5_JOINTS)

        self.prev_action = np.zeros(6, dtype=np.float32)
        self.timer_period = 0.1
        self.current_step = 0
        self.episode_reward = 0.0

        # CSV logging
        self.log_path = os.path.expanduser("~/hri_reward_log.csv")
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "reward"])

        self.episode_count = 0

    def joint_cb(self, msg):
        pos = []
        vel = []
        dpos = dict(zip(msg.name, msg.position))
        dvel = dict(zip(msg.name, msg.velocity))

        for j in UR5_JOINTS:
            pos.append(dpos.get(j, 0.0))
            vel.append(dvel.get(j, 0.0))

        self.last_pos = np.array(pos, dtype=np.float32)
        self.last_vel = np.array(vel, dtype=np.float32)
        self.state_received = True

    def target_cb(self, msg):
        self.last_target = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ], dtype=np.float32)

    def _publish_action(self, action):
        scaled = action * MAX_VEL
        new_pos = (self.last_pos + scaled * self.timer_period).tolist()

        traj = JointTrajectory()
        traj.joint_names = UR5_JOINTS
        p = JointTrajectoryPoint()
        p.positions = new_pos
        p.time_from_start = Duration(sec=0, nanosec=int(self.timer_period * 1e9))
        traj.points.append(p)

        self.cmd_pub.publish(traj)

    def step(self, action):
        if self.last_pos is None or self.last_target is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Smooth action
        action = 0.7 * action + 0.3 * self.prev_action
        self.prev_action = action

        self._publish_action(action)

        # Wait until next state
        end_time = self.get_clock().now().nanoseconds + int(self.timer_period * 1e9)
        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)

        ee = self.fk.compute_fk(self.last_pos)

        obs = np.concatenate([
            self.last_pos,
            self.last_vel,
            ee,
            self.last_target
        ])

        dist = np.linalg.norm(ee - self.last_target)

        action_penalty = 0.01 * np.sum(np.square(action))
        smooth_penalty = 0.02 * np.sum(np.square(action - self.prev_action))

        reward = -dist - action_penalty - smooth_penalty

        self.episode_reward += reward
        self.current_step += 1

        terminated = False
        truncated = (self.current_step >= 200)

        if truncated:
            with open(self.log_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow([self.episode_count, self.episode_reward])

            self.get_logger().info(f"Episode finished. Total reward: {self.episode_reward:.3f}")
            self.episode_count += 1

        return obs.astype(np.float32), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.episode_reward = 0.0
        self.prev_action = np.zeros(6)

        while (self.last_pos is None or self.last_target is None):
            rclpy.spin_once(self, timeout_sec=0.1)

        ee = self.fk.compute_fk(self.last_pos)

        obs = np.concatenate([
            self.last_pos,
            self.last_vel,
            ee,
            self.last_target
        ])

        return obs.astype(np.float32), {}

