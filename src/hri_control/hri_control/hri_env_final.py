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

        self.get_logger().info("HRI Environment FINAL (Reward-Shaped) starting...")

        # Observation: 6 joint pos + 6 vel + 7 EE pose (pos+rot) + 7 target (pos+rot) = 26
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # ROS
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_cb, 10
        )
        self.target_sub = self.create_subscription(
            PoseStamped, "/target_pose", self.target_cb, 10
        )
        self.cmd_pub = self.create_publisher(
            JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10
        )

        # State vars
        self.last_pos = None
        self.last_vel = None
        self.last_target = None
        self.state_received = False

        self.fk = FKWrapper(self, UR5_JOINTS)

        self.prev_action = np.zeros(6, dtype=np.float32)
        self.timer_period = 0.1
        self.current_step = 0
        self.episode_reward = 0.0
        self.prev_dist = 0.0 # Initialize

        # CSV logging
        self.log_path = os.path.expanduser("~/hri_reward_log.csv")
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "reward"])

        self.episode_count = 0

    # -----------------------------------------------------
    # CALLBACKS
    # -----------------------------------------------------

    def joint_cb(self, msg):
        pos, vel = [], []
        dpos = dict(zip(msg.name, msg.position))
        dvel = dict(zip(msg.name, msg.velocity))

        for j in UR5_JOINTS:
            pos.append(dpos.get(j, 0.0))
            vel.append(dvel.get(j, 0.0))

        self.last_pos = np.array(pos, dtype=np.float32)
        self.last_vel = np.array(vel, dtype=np.float32)
        self.state_received = True

    def target_cb(self, msg):
        # Now captures Position (3) + Orientation (4) = 7 items
        self.last_target = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float32,
        )

    # -----------------------------------------------------
    # ACTION -> TRAJECTORY
    # -----------------------------------------------------

    def _publish_action(self, action):
        scaled = action * MAX_VEL
        new_pos = (self.last_pos + scaled * self.timer_period).tolist()

        traj = JointTrajectory()
        traj.joint_names = UR5_JOINTS

        p = JointTrajectoryPoint()
        p.positions = new_pos
        p.time_from_start = Duration(
            sec=0,
            nanosec=int(self.timer_period * 1e9)
        )
        traj.points.append(p)

        self.cmd_pub.publish(traj)

    # -----------------------------------------------------
    # STEP
    # -----------------------------------------------------

    def step(self, action):
        if self.last_pos is None or self.last_target is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        # 1. Action Smoothing
        old_action = self.prev_action.copy()
        action = 0.2 * action + 0.8 * old_action
        self.prev_action = action

        # Apply action
        self._publish_action(action)

        # Wait for physics
        end_time = self.get_clock().now().nanoseconds + int(self.timer_period * 1e9)
        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)

        # 2. Get State
        ee_state = self.fk.compute_fk(self.last_pos)
        ee_pos = ee_state[:3]
        ee_quat = ee_state[3:]

        target_pos = self.last_target[:3]
        target_quat = self.last_target[3:]
        
        # Calculate Distances
        dist = float(np.linalg.norm(ee_pos - target_pos))
        
        # Calculate Distance Change (Shaping Reward)
        # Use the value calculated in reset() or previous step
        delta_dist = self.prev_dist - dist
        self.prev_dist = dist

        # Build Observation
        obs = np.concatenate([self.last_pos, self.last_vel, ee_state, self.last_target])

        # Reward function

        #Distance Reward
        r_dist = float(np.exp(-1.0 * dist))

        #Progress Reward
        r_progress = 10.0 * delta_dist 

        #Orientation Reward
        dot_prod = np.dot(ee_quat, target_quat)
        r_orient = float(dot_prod ** 2)

        #Penalties
        action_penalty = 0.01 * np.sum(np.square(action))
        smooth_penalty = 0.01 * np.sum(np.square(action - old_action))

        #Total reward
        #Weights: Distance(2.0) + Progress(1.0) + Orientation(0.5)
        reward = (2.0 * r_dist) + r_progress + (0.5 * r_orient) - action_penalty - smooth_penalty

        # Logging
        if self.current_step % 20 == 0:
            self.get_logger().info(f"Dist: {dist:.3f} | Prog: {r_progress:.3f} | Rew: {reward:.3f}")

        # Termination
        terminated = dist < 0.05 and r_orient > 0.9
        truncated = self.current_step >= 200

        self.episode_reward += reward
        self.current_step += 1
        
        info = {"dist": dist}

        if terminated or truncated:
            with open(self.log_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow([self.episode_count, self.episode_reward])
            self.get_logger().info(f"Episode {self.episode_count} done | Reward: {self.episode_reward:.3f}")
            self.episode_count += 1
            # Note: prev_dist reset is now handled in reset(), not here.

        return obs.astype(np.float32), reward, terminated, truncated, info

    # -----------------------------------------------------
    # RESET
    # -----------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.episode_reward = 0.0
        self.prev_action = np.zeros(6)

        while self.last_pos is None or self.last_target is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        ee_state = self.fk.compute_fk(self.last_pos)
        ee_pos = ee_state[:3]
        
        # --- FIX: Initialize prev_dist correctly for the new episode ---
        target_pos = self.last_target[:3]
        self.prev_dist = float(np.linalg.norm(ee_pos - target_pos))
        # ---------------------------------------------------------------

        obs = np.concatenate([self.last_pos, self.last_vel, ee_state, self.last_target])
        return obs.astype(np.float32), {}
