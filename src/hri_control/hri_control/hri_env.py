# This file is hri_env.py
# FINAL VERSION with RL Logic
# This class is BOTH a ROS2 Node and a Gymnasium Environment

import rclpy
from rclpy.node import Node
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker  # NEW: To see the target
import time

# The joint names for the UR5e
UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# The maximum velocity (rad/s) we'll let the agent command.
MAX_JOINT_VELOCITY = 0.5 

class HriEnv(Node, gym.Env):
    """
    A Gymnasium Environment that is also a ROS2 Node.
    This environment is for the UR5e robot in Gazebo.
    """
    
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super().__init__('hri_env_node') 
        gym.Env.__init__(self)

        self.get_logger().info("HRI Environment is starting...")

        # --- NEW: Target Hand Position ---
        self.last_target_position = np.array([0.5, 0.3, 0.5], dtype=np.float32) # Default start

        # --- Define Gym Spaces ---
        # Observation Space: 6 joint positions + 3 target (x, y, z)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32  # SHAPE IS NOW 9
        )
        
        # Action Space: 6 joint velocities (normalized between -1 and 1)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # --- ROS2 Subscribers & Publishers ---
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)
        
        # NEW: Target Position Subscriber (from hand_simulator.py)
        self.target_sub = self.create_subscription(
            Marker,
            '/target_hand_marker',
            self.target_callback,
            10)
        
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10)

        # --- State Variables ---
        self.last_joint_state = None
        self.current_step = 0
        self.timer_period = 0.1
        self.last_action_time = self.get_clock().now()
        self.episode_reward = 0.0
        self.state_received = False

        self.get_logger().info("HRI Environment initialized.")

    def joint_state_callback(self, msg):
        if not self.state_received:
            self.state_received = True
            self.get_logger().info("First joint state received.")

        if self.last_joint_state is None:
            self.last_joint_state = [0.0] * len(UR5E_JOINT_NAMES)

        msg_dict = dict(zip(msg.name, msg.position))
        
        ordered_positions = []
        for name in UR5E_JOINT_NAMES:
            if name in msg_dict:
                ordered_positions.append(msg_dict[name])
            else:
                try:
                    idx = UR5E_JOINT_NAMES.index(name)
                    ordered_positions.append(self.last_joint_state[idx])
                except (ValueError, IndexError):
                    ordered_positions.append(0.0)
        
        self.last_joint_state = ordered_positions

    # NEW: Callback to store the target's position
    def target_callback(self, msg: Marker):
        self.last_target_position = np.array([
            msg.pose.position.x, 
            msg.pose.position.y, 
            msg.pose.position.z
        ], dtype=np.float32)

    def _publish_action(self, action_velocities):
        if self.last_joint_state is None:
            self.get_logger().warn("Skipping action, no joint state received yet.")
            return

        traj_msg = JointTrajectory()
        traj_msg.joint_names = UR5E_JOINT_NAMES
        point = JointTrajectoryPoint()
        
        target_positions = [
            current + (velocity * self.timer_period) 
            for current, velocity in zip(self.last_joint_state, action_velocities)
        ]
        
        point.positions = target_positions
        
        time_to_reach_ns = int(self.timer_period * 1e9)
        point.time_from_start = Duration(sec=time_to_reach_ns // 1000000000,
                                         nanosec=time_to_reach_ns % 1000000000)

        traj_msg.points.append(point)
        self.trajectory_pub.publish(traj_msg)

    def step(self, action):
        self.current_step += 1
        
        # 1. Take the Action
        scaled_velocities = [a * MAX_JOINT_VELOCITY for a in action]
        self._publish_action(scaled_velocities)
        self.last_action_time = self.get_clock().now()

        # 2. Wait for the action to take effect
        end_time = self.last_action_time.nanoseconds + self.timer_period * 1e9
        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.001) 
        
        # 3. Get the Next State
        if not self.state_received:
            self.get_logger().error("Failed to get new joint state in step!")
            next_state = np.zeros(self.observation_space.shape, dtype=np.float32)
            return next_state, 0.0, True, False, {"error": "Failed to get state"}

        # NEW: State is 6 robot joints + 3 target coordinates
        robot_state = np.array(self.last_joint_state, dtype=np.float32)
        target_state = self.last_target_position
        next_state = np.concatenate([robot_state, target_state])

        # 4. Calculate the Reward
        
        # NOTE: This is a simple proxy for EE position.
        # A full Forward Kinematics (FK) lookup is the true solution.
        robot_ee_proxy_pos = np.array([
            self.last_joint_state[3], # wrist_1_joint
            self.last_joint_state[4], # wrist_2_joint
            self.last_joint_state[5]  # wrist_3_joint
        ])

        # a) Distance Reward (The GOAL)
        distance_to_target = np.linalg.norm(robot_ee_proxy_pos - target_state)
        # We want to minimize distance, so reward is the negative distance.
        distance_reward = -distance_to_target
        
        # b) Velocity Penalty (The SMOOTHNESS)
        # Penalize large movements to keep it smooth
        velocity_penalty = -0.01 * np.sum(np.square(scaled_velocities))

        # c) Final Reward
        reward = distance_reward + velocity_penalty
        
        self.episode_reward += reward

        # 5. Check if Done
        terminated = False
        truncated = False 
        
        if self.current_step > 150:
            truncated = True
            self.get_logger().info(f"--- EPISODE FINISHED --- Total Reward: {self.episode_reward:.2f} ---")
        
        return next_state, float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            
        self.current_step = 0
        self.episode_reward = 0.0
        
        self.get_logger().info("--- Episode Reset ---")
        
        while not self.state_received:
            self.get_logger().warn("Waiting for first joint state in reset...")
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.01)
            
        # NEW: State is 6 robot joints + 3 target coordinates
        robot_state = np.array(self.last_joint_state, dtype=np.float32)
        target_state = self.last_target_position
        initial_state = np.concatenate([robot_state, target_state])
        
        return initial_state, {}

    def close(self):
        self.get_logger().info("Closing HRI Environment.")
        self.destroy_node()
