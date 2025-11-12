#!/usr/bin/env python3

import rclpy
import threading
import numpy as np
import time

from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from gazebo_msgs.srv import ResetSimulation

import gymnasium as gym
from gymnasium import spaces

class HRI_Env(gym.Env, Node):
    """
    A custom Gymnasium environment for the UR5e robot in Gazebo.
    This environment is for the "Phase 2" test: learn to do nothing.
    """
    metadata = {'render_modes': ['human']}

    def _init_(self):
        # Initialize as a Gym Env
        super(HRI_Env, self)._init_()
        
        # Initialize as a ROS2 Node
        # We use a unique name to avoid conflicts
        Node._init_(self, 'hri_gym_env_node')
        
        # --- ROS2 Setup ---
        # We use a reentrant callback group to allow for service calls
        # within callbacks (like in the reset function)
        self.callback_group = ReentrantCallbackGroup()
        
        # ROS2 publishers & subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10,
            callback_group=self.callback_group
        )
        self.velocity_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_group_velocity_controller/commands',
            10,
            callback_group=self.callback_group
        )
        
        # ROS2 service clients
        self.reset_sim_client = self.create_client(
            ResetSimulation, 
            '/reset_simulation',
            callback_group=self.callback_group
        )
        while not self.reset_sim_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Gazebo reset service not available, waiting...')

        # --- Gym Environment Setup ---
        
        # Action: 6 joint velocities (rad/s)
        # We'll clip actions to a reasonable range
        self.max_velocity = 2.0 # rad/s
        self.action_space = spaces.Box(
            low=-self.max_velocity, 
            high=self.max_velocity, 
            shape=(6,), 
            dtype=np.float32
        )

        # Observation: 12 values (6 joint positions, 6 joint velocities)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(12,), 
            dtype=np.float32
        )

        # --- State Management ---
        # The joint names in the order used by the controller
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        self.current_joint_state = None
        self.observation = np.zeros(12, dtype=np.float32)
        self.step_count = 0
        self.max_steps_per_episode = 200

        # --- ROS2 Spinning ---
        # We must spin the ROS2 node in a separate thread
        # so that it doesn't block the Gym environment's step/reset methods
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self)
        self.executor_thread = threading.Thread(target=self.executor.spin)
        self.executor_thread.daemon = True
        self.executor_thread.start()
        
        self.get_logger().info('HRI Environment (Phase 2) Initialized')
        self.get_logger().info('Waiting 2s for joint state... ')
        time.sleep(2.0) # Give time for joint_states to arrive
        
    def _joint_state_callback(self, msg):
        """Stores the latest joint state message."""
        self.current_joint_state = msg

    def _get_observation(self):
        """
        Gets the current observation from the robot.
        This function re-orders the joint_states message to match
        our controller's joint_names order.
        """
        if self.current_joint_state is None:
            self.get_logger().warn('No joint state received yet. Returning zeros.')
            return np.zeros(12, dtype=np.float32)

        # Create a mapping from joint name to its index in the msg
        name_to_index = {name: i for i, name in enumerate(self.current_joint_state.name)}

        # Order the positions and velocities according to self.joint_names
        ordered_positions = []
        ordered_velocities = []
        for name in self.joint_names:
            if name in name_to_index:
                idx = name_to_index[name]
                ordered_positions.append(self.current_joint_state.position[idx])
                ordered_velocities.append(self.current_joint_state.velocity[idx])
            else:
                self.get_logger().warn(f"Joint '{name}' not found in /joint_states")
                ordered_positions.append(0.0)
                ordered_velocities.append(0.0)
        
        # Concatenate to form the 12-dim observation
        self.observation = np.concatenate([
            np.array(ordered_positions, dtype=np.float32),
            np.array(ordered_velocities, dtype=np.float32)
        ])
        return self.observation

    def step(self, action):
        """Execute one time step within the environment."""
        self.step_count += 1
        
        # 1. Send Action to Robot
        action = np.clip(action, -self.max_velocity, self.max_velocity)
        vel_msg = Float64MultiArray()
        vel_msg.data = action.astype(float).tolist()
        self.velocity_pub.publish(vel_msg)
        
        # 2. Wait for one "step"
        # This is a simple way to control the step frequency
        # A more robust way would use ROS2 clock
        time.sleep(0.1) # 10 Hz
        
        # 3. Get New Observation
        obs = self._get_observation()
        
        # 4. Calculate Reward (Phase 2: "Smoothness")
        # We penalize all movement. The agent's goal is to learn to
        # send [0,0,0,0,0,0] to minimize this penalty.
        joint_velocities = obs[6:]
        velocity_norm = np.linalg.norm(joint_velocities)
        reward = -velocity_norm # Penalty for moving
        
        # 5. Check for Termination
        terminated = False # This task doesn't have a "success" state
        
        # 6. Check for Truncation (Timeout)
        truncated = False
        if self.step_count >= self.max_steps_per_episode:
            truncated = True
            
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        self.step_count = 0
        
        # 1. Stop the robot
        stop_msg = Float64MultiArray()
        stop_msg.data = [0.0] * 6
        self.velocity_pub.publish(stop_msg)

        # 2. Call the Gazebo reset service
        reset_req = ResetSimulation.Request()
        future = self.reset_sim_client.call_async(reset_req)
        
        # We must spin until the service call is complete
        # This is why we need the separate thread and ReentrantCallbackGroup
        while rclpy.ok() and not future.done():
            pass # Wait
        
        if future.result() is None:
            self.get_logger().error('Failed to call Gazebo reset service')
        
        # 3. Get the initial observation
        # Give Gazebo a moment to settle after reset
        time.sleep(0.5)
        obs = self._get_observation()
        
        return obs, {}

    def render(self, mode='human'):
        # Gazebo is the renderer
        pass

    def close(self):
        """Clean up the environment."""
        self.get_logger().info('Closing HRI Environment')
        self.executor.shutdown()
        self.destroy_node()
