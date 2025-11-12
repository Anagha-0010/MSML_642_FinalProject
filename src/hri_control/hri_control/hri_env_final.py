#!/usr/bin/env python3

import rclpy
import threading
import numpy as np
import time

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, Point
from gazebo_msgs.srv import ResetSimulation

import gymnasium as gym
from gymnasium import spaces

import tf2_ros
from tf2_ros import Buffer, TransformListener

class HRI_Env_Final(gym.Env, Node):
    """
    Final Gymnasium environment for the HRI handover task.
    
    - Observation: 15-dim (6 pos, 6 vel, 3 relative_vec_to_hand)
    - Action: 6-dim (joint velocities)
    - Reward: Proximity to hand + smoothness
    """
    metadata = {'render_modes': ['human']}

    def _init_(self):
        # Initialize as a Gym Env
        super(HRI_Env_Final, self)._init_()
        
        # Initialize as a ROS2 Node
        Node._init_(self, 'hri_gym_env_final_node')
        
        # --- ROS2 Setup ---
        self.callback_group = ReentrantCallbackGroup()
        
        # TF2 listener to get end-effector position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.eef_frame = 'tool0' # UR5e's end-effector frame
        self.base_frame = 'base_link'

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
        self.hand_pose_sub = self.create_subscription(
            PoseStamped,
            '/hand_pose',
            self._hand_pose_callback,
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
        self.max_velocity = 2.0 # rad/s
        self.action_space = spaces.Box(
            low=-self.max_velocity, 
            high=self.max_velocity, 
            shape=(6,), 
            dtype=np.float32
        )

        # Observation: 15 values
        # 6 joint positions
        # 6 joint velocities
        # 3 (x,y,z) vector from end-effector to hand
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(15,), 
            dtype=np.float32
        )

        # --- State Management ---
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        self.current_joint_state = None
        self.current_hand_pose = Point(x=0.5, y=0.0, z=1.3) # Default start
        self.current_eef_pose = Point(x=0.0, y=0.0, z=0.0)
        self.observation = np.zeros(15, dtype=np.float32)
        
        self.step_count = 0
        self.max_steps_per_episode = 500 # More steps to find the hand

        # Reward weights (IMPORTANT: These need tuning)
        self.W_PROXIMITY = 1.0 # Weight for distance reward
        self.W_SMOOTHNESS = 0.1 # Weight for velocity penalty
        self.SUCCESS_BONUS = 1000.0
        self.SUCCESS_DISTANCE_THRESHOLD = 0.05 # 5 cm
        self.SUCCESS_VELOCITY_THRESHOLD = 0.1 # 10 cm/s (eef vel)
        
        # --- ROS2 Spinning ---
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self)
        self.executor_thread = threading.Thread(target=self.executor.spin)
        self.executor_thread.daemon = True
        self.executor_thread.start()
        
        self.get_logger().info('HRI Environment (FINAL) Initialized')
        self.get_logger().info('Waiting 2s for joint state and hand pose... ')
        time.sleep(2.0)
        
    def _joint_state_callback(self, msg):
        self.current_joint_state = msg

    def _hand_pose_callback(self, msg):
        self.current_hand_pose = msg.pose.position

    def _get_observation(self):
        """
        Gets the current 15-dim observation from the robot and environment.
        """
        # --- 1. Get Joint State (12 dims) ---
        if self.current_joint_state is None:
            self.get_logger().warn('No joint state received. Returning last obs.')
            return self.observation # Return last known obs

        name_to_index = {name: i for i, name in enumerate(self.current_joint_state.name)}
        ordered_positions = []
        ordered_velocities = []
        for name in self.joint_names:
            idx = name_to_index.get(name)
            if idx is not None:
                ordered_positions.append(self.current_joint_state.position[idx])
                ordered_velocities.append(self.current_joint_state.velocity[idx])
            else:
                self.get_logger().warn(f"Joint '{name}' not found. Appending 0.")
                ordered_positions.append(0.0)
                ordered_velocities.append(0.0)
        
        joint_positions = np.array(ordered_positions, dtype=np.float32)
        joint_velocities = np.array(ordered_velocities, dtype=np.float32)

        # --- 2. Get End-Effector to Hand Vector (3 dims) ---
        try:
            # Get the transform from base_link to tool0
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.eef_frame, rclpy.time.Time()
            )
            self.current_eef_pose = transform.transform.translation
            
            # Calculate the relative vector
            eef_vec = np.array([self.current_eef_pose.x, self.current_eef_pose.y, self.current_eef_pose.z])
            hand_vec = np.array([self.current_hand_pose.x, self.current_hand_pose.y, self.current_hand_pose.z])
            relative_vector = hand_vec - eef_vec
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'Could not get transform: {e}. Using last known vector.')
            # Use the last known observation's vector
            relative_vector = self.observation[12:] 

        # --- 3. Concatenate ---
        self.observation = np.concatenate([joint_positions, joint_velocities, relative_vector])
        return self.observation

    def step(self, action):
        self.step_count += 1
        
        # 1. Send Action
        action = np.clip(action, -self.max_velocity, self.max_velocity)
        vel_msg = Float64MultiArray()
        vel_msg.data = action.astype(float).tolist()
        self.velocity_pub.publish(vel_msg)
        
        time.sleep(0.1) # 10 Hz
        
        # 2. Get New Observation
        obs = self._get_observation()
        
        # 3. Calculate Reward (Phase 3: Your proposal)
        joint_velocities = obs[6:12]
        relative_vector = obs[12:]
        
        distance_to_hand = np.linalg.norm(relative_vector)
        
        # Reward for minimizing distance (negative penalty)
        proximity_reward = -distance_to_hand
        
        # Penalty for rapid movements
        smoothness_penalty = -np.linalg.norm(joint_velocities)
        
        # Total Reward
        reward = (self.W_PROXIMITY * proximity_reward) + \
                 (self.W_SMOOTHNESS * smoothness_penalty)
        
        # 4. Check for Termination (Success)
        terminated = False
        eef_velocity_norm = 0.0 # TODO: Calculate this from eef_pose diff
        
        if distance_to_hand < self.SUCCESS_DISTANCE_THRESHOLD:
            # TODO: A more accurate eef_velocity would be better
            # For now, we use joint velocities as a proxy
            if np.linalg.norm(joint_velocities) < self.SUCCESS_VELOCITY_THRESHOLD:
                self.get_logger().info('---!!! SUCCESS !!!---')
                reward += self.SUCCESS_BONUS
                terminated = True
        
        # 5. Check for Truncation (Timeout)
        truncated = False
        if self.step_count >= self.max_steps_per_episode:
            truncated = True
            
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        self.velocity_pub.publish(Float64MultiArray(data=[0.0]*6))
        
        reset_req = ResetSimulation.Request()
        future = self.reset_sim_client.call_async(reset_req)
        while rclpy.ok() and not future.done():
            pass # Wait
        if future.result() is None:
            self.get_logger().error('Failed to call Gazebo reset service')
        
        time.sleep(0.5)
        obs = self._get_observation()
        
        return obs, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.get_logger().info('Closing HRI Final Environment')
        self.executor.shutdown()
        self.destroy_node()
