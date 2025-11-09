# This file is hri_env.py
# It replaces the old hri_node.py
# This class is BOTH a ROS2 Node and a Gymnasium Environment

import rclpy
from rclpy.node import Node
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
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
# This scales the agent's action (which is from -1 to 1).
MAX_JOINT_VELOCITY = 0.5 

class HriEnv(Node, gym.Env):
    """
    A Gymnasium Environment that is also a ROS2 Node.
    This environment is for the UR5e robot in Gazebo.
    """
    
    # Gymnasium metadata
    metadata = {'render_modes': ['human']}

    def __init__(self):
        # Initialize the ROS2 Node first
        # We use a unique name to avoid conflicts if the 'train' script also inits rclpy
        super().__init__('hri_env_node') 
        
        # Initialize the Gymnasium Environment
        gym.Env.__init__(self)

        self.get_logger().info("HRI Environment is starting...")

        # --- Define Gym Spaces ---
        # Observation Space: 6 joint positions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
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
        
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10)

        # --- State Variables ---
        self.last_joint_state = None      # Stores the last received joint positions
        self.current_step = 0
        self.timer_period = 0.1           # 10Hz, this is our "step" time
        self.last_action_time = self.get_clock().now()

        # A flag to ensure we have received the first state
        self.state_received = False

        self.get_logger().info("HRI Environment initialized.")

    def joint_state_callback(self, msg):
        """
        This function is called every time a new /joint_states message is received.
        It re-orders the received joint states to match our UR5E_JOINT_NAMES list.
        """
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
                # Use the last known value if a joint is missing
                try:
                    idx = UR5E_JOINT_NAMES.index(name)
                    ordered_positions.append(self.last_joint_state[idx])
                except (ValueError, IndexError):
                    ordered_positions.append(0.0)
        
        self.last_joint_state = ordered_positions

        # Log the position of the first joint, throttled to 1 sec
        self.get_logger().info(
            f"Pan joint is at: {self.last_joint_state[0]:.4f} rad",
            throttle_duration_sec=1.0
        )

    def _publish_action(self, action_velocities):
        """
        Converts the agent's velocity action into a
        JointTrajectoryPoint and publishes it.
        """
        if self.last_joint_state is None:
            self.get_logger().warn("Skipping action, no joint state received yet.")
            return

        traj_msg = JointTrajectory()
        traj_msg.joint_names = UR5E_JOINT_NAMES

        point = JointTrajectoryPoint()
        
        # Calculate target positions based on velocities
        # target_pos = current_pos + (velocity * time_step)
        target_positions = [
            current + (velocity * self.timer_period) 
            for current, velocity in zip(self.last_joint_state, action_velocities)
        ]
        
        point.positions = target_positions
        
        # Set the time for this point to be timer_period seconds from now
        time_to_reach_ns = int(self.timer_period * 1e9)
        point.time_from_start = Duration(sec=time_to_reach_ns // 1000000000,
                                         nanosec=time_to_reach_ns % 1000000000)

        traj_msg.points.append(point)
        self.trajectory_pub.publish(traj_msg)

    def step(self, action):
        """
        This is the main RL loop step.
        The agent provides an 'action', and we return the 'next_state', 'reward', etc.
        """
        self.current_step += 1
        
        # 1. Take the Action
        # Scale the normalized action (-1 to 1) to the real velocity
        scaled_velocities = [a * MAX_JOINT_VELOCITY for a in action]
        self._publish_action(scaled_velocities)
        self.last_action_time = self.get_clock().now()

        # 2. Wait for the action to take effect
        # We spin the node to receive the new joint_state
        # This is a simple way to "wait" for the next state
        # We spin for the timer_period, ensuring callbacks are processed.
        end_time = self.last_action_time.nanoseconds + self.timer_period * 1e9
        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.01) # Spin briefly
            # A small sleep to prevent busy-waiting
            time.sleep(0.001) 
        
        # 3. Get the Next State
        # The joint_state_callback has updated self.last_joint_state
        if not self.state_received:
            self.get_logger().error("Failed to get new joint state in step!")
            # Return a default state and 0 reward, and mark as done
            next_state = np.zeros(self.observation_space.shape, dtype=np.float32)
            return next_state, 0.0, True, False, {"error": "Failed to get state"}

        next_state = np.array(self.last_joint_state, dtype=np.float32)

        # 4. Calculate the Reward
        # This is Iteration 1: "Smoothness"
        # We penalize high velocities (jerky movements)
        # The agent will learn to stay still to maximize reward (minimize penalty)
        reward = -np.sum(np.square(scaled_velocities))

        # 5. Check if Done
        # For Iteration 1, the task never ends.
        # TODO: Add a "done" condition (e.g., robot falls over)
        terminated = False
        
        # Check for truncation (time limit)
        truncated = False 
        if self.current_step > 1000: # Limit episode to 1000 steps (100 sec)
            truncated = True
            self.get_logger().info("Episode truncated due to time limit.")

        # 6. Return the standard Gym 5-tuple
        # (observation, reward, terminated, truncated, info)
        return next_state, float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """
        Called at the beginning of each episode.
        Resets the environment.
        """
        if seed is not None:
            # Handle the seed if Gymnasium requires it
            # This is simplified; a full implementation would use the seed
            super().reset(seed=seed)
            
        self.current_step = 0
        
        # TODO: Call the /reset_simulation service to put the robot back
        # For now, we just wait for the first joint state
        
        self.get_logger().info("--- Episode Reset ---")
        
        # Wait until we get a valid state
        # We spin the node to process the subscription
        while not self.state_received:
            self.get_logger().warn("Waiting for first joint state in reset...")
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.01)
            
        initial_state = np.array(self.last_joint_state, dtype=np.float32)
        
        # Return the standard 2-tuple (observation, info)
        return initial_state, {}

    def close(self):
        """
        Cleans up the node when the environment is closed.
        """
        self.get_logger().info("Closing HRI Environment.")
        self.destroy_node()

# Note: We don't have a main() function here because
# this file is a "library" to be imported by train.py
