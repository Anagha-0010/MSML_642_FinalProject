# This file is located at:
# ~/MSML642_final_project/src/hri_control/hri_control/hri_node.py
#
# Remember to re-build your project after editing this file:
# 1. cd ~/MSML642_final_project
# 2. colcon build
# 3. source install/setup.bash

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# The joint names for the UR5e
UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

class HriNode(Node):
    """
    A node to demonstrate subscribing to joint states and publishing
    to the /joint_trajectory_controller.
    """
    def __init__(self):
        super().__init__('hri_node')
        
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)
        
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10)
        
        self.get_logger().info('HRI Node has started. Subscribing to /joint_states...')
        self.get_logger().info('Will publish to /joint_trajectory_controller/joint_trajectory')
        
        self.timer_period = 0.1  # Timer period in seconds
        self.timer = self.create_timer(self.timer_period, self.send_test_command)
        
        self.last_joint_state = None
        self.joint_names_from_msg = [] # Stores the name list from the last message

    def joint_state_callback(self, msg):
        """
        This function is called every time a new /joint_states message is received.
        It re-orders the received joint states to match our UR5E_JOINT_NAMES list.
        """
        if self.last_joint_state is None:
            self.last_joint_state = [0.0] * len(UR5E_JOINT_NAMES)

        # Create a dictionary from the received message for easy lookup
        msg_dict = dict(zip(msg.name, msg.position))
        
        # Re-order the positions to match our UR5E_JOINT_NAMES
        ordered_positions = []
        for name in UR5E_JOINT_NAMES:
            if name in msg_dict:
                ordered_positions.append(msg_dict[name])
            else:
                self.get_logger().warn(f"Joint '{name}' not found in /joint_states.", throttle_duration_sec=5.0)
                # Use the last known value or 0.0 if this is the first time
                try:
                    idx = UR5E_JOINT_NAMES.index(name)
                    ordered_positions.append(self.last_joint_state[idx])
                except (ValueError, IndexError):
                     ordered_positions.append(0.0)
        
        self.last_joint_state = ordered_positions
        self.joint_names_from_msg = msg.name # Store the original names
        
        # --- ADDED FOR PROOF ---
        # Log the position of the first joint, rounded to 4 decimal places
        # This will throttle, printing only once per second.
        self.get_logger().info(
            f"Pan joint is at: {self.last_joint_state[0]:.4f} rad", 
            throttle_duration_sec=1.0
        )
        # --- END OF ADDITION ---

    def send_test_command(self):
        """
        This function is called by the timer and sends a trajectory command
        with a calculated future position.
        """
        if self.last_joint_state is None:
            self.get_logger().warn('No joint states received yet. Skipping command.')
            return

        # Create the trajectory message
        traj_msg = JointTrajectory()
        # We must send the joint names in the order the /joint_states topic uses
        # not our hard-coded list, just in case they are different.
        # But we know from echo they are the same, so UR5E_JOINT_NAMES is safe.
        traj_msg.joint_names = UR5E_JOINT_NAMES

        # Create a single trajectory point
        point = JointTrajectoryPoint()
        
        # --- THIS IS THE CORRECTED LOGIC ---
        
        current_positions = self.last_joint_state
        
        # --- CHANGED FOR PROOF ---
        # We define the velocity we *want*
        target_velocity = 0.5 # 0.5 rad/s on the first joint (WAS 0.1)
        # --- END OF CHANGE ---
        
        # Calculate the target position based on velocity
        # target_position = current_position + (velocity * time_step)
        
        # Create a full list of target velocities
        target_velocities = [0.0] * len(UR5E_JOINT_NAMES)
        target_velocities[0] = target_velocity # Set pan joint velocity
        
        target_positions = [
            current + (velocity * self.timer_period) 
            for current, velocity in zip(current_positions, target_velocities)
        ]
        
        # ONLY send positions. The controller will calculate velocity.
        point.positions = target_positions
        
        # **** WE NO LONGER SEND point.velocities ****
        
        # Set the time for this point to be timer_period seconds from now
        time_to_reach_ns = int(self.timer_period * 1e9)
        point.time_from_start = Duration(sec=time_to_reach_ns // 1000000000,
                                         nanosec=time_to_reach_ns % 1000000000)

        # --- END OF CORRECTED LOGIC ---

        traj_msg.points.append(point)
        
        # --- THIS IS THE FIX ---
        # Publish the message
        self.trajectory_pub.publish(traj_msg)
        self.get_logger().info('Sent test trajectory command.', throttle_duration_sec=1.0) # Log only once per sec
        # --- END OF FIX ---


def main(args=None):
    rclpy.init(args=args)
    
    hri_node = HriNode()
    
    try:
        rclpy.spin(hri_node)
    except KeyboardInterrupt:
        pass
    
    # Destroy the node explicitly
    hri_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


