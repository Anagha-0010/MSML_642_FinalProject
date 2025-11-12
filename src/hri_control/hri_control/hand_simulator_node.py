#!/usr/bin/env python3

import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time

class HandSimulatorNode(Node):
    """
    Simulates a moving hand by publishing a PoseStamped message
    on the /hand_pose topic.
    
    The hand moves in a horizontal circle.
    """
    def _init_(self):
        super()._init_('hand_simulator_node')
        
        self.publisher_ = self.create_publisher(PoseStamped, '/hand_pose', 10)
        
        # Simulation parameters
        self.center_x = 0.5
        self.center_y = 0.0
        self.center_z = 1.3
        self.radius = 0.2
        self.angular_velocity = 0.5 # rad/s
        self.publish_rate = 50.0 # Hz
        
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        
        # Create a timer to publish at a fixed rate
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)
        
        self.get_logger().info('Hand Simulator Started. Publishing to /hand_pose')

    def timer_callback(self):
        # Calculate elapsed time
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed_time = current_time - self.start_time
        
        # Calculate circular path
        angle = self.angular_velocity * elapsed_time
        x = self.center_x + self.radius * math.cos(angle)
        y = self.center_y + self.radius * math.sin(angle)
        z = self.center_z
        
        # Create and publish the PoseStamped message
        msg = PoseStamped()
        
        # Fill the header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link' # We are publishing relative to the robot's base
        
        # Fill the pose
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0 # Default orientation
        
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    hand_simulator_node = HandSimulatorNode()
    rclpy.spin(hand_simulator_node)
    
    hand_simulator_node.destroy_node()
    rclpy.shutdown()

if _name_ == '_main_':
    main()
