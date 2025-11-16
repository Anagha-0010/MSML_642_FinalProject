# This file is hand_simulator.py
# FINAL VERSION - Corrected for ROS 2 Humble / Gazebo Classic Bridge
# 1. Spawns a red sphere in Gazebo using /spawn_entity service
# 2. Moves it in a circle (using /gazebo/set_model_state)
# 3. Publishes an RViz "Marker" so we can see it

import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelState
# --- CHANGE 1: Use the classic gazebo_msgs service ---
from gazebo_msgs.srv import SpawnEntity 
from geometry_msgs.msg import Pose, Point, Quaternion
from visualization_msgs.msg import Marker
import numpy as np

# Blueprint (SDF) for a 5cm red sphere
SPHERE_SDF = """
<?xml version="1.0"?>
<sdf version="1.6">
  <model name="target_hand">
    <static>false</static> 
    <link name="base_link">
      <visual name="visual">
        <geometry><sphere><radius>0.05</radius></sphere></geometry>
        <material>
          <ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse>
          <specular>0 0 0 0</specular><emissive>1 0 0 1</emissive>
        </material>
      </visual>
      <collision name="collision">
        <geometry><sphere><radius>0.05</radius></sphere></geometry>
      </collision>
    </link>
  </model>
</sdf>
"""

class HandSimulatorNode(Node):
    def __init__(self):
        super().__init__('hand_simulator_node')
        
        # --- Job 1: The Spawner (Corrected with service list info) ---
        # --- CHANGE 2: Use the correct service name '/spawn_entity' ---
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        
        while not self.spawn_client.wait_for_service(timeout_sec=5.0): # Increased timeout
            self.get_logger().info('Spawn service (/spawn_entity) not available, waiting...')
            if not rclpy.ok():
                self.get_logger().error("RCLPY shut down, exiting.")
                return
        
        self.get_logger().info("Spawn service IS available. Spawning model...")

        # --- Job 2: The Mover ---
        self.publisher_ = self.create_publisher(ModelState, '/gazebo/set_model_state', 10)
        
        # --- Job 3: The RViz Visualizer ---
        self.marker_pub = self.create_publisher(Marker, '/target_hand_marker', 10)
        
        self.timer = None
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        self.spawn_hand_model()

    def spawn_hand_model(self):
        self.get_logger().info("Spawning 'target_hand' model...")
        req = SpawnEntity.Request()
        
        # --- CHANGE 3: Use the classic message structure ---
        req.name = "target_hand"
        req.xml = SPHERE_SDF
        req.initial_pose.position.x = 0.5
        req.initial_pose.position.y = 0.3
        req.initial_pose.position.z = 0.5
        
        self.spawn_client.call_async(req).add_done_callback(self.spawn_callback)

    def spawn_callback(self, future):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info("--- 'target_hand' sphere spawned successfully! ---")
                self.get_logger().info('Starting the hand-moving timer...')
                self.timer = self.create_timer(0.02, self.timer_callback) # 50Hz
            else:
                self.get_logger().error(f"Failed to spawn: {result.status_message}")
                if "exists" in result.status_message:
                    self.get_logger().warn("Model already exists. Starting timer anyway.")
                    self.timer = self.create_timer(0.02, self.timer_callback)
        except Exception as e:
            self.get_logger().error(f"Spawn service call failed: {e}")

    def timer_callback(self):
        # 1. Calculate the new position
        current_time = (self.get_clock().now().nanoseconds * 1e-9) - self.start_time
        radius = 0.2
        center_x = 0.5
        center_y = 0.3
        center_z = 0.5
        x_pos = center_x + radius * np.cos(current_time)
        y_pos = center_y + radius * np.sin(current_time)
        z_pos = center_z
        
        # 2. Publish the "Move" command for Gazebo
        msg = ModelState()
        msg.model_name = 'target_hand'
        msg.pose.position.x = x_pos
        msg.pose.position.y = y_pos
        msg.pose.position.z = z_pos
        msg.pose.orientation.w = 1.0
        self.publisher_.publish(msg)

        # 3. Publish the "Marker" command for RViz
        marker = Marker()
        marker.header.frame_id = "base_link" # Put it in the same "world" as the robot
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x_pos
        marker.pose.position.y = y_pos
        marker.pose.position.z = z_pos
        marker.scale.x = 0.1 # 10cm sphere (same as 0.05 radius)
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 1.0 # Make it red
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0 # Make it solid
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    hand_simulator_node = HandSimulatorNode()
    try:
        rclpy.spin(hand_simulator_node)
    except KeyboardInterrupt:
        pass
    hand_simulator_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
