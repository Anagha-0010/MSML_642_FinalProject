#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
import ament_index_python.packages

from gazebo_msgs.srv import SpawnEntity, SetEntityState
from gazebo_msgs.msg import EntityState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped

import numpy as np


class HandSimulatorNode(Node):
    def __init__(self):
        super().__init__("hand_simulator_node")

        # ------------------------------
        # Publishers
        # ------------------------------
        self.pose_pub = self.create_publisher(PoseStamped, "/target_pose", 10)
        self.marker_pub = self.create_publisher(Marker, "/target_hand_marker", 10)

        # ------------------------------
        # Gazebo service clients
        # ------------------------------
        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.move_client = self.create_client(SetEntityState, "/set_entity_state")

        self.get_logger().info("Waiting for Gazebo services...")

        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting /spawn_entity...")

        while not self.move_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting /set_entity_state...")

        self.get_logger().info("All Gazebo services available!")

        # Flags to avoid moving nonexistent models
        self.hand_spawned = False
        # self.obj_spawned = False  <-- DISABLED: We don't want the floating cube anymore

        # Spawn only the hand
        self.spawn_hand()
        # self.spawn_object()       <-- DISABLED

        # Timer (50 Hz update rate)
        self.start_time = self.get_clock().now().nanoseconds * 1e-9
        self.timer = self.create_timer(0.02, self.animate)

    # ---------------------------------------------------------
    # SPAWN HAND MODEL
    # ---------------------------------------------------------
    def spawn_hand(self):
        pkg_share = ament_index_python.packages.get_package_share_directory("hri_control")
        hand_path = os.path.join(pkg_share, "models", "target_hand.urdf")

        req = SpawnEntity.Request()
        req.name = "target_hand"
        req.xml = open(hand_path).read()
        req.initial_pose.position.x = 0.3
        req.initial_pose.position.y = 0.25
        req.initial_pose.position.z = 0.35

        future = self.spawn_client.call_async(req)
        future.add_done_callback(self.hand_spawn_callback)

    def hand_spawn_callback(self, future):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info("HAND successfully spawned!")
                self.hand_spawned = True
            else:
                self.get_logger().error(f"HAND spawn failed: {result.status_message}")
        except Exception as e:
            self.get_logger().error(f"HAND spawn callback error: {e}")

    # ---------------------------------------------------------
    # MAIN ANIMATION LOOP (SLOW MOTION)
    # ---------------------------------------------------------
    def animate(self):
        # Only proceed if the hand exists
        if not self.hand_spawned:
            return

        t = (self.get_clock().now().nanoseconds * 1e-9) - self.start_time

        # ----------------------
        # Trajectory parameters
        # ----------------------
        start_x = 0.3
        offer_x = 0.6
        base_y = 0.25
        base_z = 0.35

        # --- SLOW MODE: 16 Second Cycle (Doubled from 8s) ---
        cycle_duration = 16.0
        phase = t % cycle_duration
        z = base_z

        if phase < 4.0: 
            # Extending (0-4s)
            progress = phase / 4.0
            alpha = 0.5 * (1 - np.cos(np.pi * progress))
            x = start_x + alpha * (offer_x - start_x)
            y = base_y
            wrist_rot = 0.0

        elif phase < 8.0: 
            # Holding (4-8s)
            x = offer_x
            y = base_y
            z = base_z
            wrist_rot = 0.1

        elif phase < 12.0: 
            # Invitation shake (8-12s) - Slower frequency
            x = offer_x
            y = base_y + 0.02 * np.sin(1.5 * (phase - 8)) 
            wrist_rot = 0.10 * np.sin(1.0 * (phase - 8))
            z = base_z + 0.01 * np.sin(0.75 * (phase - 8))

        else: 
            # Retracting (12-16s)
            progress = (phase - 12.0) / 4.0
            alpha = 0.5 * (1 - np.cos(np.pi * progress))
            x = offer_x - alpha * (offer_x - start_x)
            y = base_y
            wrist_rot = 0.0
        
        # Add slight arc to Z motion
        if phase < 4.0 or phase >= 12.0:
            z = base_z + 0.02 * np.sin(alpha * np.pi)

        # ---------------------------------------
        # Orientation Logic
        # ---------------------------------------
        yaw = np.pi
        pitch = 0.52           # ~30 degrees
        roll = 1.57 + wrist_rot 

        # -------------------------
        # Euler -> Quaternion
        # -------------------------
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)

        q_w = cy * cr * cp + sy * sr * sp
        q_x = cy * sr * cp - sy * cr * sp
        q_y = cy * cr * sp + sy * sr * cp
        q_z = sy * cr * cp - cy * sr * sp

        # -------------------------
        # MOVE HAND
        # -------------------------
        hand_state = EntityState()
        hand_state.name = "target_hand"
        hand_state.pose.position.x = x
        hand_state.pose.position.y = y
        hand_state.pose.position.z = z
        hand_state.pose.orientation.w = q_w
        hand_state.pose.orientation.x = q_x
        hand_state.pose.orientation.y = q_y
        hand_state.pose.orientation.z = q_z

        self.move_client.call_async(SetEntityState.Request(state=hand_state))

        # -------------------------
        # RVIZ MARKER
        # -------------------------
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = hand_state.pose
        marker.scale.x = marker.scale.y = marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        self.marker_pub.publish(marker)

        # -------------------------
        # RL target pose
        # -------------------------
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose = hand_state.pose

        self.pose_pub.publish(pose)


def main():
    rclpy.init()
    node = HandSimulatorNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
