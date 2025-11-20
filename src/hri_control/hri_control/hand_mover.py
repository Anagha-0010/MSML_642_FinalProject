import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import ModelState
import numpy as np

class HandMover(Node):
    def __init__(self):
        super().__init__("hand_mover")

        self.cli = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.get_logger().info("Waiting for /gazebo/set_entity_state...")

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available — waiting...")

        self.t = 0.0
        self.timer = self.create_timer(0.05, self.update_hand)

    def update_hand(self):
        req = SetEntityState.Request()
        state = ModelState()
        state.name = "target_hand"   # MUST match model name spawned in Gazebo

        # Animate in a slow figure-8 motion
        state.pose.position.x = 0.45 + 0.05 * np.sin(self.t)
        state.pose.position.y = 0.15 + 0.08 * np.sin(self.t * 0.7)
        state.pose.position.z = 0.28 + 0.03 * np.cos(self.t * 0.5)

        state.pose.orientation.w = 1.0

        req.state = state
        self.cli.call_async(req)

        self.t += 0.05


def main():
    rclpy.init()
    node = HandMover()
    rclpy.spin(node)


if __name__ == "__main__":
    main()

