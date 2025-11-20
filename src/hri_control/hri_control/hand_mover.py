import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import numpy as np

class HandMover(Node):
    def __init__(self):
        super().__init__("hand_mover")

        # Correct service name
        self.cli = self.create_client(SetModelState, "/gazebo/set_model_state")
        self.get_logger().info("Waiting for /gazebo/set_model_state...")

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available — waiting...")

        self.t = 0.0
        self.timer = self.create_timer(0.05, self.update_hand)

    def update_hand(self):
        req = SetModelState.Request()
        state = ModelState()

        # Correct field name
        state.model_name = "target_hand"

        # Simple circular path
        state.pose.position.x = 0.5 + 0.15 * np.sin(self.t)
        state.pose.position.y = 0.3 + 0.15 * np.cos(self.t)
        state.pose.position.z = 0.5

        state.pose.orientation.w = 1.0

        req.model_state = state
        self.cli.call_async(req)

        self.t += 0.05


def main():
    rclpy.init()
    node = HandMover()
    rclpy.spin(node)


if __name__ == "__main__":
    main()

