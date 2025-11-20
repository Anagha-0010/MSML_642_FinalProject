import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
import numpy as np

class HandMover(Node):
    def __init__(self):
        super().__init__("hand_mover")

        self.cli = self.create_client(SetEntityState, "/set_entity_state")
        self.get_logger().info("Waiting for /set_entity_state...")

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available — waiting...")

        self.t = 0.0
        self.timer = self.create_timer(0.05, self.update_hand)

    def update_hand(self):
        req = SetEntityState.Request()
        state = EntityState()
        state.name = "target_hand"

        # Smooth circular motion
        state.pose.position.x = 0.45 + 0.05 * np.sin(self.t)
        state.pose.position.y = 0.30 + 0.05 * np.cos(self.t)
        state.pose.position.z = 0.50

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

