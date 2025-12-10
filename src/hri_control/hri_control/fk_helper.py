import numpy as np
from urdf_parser_py.urdf import URDF
import PyKDL as kdl
import rclpy
from rcl_interfaces.srv import GetParameters

#querying the pubulisher for the robot description
def get_urdf_from_robot_state_publisher(node):
    client = node.create_client(GetParameters, "/robot_state_publisher/get_parameters")

    if not client.wait_for_service(timeout_sec=3.0):
        raise RuntimeError("robot_state_publisher is not available")

    req = GetParameters.Request()
    req.names = ["robot_description"]

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)

    if not future.result():
        raise RuntimeError("Failed to call get_parameters")

    xml = future.result().values[0].string_value
    if not isinstance(xml, str) or len(xml) < 20:
        raise RuntimeError("robot_description is invalid")

    return xml


#Building a kdl tree from urdf
def build_kdl_tree_from_urdf(urdf: URDF):
    root = urdf.get_root()
    tree = kdl.Tree(root)

    for joint in urdf.joints:
        if joint.type not in ["revolute", "continuous", "prismatic", "fixed"]:
            continue
            #exrcting the origin and rotatuon
        xyz = joint.origin.xyz if joint.origin else [0, 0, 0]
        rpy = joint.origin.rpy if joint.origin else [0, 0, 0]

        frame = kdl.Frame(
            kdl.Rotation.RPY(*rpy),
            kdl.Vector(*xyz)
        )

        #Joint axis
        if joint.axis:
            axis = kdl.Vector(*joint.axis)
        else:
            axis = kdl.Vector(0, 0, 1)

        #Creating joints
        if joint.type == "fixed":
            kdl_joint = kdl.Joint(joint.name, kdl.Joint.Fixed)
        else:
            origin_vec = kdl.Vector(0, 0, 0)
            kdl_joint = kdl.Joint(joint.name, origin_vec, axis, kdl.Joint.RotAxis)

        #Creating segment with zero inertia
        segment = kdl.Segment(
            joint.child,
            kdl_joint,
            frame,
            kdl.RigidBodyInertia()
        )

        try:
            tree.addSegment(segment, joint.parent)
        except RuntimeError:
            pass

    return tree

def find_ee_link_from_joint_list(urdf_robot: URDF, joint_names):
    for j in reversed(joint_names):
        if j in urdf_robot.joint_map:
            return urdf_robot.joint_map[j].child
    return urdf_robot.get_root()

class FKWrapper:
    def __init__(self, node, joint_names, base_link=None):
        self.node = node
        self.joint_names = joint_names

        #fetching the urdf xml
        urdf_xml = get_urdf_from_robot_state_publisher(node)

        #parsing the urdf
        self.urdf = URDF.from_xml_string(urdf_xml)

        #determinig the frames
        self.base_link = base_link if base_link else self.urdf.get_root()
        self.ee_link = find_ee_link_from_joint_list(self.urdf, joint_names)

        node.get_logger().info(f"[FK] Base link = {self.base_link}")
        node.get_logger().info(f"[FK] End effector link = {self.ee_link}")

        self.tree = build_kdl_tree_from_urdf(self.urdf)

        if not self.tree.getChain(self.base_link, self.ee_link):
            raise RuntimeError(f"No KDL chain from {self.base_link} to {self.ee_link}")

        #extracting chain
        self.chain = self.tree.getChain(self.base_link, self.ee_link)
        self.n_joints = self.chain.getNrOfJoints()

        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)

    def compute_fk(self, joint_positions):
        if len(joint_positions) < self.n_joints:
            raise ValueError("Not enough joint positions for FK")

        q = kdl.JntArray(self.n_joints)
        for i in range(self.n_joints):
            q[i] = float(joint_positions[i])

        frame = kdl.Frame()
        self.fk_solver.JntToCart(q, frame)

        rx, ry, rz, rw = frame.M.GetQuaternion()

        #Returning 3 Pos + 4 Rot
        return np.array([
            frame.p.x(), frame.p.y(), frame.p.z(),
            rx, ry, rz, rw
        ], dtype=np.float32)
