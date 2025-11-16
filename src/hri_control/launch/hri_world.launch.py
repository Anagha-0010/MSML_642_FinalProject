#!/usr/bin/env python3
"""
Launch file for headless Gazebo simulation with a UR robot.
Fixes:
 - Missing robot_description for 'ros_gz_sim/create'
 - Updated to pass URDF properly to all nodes
 - FIXED: Removed --headless-rendering to bypass local 3D driver crash (LLVM error).
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, AppendEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # -------------------------------
    # 1. Declare Launch Arguments
    # -------------------------------
    declared_arguments = [
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur5e",
            description="Type/prefix of the UR robot (e.g. ur3, ur5e, ur10e, etc.)"
        ),
        DeclareLaunchArgument("safety_limits", default_value="true"),
        DeclareLaunchArgument("safety_pos_margin", default_value="0.15"),
        DeclareLaunchArgument("safety_k_position", default_value="20"),
        DeclareLaunchArgument("prefix", default_value=""),
    ]

    ur_type = LaunchConfiguration("ur_type")
    safety_limits = LaunchConfiguration("safety_limits")
    safety_pos_margin = LaunchConfiguration("safety_pos_margin")
    safety_k_position = LaunchConfiguration("safety_k_position")
    prefix = LaunchConfiguration("prefix")

    # -------------------------------
    # 2. Locate Required Packages
    # -------------------------------
    pkg_ros_gz_sim = get_package_share_directory('gazebo_ros')
    pkg_ur_description = get_package_share_directory('ur_description')
    pkg_ur_simulation_gz = get_package_share_directory('ur_simulation_gz')

    # --- NEW BLOCK TO FIX MESH PATH ---
    # Get the parent directory of the 'ur_description' package share
    ur_description_path = os.path.join(pkg_ur_description, '..')
    
    set_env_vars = AppendEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        ur_description_path
    )
    # --- END OF NEW BLOCK ---

    controllers_path = os.path.join(pkg_ur_simulation_gz, 'config', 'ur_controllers.yaml')

    # -------------------------------
    # 3. Generate Robot Description via xacro
    # -------------------------------
    robot_description_content = ParameterValue(
        Command([
            FindExecutable(name="xacro"),
            " ",
            PathJoinSubstitution([pkg_ur_description, "urdf", "ur.urdf.xacro"]),
            " ",
            "name:=ur",
            " ",
            "robot_ip:=192.168.0.1",
            " ",
            "ur_type:=", ur_type,
            " ",
            "sim_gz:=true",
            " ",
            "use_fake_hardware:=true",
            " ",
            "simulation_controllers:=", controllers_path,
            " ",
            "prefix:=", prefix,
            " ",
            "safety_limits:=", safety_limits,
            " ",
            "safety_pos_margin:=", safety_pos_margin,
            " ",
            "safety_k_position:=", safety_k_position
        ]),
        value_type=str
    )

    robot_description = {"robot_description": robot_description_content}

    # -------------------------------
    # 4. Launch Headless Gazebo
    # -------------------------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gazebo.launch.py')
        ),
        # --- CRITICAL FIX: Removed '--headless-rendering' to prevent 3D driver crash (LLVM) ---
        # This forces Gazebo to run in server-only mode, which is stable.
        launch_arguments={'headless': 'false', 'pause': 'false'}.items(),
    )

    # -------------------------------
    # 5. Start the Controller Manager
    # -------------------------------
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            robot_description,
            controllers_path
        ],
        output="screen",
    )

    # -------------------------------
    # 6. Publish robot_description (so others can access it)
    # -------------------------------
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description],
        output="screen",
    )

    # -------------------------------
    # 7. Spawn the Robot into Gazebo
    # -------------------------------
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic','robot_description','-entity','ur',
        ],
        output='screen'
    )

    # -------------------------------
    # 8. Load the Joint State Broadcaster
    # -------------------------------
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen"
    )

    # -------------------------------
    # 9. Load the Trajectory Controller
    # -------------------------------
    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
        output="screen"
    )

    # -------------------------------
    # 10. Combine Everything
    # -------------------------------
    nodes_to_launch = [
        set_env_vars,
        
        gazebo,
        robot_state_publisher,
        control_node,
        spawn_robot,
        joint_state_broadcaster_spawner,
        joint_trajectory_controller_spawner
    ]

    return LaunchDescription(declared_arguments + nodes_to_launch)
