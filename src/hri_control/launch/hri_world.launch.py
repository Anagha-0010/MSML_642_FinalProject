#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    AppendEnvironmentVariable
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import xacro


def generate_launch_description():

    # -------------------------------
    # 1. Declare Launch Arguments
    # -------------------------------
    declared_arguments = [
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur5e",
            description="Type of UR robot"
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
    # 2. Locate Packages
    # -------------------------------
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    pkg_ur_description = get_package_share_directory("ur_description")
    pkg_hri_control = get_package_share_directory("hri_control")

    # -------------------------------
    # 3. Gazebo model & plugin paths
    # -------------------------------
    ur_description_path = os.path.join(pkg_ur_description, "..")
    hri_models_path = os.path.join(pkg_hri_control, "models")

    set_model_path = AppendEnvironmentVariable(
        "GAZEBO_MODEL_PATH",
        f"{ur_description_path}:{hri_models_path}"
    )

    export_api_plugin = AppendEnvironmentVariable(
        "GAZEBO_PLUGIN_PATH",
        os.path.join(pkg_gazebo_ros, "lib")
    )

    # Controller parameters
    ur_controller_params_path = os.path.join(
        pkg_hri_control, "config", "ur_simple_controllers.yaml"
    )

    # Path to URDF control xacro
    ur_control_xacro = os.path.join(
        pkg_ur_description, "urdf", "ur_ros2_control.xacro"
    )

    # -------------------------------------
    # 4. Generate UR Robot Description
    # -------------------------------------
    ur_robot_description_content = ParameterValue(
        Command([
            FindExecutable(name="xacro"), " ",
            PathJoinSubstitution([pkg_ur_description, "urdf", "ur.urdf.xacro"]),
            " ", "name:=ur",
            " ", "ur_type:=", ur_type,
            " ", "sim_gazebo:=true",
            " ", "use_fake_hardware:=false",
            " ", "ros2_control_xacro_file:=", ur_control_xacro,
            " ", "simulation_controllers:=", ur_controller_params_path,
            " ", "prefix:=", prefix,
            " ", "safety_limits:=", safety_limits,
            " ", "safety_pos_margin:=", safety_pos_margin,
            " ", "safety_k_position:=", safety_k_position
        ]),
        value_type=str
    )

    ur_robot_description = {"robot_description": ur_robot_description_content}

    # UR State Publisher
    ur_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[ur_robot_description],
        output="screen",
    )

    # UR Spawner
    ur_spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "robot_description",
            "-entity", "ur"
        ],
        output="screen"
    )

    # UR Controllers
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager"
        ],
        output="screen"
    )

    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_trajectory_controller",
            "-c", "/controller_manager"
        ],
        output="screen"
    )

    # -------------------------------
    # 5. Launch Gazebo with world file
    # -------------------------------
    world_path = os.path.join(pkg_hri_control, "worlds", "hri_world.world")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "verbose": "true",
            "paused": "false",
            "use_sim_time": "true",
            'world': os.path.join(pkg_hri_control, 'worlds', 'hri_world.world'),
            'extra_gazebo_args': '-s libgazebo_ros_api_plugin.so -s libgazebo_ros_state.so'

        }.items()
    )

    # -------------------------------
    # 6. Combine Everything
    # -------------------------------
    nodes_to_launch = [
        set_model_path,
        export_api_plugin,
        gazebo,

        ur_robot_state_publisher,
        ur_spawn_robot,
        joint_state_broadcaster_spawner,
        joint_trajectory_controller_spawner
    ]

    return LaunchDescription(declared_arguments + nodes_to_launch)

