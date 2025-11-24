#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    AppendEnvironmentVariable,
    TimerAction
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


def generate_launch_description():

    # -------------------------------
    # Arguments
    # -------------------------------
    declared_arguments = [
        DeclareLaunchArgument("ur_type", default_value="ur5e"),
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
    # Locate packages
    # -------------------------------
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    pkg_ur_description = get_package_share_directory("ur_description")
    pkg_hri_control = get_package_share_directory("hri_control")

    # -------------------------------
    # Gazebo model paths
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

    # Controller config
    ur_controller_params_path = os.path.join(
        pkg_hri_control, "config", "ur_simple_controllers.yaml"
    )

    # xacro control file
    ur_control_xacro = os.path.join(
        pkg_ur_description, "urdf", "ur_ros2_control.xacro"
    )

    # -------------------------------
    # Generate UR robot_description
    # -------------------------------
    ur_robot_description_content = ParameterValue(
        Command([
            FindExecutable(name="xacro"), " ",
            os.path.join(pkg_ur_description, "urdf", "ur.urdf.xacro"),
            " ", "name:=ur",
            " ", "ur_type:=", ur_type,
            " ", "sim_gazebo:=true",
            " ", "use_fake_hardware:=false",
            " ", "ros2_control_xacro_file:=", ur_control_xacro,
            " ", "simulation_controllers:=", ur_controller_params_path,
            " ", "prefix:=", prefix,
            " ", "safety_limits:=", safety_limits,
            " ", "safety_pos_margin:=", safety_pos_margin,
            " ", "safety_k_position:=", safety_k_position,
        ]),
        value_type=str
    )

    ur_robot_description = {"robot_description": ur_robot_description_content}

    # State publisher
    ur_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[ur_robot_description],
        output="screen",
    )

    # Spawn robot
    ur_spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-topic", "robot_description", "-entity", "ur"],
        output="screen"
    )

    # Controllers
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
    # Gazebo world
    # -------------------------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "verbose": "true",
            "paused": "false",
            "use_sim_time": "true",
            "world": os.path.join(pkg_hri_control, "worlds", "hri_world.world")
        }.items()
    )

    # -------------------------------
    # 💥 Delay robot spawn (fix falling)
    # -------------------------------
    delayed_robot_spawn = TimerAction(
        period=3.0,  # wait 3 seconds
        actions=[
            ur_robot_state_publisher,
            ur_spawn_robot,
            joint_state_broadcaster_spawner,
            joint_trajectory_controller_spawner
        ]
    )

    # -------------------------------
    # Launch Description
    # -------------------------------
    nodes_to_launch = [
        set_model_path,
        export_api_plugin,
        gazebo,
        delayed_robot_spawn
    ]

    return LaunchDescription(declared_arguments + nodes_to_launch)

