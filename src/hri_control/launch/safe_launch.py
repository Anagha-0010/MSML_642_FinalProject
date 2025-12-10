#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, AppendEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    #getting packages directories
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    pkg_ur_description = get_package_share_directory("ur_description")
    pkg_hri_control = get_package_share_directory("hri_control")

    #paths used to set the model and plugin
    ur_description_path = os.path.join(pkg_ur_description, "..")
    hri_models_path = os.path.join(pkg_hri_control, "models")
    
    set_model_path = AppendEnvironmentVariable("GAZEBO_MODEL_PATH", f"{ur_description_path}:{hri_models_path}")
    export_api_plugin = AppendEnvironmentVariable("GAZEBO_PLUGIN_PATH", os.path.join(pkg_gazebo_ros, "lib"))

    xacro_file = os.path.join(pkg_hri_control, "urdf", "safe_demo.xacro")

    #this is the robot description
    ur_robot_description_content = ParameterValue(
        Command([
            FindExecutable(name="xacro"), " ",
            xacro_file, 
            " ", "name:=ur",
            " ", "ur_type:=ur5e",
            " ", "sim_gazebo:=true",
        ]),
        value_type=str
    )
    ur_robot_description = {"robot_description": ur_robot_description_content}

    #nodes to spaw the robot and its joints
    ur_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[ur_robot_description],
        output="screen",
    )

    ur_spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-topic", "robot_description", "-entity", "ur"],
        output="screen"
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen"
    )

    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
        output="screen"
    )

    #generating the gazebo world 
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, "launch", "gazebo.launch.py")),
        launch_arguments={
            "verbose": "true",
            "paused": "false",
            "use_sim_time": "true",
            "world": os.path.join(pkg_hri_control, "worlds", "hri_world.world")
        }.items()
    )

    #delaying the spawn so that gazebo is ready when we spawn the hand model 
    delayed_robot_spawn = TimerAction(
        period=10.0,
        actions=[ur_robot_state_publisher, ur_spawn_robot, joint_state_broadcaster_spawner, joint_trajectory_controller_spawner]
    )

    return LaunchDescription([set_model_path, export_api_plugin, gazebo, delayed_robot_spawn])
