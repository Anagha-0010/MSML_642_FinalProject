#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, AppendEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
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
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_ur_description = get_package_share_directory('ur_description')
    pkg_ur_simulation_gz = get_package_share_directory('ur_simulation_gz')
    pkg_hri_control = get_package_share_directory('hri_control')

    # --- BLOCK TO FIX MESH PATH ---
    ur_description_path = os.path.join(pkg_ur_description, '..')
    hri_models_path = os.path.join(pkg_hri_control, 'models')
    combined_model_path = ur_description_path + ':' + hri_models_path
    
    set_env_vars = AppendEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        combined_model_path
    )
    # --- END OF BLOCK ---

    # --- THIS IS THE START OF THE FIX ---
    # Path to the controller PARAMETERS (our simple YAML)
    ur_controller_params_path = os.path.join(
        pkg_hri_control, 'config', 'ur_simple_controllers.yaml'
    )

    # Path to the ros2_control XACRO file (part of ur_description)
    ur_simulation_controllers_xacro_path = os.path.join(
        pkg_ur_description, 'urdf', 'ur_ros2_control.xacro'
    )
    # --- THIS IS THE END OF THE FIX ---


    # -------------------------------------
    # 3. ROBOT 1: THE UR ARM
    # -------------------------------------
    
    # Generate the UR robot description
    ur_robot_description_content = ParameterValue(
        Command([
            FindExecutable(name="xacro"),
            " ",
            PathJoinSubstitution([pkg_ur_description, "urdf", "ur.urdf.xacro"]), # Back to original
            " ", "name:=ur",
            " ", "ur_type:=", ur_type,
            " ", "sim_gazebo:=true",
            " ", "use_fake_hardware:=false",
            # --- ADD THESE LINES ---
            " ", "hardware_interface:=PositionJointInterface",
            " ", "robot_description_controller:=", ur_simulation_controllers_xacro_path, # Use the correct variable
            # --- END ADDITIONS ---
            " ", "simulation_controllers:=", ur_simulation_controllers_xacro_path, 
            # ...
            " ", "simulation_controllers:=", ur_simulation_controllers_xacro_path, # <-- FIXED
            " ", "prefix:=", prefix,
            " ", "safety_limits:=", safety_limits,
            " ", "safety_pos_margin:=", safety_pos_margin,
            " ", "safety_k_position:=", safety_k_position
        ]),
        value_type=str
    )
    
    ur_robot_description = {"robot_description": ur_robot_description_content}

    # UR Robot State Publisher
    ur_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[ur_robot_description],
        output="screen",
    )

    # UR Controller Manager
    ur_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ur_robot_description, ur_controller_params_path], # <-- FIXED
        output="screen",
    )

    # UR Spawner
    ur_spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description', # This is the default topic
            '-entity', 'ur',               # The name of the robot in Gazebo
        ],
        output='screen'
    )

    # UR Controllers
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

    # -------------------------------------
    # 4. ROBOT 2: THE SHADOW HAND
    # -------------------------------------

    # Load the hand's URDF file directly
    hand_urdf_path = os.path.join(pkg_hri_control, 'models', 'shadow_hand_right.urdf')
    hand_robot_description_content = xacro.process_file(hand_urdf_path).toxml()
    hand_robot_description = {"robot_description": hand_robot_description_content}

    # Hand Robot State Publisher
    hand_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='shadow_hand', # <-- IMPORTANT
        parameters=[hand_robot_description],
        output='screen',
    )
    
    # Hand Spawner
    hand_spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', '/shadow_hand/robot_description', # <-- IMPORTANT (uses namespace)
            '-entity', 'shadow_hand',                   # The name in Gazebo
            '-x', '1.0',  # Spawn it 1 meter away so it's not inside the UR
            '-y', '0.0',
            '-z', '0.5'
        ],
        output='screen'
    )


    # -------------------------------
    # 5. Launch Gazebo (Classic)
    # -------------------------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'headless': 'false', 'pause': 'false'}.items(),
    )

    # -------------------------------
    # 6. Combine Everything
    # -------------------------------
    nodes_to_launch = [
        set_env_vars,
        gazebo,
        
        # UR Robot Nodes
        ur_robot_state_publisher,
        ur_control_node,
        ur_spawn_robot,
        joint_state_broadcaster_spawner,
        joint_trajectory_controller_spawner,
        
        # Hand Robot Nodes
        hand_robot_state_publisher,
        hand_spawn_robot
    ]

    return LaunchDescription(declared_arguments + nodes_to_launch)
