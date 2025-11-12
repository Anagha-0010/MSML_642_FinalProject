# Complete, state-of-the-art launch file for VELOCITY control (FINAL FIX)

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    # --- 1. Launch Arguments ---
    use_sim_time = LaunchConfiguration('use_sim_time', default=True)

    # --- 2. Find Key Files & Directories ---
    pkg_hri_control = FindPackageShare('hri_control')
    pkg_ur_description = FindPackageShare('ur_description') # Still needed for YAML paths
    pkg_gazebo_ros = FindPackageShare('gazebo_ros')

    # Path to our controller configuration
    controller_config_path = PathJoinSubstitution(
        [pkg_hri_control, "config", "ur5e_velocity_controllers.yaml"]
    )

    # Path to our LOCAL, FIXED robot model (URDF)
    urdf_xacro_path = PathJoinSubstitution(
        [pkg_hri_control, "config", "ur.urdf.xacro"]
    )
    
    # Get path to xacro executable
    xacro_binary = FindExecutable(name='xacro')

    # --- CRITICAL FIX: Find Paths to Parameter Files ---
    joint_limits_path = PathJoinSubstitution(
        [pkg_ur_description, "config", "ur5e", "joint_limits.yaml"]
    )
    kinematics_path = PathJoinSubstitution(
        [pkg_ur_description, "config", "ur5e", "default_kinematics.yaml"]
    )
    physical_params_path = PathJoinSubstitution(
        [pkg_ur_description, "config", "ur5e", "physical_parameters.yaml"]
    )
    visual_params_path = PathJoinSubstitution(
        [pkg_ur_description, "config", "ur5e", "visual_parameters.yaml"]
    )
    initial_positions_path = PathJoinSubstitution(
        [pkg_ur_description, "config", "initial_positions.yaml"]
    )

    # Use Command substitution to generate the robot description at launch time
    robot_description_content = Command([
        xacro_binary, ' ',
        urdf_xacro_path, ' ', # This is our local file: .../config/ur.urdf.xacro

        # --- Arguments (SAFETY_ON FIX HERE) ---
        'robot_name:=ur5e', ' ',
        'ur_type:=ur5e', ' ',
        'sim_gazebo:=true', ' ',
        'simulation_controllers:=', controller_config_path, ' ',
        'safety_on:=true', ' ', # <<< FINAL FIX: RENAMED ARGUMENT
        'safety_pause_margin:=0.15',' ',
        'safety_k_position:=20',' ',
        'ros2_control:=true', ' ',
        
        # --- Default Arguments ---
        'sim_ignition:=false', ' ',
        'ros2_control_plugin:=gazebo_ros2_control/GazeboSystem', ' ',
        'use_fake_hardware:=false', ' ',
        'fake_sensor_commands:=false', ' ',
        'initial_positions:={}', ' ',
        'prefix:=', ' ',
        'parent_link:=base_link', ' ',

        # --- Paths to parameter files ---
        'joint_limits_parameters_file:=', joint_limits_path, ' ',
        'kinematics_parameters_file:=', kinematics_path, ' ',
        'physical_parameters_file:=', physical_params_path, ' ',
        'visual_parameters_file:=', visual_params_path, ' ',
        'initial_positions_file:=', initial_positions_path
    ])
    
    # --- 3. Define Key Nodes & Actions ---

    # Start Gazebo
    start_gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_gazebo_ros, 'launch', 'gzserver.launch.py'])
        )
    )
    
    start_gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_gazebo_ros, 'launch', 'gzclient.launch.py'])
        )
    )

    # Robot State Publisher
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description_content # Pass the Command object here
        }]
    )

    # Spawn Robot Entity
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'ur5e', '-z', '1.0'],
        output='screen'
    )

    # --- ros2_control nodes ---

    # Load Joint State Broadcaster
    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'start',
             'joint_state_broadcaster'],
        output='screen'
    )

    # Load Joint Group Velocity Controller
    load_joint_group_velocity_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'start',
             'joint_group_velocity_controller'],
        output='screen'
    )
    
    # --- 4. Assemble the Launch Description ---
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        
        start_gazebo_server,
        start_gazebo_client,
        node_robot_state_publisher,
        spawn_entity,
        
        # We chain the controller loads to happen after the robot is spawned
        # This is a more robust way to launch
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[load_joint_state_broadcaster],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[load_joint_group_velocity_controller],
            )
        ),
    ])
