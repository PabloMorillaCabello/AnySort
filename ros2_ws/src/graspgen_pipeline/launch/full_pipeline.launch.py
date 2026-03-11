"""
Launch file for the full GraspGen pipeline:
  1. Orbbec Gemini 2 camera driver (RGB-D) — from OrbbecSDK_ROS2
  2. Camera relay / synchronizer
  3. SAM3 segmentation node
  4. GraspGen grasp pose generator
  5. MoveIt2 motion planner
  6. Robotiq 3F gripper driver
  7. Pipeline orchestrator
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ---- Arguments ----
    text_prompt_arg = DeclareLaunchArgument(
        "text_prompt",
        default_value="object",
        description="Text prompt for SAM3 segmentation",
    )
    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip",
        default_value="192.168.1.100",
        description="IP address of the UR robot",
    )
    use_sim_arg = DeclareLaunchArgument(
        "use_sim",
        default_value="false",
        description="Use simulated robot (no real hardware)",
    )
    launch_camera_arg = DeclareLaunchArgument(
        "launch_camera",
        default_value="true",
        description="Launch Orbbec camera driver (set false if already running)",
    )

    # ---- Config paths ----
    pipeline_config = os.path.join(
        get_package_share_directory("graspgen_pipeline"),
        "config",
        "pipeline_params.yaml",
    )

    # ---- 1. Orbbec Gemini 2 Camera Driver ----
    orbbec_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("orbbec_camera"),
                "launch",
                "gemini2.launch.py",
            )
        ),
        condition=IfCondition(LaunchConfiguration("launch_camera")),
    )

    # ---- 2. Camera relay / synchronizer ----
    camera_node = Node(
        package="graspgen_pipeline",
        executable="camera_node.py",
        name="camera_node",
        parameters=[pipeline_config],
        output="screen",
    )

    # ---- 3. SAM3 Segmentation ----
    segmentation_node = Node(
        package="graspgen_pipeline",
        executable="segmentation_node.py",
        name="segmentation_node",
        parameters=[
            pipeline_config,
            {"text_prompt": LaunchConfiguration("text_prompt")},
        ],
        output="screen",
    )

    # ---- 4. GraspGen Grasp Generator ----
    grasp_generator_node = Node(
        package="graspgen_pipeline",
        executable="grasp_generator_node.py",
        name="grasp_generator_node",
        parameters=[pipeline_config],
        output="screen",
    )

    # ---- 5. Motion Planner ----
    motion_planner_node = Node(
        package="graspgen_pipeline",
        executable="motion_planner_node.py",
        name="motion_planner_node",
        parameters=[
            pipeline_config,
            {"robot_ip": LaunchConfiguration("robot_ip")},
            {"use_sim": LaunchConfiguration("use_sim")},
        ],
        output="screen",
    )

    # ---- 6. Robotiq 3F Gripper ----
    gripper_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("robotiq_3f_driver"),
                "launch",
                "gripper.launch.py",
            )
        ),
        condition=UnlessCondition(LaunchConfiguration("use_sim")),
    )

    # ---- 7. Pipeline Orchestrator ----
    pipeline_orchestrator = Node(
        package="graspgen_pipeline",
        executable="pipeline_orchestrator.py",
        name="pipeline_orchestrator",
        parameters=[pipeline_config],
        output="screen",
    )

    return LaunchDescription(
        [
            # Arguments
            text_prompt_arg,
            robot_ip_arg,
            use_sim_arg,
            launch_camera_arg,
            # Nodes (in pipeline order)
            orbbec_launch,
            camera_node,
            segmentation_node,
            grasp_generator_node,
            motion_planner_node,
            gripper_node,
            pipeline_orchestrator,
        ]
    )
