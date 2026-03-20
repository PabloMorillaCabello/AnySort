"""
Launch file for Orbbec Gemini 2 camera via OrbbecSDK_ROS2.

Usage:
  ros2 launch graspgen_pipeline orbbec_camera.launch.py
  ros2 launch graspgen_pipeline orbbec_camera.launch.py enable_ir:=true

Or use the built-in OrbbecSDK_ROS2 launch file directly:
  ros2 launch orbbec_camera gemini2.launch.py

Published topics:
  /camera/color/image_raw     (sensor_msgs/Image, bgr8)
  /camera/depth/image_raw     (sensor_msgs/Image, 16UC1, mm)
  /camera/ir/image_raw        (sensor_msgs/Image, mono16)
  /camera/color/camera_info   (sensor_msgs/CameraInfo)
  /camera/depth/camera_info   (sensor_msgs/CameraInfo)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    orbbec_share = get_package_share_directory('orbbec_camera')

    return LaunchDescription([
        # Tuneable parameters
        DeclareLaunchArgument('camera_name', default_value='camera',
                              description='Camera namespace'),
        DeclareLaunchArgument('serial_number', default_value='',
                              description='Camera serial (empty = auto-detect)'),
        DeclareLaunchArgument('enable_color', default_value='true',
                              description='Enable RGB stream'),
        DeclareLaunchArgument('enable_depth', default_value='true',
                              description='Enable depth stream'),
        DeclareLaunchArgument('enable_ir', default_value='false',
                              description='Enable infrared stream'),

        # Include the official OrbbecSDK_ROS2 Gemini 2 launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(orbbec_share, 'launch', 'gemini2.launch.py')
            ),
            launch_arguments={
                'camera_name': LaunchConfiguration('camera_name'),
                'serial_number': LaunchConfiguration('serial_number'),
                'enable_color': LaunchConfiguration('enable_color'),
                'enable_depth': LaunchConfiguration('enable_depth'),
                'enable_ir': LaunchConfiguration('enable_ir'),
            }.items(),
        ),
    ])
