from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("baudrate", default_value="115200"),
        Node(
            package="robotiq_3f_driver",
            executable="gripper_node.py",
            name="robotiq_3f_gripper",
            parameters=[{
                "port": LaunchConfiguration("port"),
                "baudrate": LaunchConfiguration("baudrate"),
            }],
            output="screen",
        ),
    ])
