cmake_minimum_required(VERSION 3.8)
project(robotiq_3f_driver)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
  robotiq_3f_driver/gripper_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
