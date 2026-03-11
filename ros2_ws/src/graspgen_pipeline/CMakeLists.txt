cmake_minimum_required(VERSION 3.8)
project(graspgen_pipeline)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install launch files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install Python scripts as executables
install(PROGRAMS
  graspgen_pipeline/camera_node.py
  graspgen_pipeline/segmentation_node.py
  graspgen_pipeline/grasp_generator_node.py
  graspgen_pipeline/motion_planner_node.py
  graspgen_pipeline/pipeline_orchestrator.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
