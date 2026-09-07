cmake_minimum_required(VERSION 3.12)
project(drone_visualisation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# This is needed for Qt5
set(CMAKE_AUTOMOC ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(interfaces REQUIRED)
find_package(rviz_common REQUIRED)
find_package(rviz_rendering REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

# Include directories
include_directories(include)

# Build the RViz plugin library
add_library(drone_visualisation_plugins SHARED
  src/arm_panel.cpp
  include/drone_visualisation/arm_panel.hpp
)

target_include_directories(drone_visualisation_plugins PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(drone_visualisation_plugins
  rclcpp
  std_msgs
  interfaces
  rviz_common
  rviz_rendering
  pluginlib
)

target_link_libraries(drone_visualisation_plugins
  Qt5::Widgets
)

# Install the plugin library
install(TARGETS drone_visualisation_plugins
  EXPORT drone_visualisation_plugins
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install plugin description file
pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

# Install launch/URDF/meshes (ok if some folders are empty)
install(DIRECTORY
  launch
  urdf
  meshes
  rviz
  DESTINATION share/${PROJECT_NAME}
)

# Register the package in the ament index (so package:// resolves)
install(FILES
  resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

# Ship package.xml alongside the assets (handy for inspection)
install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
