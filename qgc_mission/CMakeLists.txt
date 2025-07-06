# 设置 CMake 最低版本和项目名称
cmake_minimum_required(VERSION 3.8)
project(qgc_mission)

# 为 ROS 2 Humble 设置 C++17 标准
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# 添加编译警告选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
# 查找所有需要的 ROS 2 功能包
find_package(rclcpp REQUIRED)
find_package(mavros REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(ai_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
# 查找所有需要的系统库
find_package(Eigen3 REQUIRED)
# 手动查找 GeographicLib
find_path(GeographicLib_INCLUDE_DIR NAMES GeographicLib/Geocentric.hpp)
find_library(GeographicLib_LIBRARY NAMES Geographic)
if(GeographicLib_INCLUDE_DIR AND GeographicLib_LIBRARY)
  add_library(GeographicLib::GeographicLib INTERFACE IMPORTED)
  target_include_directories(GeographicLib::GeographicLib INTERFACE ${GeographicLib_INCLUDE_DIR})
  target_link_libraries(GeographicLib::GeographicLib INTERFACE ${GeographicLib_LIBRARY})
else()
  message(FATAL_ERROR "Could not find GeographicLib library or headers manually.")
endif()


add_executable(qgc_mission_ros2_node src/qgc_mission_ros2.cpp)

ament_target_dependencies(qgc_mission_ros2_node
  rclcpp
  mavros
  mavros_msgs
  ai_msgs
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
)

target_link_libraries(qgc_mission_ros2_node
  Eigen3::Eigen
  GeographicLib::GeographicLib
)


install(TARGETS
  qgc_mission_ros2_node
  DESTINATION lib/${PROJECT_NAME}
)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}
  )
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()