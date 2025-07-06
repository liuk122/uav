cmake_minimum_required(VERSION 3.8)
project(object_position)

add_compile_options(-std=c++17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(ai_msgs REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} src/object_position.cpp)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  tf2_ros
  sensor_msgs
  cv_bridge
  image_transport
  geometry_msgs
  ai_msgs
  tf2_geometry_msgs
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
)

install(TARGETS
    ${PROJECT_NAME}
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
