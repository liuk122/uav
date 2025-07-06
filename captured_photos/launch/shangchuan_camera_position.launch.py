
import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 获取当前工作空间的路径
    pkg_dir = os.path.dirname(os.path.abspath(__file__))

    # 定义要启动的节点
    
    shangchuan_camera_position_node = Node(
        package='captured_photos',
        executable='shangchuan_camera_position.py',
        name='multimedia_and_position_uploader',
        output='screen'
    )

    # 创建LaunchDescription对象
    ld = LaunchDescription()

    # 将节点添加到LaunchDescription中
    
    ld.add_action(shangchuan_camera_position_node)

    return ld
