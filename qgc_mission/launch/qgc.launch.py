import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource, AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():

    # --- 1. 定位功能包路径 ---
    your_pkg_name = 'qgc_mission' 
    mavros_pkg_share = get_package_share_directory('mavros')
    realsense_pkg_share = get_package_share_directory('realsense2_camera')

    # --- 2. MAVROS 启动配置 ---
    fcu_url_arg = DeclareLaunchArgument(
        'fcu_url',
        default_value='/dev/ttyUSB0:921600',
        description='MAVROS FCU connection URL.'
    )
    
    # 【最终关键修正】: 使用您确认存在的文件名 'px4.launch'
    mavros_launch_path = PathJoinSubstitution([
        mavros_pkg_share, 'launch', 'px4.launch'
    ])

    mavros_launch = IncludeLaunchDescription(
        # AnyLaunchDescriptionSource 可以处理 .launch, .launch.py, .launch.xml 等多种文件
        AnyLaunchDescriptionSource(mavros_launch_path),
        launch_arguments={
            'fcu_url': LaunchConfiguration('fcu_url'),
            'gcs_url': ''
        }.items()
    )

    # --- 3. RealSense 相机节点启动配置 ---
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                realsense_pkg_share,
                'launch',
                'rs_launch.py'
            ])
        )
    )
    
    # --- 4. 启动我们自己的任务节点 ---
    qgc_mission_node = Node(
        package=your_pkg_name,
        executable='qgc_mission_ros2_node',
        name='qgc_mission_node',
        output='screen'
    )

    # ==================== 【新增节点】 ====================
    # --- 5. 启动照片与位置上传节点 ---
    shangchuan_photo_position_node = Node(
        package='captured_photos',
        executable='shangchuan_photo_position.py',
        name='multimedia_and_position_uploader',
        output='log'
    )
    # =======================================================

    # --- 6. 整合并返回完整的 LaunchDescription ---
    # 将所有需要启动的动作（参数、包含的launch文件、节点）都放入列表中
    return LaunchDescription([
        fcu_url_arg,
        mavros_launch,
        realsense_launch,
        qgc_mission_node,
        shangchuan_photo_position_node  # <-- 将新节点添加到启动列表
    ])