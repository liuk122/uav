#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
from cv_bridge import CvBridge
import cv2
import os
import time
import threading
import queue
import copy
from collections import deque

# 导入地平线平台的消息类型
from ai_msgs.msg import PerceptionTargets

# 导入阿里云 OSS SDK
import oss2

# 【修改 1】: 导入QoS相关模块，这是解决收不到位置数据的关键
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class CloudUploaderNode(Node):
    def __init__(self):
        super().__init__('cloud_uploader_node')

        # --- 基础配置 ---
        self.SAVE_FOLDER = '/home/sunrise/ros2_ws/captured_data'
        if not os.path.exists(self.SAVE_FOLDER):
            os.makedirs(self.SAVE_FOLDER)
        self.get_logger().info(f"本地数据将保存至: {self.SAVE_FOLDER}")

        # --- 阿里云 OSS 配置 ---
        self.OSS_ACCESS_KEY_ID = ''
        self.OSS_ACCESS_KEY_SECRET = ''
        self.OSS_ENDPOINT = 'oss-cn-beijing.aliyuncs.com'
        self.OSS_BUCKET_NAME = ''
        self.OSS_VIDEO_FOLDER = ''
        self.OSS_POSITION_FOLDER = ''

        try:
            auth = oss2.Auth(self.OSS_ACCESS_KEY_ID, self.OSS_ACCESS_KEY_SECRET)
            self.bucket = oss2.Bucket(auth, self.OSS_ENDPOINT, self.OSS_BUCKET_NAME)
            self.get_logger().info("阿里云 OSS 客户端初始化成功。")
        except Exception as e:
            self.get_logger().error(f"阿里云 OSS 客户端初始化失败: {e}")
            raise e

        # --- ROS 2 & CV 相关初始化 ---
        self.bridge = CvBridge()
        self.latest_position = None
        self.detection_lock = threading.Lock()
        self.recording = False

        # 图像缓冲区和线程安全队列
        self.image_queue = deque(maxlen=200)  # 增加队列长度，以缓存足够预录制和录制时间的帧
        self.image_lock = threading.Lock()
        self.image_counter = 0

        # 【新增】: 增加位置数据状态变量，用于判断数据是否有效
        self.position_available = False
        self.last_position_time = 0

        # 录制线程
        self.recording_thread = None

        # --- 视频录制相关变量 ---
        self.video_count = 0
        self.last_record_time = 0
        self.COOLDOWN_TIME = 10  # 增加冷却时间避免频繁触发
        self.RECORD_DURATION = 10
        self.FPS = 15.0 # 明确为浮点数
        self.FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
        self.TIMESTAMP_TOLERANCE = 5.0 # 位置数据时间容差(秒)


        # --- 异步上传队列 ---
        self.upload_queue = queue.Queue(maxsize=200)
        self.uploader_thread = threading.Thread(target=self.oss_uploader_worker)
        self.uploader_thread.daemon = True
        self.uploader_thread.start()

        # 【修改 2】: 定义一个专门用于传感器数据的QoS配置
        sensor_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # --- 创建 ROS 2 订阅者 ---
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 50)
        self.create_subscription(PerceptionTargets, '/hobot_dnn_detection', self.detection_callback, 10)
        # 【修改 3】: 在订阅位置话题时，应用我们定义的sensor_qos_profile
        self.create_subscription(
            NavSatFix,
            '/mavros/global_position/global',
            self.position_callback,
            qos_profile=sensor_qos_profile # 应用QoS配置
        )

        self.get_logger().info("云端上传节点已启动 (ROS 2 Humble)。正在等待GPS信号...")

    def oss_uploader_worker(self):
        """异步上传工作线程"""
        self.get_logger().info("OSS 上传工作线程启动。")
        while rclpy.ok():
            try:
                file_type, local_path, oss_path, max_retries = self.upload_queue.get(timeout=1.0)
                success = False
                for attempt in range(max_retries):
                    try:
                        result = self.bucket.put_object_from_file(oss_path, local_path)
                        if 200 <= result.status < 300:
                            self.get_logger().info(f"{file_type} 上传成功: {oss_path}")
                            success = True
                            os.remove(local_path)
                            self.get_logger().info(f"本地文件已删除: {local_path}")
                            break
                        else:
                            self.get_logger().error(f"OSS 上传失败 (状态码: {result.status}): {oss_path}")
                    except Exception as e:
                        self.get_logger().error(f"{file_type} 上传重试 {attempt + 1}/{max_retries} 失败: {e}")
                    time.sleep(2 ** attempt)
                if not success:
                    self.get_logger().error(f"最终放弃上传 {file_type}: {local_path}")
                self.upload_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"上传线程异常: {e}")

    def image_callback(self, msg):
        """持续更新最新图像帧"""
        try:
            # 不再需要 self.image_buffer，直接使用队列
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.image_lock:
                self.image_queue.append((time.time(), img))
            self.image_counter += 1
            
            if self.image_counter % 60 == 0:
                self.get_logger().info(f"收到图像 #{self.image_counter}, 队列大小: {len(self.image_queue)}")
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")

    def position_callback(self, msg):
        """持续更新最新 GPS 位置并标记可用"""
        if not self.position_available:
            self.get_logger().info("首次成功接收到GPS信号！位置数据现在可用。")
        
        with self.detection_lock: # 使用锁确保线程安全
            self.latest_position = msg
            self.position_available = True
            self.last_position_time = time.time()

    def detection_callback(self, msg: PerceptionTargets):
        """处理检测结果并触发录制"""
        # 如果正在录制或冷却中，则直接返回
        if self.recording or (time.time() - self.last_record_time < self.COOLDOWN_TIME):
            return

        if not self.detection_lock.acquire(blocking=False):
            return

        try:
            person_detected = any(target.type == 'person' and any(roi.confidence > 0.5 for roi in target.rois) for target in msg.targets)

            if not person_detected:
                return

            self.get_logger().info("检测到 'person'，准备触发视频录制...")

            # 【修改 4】: 增加对位置数据有效性的严格检查
            current_time = time.time()
            position_to_save = None
            position_age = current_time - self.last_position_time
            if self.position_available and position_age <= self.TIMESTAMP_TOLERANCE:
                position_to_save = copy.deepcopy(self.latest_position)
                self.get_logger().info(f"已锁定有效位置数据 (数据龄: {position_age:.2f}秒) 用于本次录制。")
            else:
                if not self.position_available:
                    self.get_logger().error("无可用位置数据，将只上传视频。请检查GPS设备和MAVROS。")
                else:
                    self.get_logger().warn(f"位置数据已过期 (数据龄: {position_age:.2f}秒)，本次录制将不关联位置信息。")

            self.last_record_time = current_time
            self.recording = True # 设置录制标志

            # 在新线程中执行录制，避免阻塞回调函数
            self.recording_thread = threading.Thread(target=self.record_video, args=(position_to_save,))
            self.recording_thread.daemon = True
            self.recording_thread.start()

        except Exception as e:
            self.get_logger().error(f"处理检测回调时发生严重错误: {e}")
        finally:
            self.detection_lock.release()

    def record_video(self, position_to_save):
        """在单独线程中录制视频"""
        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            base_filename = f'capture_{timestamp_str}_{self.video_count}'
            video_filename = f'{base_filename}.mp4'
            local_video_path = os.path.join(self.SAVE_FOLDER, video_filename)

            with self.image_lock:
                if not self.image_queue:
                    self.get_logger().error("录制开始时图像队列为空，取消录制。")
                    self.recording = False
                    return
                # 使用队列中最后一帧的尺寸来初始化录制器
                _, last_img = self.image_queue[-1]
                height, width, _ = last_img.shape

            out = cv2.VideoWriter(local_video_path, self.FOURCC, self.FPS, (width, height))
            if not out.isOpened():
                self.get_logger().error(f"无法打开 VideoWriter，请检查编码器({self.FOURCC})或路径权限。")
                self.recording = False
                return

            self.get_logger().info(f"开始录制视频: {local_video_path}")
            start_time = time.time()
            frame_count = 0
            
            # 【修复 5】: 修复视频录制为静止画面的BUG
            # 录制开始时，记录下当前队列中最后一帧的时间戳
            with self.image_lock:
                last_written_ts, _ = self.image_queue[-1]

            # 录制循环
            while (time.time() - start_time) < self.RECORD_DURATION:
                frame_to_write = None
                with self.image_lock:
                    # 从后向前遍历队列，寻找比上一帧更新的图像
                    for i in range(len(self.image_queue) - 1, -1, -1):
                        ts, img = self.image_queue[i]
                        if ts > last_written_ts:
                            frame_to_write = img
                            last_written_ts = ts
                            break # 找到最新的一帧就跳出
                
                if frame_to_write is not None:
                    out.write(frame_to_write)
                    frame_count += 1
                
                # 精确控制帧率
                time.sleep(max(0, 1.0/self.FPS - (time.time() - start_time - frame_count/self.FPS)))

            out.release()
            actual_duration = time.time() - start_time
            self.get_logger().info(f"视频录制完成，共录制 {frame_count} 帧，时长: {actual_duration:.2f}秒")

            # 上传视频
            oss_video_path = f"{self.OSS_VIDEO_FOLDER}{video_filename}"
            self.upload_queue.put(("视频", local_video_path, oss_video_path, 3))
            
            # 保存并上传位置数据
            if position_to_save:
                pos_filename = f'{base_filename}.txt'
                local_pos_path = os.path.join(self.SAVE_FOLDER, pos_filename)
                with open(local_pos_path, 'w') as f:
                    f.write("timestamp,latitude,longitude,altitude\n")
                    ts = position_to_save.header.stamp.sec + position_to_save.header.stamp.nanosec / 1e9
                    f.write(f"{ts},{position_to_save.latitude},{position_to_save.longitude},{position_to_save.altitude}\n")
                self.get_logger().info(f"关联的位置数据已保存: {local_pos_path}")
                oss_pos_path = f"{self.OSS_POSITION_FOLDER}{pos_filename}"
                self.upload_queue.put(("位置数据", local_pos_path, oss_pos_path, 3))
            else:
                self.get_logger().warn("本次录制无有效的关联位置数据。")

            self.video_count += 1

        except Exception as e:
            self.get_logger().error(f"视频录制过程中发生严重错误: {e}")
        finally:
            self.recording = False # 确保录制标志在任何情况下都能被重置

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CloudUploaderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n程序被中断...")
    finally:
        if node:
            node.get_logger().info("节点关闭，等待剩余上传任务完成...")
            # 确保录制线程已结束
            if node.recording_thread and node.recording_thread.is_alive():
                node.recording = False # 发送停止信号
                node.recording_thread.join(timeout=2.0) # 等待线程结束

            node.upload_queue.join()
            node.get_logger().info("所有上传任务已完成。")
            node.destroy_node()
        rclpy.shutdown()
        print("程序已安全退出。")

if __name__ == '__main__':
    main()