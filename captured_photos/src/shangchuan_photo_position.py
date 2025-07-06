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

# 导入地平线平台的消息类型
from ai_msgs.msg import PerceptionTargets

# 导入阿里云OSS SDK
import oss2

# 【修改 1】: 导入QoS相关模块，这是解决收不到位置数据的关键
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class CloudUploaderNode(Node):
    def __init__(self):
        super().__init__('cloud_uploader_node')

        # --- 基础配置 ---
        self.SAVE_FOLDER = '/home/sunrise/ros2_ws/captured_photo'
        if not os.path.exists(self.SAVE_FOLDER):
            os.makedirs(self.SAVE_FOLDER)
        self.get_logger().info(f"本地数据将保存至: {self.SAVE_FOLDER}")

        # --- 阿里云OSS配置 ---
        self.OSS_ACCESS_KEY_ID = ''
        self.OSS_ACCESS_KEY_SECRET = ''
        self.OSS_ENDPOINT = 'oss-cn-beijing.aliyuncs.com'
        self.OSS_BUCKET_NAME = ''
        self.OSS_IMAGE_FOLDER = ''
        self.OSS_POSITION_FOLDER = ''

        try:
            auth = oss2.Auth(self.OSS_ACCESS_KEY_ID, self.OSS_ACCESS_KEY_SECRET)
            self.bucket = oss2.Bucket(auth, self.OSS_ENDPOINT, self.OSS_BUCKET_NAME)
            self.get_logger().info("阿里云OSS客户端初始化成功。")
        except Exception as e:
            self.get_logger().error(f"阿里云OSS客户端初始化失败: {e}")
            raise e

        # --- ROS 2 & CV 相关初始化 ---
        self.bridge = CvBridge()
        self.latest_position = None
        self.position_available = False
        self.last_position_time = 0
        self.detection_lock = threading.Lock()
        self.capturing = False

        self.image_queue = queue.Queue(maxsize=100)
        self.image_counter = 0

        self.capturing_condition = threading.Condition()
        self.photo_count = 0
        # 【修改 2】: 增加一个变量，用于在检测时锁定位置信息，安全地传递给拍照线程
        self.position_for_capture = None

        # --- 拍照相关变量 ---
        self.photo_count_total = 0
        self.last_capture_time = 0
        self.COOLDOWN_TIME = 5
        self.INTERVAL_TIME = 3
        self.MAX_PHOTOS = 5
        self.TIMESTAMP_TOLERANCE = 5.0

        # --- 异步上传队列 ---
        self.upload_queue = queue.Queue(maxsize=200)
        self.uploader_thread = threading.Thread(target=self.oss_uploader_worker)
        self.uploader_thread.daemon = True
        self.uploader_thread.start()

        # 【修改 3】: 定义一个专门用于传感器数据的QoS配置
        sensor_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE # 对于传感器数据通常是VOLATILE
        )

        # --- 创建ROS 2订阅者 ---
        # 使用默认的QoS (usually a larger queue size is better for images)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 50)
        self.create_subscription(PerceptionTargets, '/hobot_dnn_detection', self.detection_callback, 10)
        # 【修改 4】: 在订阅位置话题时，应用我们定义的sensor_qos_profile
        self.create_subscription(
            NavSatFix,
            '/mavros/global_position/global',
            self.position_callback,
            qos_profile=sensor_qos_profile  # 应用QoS配置
        )

        # 拍照工作线程
        self.capturing_worker = threading.Thread(target=self.capturing_worker_thread)
        self.capturing_worker.daemon = True
        self.capturing_worker.start()

        self.get_logger().info("云端上传节点已启动 (ROS 2 Humble)。正在等待GPS信号...")

    def oss_uploader_worker(self):
        self.get_logger().info("OSS上传工作线程启动。")
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
                            os.remove(local_path) # 上传成功后可以删除本地文件
                            self.get_logger().info(f"本地文件已删除: {local_path}")
                            break
                        else:
                            self.get_logger().error(f"OSS上传失败 (状态码: {result.status}): {oss_path}")
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
        try:
            # 只在需要拍照时才进行图像转换和入队，降低CPU负载
            if not self.capturing and self.image_queue.full():
                return # 如果不处于拍照状态且队列已满，则不处理新图像

            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_counter += 1
            
            # 使用 put_nowait 并处理 Full 异常，确保队列操作是非阻塞的
            try:
                self.image_queue.put_nowait(img)
            except queue.Full:
                try:
                    self.image_queue.get_nowait()
                    self.image_queue.put_nowait(img)
                except queue.Empty:
                    pass
        except Exception as e:
            self.get_logger().error(f"图像回调处理失败: {e}")

    def position_callback(self, msg):
        # 第一次收到位置数据时，打印一条特殊信息
        if not self.position_available:
            self.get_logger().info("首次成功接收到GPS信号！位置数据现在可用。")
        
        self.latest_position = msg
        self.position_available = True
        self.last_position_time = time.time()
        
        # 打印详细的位置数据 (可以降低频率，避免刷屏)
        if self.image_counter % 30 == 0: # 每30帧图像的时间大约打印一次
             if hasattr(msg, 'status') and msg.status.status >= 0:
                 self.get_logger().info(f"收到有效GPS位置: lat={msg.latitude:.6f}, lon={msg.longitude:.6f}, alt={msg.altitude:.2f}")
             else:
                 self.get_logger().warn(f"收到GPS位置但状态未知: lat={msg.latitude:.6f}, lon={msg.longitude:.6f}")

    def detection_callback(self, msg: PerceptionTargets):
        if not self.detection_lock.acquire(blocking=False):
            return

        try:
            current_time = time.time()
            if current_time - self.last_capture_time < self.COOLDOWN_TIME:
                return

            person_detected = any(target.type == 'person' and any(roi.confidence > 0.5 for roi in target.rois) for target in msg.targets)

            if not person_detected:
                return
            
            self.get_logger().info("检测到 'person'，准备触发拍照...")

            # 【修改 5】: 关键逻辑修改。在检测到人的瞬间，就决定用哪个位置信息，并将其传递给拍照线程。
            position_age = current_time - self.last_position_time
            
            temp_position_for_capture = None
            if self.position_available and position_age <= self.TIMESTAMP_TOLERANCE:
                temp_position_for_capture = copy.deepcopy(self.latest_position)
                self.get_logger().info(f"已锁定有效位置数据 (数据龄: {position_age:.2f}秒) 用于本次拍照。")
            else:
                if not self.position_available:
                    self.get_logger().error("无可用位置数据，将只上传图片。请检查GPS设备和MAVROS。")
                else: # 位置数据过期
                    self.get_logger().warn(f"位置数据已过期 (数据龄: {position_age:.2f}秒)，本次拍照将不关联位置信息。")
            
            self.last_capture_time = current_time

            with self.capturing_condition:
                self.capturing = True
                self.photo_count = 0
                # 将捕获到的位置信息安全地赋值给共享变量
                self.position_for_capture = temp_position_for_capture
                self.capturing_condition.notify_all() # 唤醒拍照线程

            self.get_logger().info("已通知拍照线程开始工作。")

        except Exception as e:
            self.get_logger().error(f"处理检测回调时发生严重错误: {e}")
        finally:
            self.detection_lock.release()

    def capturing_worker_thread(self):
        self.get_logger().info("拍照工作线程启动。")

        while rclpy.ok():
            with self.capturing_condition:
                while not self.capturing:
                    self.capturing_condition.wait()

                # 【修改 6】: 拍照线程直接使用从检测线程传递过来的位置信息，不再自己重新判断。
                # 这样可以保证一个检测事件触发的所有照片都使用同一个时间点的位置。
                position_to_save = self.position_for_capture
                self.get_logger().info(f"拍照线程开始执行，将连续拍摄 {self.MAX_PHOTOS} 张照片。")
                if position_to_save:
                    self.get_logger().info("本次拍照将关联位置数据。")
                else:
                    self.get_logger().warn("本次拍照无关联位置数据。")

            # 拍照循环
            capture_start_time = time.time()
            while self.capturing and self.photo_count < self.MAX_PHOTOS:
                try:
                    # 从队列中获取最新帧
                    img = self.image_queue.get(timeout=1.0)

                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    base_filename = f'photo_{timestamp_str}_{self.photo_count_total}'

                    img_filename = f'{base_filename}.jpg'
                    local_img_path = os.path.join(self.SAVE_FOLDER, img_filename)
                    cv2.imwrite(local_img_path, img)
                    self.get_logger().info(f"保存图片: {local_img_path}")

                    oss_img_path = f"{self.OSS_IMAGE_FOLDER}{img_filename}"
                    self.upload_queue.put(("图片", local_img_path, oss_img_path, 3))

                    # 使用之前锁定的位置数据
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

                    self.photo_count += 1
                    self.photo_count_total += 1

                    # 等待下一个拍照间隔
                    time.sleep(self.INTERVAL_TIME)

                except queue.Empty:
                    self.get_logger().warn("图像队列为空，等待新帧...")
                    time.sleep(0.5) # 等待一下，避免CPU空转
                except Exception as e:
                    self.get_logger().error(f"拍照循环中发生错误: {e}")

            # 拍照结束
            if self.capturing:
                self.get_logger().info(f"连续拍照完成，共拍摄 {self.photo_count} 张照片。")
                with self.capturing_condition:
                    self.capturing = False
                    # 清理，等待下一次触发
                    self.position_for_capture = None

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
            # 停止接受新任务
            node.capturing = False
            # 等待队列中的所有任务被处理
            node.upload_queue.join()
            node.get_logger().info("所有上传任务已完成。")
            node.destroy_node()
        rclpy.shutdown()
        print("程序已安全退出。")

if __name__ == '__main__':
    main()