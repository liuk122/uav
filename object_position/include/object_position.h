#pragma once // 防止头文件被重复包含

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// 关键: 包含新的 ai_msgs 消息头文件
#include "ai_msgs/msg/perception_targets.hpp"

class ObjectTracker : public rclcpp::Node
{
public:
    // 构造函数，继承 rclcpp::Node
    explicit ObjectTracker(const rclcpp::NodeOptions & options);
    ~ObjectTracker();

private:
    // == 回调函数声明 ==
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void depthImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    // 关键: 修改 objectCallback 以接收新的消息类型
    void objectCallback(const ai_msgs::msg::PerceptionTargets::SharedPtr msg);
    void colorImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);

    // == 内部逻辑函数声明 ==
    void processDetectedObject();
    void handleNoObject();
    void publishObjectPosition();
    void publishEmptyPosition();

    // == ROS2 成员变量 ==
    // 关键: 发布者类型修正为 PoseStamped
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr positionPub_;
    
    // 订阅者使用 ROS2 的智能指针类型
    rclcpp::Subscription<ai_msgs::msg::PerceptionTargets>::SharedPtr objectSub_;
    image_transport::Subscriber depthImageSub_;
    image_transport::Subscriber colorImageSub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cameraInfoSub_;
    
    // image_transport 实例
    std::shared_ptr<image_transport::ImageTransport> imageTransport_;

    // == 数据和状态成员变量 (原全局变量和类成员已整合) ==
    bool isObjectDetected_;
    bool hasObjectPosition_;
    int noObjectCounter_;
    std::string frameId_;

    cv::Mat depthImage_;
    cv::Point targetPoint_; // 目标中心像素坐标
    sensor_msgs::msg::CameraInfo cameraInfo_;
    
    // 存储计算出的3D坐标
    double objectX_, objectY_, objectZ_;
};

