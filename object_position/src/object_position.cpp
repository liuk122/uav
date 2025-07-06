#include "object_position.h" 

using namespace std::placeholders;

// 构造函数: 初始化所有成员变量和 ROS2 的发布者/订阅者
ObjectTracker::ObjectTracker(const rclcpp::NodeOptions & options) 
    : Node("object_position", options), // 初始化节点名称
      isObjectDetected_(false),
      hasObjectPosition_(false),
      noObjectCounter_(0),
      frameId_("no_object"),
      depthImage_(cv::Mat::zeros(480, 640, CV_16UC1)),
      targetPoint_(cv::Point(0, 0)) 
{
    // 创建 image_transport 实例
    imageTransport_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());

    // 初始化发布者
    positionPub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/object_position", 10);
    
    // 订阅地平线 dnn_node_example 的话题和消息类型
    objectSub_ = this->create_subscription<ai_msgs::msg::PerceptionTargets>(
        "/hobot_dnn_detection", 10, std::bind(&ObjectTracker::objectCallback, this, _1));

    depthImageSub_ = imageTransport_->subscribe(
        "/d435/depth/image_rect_raw", 1, std::bind(&ObjectTracker::depthImageCallback, this, _1));

    colorImageSub_ = imageTransport_->subscribe(
        "/d435/color/image_raw", 1, std::bind(&ObjectTracker::colorImageCallback, this, _1));

    cameraInfoSub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/d435/depth/camera_info", 10, std::bind(&ObjectTracker::cameraInfoCallback, this, _1));
    
    RCLCPP_INFO(this->get_logger(), "Object position tracker node has been started.");
}

ObjectTracker::~ObjectTracker() {
    RCLCPP_INFO(this->get_logger(), "ObjectTracker destroyed");
}

void ObjectTracker::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    cameraInfo_ = *msg;
}

void ObjectTracker::depthImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    try {
        cv_bridge::CvImagePtr cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        depthImage_ = cvPtr->image;
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "CV Bridge error: %s", e.what());
    }
}


void ObjectTracker::objectCallback(const ai_msgs::msg::PerceptionTargets::SharedPtr msg) {
    if (msg->targets.empty() || msg->targets[0].rois.empty()) {
        return; // 如果没有检测到目标或ROI，则直接返回
    }

    const auto& target = msg->targets[0];
    const auto& roi = target.rois[0].rect; // roi 是 sensor_msgs::msg::RegionOfInterest

    frameId_ = target.type;
    isObjectDetected_ = true;

    // 计算包围框的中心点
    targetPoint_.x = roi.x_offset + roi.width / 2;
    targetPoint_.y = roi.y_offset + roi.height / 2;
}


void ObjectTracker::colorImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    (void)msg; // 避免 "unused parameter" 警告
    if (isObjectDetected_) {
        processDetectedObject();
    } else {
        handleNoObject();
    }
}

void ObjectTracker::processDetectedObject() {
    isObjectDetected_ = false;
    noObjectCounter_ = 0;

    // 检查 targetPoint 是否在深度图范围内
    if (targetPoint_.y >= depthImage_.rows || targetPoint_.x >= depthImage_.cols) {
        RCLCPP_WARN(this->get_logger(), "Target point is outside the depth image bounds.");
        return;
    }
    
    // 计算3D坐标
    // Z: 深度信息 (从毫米转换为米)
    objectZ_ = 0.001 * depthImage_.at<u_int16_t>(targetPoint_.y, targetPoint_.x);
    // X, Y: 直接使用像素坐标中心
    objectX_ = targetPoint_.x;
    objectY_ = targetPoint_.y;

    if (objectZ_ > 0) { // 仅在深度有效时发布
        publishObjectPosition();
    }
}

void ObjectTracker::handleNoObject() {
    noObjectCounter_++;
    if (noObjectCounter_ >= 5) {
        noObjectCounter_ = 0;
        if(hasObjectPosition_){ // 只有在之前有位置时才发布一次空位置
             RCLCPP_INFO(this->get_logger(), "Object lost, publishing empty position.");
             hasObjectPosition_ = false;
             publishEmptyPosition();
        }
    }
}

void ObjectTracker::publishObjectPosition() {
    hasObjectPosition_ = true;
    auto objectPose = std::make_unique<geometry_msgs::msg::PoseStamped>();

    objectPose->header.frame_id = frameId_;
    objectPose->header.stamp = this->get_clock()->now(); // ROS2 获取时间的方式
    objectPose->pose.position.x = objectX_;
    objectPose->pose.position.y = objectY_;
    objectPose->pose.position.z = objectZ_;
    
    // ROS2 中创建0旋转的四元数
    tf2::Quaternion q;
    q.setRPY(0, 0, 0); // Roll, Pitch, Yaw
    objectPose->pose.orientation = tf2::toMsg(q);

    positionPub_->publish(std::move(objectPose));
}

void ObjectTracker::publishEmptyPosition() {
    auto objectPose = std::make_unique<geometry_msgs::msg::PoseStamped>();
    
    objectPose->header.frame_id = "camera_color_optical_frame";
    objectPose->header.stamp = this->get_clock()->now();
    objectPose->pose.position.x = 0;
    objectPose->pose.position.y = 0;
    objectPose->pose.position.z = 0;
    
    tf2::Quaternion q;
    q.setRPY(0, 0, 0);
    objectPose->pose.orientation = tf2::toMsg(q);
    
    positionPub_->publish(std::move(objectPose));
}


// ROS2 标准的 main 函数
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    auto node = std::make_shared<ObjectTracker>(options);
    rclcpp::spin(node); // 启动ROS2事件循环
    rclcpp::shutdown();
    return 0;
}
