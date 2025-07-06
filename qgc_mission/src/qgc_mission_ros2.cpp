#include "rclcpp/rclcpp.hpp"
#include <Eigen/Eigen>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <memory>
#include <array>
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "mavros_msgs/msg/state.hpp"
#include "mavros_msgs/msg/position_target.hpp"
#include "mavros_msgs/msg/waypoint_list.hpp"
#include "mavros_msgs/msg/home_position.hpp"
#include "mavros_msgs/srv/command_bool.hpp"
#include "mavros_msgs/srv/set_mode.hpp"
#include "mavros_msgs/srv/command_long.hpp"
#include "ai_msgs/msg/perception_targets.hpp"

// TF2 and Eigen Conversion
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include <mavros/frame_tf.hpp>

// GeographicLib for GPS transformations
#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/Constants.hpp>

using namespace std;
using namespace std::chrono_literals;

// 卡尔曼滤波器类
class KalmanFilter {
public:
    KalmanFilter() {
        x_ = Eigen::Vector4d::Zero();
        P_ = Eigen::Matrix4d::Identity();
        F_ = Eigen::Matrix4d::Identity();
        Q_ = Eigen::Matrix4d::Identity() * 0.01;
        Q_(0, 0) = 0.001; Q_(1, 1) = 0.001;
        H_ = Eigen::Matrix<double, 2, 4>::Zero();
        H_(0, 0) = 1; H_(1, 1) = 1;
        R_ = Eigen::Matrix2d::Identity() * 0.1;
    }

    void init(const Eigen::Vector2d& initial_measurement) {
        x_ << initial_measurement(0), initial_measurement(1), 0, 0;
        P_ = Eigen::Matrix4d::Identity();
    }

    void predict(double dt) {
        F_(0, 2) = dt; F_(1, 3) = dt;
        x_ = F_ * x_;
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    void update(const Eigen::Vector2d& measurement) {
        Eigen::Matrix2d S = H_ * P_ * H_.transpose() + R_;
        Eigen::Matrix<double, 4, 2> K = P_ * H_.transpose() * S.inverse();
        Eigen::Vector2d y = measurement - H_ * x_;
        x_ = x_ + K * y;
        Eigen::Matrix4d I = Eigen::Matrix4d::Identity();
        P_ = (I - K * H_) * P_;
    }

    Eigen::Vector4d get_state() const { return x_; }

private:
    Eigen::Vector4d x_; Eigen::Matrix4d P_; Eigen::Matrix4d F_;
    Eigen::Matrix<double, 2, 4> H_; Eigen::Matrix4d Q_; Eigen::Matrix2d R_;
};


class QGCMissionNode : public rclcpp::Node
{
public:
    QGCMissionNode() : Node("qgc_mission_node")
    {
        auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
        auto reliable_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

        state_sub_ = this->create_subscription<mavros_msgs::msg::State>("/mavros/state", reliable_qos, std::bind(&QGCMissionNode::state_cb, this, std::placeholders::_1));
        local_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/mavros/local_position/odom", sensor_qos, std::bind(&QGCMissionNode::local_odom_cb, this, std::placeholders::_1));
        waypoint_sub_ = this->create_subscription<mavros_msgs::msg::WaypointList>("/mavros/mission/waypoints", reliable_qos, std::bind(&QGCMissionNode::waypoints_cb, this, std::placeholders::_1));
        home_pos_sub_ = this->create_subscription<mavros_msgs::msg::HomePosition>("/mavros/home_position/home", reliable_qos, std::bind(&QGCMissionNode::home_pos_cb, this, std::placeholders::_1));
        local_pos_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("/mavros/local_position/pose", sensor_qos, std::bind(&QGCMissionNode::local_pos_cb, this, std::placeholders::_1));
        gps_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>("/mavros/global_position/global", sensor_qos, std::bind(&QGCMissionNode::gps_cb, this, std::placeholders::_1));
        perception_sub_ = this->create_subscription<ai_msgs::msg::PerceptionTargets>("/hobot_dnn_detection", sensor_qos, std::bind(&QGCMissionNode::perception_cb, this, std::placeholders::_1));
        local_pos_pub_ = this->create_publisher<mavros_msgs::msg::PositionTarget>("/mavros/setpoint_raw/local", reliable_qos);
        arming_client_ = this->create_client<mavros_msgs::srv::CommandBool>("mavros/cmd/arming");
        set_mode_client_ = this->create_client<mavros_msgs::srv::SetMode>("mavros/set_mode");
        ctrl_pwm_client_ = this->create_client<mavros_msgs::srv::CommandLong>("mavros/cmd/command");

        RCLCPP_INFO(this->get_logger(), "QGC Mission Node (P+FF Controller) initialized.");
    }

    void run();

private:
    float ALTITUDE = 10.0;
    float ALTITUDE_Throw = 8.0;

    //P+FF控制器参数
    float Kp_pos_ = 0.005; // P反馈项的增益
    float Kf_vel_ = 0.7;   // FF前馈项的增益 (0.0到1.0之间)
    float max_track_speed = 2.0;

    static const int HISTORY_SIZE = 10;
    std::array<bool, HISTORY_SIZE> detection_history_buffer_{};
    int detection_history_index_ = 0;

    // 状态与数据变量
    vector<geometry_msgs::msg::PoseStamped> pose;
    mavros_msgs::msg::PositionTarget pos_target;
    mavros_msgs::msg::State current_state;
    mavros_msgs::msg::HomePosition home_pos;
    geometry_msgs::msg::PoseStamped local_pos;
    nav_msgs::msg::Odometry local_odom;
    mavros_msgs::msg::WaypointList waypoints;
    Eigen::Vector3d current_gps;
    bool final_person_found_in_current_analysis = false;
    int consecutive_target_loss_frames_ = 0;
    bool payload_dropped_ = false;
    tf2::Quaternion quat;
    double roll = 0.0, pitch = 0.0, yaw = 0.0;
    float init_position_x_take_off = 0, init_position_y_take_off = 0, init_position_z_take_off = 0;
    bool flag_init_position = false;
    bool flag_waypoints_receive = false;
    
    // 卡尔曼滤波器相关
    KalmanFilter kf_;
    bool kf_initialized_ = false;
    rclcpp::Time last_perception_time_;
    

    rclcpp::Time lib_mission_success_time_record;
    bool lib_time_record_start_flag = false;
    
    rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr local_odom_sub_;
    rclcpp::Subscription<mavros_msgs::msg::WaypointList>::SharedPtr waypoint_sub_;
    rclcpp::Subscription<mavros_msgs::msg::HomePosition>::SharedPtr home_pos_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr local_pos_sub_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub_;
    rclcpp::Subscription<ai_msgs::msg::PerceptionTargets>::SharedPtr perception_sub_;
    rclcpp::Publisher<mavros_msgs::msg::PositionTarget>::SharedPtr local_pos_pub_;
    rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arming_client_;
    rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr set_mode_client_;
    rclcpp::Client<mavros_msgs::srv::CommandLong>::SharedPtr ctrl_pwm_client_;

    // 函数声明...
    void state_cb(const mavros_msgs::msg::State::SharedPtr msg);
    void home_pos_cb(const mavros_msgs::msg::HomePosition::SharedPtr msg);
    void local_odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg);
    void local_pos_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void waypoints_cb(const mavros_msgs::msg::WaypointList::SharedPtr msg);
    void gps_cb(const sensor_msgs::msg::NavSatFix::SharedPtr msg);
    void perception_cb(const ai_msgs::msg::PerceptionTargets::SharedPtr msg);
    bool object_recognize_track_vel(float altitude, float error_max);
    bool lib_time_record_func(float time_duration, rclcpp::Time time_now);
    void lib_pwm_control(int pwm_channel_5, int pwm_channel_6);
    template<typename SrvT>
    bool call_service(typename rclcpp::Client<SrvT>::SharedPtr client, typename SrvT::Request::SharedPtr request, const std::string& service_name);
};

void QGCMissionNode::state_cb(const mavros_msgs::msg::State::SharedPtr msg){ current_state = *msg; }
void QGCMissionNode::home_pos_cb(const mavros_msgs::msg::HomePosition::SharedPtr msg){ home_pos = *msg; }
void QGCMissionNode::waypoints_cb(const mavros_msgs::msg::WaypointList::SharedPtr msg){ if (!msg->waypoints.empty()) { flag_waypoints_receive = true; waypoints = *msg; } }
void QGCMissionNode::gps_cb(const sensor_msgs::msg::NavSatFix::SharedPtr msg){ current_gps = { msg->latitude, msg->longitude, msg->altitude }; }
void QGCMissionNode::local_pos_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg) { local_pos = *msg; }
void QGCMissionNode::local_odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    local_odom = *msg;
    if (flag_init_position == false && (local_odom.pose.pose.position.z != 0)) {
        init_position_x_take_off = local_odom.pose.pose.position.x;
        init_position_y_take_off = local_odom.pose.pose.position.y;
        init_position_z_take_off = local_odom.pose.pose.position.z;
        flag_init_position = true;
    }
    tf2::fromMsg(local_odom.pose.pose.orientation, quat);
    tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
}

void QGCMissionNode::perception_cb(const ai_msgs::msg::PerceptionTargets::SharedPtr msg)
{
    bool person_detected_in_this_frame_raw = false;
    const ai_msgs::msg::Target* person_target = nullptr;
    for (const auto& target : msg->targets) {
        if (target.type == "person" && !target.rois.empty()) {
            person_detected_in_this_frame_raw = true;
            person_target = &target;
            break;
        }
    }
    
    if (person_detected_in_this_frame_raw && person_target != nullptr) {
        consecutive_target_loss_frames_ = 0;
        const auto& roi = person_target->rois[0].rect;
        Eigen::Vector2d measurement(
            static_cast<double>(roi.x_offset) + static_cast<double>(roi.width) / 2.0,
            static_cast<double>(roi.y_offset) + static_cast<double>(roi.height) / 2.0
        );
        
        // 【新逻辑】: 只有在KF未初始化，或者目标丢失超过30帧后，才重新初始化KF
        if (!kf_initialized_ || consecutive_target_loss_frames_ > 30) {
            kf_.init(measurement);
            kf_initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "卡尔曼滤波器已(重新)初始化！");
        } else {
            double dt = (this->get_clock()->now() - last_perception_time_).seconds();
            if (dt > 0.0 && dt < 1.0) {
                kf_.predict(dt);
                kf_.update(measurement);
            }
        }
        last_perception_time_ = this->get_clock()->now();

    } else {
        consecutive_target_loss_frames_++;
    }

    detection_history_buffer_[detection_history_index_] = person_detected_in_this_frame_raw;
    detection_history_index_ = (detection_history_index_ + 1) % HISTORY_SIZE;
    int detected_count = 0;
    for(int i = 0; i < HISTORY_SIZE; ++i) {
        if (detection_history_buffer_[i]) {
            detected_count++;
        }
    }
    if (detected_count >= 6) { // 使用您设定的阈值
        if (!final_person_found_in_current_analysis) { RCLCPP_INFO(this->get_logger(), "稳定识别到目标!"); }
        final_person_found_in_current_analysis = true;
    } else {
        if (final_person_found_in_current_analysis) { RCLCPP_INFO(this->get_logger(), "稳定识别丢失。"); }
        final_person_found_in_current_analysis = false;
    }
}

bool QGCMissionNode::object_recognize_track_vel(float altitude, float error_max)
{
    if (!kf_initialized_) {
        pos_target.velocity.x = 0;
        pos_target.velocity.y = 0;
        pos_target.coordinate_frame = 8;
        pos_target.position.z = init_position_z_take_off + altitude;
        return false;
    }

    // 补偿延迟 ---
    double dt = (this->get_clock()->now() - last_perception_time_).seconds();
    if (dt > 0.0 && dt < 1.0) {
        kf_.predict(dt);
        last_perception_time_ = this->get_clock()->now();
    }

    // 预测出的完整状态 ---
    Eigen::Vector4d predicted_state = kf_.get_state();
    double pos_x_pred = predicted_state(0);
    double pos_y_pred = predicted_state(1);
    double vel_x_pred = predicted_state(2); // 预测出的X方向像素速度 (pixel/s)
    double vel_y_pred = predicted_state(3); // 预测出的Y方向像素速度 (pixel/s)

    // 检查是否已经对准目标
    if (fabs(pos_x_pred - 320) < error_max && fabs(pos_y_pred - 240) < error_max) {
        pos_target.velocity.x = 0;
        pos_target.velocity.y = 0;
        pos_target.coordinate_frame = 8;
        pos_target.position.z = init_position_z_take_off + altitude;
        return true; // 返回 true, 表示已对准
    }
    
    // 计算P+FF控制律 ---
    // P (Proportional) 反馈项: 修正位置误差
    float pos_error_x = pos_x_pred - 320.0;
    float pos_error_y = pos_y_pred - 240.0;
    float p_term_vy = -Kp_pos_ * pos_error_x; // 图像x轴误差 -> 无人机y轴速度
    float p_term_vx = -Kp_pos_ * pos_error_y; // 图像y轴误差 -> 无人机x轴速度

    // 将KF预测出的像素速度，通过增益映射到无人机的速度指令
    float ff_term_vy = -Kf_vel_ * (vel_x_pred * 0.002); 
    float ff_term_vx = -Kf_vel_ * (vel_y_pred * 0.002);

    // 最终指令 = P反馈 + FF前馈
    float final_velocity_x = p_term_vx + ff_term_vx;
    float final_velocity_y = p_term_vy + ff_term_vy;

    final_velocity_x = std::clamp(final_velocity_x, -max_track_speed, max_track_speed);
    final_velocity_y = std::clamp(final_velocity_y, -max_track_speed, max_track_speed);
    pos_target.velocity.x = final_velocity_x;
    pos_target.velocity.y = final_velocity_y;
    
    pos_target.type_mask = 1 + 2 + 64 + 128 + 256 + 512 + 1024 + 2048; // 速度控制
    pos_target.coordinate_frame = 8; // BODY_NED
    pos_target.position.z = init_position_z_take_off + altitude; // 使用传入的固定高度
    
    return false; // 表示追踪仍在进行
}

// --- 辅助函数 ---
bool QGCMissionNode::lib_time_record_func(float time_duration, rclcpp::Time time_now)
{
    if (lib_time_record_start_flag == false) {
        lib_mission_success_time_record = time_now;
        lib_time_record_start_flag = true;
    }
    if ((this->get_clock()->now() - lib_mission_success_time_record).seconds() > time_duration) {
        lib_time_record_start_flag = false;
        return true;
    } else {
        return false;
    }
}
void QGCMissionNode::lib_pwm_control(int pwm_channel_5, int pwm_channel_6)
{
    auto request = std::make_shared<mavros_msgs::srv::CommandLong::Request>();
    request->command = 187;
    request->param1 = ((static_cast<float>(pwm_channel_5) / 50.0) - 1.0);
    request->param2 = ((static_cast<float>(pwm_channel_6) / 50.0) - 1.0);
    ctrl_pwm_client_->async_send_request(request);
}
template<typename SrvT>
bool QGCMissionNode::call_service(typename rclcpp::Client<SrvT>::SharedPtr client, typename SrvT::Request::SharedPtr request, const std::string& service_name) {
    if (!client->wait_for_service(1s)) {
        RCLCPP_ERROR(this->get_logger(), "Service not available: %s", service_name.c_str());
        return false;
    }
    auto future = client->async_send_request(request);
    if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future, 1s) != rclcpp::FutureReturnCode::SUCCESS) {
        RCLCPP_ERROR(this->get_logger(), "Failed to call service %s", service_name.c_str());
        return false;
    }
    if constexpr (std::is_same_v<SrvT, mavros_msgs::srv::CommandBool>) { return future.get()->success; }
    if constexpr (std::is_same_v<SrvT, mavros_msgs::srv::SetMode>) { return future.get()->mode_sent; }
    return true;
}

// --- 主运行函数 ---
void QGCMissionNode::run()
{
    rclcpp::Rate rate(20.0);
    
    RCLCPP_INFO(this->get_logger(), "等待Mavros连接...");
    while (rclcpp::ok() && !current_state.connected) { rclcpp::spin_some(this->get_node_base_interface()); rate.sleep(); }
    RCLCPP_INFO(this->get_logger(), "Mavros已连接.");
    RCLCPP_INFO(this->get_logger(), "等待获取初始位置...");
    while (rclcpp::ok() && !flag_init_position) { rclcpp::spin_some(this->get_node_base_interface()); rate.sleep(); }
    RCLCPP_INFO(this->get_logger(), "已捕获初始位置: [%.2f, %.2f, %.2f]", init_position_x_take_off, init_position_y_take_off, init_position_z_take_off);
    RCLCPP_INFO(this->get_logger(), "等待从QGC接收航点...");
    while (rclcpp::ok() && !flag_waypoints_receive) { rclcpp::spin_some(this->get_node_base_interface()); rate.sleep(); }
    RCLCPP_INFO(this->get_logger(), "收到 %ld 个航点，正在处理...", waypoints.waypoints.size());
    for (const auto& wp : waypoints.waypoints) {
        geometry_msgs::msg::PoseStamped p;
        GeographicLib::Geocentric earth(GeographicLib::Constants::WGS84_a(), GeographicLib::Constants::WGS84_f());
        Eigen::Vector3d goal_gps(wp.x_lat, wp.y_long, 0), current_ecef, goal_ecef;
        earth.Forward(current_gps.x(), current_gps.y(), current_gps.z(), current_ecef.x(), current_ecef.y(), current_ecef.z());
        earth.Forward(goal_gps.x(), goal_gps.y(), goal_gps.z(), goal_ecef.x(), goal_ecef.y(), goal_ecef.z());
        Eigen::Vector3d ecef_offset = goal_ecef - current_ecef;
        Eigen::Vector3d enu_offset = mavros::ftf::transform_frame_ecef_enu(ecef_offset, current_gps);
        p.pose.position.x = init_position_x_take_off + enu_offset.x();
        p.pose.position.y = init_position_y_take_off + enu_offset.y();
        pose.push_back(p);
    }
    RCLCPP_INFO(this->get_logger(), "处理完成 %ld 个航点。", pose.size());

    enum class MissionState { ARM_AND_SET_MODE, TAKEOFF, WAYPOINT_MISSION, TRACKING };
    MissionState current_mission_state = MissionState::ARM_AND_SET_MODE;
    rclcpp::Time last_request_time = this->get_clock()->now();
    int current_waypoint_index = 0;
    RCLCPP_INFO(this->get_logger(), "进入主任务循环...");
    while(rclcpp::ok())
    {
        switch(current_mission_state)
        {
            case MissionState::ARM_AND_SET_MODE:
                if (current_state.mode != "OFFBOARD" && (this->get_clock()->now() - last_request_time > 2s)) {
                    auto offb_set_mode = std::make_shared<mavros_msgs::srv::SetMode::Request>();
                    offb_set_mode->custom_mode = "OFFBOARD";
                    call_service<mavros_msgs::srv::SetMode>(set_mode_client_, offb_set_mode, "SetMode");
                    last_request_time = this->get_clock()->now();
                } else if (!current_state.armed && (this->get_clock()->now() - last_request_time > 2s)) {
                    auto arm_cmd = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
                    arm_cmd->value = true;
                    call_service<mavros_msgs::srv::CommandBool>(arming_client_, arm_cmd, "Arming");
                    last_request_time = this->get_clock()->now();
                }
                pos_target.coordinate_frame = 1;
                pos_target.type_mask = 8 + 16 + 32 + 64 + 128 + 256 + 512 + 1024 + 2048;
                pos_target.position.x = init_position_x_take_off;
                pos_target.position.y = init_position_y_take_off;
                pos_target.position.z = init_position_z_take_off;
                if (current_state.mode == "OFFBOARD" && current_state.armed) {
                    RCLCPP_INFO(this->get_logger(), "成功进入OFFBOARD并已解锁！切换到起飞阶段。");
                    lib_pwm_control(0, 0);
                    rclcpp::sleep_for(200ms); // 短暂延时以确保指令执行
                    current_mission_state = MissionState::TAKEOFF;
                }
                break;
            case MissionState::TAKEOFF:
                { 
                    double takeoff_target_z = init_position_z_take_off + ALTITUDE;
                    pos_target.coordinate_frame = 1;
                    pos_target.type_mask = 8 + 16 + 32 + 64 + 128 + 256 + 512 + 1024 + 2048;
                    pos_target.position.x = init_position_x_take_off;
                    pos_target.position.y = init_position_y_take_off;
                    pos_target.position.z = takeoff_target_z;
                    if (fabs(local_pos.pose.position.z - takeoff_target_z) < 0.5) {
                        RCLCPP_INFO(this->get_logger(), "已到达目标高度，起飞完成。切换到航线任务阶段。");
                        current_mission_state = MissionState::WAYPOINT_MISSION;
                    }
                }
                break;
            case MissionState::WAYPOINT_MISSION:
                if (static_cast<size_t>(current_waypoint_index) >= pose.size()) {
                    RCLCPP_INFO(this->get_logger(), "所有航点完成，进入目标追踪阶段。");
                    current_mission_state = MissionState::TRACKING;
                    break;
                }
                if (final_person_found_in_current_analysis) {
                    RCLCPP_INFO(this->get_logger(), "发现目标！中断航线飞行，切换到目标追踪阶段。");
                    current_mission_state = MissionState::TRACKING;
                    break;
                }
                pos_target.coordinate_frame = 1;
                pos_target.type_mask = 8 + 16 + 32 + 64 + 128 + 256 + 512 + 1024 + 2048;
                pos_target.position.x = pose[current_waypoint_index].pose.position.x;
                pos_target.position.y = pose[current_waypoint_index].pose.position.y;
                pos_target.position.z = init_position_z_take_off + ALTITUDE;
                {
                    double dist_to_wp = std::sqrt(std::pow(local_pos.pose.position.x - pose[current_waypoint_index].pose.position.x, 2) + std::pow(local_pos.pose.position.y - pose[current_waypoint_index].pose.position.y, 2));
                    if (dist_to_wp < 1.0) {
                        RCLCPP_INFO(this->get_logger(), "已到达航点 %d", current_waypoint_index + 1);
                        current_waypoint_index++;
                    }
                }
                break;
            case MissionState::TRACKING:
                if (object_recognize_track_vel(ALTITUDE_Throw, 30)) {
                    if (!payload_dropped_) {
                        if (lib_time_record_func(3.0, this->get_clock()->now())) {
                            RCLCPP_INFO(this->get_logger(), "目标对准超过3秒，开始投放。");
                            lib_pwm_control(100, 100);
                            payload_dropped_ = true;
                            RCLCPP_INFO(this->get_logger(), "投放完毕！继续追踪目标...");
                        }
                    }
                }
                break;
        }
        local_pos_pub_->publish(pos_target);
        rclcpp::spin_some(this->get_node_base_interface());
        rate.sleep();
    }
}

// 主函数
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto mission_node = std::make_shared<QGCMissionNode>();
    mission_node->run();
    rclcpp::shutdown();
    return 0;
}