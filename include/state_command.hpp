#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <memory>

// Forward declarations
class MiDaSDepth;
class FastSAMSegmenter;
class ApriltagUtils;
namespace utils {
    class HandLandmarkerMP;
    class IMUServer;
}
class IMUTracker;

// Enum for system states
enum class SystemState {
    IDLE,           // Initial state - initialize and verify components
    TRACKING,       // Active tracking state using AprilTag and MiDaS
    ERROR,          // Error state
    SHUTDOWN        // Clean shutdown state
};

// Struct to hold shared state data between states
struct StateCommand {
    // Current system state
    SystemState current_state = SystemState::IDLE;
    
    // RGB frame from camera
    cv::Mat rgb_frame;
    
    // Depth map from MiDaS
    cv::Mat depth_map;
    
    // AprilTag detection data
    bool apriltag_detected = false;
    int apriltag_id = -1;
    Eigen::Vector3f apriltag_position = Eigen::Vector3f::Zero();
    Eigen::Matrix3f apriltag_rotation = Eigen::Matrix3f::Identity();
    double scaling_factor = -1.0;
    
    // IMU transform matrix (4x4)
    Eigen::Matrix4f imu_transform = Eigen::Matrix4f::Identity();
    
    // Component initialization flags
    bool midas_initialized = false;
    bool fastsam_initialized = false;
    bool mediapipe_initialized = false;
    bool apriltag_initialized = false;
    bool imu_initialized = false;
    
    // Shared component pointers (to avoid re-initialization)
    MiDaSDepth* midas_utils = nullptr;
    FastSAMSegmenter* fastsam_utils = nullptr;
    ApriltagUtils* apriltag_utils = nullptr;
    utils::HandLandmarkerMP* mediapipe_utils = nullptr;
    utils::IMUServer* imu_server = nullptr;
    IMUTracker* imu_tracker = nullptr;
    
    // Camera calibration
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    
    // Error message
    std::string error_message;
    
    StateCommand() = default;
};
