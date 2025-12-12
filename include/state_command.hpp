#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class StateCommand {
public:
    // RGB frame from camera
    cv::Mat RGBframe;
    
    // Depth map
    cv::Mat depth_map;
    
    // Scaling factor from AprilTag
    double scaling_factor = -1.0;
    
    // IMU transform matrix (4x4)
    Eigen::Matrix4f imu_transform = Eigen::Matrix4f::Identity();
    
    StateCommand() = default;
};
