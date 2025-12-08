#pragma once

#include <iostream>
#include <array>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"  // for cv::Mat (changed to quotes)
#include <Eigen/Dense>         // for Eigen::Matrix4f
#include <nlohmann/json.hpp>   // for nlohmann::json

// --- Enum for PickState ---
enum class PickState {
    IDLE = 0,
    OBJECT = 1,
    TRACKING = 2,
    PICK = 3
};

// --- Struct for IMU transform ---
struct ImuTransform {
    struct Translation {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    } translation;

    struct Rotation {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double w = 0.0;
    } rotation;

    // Constructor to parse from nlohmann::json
    ImuTransform() = default;
    ImuTransform(const nlohmann::json& json_data) {
        if (json_data.contains("transform")) {
            const auto& transform = json_data["transform"];
            if (transform.contains("position")) {
                const auto& pos = transform["position"];
                translation.x = pos.value("x", 0.0);
                translation.y = pos.value("y", 0.0);
                translation.z = pos.value("z", 0.0);
            }
            if (transform.contains("orientation")) { // Assuming orientation for rotation
                const auto& orient = transform["orientation"];
                rotation.x = orient.value("x", 0.0);
                rotation.y = orient.value("y", 0.0);
                rotation.z = orient.value("z", 0.0);
                rotation.w = orient.value("w", 0.0);
            }
        }
    }
};

// --- StateCommand container class ---
class StateCommand {
public:
    // Constructor
    StateCommand();

    // Members
    cv::Mat RGBframe;              // Image frame
    double scaling_factor = 1.0;   // scaling factor (if needed)
    cv::Mat depth_map;             // depth image

    ImuTransform imu_tf;           // IMU transform

    Eigen::Matrix4f camera_pos;    // 4x4 transform (camera pose)
    Eigen::Matrix4f saved_3d_pos;  // saved 3D transform
    Eigen::Matrix4f track_3d_pos;  // tracked 3D transform

    // If you want to store state explicitly
    PickState state = PickState::IDLE;
};

extern StateCommand state_command;