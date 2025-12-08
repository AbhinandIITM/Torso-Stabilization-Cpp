#pragma once

#include <deque>
#include <cstddef> // For size_t
#include <Eigen/Dense>
#include <optional>
// ---------------- IMUTracker -----------------
// Declares the IMUTracker class.
// This class processes raw accelerometer and gyroscope data to estimate
// the device's orientation and position in 3D space.
class IMUTracker {
public:
    // Constructor with an optional buffer size for the accelerometer filter.
    IMUTracker(size_t buffer_size = 5);

    // Updates the tracker's state with new sensor data and a timestamp.
    // Returns the new 4x4 transformation matrix.
    Eigen::Matrix4d update(const Eigen::Vector3d& accel,
                           const Eigen::Vector3d& gyro,
                           double timestamp);

    // Returns the current 4x4 transformation matrix representing the tracker's pose.
    Eigen::Matrix4d get_transform() const;

private:
    // Applies a simple moving average filter to the accelerometer data.
    Eigen::Vector3d filter_accel(const Eigen::Vector3d& accel);

    // State variables
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Quaterniond orientation;
    //double last_timestamp;
    std::optional<double> last_timestamp;
    // Configuration and buffer for filtering
    size_t buffer_size;
    std::deque<Eigen::Vector3d> accel_buffer;
};
