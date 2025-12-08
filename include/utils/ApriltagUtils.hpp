#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Forward declarations for the C-based apriltag library structs.
// This avoids including the C headers in our C++ header, which is good practice.
struct apriltag_detector;
struct apriltag_family;

// A struct to hold the results of a single AprilTag detection in a C++-friendly format.
struct TagDetection {
    int id;
    cv::Mat pose_R;
    cv::Mat pose_t;
    std::vector<cv::Point2f> corners;
    cv::Point2f center;
    double depth;
};

// This class encapsulates the logic for detecting AprilTags, estimating their pose,
// and calculating a scaling factor for depth maps.
class ApriltagUtils {
public:
    // Constructor: Initializes the detector with calibration data, tag family, and size.
    ApriltagUtils(const std::string& calib_data_path, const std::string& family, double tag_size);

    // Destructor: Cleans up the resources used by the apriltag C library.
    ~ApriltagUtils();

    // Detects tags in a given frame and returns a vector of detection results.
    std::vector<TagDetection> get_tags(const cv::Mat& frame);

    // Calculates a scaling factor for a relative depth map based on the known size
    // of the detected AprilTags. It also draws visualizations on the provided frame.
    void get_scaling_factor(const std::vector<TagDetection>& tags, cv::Mat& frame, const cv::Mat& relative_depth_map, double& out_scaling_factor);

private:
    // C library pointers
    apriltag_detector* at_detector;
    apriltag_family* tag_family;

    // Camera parameters
    cv::Mat cam_matrix;
    cv::Mat dist_coeffs;
    double tag_size_meters;
};
