#ifndef APRILTAG_UTILS_HPP
#define APRILTAG_UTILS_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Forward declarations for AprilTag C library types
struct apriltag_detector;
struct apriltag_family;

// Structure to hold tag detection results
struct TagDetection {
    int id;
    std::vector<cv::Point2f> corners;
    cv::Point2f center;
    cv::Mat pose_R;  // 3x3 rotation matrix
    cv::Mat pose_t;  // 3x1 translation vector
    double depth;    // Distance from camera in meters
};

class ApriltagUtils {
public:
    ApriltagUtils(const std::string& calib_data_path,
                  const std::string& family,
                  double tag_size);
    
    ~ApriltagUtils();
    
    // Detect AprilTags in frame and return detection results
    std::vector<TagDetection> get_tags(const cv::Mat& frame);
    
    // Calculate scaling factor for depth maps using detected tags
    void get_scaling_factor(const std::vector<TagDetection>& tags,
                           cv::Mat& frame,
                           const cv::Mat& relative_depth_map,
                           double& out_scaling_factor);

private:
    apriltag_detector* at_detector;
    apriltag_family* tag_family;
    std::string family_name;
    cv::Mat cam_matrix;
    cv::Mat dist_coeffs;
    double tag_size_meters;
};

#endif // APRILTAG_UTILS_HPP
