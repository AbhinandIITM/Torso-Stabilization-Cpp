#include "utils/apriltag_utils.hpp"
#include <iostream>

// Include the C-based AprilTag library headers
extern "C" {
    #include "apriltag.h"
    #include "tag36h11.h"
    #include "tag25h9.h"
    #include "tag16h5.h"
    #include "tagCircle21h7.h"
    #include "tagCircle49h12.h"
    #include "tagCustom48h12.h"
    #include "tagStandard41h12.h"
    #include "tagStandard52h13.h"
    #include "apriltag_pose.h"
    #include "common/homography.h"
}

ApriltagUtils::ApriltagUtils(const std::string& calib_data_path,
                             const std::string& family,
                             double tag_size)
    : at_detector(nullptr),
      tag_family(nullptr),
      family_name(family),
      tag_size_meters(tag_size)
{
    std::cout << "Initializing AprilTag detector...\n";
    
    // Load camera calibration
    cv::FileStorage fs(calib_data_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open calibration file: " + calib_data_path);
    }
    
    fs["camera_matrix"] >> cam_matrix;
    fs["distortion_coefficients"] >> dist_coeffs;
    fs.release();
    
    if (cam_matrix.empty() || dist_coeffs.empty()) {
        throw std::runtime_error("Camera matrix or distortion coefficients not found in calibration file");
    }
    
    std::cout << "Camera calibration loaded\n";
    std::cout << "Camera Matrix:\n" << cam_matrix << "\n";
    
    // Create AprilTag detector
    at_detector = apriltag_detector_create();
    
    // Create tag family based on input
    if (family == "tag36h11") {
        tag_family = tag36h11_create();
    } else if (family == "tag25h9") {
        tag_family = tag25h9_create();
    } else if (family == "tag16h5") {
        tag_family = tag16h5_create();
    } else if (family == "tagCircle21h7") {
        tag_family = tagCircle21h7_create();
    } else if (family == "tagCircle49h12") {
        tag_family = tagCircle49h12_create();
    } else if (family == "tagCustom48h12") {
        tag_family = tagCustom48h12_create();
    } else if (family == "tagStandard41h12") {
        tag_family = tagStandard41h12_create();
    } else if (family == "tagStandard52h13") {
        tag_family = tagStandard52h13_create();
    } else {
        apriltag_detector_destroy(at_detector);
        throw std::runtime_error("Unsupported tag family: " + family);
    }
    
    // Add family to detector
    apriltag_detector_add_family(at_detector, tag_family);
    
    // ========================================
    // OPTIMIZED PARAMETERS FOR MULTIPLE TAGS
    // ========================================
    at_detector->quad_decimate = 2.0;      // Higher = faster but less accurate (1.0-4.0)
    at_detector->quad_sigma = 0.0;         // No blur for speed
    at_detector->nthreads = 4;             // Use all cores
    at_detector->debug = 0;                // No debug
    at_detector->refine_edges = 1;         // Better accuracy with minimal cost
    at_detector->decode_sharpening = 0.25; // Moderate sharpening
    
    std::cout << "AprilTag detector initialized with family: " << family << "\n";
    std::cout << "Tag size: " << tag_size_meters << " meters (" << (tag_size_meters * 100) << " cm)\n";
    std::cout << "Performance settings: quad_decimate=" << at_detector->quad_decimate 
              << ", nthreads=" << at_detector->nthreads << "\n";
}

ApriltagUtils::~ApriltagUtils() {
    std::cout << "Destroying AprilTag detector...\n";
    
    if (tag_family) {
        apriltag_detector_remove_family(at_detector, tag_family);
        
        // Destroy the appropriate family
        if (family_name == "tag36h11") {
            tag36h11_destroy(tag_family);
        } else if (family_name == "tag25h9") {
            tag25h9_destroy(tag_family);
        } else if (family_name == "tag16h5") {
            tag16h5_destroy(tag_family);
        } else if (family_name == "tagCircle21h7") {
            tagCircle21h7_destroy(tag_family);
        } else if (family_name == "tagCircle49h12") {
            tagCircle49h12_destroy(tag_family);
        } else if (family_name == "tagCustom48h12") {
            tagCustom48h12_destroy(tag_family);
        } else if (family_name == "tagStandard41h12") {
            tagStandard41h12_destroy(tag_family);
        } else if (family_name == "tagStandard52h13") {
            tagStandard52h13_destroy(tag_family);
        }
    }
    
    if (at_detector) {
        apriltag_detector_destroy(at_detector);
    }
    
    std::cout << "AprilTag detector destroyed\n";
}

std::vector<TagDetection> ApriltagUtils::get_tags(const cv::Mat& frame) {
    std::vector<TagDetection> detections;
    
    if (frame.empty()) {
        std::cerr << "Warning: Empty frame provided to get_tags\n";
        return detections;
    }
    
    // Convert to grayscale if needed
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame.clone();
    }
    
    // Create image_u8_t structure for AprilTag library
    image_u8_t im = {
        .width = gray.cols,
        .height = gray.rows,
        .stride = gray.cols,
        .buf = gray.data
    };
    
    // Detect tags
    zarray_t* detections_raw = apriltag_detector_detect(at_detector, &im);
    
    // Extract camera intrinsics
    double fx = cam_matrix.at<double>(0, 0);
    double fy = cam_matrix.at<double>(1, 1);
    double cx = cam_matrix.at<double>(0, 2);
    double cy = cam_matrix.at<double>(1, 2);
    
    // Process each detection
    for (int i = 0; i < zarray_size(detections_raw); i++) {
        apriltag_detection_t* det;
        zarray_get(detections_raw, i, &det);
        
        TagDetection tag_det;
        tag_det.id = det->id;
        
        // Extract corner points
        for (int j = 0; j < 4; j++) {
            tag_det.corners.push_back(cv::Point2f(det->p[j][0], det->p[j][1]));
        }
        
        // Calculate center
        tag_det.center = cv::Point2f(det->c[0], det->c[1]);
        
        // ========================================
        // Use AprilTag's native pose estimation
        // ========================================
        apriltag_detection_info_t info;
        info.det = det;
        info.tagsize = tag_size_meters;
        info.fx = fx;
        info.fy = fy;
        info.cx = cx;
        info.cy = cy;
        
        // Estimate pose using AprilTag's optimized method
        apriltag_pose_t pose;
        double err = estimate_tag_pose(&info, &pose);
        
        // Extract rotation matrix
        tag_det.pose_R = cv::Mat(3, 3, CV_64F);
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                tag_det.pose_R.at<double>(row, col) = MATD_EL(pose.R, row, col);
            }
        }
        
        // Extract translation vector
        tag_det.pose_t = cv::Mat(3, 1, CV_64F);
        tag_det.pose_t.at<double>(0, 0) = MATD_EL(pose.t, 0, 0);
        tag_det.pose_t.at<double>(1, 0) = MATD_EL(pose.t, 1, 0);
        tag_det.pose_t.at<double>(2, 0) = MATD_EL(pose.t, 2, 0);
        
        // Calculate depth (Z component = distance along camera axis)
        // This is the most accurate distance metric for AprilTags
        tag_det.depth = MATD_EL(pose.t, 2, 0);  // Z component only
        
        // Cleanup pose matrices
        matd_destroy(pose.R);
        matd_destroy(pose.t);
        
        detections.push_back(tag_det);
    }
    
    // Clean up detections
    apriltag_detections_destroy(detections_raw);
    
    return detections;
}

void ApriltagUtils::get_scaling_factor(const std::vector<TagDetection>& tags,
                                       cv::Mat& frame,
                                       const cv::Mat& relative_depth_map,
                                       double& out_scaling_factor) {
    if (tags.empty()) {
        std::cerr << "Warning: No tags provided for scaling factor calculation\n";
        out_scaling_factor = -1.0;
        return;
    }
    
    if (relative_depth_map.empty()) {
        std::cerr << "Warning: Empty depth map provided\n";
        out_scaling_factor = -1.0;
        return;
    }
    
    double total_scaling_factor = 0.0;
    int valid_tags = 0;
    
    for (const auto& tag : tags) {
        if (tag.depth <= 0) {
            continue;
        }
        
        // Get depth map value at tag center
        int cx = static_cast<int>(tag.center.x);
        int cy = static_cast<int>(tag.center.y);
        
        // Ensure coordinates are within bounds
        if (cx < 0 || cx >= relative_depth_map.cols ||
            cy < 0 || cy >= relative_depth_map.rows) {
            continue;
        }
        
        // Get relative depth value
        float relative_depth = relative_depth_map.at<float>(cy, cx);
        
        if (relative_depth > 0) {
            // Calculate scaling factor: true_depth / relative_depth
            double scale = tag.depth / relative_depth;
            total_scaling_factor += scale;
            valid_tags++;
            
            // Draw visualization on frame
            if (!frame.empty()) {
                // Draw tag corners
                for (size_t i = 0; i < tag.corners.size(); i++) {
                    cv::line(frame,
                            tag.corners[i],
                            tag.corners[(i + 1) % 4],
                            cv::Scalar(0, 255, 0), 2);
                }
                
                // Draw center point
                cv::circle(frame, tag.center, 5, cv::Scalar(255, 0, 0), -1);
                
                // Draw tag ID and depth in cm
                std::string label = "ID:" + std::to_string(tag.id) +
                                  " " + std::to_string(static_cast<int>(tag.depth * 100)) + "cm";
                cv::putText(frame, label,
                           cv::Point(tag.center.x + 10, tag.center.y - 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5,
                           cv::Scalar(0, 255, 255), 2);
                
                // Draw coordinate axes
                std::vector<cv::Point3f> axis_points;
                axis_points.push_back(cv::Point3f(0, 0, 0));
                axis_points.push_back(cv::Point3f(tag_size_meters, 0, 0));
                axis_points.push_back(cv::Point3f(0, tag_size_meters, 0));
                axis_points.push_back(cv::Point3f(0, 0, tag_size_meters));
                
                std::vector<cv::Point2f> image_points;
                cv::Mat rvec;
                cv::Rodrigues(tag.pose_R, rvec);
                cv::projectPoints(axis_points, rvec, tag.pose_t,
                                cam_matrix, dist_coeffs, image_points);
                
                // Draw axes
                cv::line(frame, image_points[0], image_points[1], cv::Scalar(0, 0, 255), 2);
                cv::line(frame, image_points[0], image_points[2], cv::Scalar(0, 255, 0), 2);
                cv::line(frame, image_points[0], image_points[3], cv::Scalar(255, 0, 0), 2);
            }
        }
    }
    
    if (valid_tags > 0) {
        out_scaling_factor = total_scaling_factor / valid_tags;
        std::cout << "Scaling factor calculated from " << valid_tags
                  << " tags: " << out_scaling_factor << "\n";
    } else {
        out_scaling_factor = -1.0;
        std::cerr << "Warning: No valid tags for scaling factor calculation\n";
    }
}
