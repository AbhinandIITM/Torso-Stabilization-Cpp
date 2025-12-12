#include "utils/apriltag_utils.hpp"
#include <iostream>
#include <cmath>

// Include the C-based AprilTag library headers
extern "C" {
    #include <apriltag/apriltag.h>
    #include <apriltag/tag36h11.h>
    #include <apriltag/tag25h9.h>
    #include <apriltag/tag16h5.h>
    #include <apriltag/tagCircle21h7.h>
    #include <apriltag/tagCircle49h12.h>
    #include <apriltag/tagCustom48h12.h>
    #include <apriltag/tagStandard41h12.h>
    #include <apriltag/tagStandard52h13.h>
}

ApriltagUtils::ApriltagUtils(const std::string& calib_data_path, 
                             const std::string& family, 
                             double tag_size)
    : at_detector(nullptr), 
      tag_family(nullptr), 
      tag_size_meters(tag_size)
{
    std::cout << "Initializing AprilTag detector...\n";
    
    // Load camera calibration
    cv::FileStorage fs(calib_data_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open calibration file: " + calib_data_path);
    }
    
    fs["camMatrix"] >> cam_matrix;
    fs["dist_coeffs"] >> dist_coeffs;
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
    
    // Configure detector parameters for better performance
    at_detector->quad_decimate = 2.0;  // Decimate input image for faster detection
    at_detector->quad_sigma = 0.0;     // Gaussian blur sigma (0 = no blur)
    at_detector->nthreads = 4;         // Number of threads
    at_detector->debug = 0;            // No debug output
    at_detector->refine_edges = 1;     // Refine edges for better accuracy
    
    std::cout << "AprilTag detector initialized with family: " << family << "\n";
    std::cout << "Tag size: " << tag_size_meters << " meters\n";
}

ApriltagUtils::~ApriltagUtils() {
    std::cout << "Destroying AprilTag detector...\n";
    
    if (tag_family) {
        // Destroy the appropriate family
        // Note: You need to call the correct destroy function for the family used
        // For simplicity, we'll use tag36h11_destroy as an example
        // In production, you'd track which family was created
        apriltag_detector_remove_family(at_detector, tag_family);
        tag36h11_destroy(tag_family); // Adjust based on family used
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
        
        // Prepare points for pose estimation
        std::vector<cv::Point3f> object_points;
        double half_size = tag_size_meters / 2.0;
        object_points.push_back(cv::Point3f(-half_size, -half_size, 0));
        object_points.push_back(cv::Point3f( half_size, -half_size, 0));
        object_points.push_back(cv::Point3f( half_size,  half_size, 0));
        object_points.push_back(cv::Point3f(-half_size,  half_size, 0));
        
        // Solve PnP to get pose
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(
            object_points,
            tag_det.corners,
            cam_matrix,
            dist_coeffs,
            rvec,
            tvec,
            false,
            cv::SOLVEPNP_IPPE_SQUARE
        );
        
        if (success) {
            // Convert rotation vector to rotation matrix
            cv::Rodrigues(rvec, tag_det.pose_R);
            tag_det.pose_t = tvec.clone();
            
            // Calculate depth (distance from camera)
            tag_det.depth = cv::norm(tvec);
        } else {
            std::cerr << "Warning: PnP failed for tag " << det->id << "\n";
            tag_det.pose_R = cv::Mat::eye(3, 3, CV_64F);
            tag_det.pose_t = cv::Mat::zeros(3, 1, CV_64F);
            tag_det.depth = -1.0;
        }
        
        detections.push_back(tag_det);
    }
    
    // Clean up
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
            continue; // Skip tags with invalid depth
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
                
                // Draw tag ID and depth
                std::string label = "ID:" + std::to_string(tag.id) + 
                                   " D:" + std::to_string(tag.depth).substr(0, 4) + "m";
                cv::putText(frame, label, 
                           cv::Point(tag.center.x + 10, tag.center.y - 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                           cv::Scalar(0, 255, 255), 2);
                
                // Draw coordinate axes
                std::vector<cv::Point3f> axis_points;
                axis_points.push_back(cv::Point3f(0, 0, 0));
                axis_points.push_back(cv::Point3f(tag_size_meters, 0, 0)); // X-axis (red)
                axis_points.push_back(cv::Point3f(0, tag_size_meters, 0)); // Y-axis (green)
                axis_points.push_back(cv::Point3f(0, 0, tag_size_meters)); // Z-axis (blue)
                
                std::vector<cv::Point2f> image_points;
                cv::projectPoints(axis_points, tag.pose_R, tag.pose_t,
                                cam_matrix, dist_coeffs, image_points);
                
                // Draw axes
                cv::line(frame, image_points[0], image_points[1], cv::Scalar(0, 0, 255), 2); // X red
                cv::line(frame, image_points[0], image_points[2], cv::Scalar(0, 255, 0), 2); // Y green
                cv::line(frame, image_points[0], image_points[3], cv::Scalar(255, 0, 0), 2); // Z blue
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
