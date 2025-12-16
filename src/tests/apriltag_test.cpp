#include "utils/apriltag_utils.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>

int main(int argc, char** argv) {
    std::cout << "\n========================================\n";
    std::cout << "  AprilTag Detection Test\n";
    std::cout << "  Optimized for Multiple Tags\n";
    std::cout << "========================================\n\n";
    
    // Configuration parameters
    const std::string CALIB_DATA_PATH = "src/calib_2.yaml";
    const std::string TAG_FAMILY = "tag36h11";
    const double TAG_SIZE = 0.05;  // 5 cm in meters
    const int CAMERA_ID = 2;
    
    try {
        // Load camera calibration to display info
        cv::FileStorage fs(CALIB_DATA_PATH, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "ERROR: Failed to open calibration file: " << CALIB_DATA_PATH << "\n";
            return 1;
        }
        
        cv::Mat cam_matrix, dist_coeffs;
        fs["camera_matrix"] >> cam_matrix;
        fs["distortion_coefficients"] >> dist_coeffs;
        fs.release();
        
        if (cam_matrix.empty() || dist_coeffs.empty()) {
            std::cerr << "ERROR: Camera matrix or distortion coefficients not found in calibration file\n";
            return 1;
        }
        
        // Extract camera parameters for display
        double fx = cam_matrix.at<double>(0, 0);
        double fy = cam_matrix.at<double>(1, 1);
        double cx = cam_matrix.at<double>(0, 2);
        double cy = cam_matrix.at<double>(1, 2);
        
        // Initialize AprilTag detector
        std::cout << "Initializing AprilTag detector...\n";
        std::cout << "  - Calibration file: " << CALIB_DATA_PATH << "\n";
        std::cout << "  - Tag family: " << TAG_FAMILY << "\n";
        std::cout << "  - Tag size: " << TAG_SIZE << " meters (" << (TAG_SIZE * 100) << " cm)\n";
        std::cout << "  - Camera matrix: fx=" << std::fixed << std::setprecision(2) 
                  << fx << ", fy=" << fy << "\n";
        std::cout << "  - Principal point: cx=" << cx << ", cy=" << cy << "\n";
        std::cout << "  - Using AprilTag native pose estimation (most accurate)\n\n";
        
        ApriltagUtils apriltag_detector(CALIB_DATA_PATH, TAG_FAMILY, TAG_SIZE);
        
        // Open camera with V4L2 backend
        std::cout << "Opening camera " << CAMERA_ID << " with V4L2...\n";
        cv::VideoCapture cap(CAMERA_ID, cv::CAP_V4L2);
        
        if (!cap.isOpened()) {
            std::cerr << "\nERROR: Could not open camera " << CAMERA_ID << "!\n";
            return 1;
        }
        
        // Try to read a test frame
        cv::Mat test_frame;
        if (!cap.read(test_frame) || test_frame.empty()) {
            std::cerr << "\nERROR: Camera opened but can't read frames!\n";
            return 1;
        }
        
        std::cout << "Camera " << CAMERA_ID << " opened successfully!\n";
        
        // Set camera properties
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(cv::CAP_PROP_FPS, 30);
        
        std::cout << "\nStarting detection loop...\n";
        std::cout << "Controls: 'q' to quit, 's' to save frame\n\n";
        
        cv::Mat frame;
        int frame_count = 0;
        
        // For FPS calculation
        auto last_fps_time = std::chrono::high_resolution_clock::now();
        int fps_frame_count = 0;
        double current_fps = 0.0;
        
        while (true) {
            // Capture frame
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "ERROR: Empty frame captured!\n";
                break;
            }
            
            frame_count++;
            fps_frame_count++;
            
            auto detect_start = std::chrono::high_resolution_clock::now();
            
            // Detect AprilTags
            std::vector<TagDetection> detections = apriltag_detector.get_tags(frame);
            
            auto detect_end = std::chrono::high_resolution_clock::now();
            double detect_ms = std::chrono::duration<double, std::milli>(detect_end - detect_start).count();
            
            // Calculate FPS every second
            auto current_time = std::chrono::high_resolution_clock::now();
            auto fps_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - last_fps_time).count();
            
            if (fps_duration >= 1000) {
                current_fps = fps_frame_count * 1000.0 / fps_duration;
                fps_frame_count = 0;
                last_fps_time = current_time;
            }
            
            // Draw detections on frame (minimal for performance)
            cv::Mat display_frame = frame;  // No clone for speed
            
            for (const auto& tag : detections) {
                // Draw corners (simple lines)
                for (size_t i = 0; i < 4; i++) {
                    size_t next = (i + 1) % 4;
                    cv::line(display_frame, tag.corners[i], tag.corners[next], 
                            cv::Scalar(0, 255, 0), 2);
                }
                
                // Draw center
                cv::circle(display_frame, tag.center, 5, cv::Scalar(0, 0, 255), -1);
                
                // Draw tag ID and distance in cm
                std::ostringstream label_stream;
                label_stream << "ID:" << tag.id << " " 
                            << std::fixed << std::setprecision(1) 
                            << (tag.depth * 100.0) << "cm";
                std::string label = label_stream.str();
                
                cv::putText(display_frame, label,
                          cv::Point(tag.center.x + 10, tag.center.y - 10),
                          cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                          cv::Scalar(0, 255, 255), 2);
                
                // Print detection info every 60 frames
                if (frame_count % 60 == 0) {
                    std::cout << "Tag ID " << tag.id
                             << ": " << std::fixed << std::setprecision(1) 
                             << (tag.depth * 100.0) << " cm"
                             << " at (" << static_cast<int>(tag.center.x) 
                             << ", " << static_cast<int>(tag.center.y) << ")\n";
                }
            }
            
            // Draw compact info panel
            cv::rectangle(display_frame, cv::Point(5, 5), 
                        cv::Point(220, 85), cv::Scalar(0, 0, 0), cv::FILLED);
            
            std::ostringstream fps_stream, tags_stream, time_stream;
            fps_stream << "FPS: " << std::fixed << std::setprecision(1) << current_fps;
            tags_stream << "Tags: " << detections.size();
            time_stream << "Detect: " << std::fixed << std::setprecision(1) << detect_ms << "ms";
            
            cv::putText(display_frame, fps_stream.str(), cv::Point(15, 25),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            cv::putText(display_frame, tags_stream.str(), cv::Point(15, 45),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            cv::putText(display_frame, time_stream.str(), cv::Point(15, 65),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            
            // Show frame
            cv::imshow("AprilTag Detection", display_frame);
            
            // Handle keyboard input
            char key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) {
                std::cout << "\nQuitting...\n";
                break;
            } else if (key == 's' || key == 'S') {
                std::string filename = "apriltag_" + std::to_string(frame_count) + ".jpg";
                cv::imwrite(filename, frame);  // Save original, not display_frame
                std::cout << "Saved to " << filename << "\n";
            }
        }
        
        // Cleanup
        cap.release();
        cv::destroyAllWindows();
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    
    std::cout << "Test completed!\n";
    return 0;
}
