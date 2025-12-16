#include "utils/apriltag_utils.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>


int main(int argc, char** argv) {
    std::cout << "\n========================================\n";
    std::cout << "  AprilTag Detection Test\n";
    std::cout << "========================================\n\n";
    
    // Configuration parameters
    const std::string CALIB_DATA_PATH = "src/calib_2.yaml";
    const std::string TAG_FAMILY = "tag36h11";
    const double TAG_SIZE = 5;  
    const int CAMERA_ID = 2;  // Fixed camera ID (change to 1 if needed)
    
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
        std::cout << "  - Tag size: " << TAG_SIZE << " meters\n";
        std::cout << "  - Camera matrix: fx=" << std::fixed << std::setprecision(2) 
                  << fx << ", fy=" << fy << "\n";
        std::cout << "  - Principal point: cx=" << cx << ", cy=" << cy << "\n";
        std::cout << "  - Distortion coefficients: [";
        for (int i = 0; i < dist_coeffs.cols; i++) {
            std::cout << dist_coeffs.at<double>(0, i);
            if (i < dist_coeffs.cols - 1) std::cout << ", ";
        }
        std::cout << "]\n\n";
        
        ApriltagUtils apriltag_detector(CALIB_DATA_PATH, TAG_FAMILY, TAG_SIZE);
        
        // Open camera with V4L2 backend
        std::cout << "Opening camera " << CAMERA_ID << " with V4L2...\n";
        cv::VideoCapture cap(CAMERA_ID, cv::CAP_V4L2);
        
        if (!cap.isOpened()) {
            std::cerr << "\nERROR: Could not open camera " << CAMERA_ID << "!\n";
            std::cerr << "Troubleshooting:\n";
            std::cerr << "  1. Check permissions: groups | grep video\n";
            std::cerr << "  2. If not in video group: sudo usermod -a -G video $USER\n";
            std::cerr << "  3. Check cameras: ls -l /dev/video*\n";
            std::cerr << "  4. Try changing CAMERA_ID to 0 or 1 in the code\n";
            return 1;
        }
        
        // Try to read a test frame
        cv::Mat test_frame;
        if (!cap.read(test_frame) || test_frame.empty()) {
            std::cerr << "\nERROR: Camera opened but can't read frames!\n";
            return 1;
        }
        
        std::cout << "Camera " << CAMERA_ID << " opened successfully!\n";
        
        // Set camera properties (optional)
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(cv::CAP_PROP_FPS, 30);
        
        std::cout << "\nCamera properties:\n";
        std::cout << "  - Resolution: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) 
                  << "x" << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << "\n";
        std::cout << "  - FPS: " << cap.get(cv::CAP_PROP_FPS) << "\n\n";
        
        std::cout << "Starting detection loop...\n";
        std::cout << "Controls:\n";
        std::cout << "  - Press 'q' to quit\n";
        std::cout << "  - Press 's' to save current frame\n\n";
        
        cv::Mat frame;
        int frame_count = 0;  // Keep for internal use only
        
        // For FPS calculation
        auto last_fps_time = std::chrono::high_resolution_clock::now();
        int fps_frame_count = 0;
        double current_fps = 0.0;
        
        while (true) {
            auto frame_start = std::chrono::high_resolution_clock::now();
            
            // Capture frame
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "ERROR: Empty frame captured!\n";
                break;
            }
            
            frame_count++;
            fps_frame_count++;
            
            // Detect AprilTags
            std::vector<TagDetection> detections = apriltag_detector.get_tags(frame);
            
            // Calculate FPS every second
            auto current_time = std::chrono::high_resolution_clock::now();
            auto fps_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - last_fps_time).count();
            
            if (fps_duration >= 1000) {
                current_fps = fps_frame_count * 1000.0 / fps_duration;
                fps_frame_count = 0;
                last_fps_time = current_time;
            }
            
            // Draw detections on frame
            cv::Mat display_frame = frame.clone();
            
            for (const auto& tag : detections) {
                // Draw corners with different colors
                std::vector<cv::Scalar> corner_colors = {
                    cv::Scalar(255, 0, 0),    // Blue
                    cv::Scalar(0, 255, 0),    // Green
                    cv::Scalar(0, 0, 255),    // Red
                    cv::Scalar(255, 255, 0)   // Cyan
                };
                
                for (size_t i = 0; i < tag.corners.size(); i++) {
                    cv::circle(display_frame, tag.corners[i], 6, 
                             corner_colors[i], -1);
                    
                    // Draw lines between corners
                    size_t next = (i + 1) % tag.corners.size();
                    cv::line(display_frame, tag.corners[i], tag.corners[next], 
                            cv::Scalar(0, 255, 0), 2);
                }
                
                // Draw center
                cv::circle(display_frame, tag.center, 8, 
                         cv::Scalar(0, 0, 255), -1);
                
                // Draw tag ID with background
                std::string id_text = "ID: " + std::to_string(tag.id);
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(id_text, cv::FONT_HERSHEY_SIMPLEX, 
                                                     0.8, 2, &baseline);
                cv::Point text_origin(tag.center.x + 15, tag.center.y - 15);
                cv::rectangle(display_frame, 
                            text_origin + cv::Point(0, baseline),
                            text_origin + cv::Point(text_size.width, -text_size.height),
                            cv::Scalar(0, 0, 0), cv::FILLED);
                cv::putText(display_frame, id_text, text_origin,
                          cv::FONT_HERSHEY_SIMPLEX, 0.8, 
                          cv::Scalar(0, 255, 255), 2);
                
                // Draw depth if available
                // Draw depth if available
                if (tag.depth > 0) {
                    std::ostringstream depth_stream;
                    depth_stream << std::fixed << std::setprecision(2) << (tag.depth * 1000);  // Convert to mm
                    std::string depth_text = depth_stream.str() + "mm";
                    cv::putText(display_frame, depth_text,
                                cv::Point(tag.center.x + 15, tag.center.y + 15),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                                cv::Scalar(255, 255, 0), 2);
                }

                
                // Print detection info every 30 frames
                if (frame_count % 30 == 0) {
                    std::cout << "Tag ID: " << tag.id
                             << " at (" << static_cast<int>(tag.center.x) 
                             << ", " << static_cast<int>(tag.center.y) << ")";
                    if (tag.depth > 0) {
                        std::cout << ", depth: " << tag.depth << "m";
                    }
                    std::cout << "\n";
                }
            }
            
            // Draw minimal info panel - only FPS
            cv::rectangle(display_frame, cv::Point(5, 5), 
                        cv::Point(150, 45), cv::Scalar(0, 0, 0), cv::FILLED);
            
            std::string fps_text = "FPS: " + std::to_string(static_cast<int>(current_fps));
            
            cv::putText(display_frame, fps_text, cv::Point(15, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
            
            // Show frame
            cv::imshow("AprilTag Detection Test", display_frame);
            
            // Handle keyboard input
            char key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) {  // 27 = ESC
                std::cout << "\nQuitting...\n";
                break;
            } else if (key == 's' || key == 'S') {
                std::string filename = "apriltag_frame_" + 
                    std::to_string(frame_count) + ".jpg";
                cv::imwrite(filename, display_frame);
                std::cout << "Saved frame to " << filename << "\n";
            }
        }
        
        // Cleanup
        cap.release();
        cv::destroyAllWindows();
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV ERROR: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    
    std::cout << "Test completed successfully!\n";
    return 0;
}
