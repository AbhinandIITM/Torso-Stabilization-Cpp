#include "include/tracking_state.hpp"
#include <iostream>
#include <chrono>

TrackingState::TrackingState(StateCommand& state_command,
                             cv::VideoCapture& cap)
    : state_command_(state_command),
      cap_(cap),
      tracking_active_(false),
      frames_without_detection_(0) {
    
    std::cout << "TrackingState: Constructor called\n";
    
    // Get references to shared components from StateCommand
    midas_utils_ = state_command_.midas_utils;
    apriltag_utils_ = state_command_.apriltag_utils;
    mediapipe_utils_ = state_command_.mediapipe_utils;
    camera_matrix_ = state_command_.camera_matrix;
    dist_coeffs_ = state_command_.dist_coeffs;
    
    // Verify components are initialized
    if (!midas_utils_ || !apriltag_utils_) {
        throw std::runtime_error("TrackingState: Required components not initialized!");
    }
    
    std::cout << "TrackingState: All components verified\n";
}

SystemState TrackingState::run() {
    std::cout << "TrackingState: Starting tracking loop...\n";
    std::cout << "Place AprilTag in view to begin tracking\n";
    std::cout << "Press 'i' to return to IDLE, 'q' or ESC to quit\n\n";
    
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    while (true) {
        // Capture frame
        if (!cap_.read(current_frame_)) {
            std::cerr << "Error: Failed to capture frame\n";
            state_command_.error_message = "Camera capture failed";
            return SystemState::ERROR;
        }
        
        frame_count++;
        
        // Process frame
        processFrame();
        
        // Visualize
        visualizeTracking();
        
        // Display
        cv::imshow("Torso Stabilization - Tracking State", current_frame_);
        
        // Calculate and display FPS every 30 frames
        if (frame_count % 30 == 0) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - start_time).count();
            float fps = (30.0f * 1000.0f) / elapsed;
            
            std::cout << "Frame " << frame_count 
                      << " - FPS: " << fps
                      << " - Tracking: " << (tracking_active_ ? "ACTIVE" : "SEARCHING")
                      << " - AprilTag: " << (state_command_.apriltag_detected ? "DETECTED" : "NOT FOUND");
            
            if (state_command_.apriltag_detected) {
                std::cout << " [ID:" << state_command_.apriltag_id 
                          << " | Depth:" << state_command_.apriltag_position.z() << "m]";
            }
            std::cout << " - Hands: " << hand_landmarks_.size();
            std::cout << "\n";
            
            start_time = current_time;
        }
        
        // Check for state transitions
        char key = cv::waitKey(1);
        if (key == 'i' || key == 'I') {
            std::cout << "Returning to IDLE state...\n";
            return SystemState::IDLE;
        } else if (key == 'q' || key == 'Q' || key == 27) {
            std::cout << "Shutdown requested\n";
            return SystemState::SHUTDOWN;
        }
    }
    
    return SystemState::IDLE;
}

void TrackingState::processFrame() {
    // Step 1: Detect AprilTag
    bool tag_detected = detectAprilTag();
    
    // Step 2: Estimate depth with MiDaS
    estimateDepth();
    
    // Step 3: Segment objects with FastSAM
    segmentObjects();
    
    // Step 4: Detect hand landmarks
    detectHands();
    
    // Step 5: Update tracking state
    updateTracking();
}

bool TrackingState::detectAprilTag() {
    try {
        std::vector<TagDetection> detections = apriltag_utils_->get_tags(current_frame_);
        
        if (!detections.empty()) {
            // Use first detected tag
            const TagDetection& detection = detections[0];
            
            state_command_.apriltag_detected = true;
            state_command_.apriltag_id = detection.id;
            
            // Extract position from pose_t (3x1 translation vector)
            state_command_.apriltag_position = Eigen::Vector3f(
                detection.pose_t.at<double>(0, 0),
                detection.pose_t.at<double>(1, 0),
                detection.pose_t.at<double>(2, 0)
            );
            
            // Extract rotation from pose_R (3x3 rotation matrix)
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    state_command_.apriltag_rotation(i, j) = 
                        detection.pose_R.at<double>(i, j);
                }
            }
            
            // Use depth from TagDetection
            state_command_.scaling_factor = 1.0 / detection.depth;
            
            frames_without_detection_ = 0;
            return true;
        } else {
            state_command_.apriltag_detected = false;
            frames_without_detection_++;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "AprilTag detection error: " << e.what() << "\n";
        state_command_.apriltag_detected = false;
        return false;
    }
}

void TrackingState::estimateDepth() {
    try {
        state_command_.depth_map = midas_utils_->getDepthMap(current_frame_);
    } catch (const std::exception& e) {
        std::cerr << "Depth estimation error: " << e.what() << "\n";
        state_command_.depth_map = cv::Mat();
    }
}

void TrackingState::segmentObjects() {
    try {
        if (state_command_.fastsam_utils) {
            fastsam_result_ = state_command_.fastsam_utils->segment(current_frame_);
        }
    } catch (const std::exception& e) {
        std::cerr << "FastSAM segmentation error: " << e.what() << "\n";
    }
}

void TrackingState::detectHands() {
    try {
        if (mediapipe_utils_) {
            hand_landmarks_ = mediapipe_utils_->Detect(current_frame_);
            
            // Get smoothed index fingertip
            smoothed_index_tip_ = mediapipe_utils_->GetSmoothedIndexTip(current_frame_);
        }
    } catch (const std::exception& e) {
        std::cerr << "MediaPipe hand detection error: " << e.what() << "\n";
        hand_landmarks_.clear();
        smoothed_index_tip_ = std::nullopt;
    }
}


void TrackingState::updateTracking() {
    if (state_command_.apriltag_detected) {
        if (!tracking_active_) {
            std::cout << "🎯 Tracking activated! Tag ID: " 
                      << state_command_.apriltag_id << "\n";
        }
        tracking_active_ = true;
    } else if (frames_without_detection_ > MAX_FRAMES_WITHOUT_DETECTION) {
        if (tracking_active_) {
            std::cout << "⚠️  Tracking lost after " 
                      << frames_without_detection_ << " frames\n";
        }
        tracking_active_ = false;
    }
}

void TrackingState::drawHandSkeleton(const std::vector<utils::HandPoint>& landmarks) {
    int frame_width = current_frame_.cols;
    int frame_height = current_frame_.rows;
    
    // MediaPipe hand connections (21 landmarks, 0-20)
    const std::vector<std::pair<int, int>> connections = {
        // Thumb
        {0, 1}, {1, 2}, {2, 3}, {3, 4},
        // Index finger
        {0, 5}, {5, 6}, {6, 7}, {7, 8},
        // Middle finger
        {0, 9}, {9, 10}, {10, 11}, {11, 12},
        // Ring finger
        {0, 13}, {13, 14}, {14, 15}, {15, 16},
        // Pinky
        {0, 17}, {17, 18}, {18, 19}, {19, 20},
        // Palm
        {5, 9}, {9, 13}, {13, 17}
    };
    
    // Draw connections (skeleton)
    for (const auto& connection : connections) {
        int idx1 = connection.first;
        int idx2 = connection.second;
        
        if (idx1 < landmarks.size() && idx2 < landmarks.size()) {
            cv::Point pt1(landmarks[idx1].x * frame_width, 
                         landmarks[idx1].y * frame_height);
            cv::Point pt2(landmarks[idx2].x * frame_width, 
                         landmarks[idx2].y * frame_height);
            
            cv::line(current_frame_, pt1, pt2, cv::Scalar(0, 255, 0), 2);
        }
    }
    
    // Draw landmarks (joints)
    for (size_t i = 0; i < landmarks.size(); ++i) {
        cv::Point pt(landmarks[i].x * frame_width, 
                    landmarks[i].y * frame_height);
        
        // Different colors for different fingers
        cv::Scalar color;
        if (i == 0) {
            color = cv::Scalar(255, 0, 0);  // Wrist - Blue
        } else if (i >= 1 && i <= 4) {
            color = cv::Scalar(255, 255, 0);  // Thumb - Cyan
        } else if (i >= 5 && i <= 8) {
            color = cv::Scalar(0, 255, 255);  // Index - Yellow
        } else if (i >= 9 && i <= 12) {
            color = cv::Scalar(255, 0, 255);  // Middle - Magenta
        } else if (i >= 13 && i <= 16) {
            color = cv::Scalar(128, 0, 255);  // Ring - Purple
        } else {
            color = cv::Scalar(0, 128, 255);  // Pinky - Orange
        }
        
        cv::circle(current_frame_, pt, 5, color, -1);
        cv::circle(current_frame_, pt, 6, cv::Scalar(255, 255, 255), 1);
    }
}


void TrackingState::visualizeTracking() {
    // Create overlay for segmentation masks
    cv::Mat overlay = current_frame_.clone();
    
    // Draw FastSAM segmentation masks
    if (!fastsam_result_.masks.empty()) {
        // Generate random colors for each object (consistent across frames)
        static std::vector<cv::Scalar> colors;
        if (colors.size() < fastsam_result_.masks.size()) {
            colors.clear();
            for (size_t i = 0; i < fastsam_result_.masks.size(); ++i) {
                cv::Scalar color(
                    rand() % 200 + 55,  // 55-255
                    rand() % 200 + 55,
                    rand() % 200 + 55
                );
                colors.push_back(color);
            }
        }
        
        // Draw each segmentation mask
        for (size_t i = 0; i < fastsam_result_.masks.size(); ++i) {
            const auto& mask = fastsam_result_.masks[i];
            const auto& box = fastsam_result_.boxes[i];
            float score = fastsam_result_.scores[i];
            
            // Apply colored mask overlay
            cv::Mat colored_mask = cv::Mat::zeros(mask.size(), CV_8UC3);
            colored_mask.setTo(colors[i % colors.size()], mask);
            
            // Blend mask with original frame
            cv::addWeighted(overlay, 1.0, colored_mask, 0.3, 0, overlay);
            
            // Draw bounding box
            cv::rectangle(overlay, box, colors[i % colors.size()], 2);
            
            // Draw label with confidence
            std::string label = cv::format("Obj %zu: %.2f", i, score);
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                                  0.5, 1, &baseline);
            
            // Background for text
            cv::rectangle(overlay, 
                         cv::Point(box.x, box.y - text_size.height - 5),
                         cv::Point(box.x + text_size.width, box.y),
                         colors[i % colors.size()], -1);
            
            // Text
            cv::putText(overlay, label, cv::Point(box.x, box.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
        
        // Blend overlay with original frame
        cv::addWeighted(current_frame_, 0.6, overlay, 0.4, 0, current_frame_);
    }
    
    // Draw hand skeletons
    for (const auto& [hand_id, landmarks] : hand_landmarks_) {
        drawHandSkeleton(landmarks);
    }
    
    // Draw smoothed index fingertip with larger highlight
    if (smoothed_index_tip_.has_value()) {
        cv::Point tip = smoothed_index_tip_.value();
        
        // Draw larger circle for smoothed tip
        cv::circle(current_frame_, tip, 12, cv::Scalar(0, 0, 255), -1);  // Red filled circle
        cv::circle(current_frame_, tip, 14, cv::Scalar(255, 255, 255), 2);  // White border
        
        // Draw crosshair
        int cross_size = 20;
        cv::line(current_frame_, 
                 cv::Point(tip.x - cross_size, tip.y), 
                 cv::Point(tip.x + cross_size, tip.y),
                 cv::Scalar(0, 255, 0), 2);
        cv::line(current_frame_, 
                 cv::Point(tip.x, tip.y - cross_size), 
                 cv::Point(tip.x, tip.y + cross_size),
                 cv::Scalar(0, 255, 0), 2);
        
        // Label
        cv::putText(current_frame_, "Index Tip", 
                    cv::Point(tip.x + 20, tip.y - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }
    
    // Draw tracking status header
    std::string status = tracking_active_ ? "TRACKING ACTIVE" : "SEARCHING FOR TAG";
    cv::Scalar status_color = tracking_active_ ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
    
    cv::putText(current_frame_, status, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2);
    
    if (state_command_.apriltag_detected) {
        // Draw AprilTag ID
        std::string tag_info = "Tag ID: " + std::to_string(state_command_.apriltag_id);
        cv::putText(current_frame_, tag_info, cv::Point(10, 70),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        // Draw 3D position
        std::string pos_info = cv::format("Position: [%.2f, %.2f, %.2f]m",
                                          state_command_.apriltag_position.x(),
                                          state_command_.apriltag_position.y(),
                                          state_command_.apriltag_position.z());
        cv::putText(current_frame_, pos_info, cv::Point(10, 100),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        
        // Draw rotation angles (convert rotation matrix to Euler angles)
        Eigen::Vector3f euler = state_command_.apriltag_rotation.eulerAngles(0, 1, 2);
        euler = euler * 180.0f / M_PI;  // Convert to degrees
        
        std::string rot_info = cv::format("Rotation: [%.1f°, %.1f°, %.1f°]",
                                          euler.x(), euler.y(), euler.z());
        cv::putText(current_frame_, rot_info, cv::Point(10, 130),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
    
    // Draw object and hand count
    int y_offset = 160;
    if (!fastsam_result_.masks.empty()) {
        std::string obj_count = cv::format("Objects: %zu", fastsam_result_.masks.size());
        cv::putText(current_frame_, obj_count, cv::Point(10, y_offset),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 200, 0), 2);
        y_offset += 30;
    }
    
    if (!hand_landmarks_.empty()) {
        std::string hand_count = cv::format("Hands: %zu", hand_landmarks_.size());
        cv::putText(current_frame_, hand_count, cv::Point(10, y_offset),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    }
    
    // Draw depth map overlay (small corner visualization)
    if (!state_command_.depth_map.empty()) {
        cv::Mat depth_vis;
        cv::normalize(state_command_.depth_map, depth_vis, 0, 255, cv::NORM_MINMAX);
        depth_vis.convertTo(depth_vis, CV_8U);
        cv::applyColorMap(depth_vis, depth_vis, cv::COLORMAP_JET);
        
        int preview_size = 200;
        int margin = 10;
        
        // Check if preview fits in the frame
        if (current_frame_.cols > preview_size + margin && 
            current_frame_.rows > preview_size + margin) {
            
            cv::Mat roi = current_frame_(cv::Rect(
                current_frame_.cols - preview_size - margin,
                current_frame_.rows - preview_size - margin,
                preview_size,
                preview_size
            ));
            
            cv::Mat resized_depth;
            cv::resize(depth_vis, resized_depth, roi.size());
            resized_depth.copyTo(roi);
            
            // Label
            cv::putText(current_frame_, "Depth Map", 
                        cv::Point(current_frame_.cols - preview_size - margin, 
                                 current_frame_.rows - preview_size - margin - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }
    
    // Draw tracking status indicator
    if (tracking_active_) {
        cv::circle(current_frame_, cv::Point(current_frame_.cols - 30, 30), 
                   15, cv::Scalar(0, 255, 0), -1);
        cv::putText(current_frame_, "LOCK", 
                    cv::Point(current_frame_.cols - 100, 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    } else {
        cv::circle(current_frame_, cv::Point(current_frame_.cols - 30, 30), 
                   15, cv::Scalar(0, 165, 255), -1);
        cv::putText(current_frame_, "SEARCH", 
                    cv::Point(current_frame_.cols - 120, 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 2);
    }
    
    // Instructions footer
    cv::putText(current_frame_, "Press 'i' for IDLE | 'q' to QUIT",
                cv::Point(10, current_frame_.rows - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}
