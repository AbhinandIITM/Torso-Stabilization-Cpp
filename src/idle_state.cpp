#include "include/idle_state.hpp"
#include <iostream>
#include <chrono>
#include <thread>

IdleState::IdleState(StateCommand& state_command,
                     cv::VideoCapture& cap,
                     const std::string& calib_path,
                     const std::string& midas_model_path)
    : state_command_(state_command),
      cap_(cap),
      calib_path_(calib_path),
      midas_model_path_(midas_model_path) {
    
    std::cout << "IdleState: Constructor called\n";
}

SystemState IdleState::run() {
    std::cout << "IdleState: Starting initialization sequence...\n\n";
    
    // Check if components are already initialized
    bool need_init = !state_command_.midas_initialized || 
                     !state_command_.apriltag_initialized;
    
    if (need_init) {
        // Step 1: Initialize all components
        if (!initializeComponents()) {
            state_command_.error_message = "Component initialization failed";
            return SystemState::ERROR;
        }
        
        std::cout << "\n";
        
        // Step 2: Verify components work correctly (optimized to 2 frames)
        if (!verifyComponents()) {
            state_command_.error_message = "Component verification failed";
            return SystemState::ERROR;
        }
    } else {
        std::cout << "Components already initialized, skipping initialization\n\n";
    }
    
    std::cout << "\n========================================\n";
    std::cout << "  ALL COMPONENTS READY\n";
    std::cout << "========================================\n";
    std::cout << "Press 't' to enter TRACKING state\n";
    std::cout << "Press 'q' or ESC to quit\n\n";
    
    // Wait for user input
    cv::Mat display_frame;
    while (true) {
        if (!cap_.read(display_frame)) {
            state_command_.error_message = "Failed to read camera frame";
            return SystemState::ERROR;
        }
        
        displayStatus(display_frame);
        cv::imshow("Torso Stabilization - Idle State", display_frame);
        
        char key = cv::waitKey(30);
        if (key == 't' || key == 'T') {
            std::cout << "Transitioning to TRACKING state...\n";
            releaseComponents();  // Transfer ownership to StateCommand
            return SystemState::TRACKING;
        } else if (key == 'q' || key == 'Q' || key == 27) {
            std::cout << "Shutdown requested\n";
            cleanup();
            return SystemState::SHUTDOWN;
        }
    }
}

bool IdleState::initializeComponents() {
    std::cout << "Initializing components...\n";
    std::cout << "----------------------------------------\n";
    
    // Load camera calibration
    std::cout << "[1/6] Loading camera calibration... ";
    cv::FileStorage fs(calib_path_, cv::FileStorage::READ);
    if (fs.isOpened()) {
        fs["camera_matrix"] >> camera_matrix_;
        fs["distortion_coefficients"] >> dist_coeffs_;
        fs.release();
        
        // Store in StateCommand
        state_command_.camera_matrix = camera_matrix_.clone();
        state_command_.dist_coeffs = dist_coeffs_.clone();
        
        std::cout << "✓ OK\n";
    } else {
        std::cout << "✗ FAILED\n";
        std::cerr << "      Could not load calibration file: " << calib_path_ << "\n";
        return false;
    }
    
    // Initialize MiDaS
    std::cout << "[2/6] Initializing MiDaS depth estimation... ";
    try {
        midas_utils_ = std::make_unique<MiDaSDepth>(midas_model_path_);
        state_command_.midas_initialized = true;
        std::cout << "✓ OK\n";
    } catch (const std::exception& e) {
        std::cout << "✗ FAILED\n";
        std::cerr << "      " << e.what() << "\n";
        return false;
    }
    
    // Initialize FastSAM
    std::cout << "[3/6] Initializing FastSAM segmentation... ";
    try {
        fastsam_utils_ = std::make_unique<FastSAMSegmenter>(
            "src/models/FastSAM-x.torchscript",
            true,    // Use CUDA if available
            640,     // Input size
            0.25f,   // Confidence threshold
            0.45f    // IOU threshold
        );
        state_command_.fastsam_initialized = true;
        std::cout << "✓ OK\n";
    } catch (const std::exception& e) {
        std::cout << "✗ FAILED\n";
        std::cerr << "      " << e.what() << "\n";
        return false;
    }
    
    // Initialize MediaPipe
    std::cout << "[4/6] Initializing MediaPipe HandLandmarker... ";
    try {
        pipe_utils_ = std::make_unique<utils::HandLandmarkerMP>();
        state_command_.mediapipe_initialized = true;
        std::cout << "✓ OK\n";
    } catch (const std::exception& e) {
        std::cout << "✗ FAILED\n";
        std::cerr << "      " << e.what() << "\n";
        return false;
    }
    
    // Initialize AprilTag
    std::cout << "[5/6] Initializing AprilTag detector... ";
    try {
        apriltag_utils_ = std::make_unique<ApriltagUtils>(
            calib_path_,
            "tag36h11",  // AprilTag family
            0.15         // Tag size in meters (15cm)
        );
        state_command_.apriltag_initialized = true;
        std::cout << "✓ OK\n";
    } catch (const std::exception& e) {
        std::cout << "✗ FAILED\n";
        std::cerr << "      " << e.what() << "\n";
        return false;
    }
    
    // Initialize IMU
    std::cout << "[6/6] Initializing IMU server... ";
    try {
        imu_server_ = std::make_unique<utils::IMUServer>(8080);
        imu_tracker_ = std::make_unique<IMUTracker>();
        state_command_.imu_initialized = true;
        std::cout << "✓ OK\n";
    } catch (const std::exception& e) {
        std::cout << "✗ FAILED\n";
        std::cerr << "      " << e.what() << "\n";
        return false;
    }
    
    std::cout << "----------------------------------------\n";
    std::cout << "All components initialized successfully!\n";
    return true;
}

bool IdleState::verifyComponents() {
    std::cout << "Verifying components...\n";
    std::cout << "----------------------------------------\n";
    
    cv::Mat test_frame;
    int verification_frames = 2;  // Optimized to 2 frames
    
    for (int i = 0; i < verification_frames; ++i) {
        if (!cap_.read(test_frame)) {
            std::cerr << "Failed to capture verification frame\n";
            return false;
        }
        
        std::cout << "Verification frame " << (i + 1) << "/" << verification_frames << ":\n";
        
        // Test MiDaS
        std::cout << "  - MiDaS depth... ";
        try {
            cv::Mat depth = midas_utils_->getDepthMap(test_frame);
            if (depth.empty()) {
                std::cout << "✗ Empty depth map\n";
                return false;
            }
            std::cout << "✓ " << depth.rows << "x" << depth.cols << "\n";
        } catch (const std::exception& e) {
            std::cout << "✗ " << e.what() << "\n";
            return false;
        }
        
        // Test AprilTag
        std::cout << "  - AprilTag detection... ";
        try {
            auto detections = apriltag_utils_->get_tags(test_frame);
            std::cout << "✓ " << detections.size() << " tags detected\n";
        } catch (const std::exception& e) {
            std::cout << "✗ " << e.what() << "\n";
            return false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "----------------------------------------\n";
    std::cout << "All components verified successfully!\n";
    return true;
}

void IdleState::releaseComponents() {
    // Transfer ownership to StateCommand
    state_command_.midas_utils = midas_utils_.release();
    state_command_.fastsam_utils = fastsam_utils_.release();
    state_command_.apriltag_utils = apriltag_utils_.release();
    state_command_.mediapipe_utils = pipe_utils_.release();
    state_command_.imu_server = imu_server_.release();
    state_command_.imu_tracker = imu_tracker_.release();
}

void IdleState::displayStatus(const cv::Mat& frame) {
    cv::Mat display = frame.clone();
    
    int y_offset = 30;
    int line_height = 35;
    
    cv::putText(display, "IDLE STATE - System Ready", 
                cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX, 
                1.0, cv::Scalar(0, 255, 0), 2);
    y_offset += line_height + 10;
    
    // Component status
    auto drawStatus = [&](const std::string& name, bool status) {
        cv::Scalar color = status ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        std::string symbol = status ? "[OK]" : "[FAIL]";
        cv::putText(display, symbol + " " + name, 
                    cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX, 
                    0.6, color, 2);
        y_offset += line_height;
    };
    
    drawStatus("MiDaS", state_command_.midas_initialized);
    drawStatus("FastSAM", state_command_.fastsam_initialized);
    drawStatus("MediaPipe", state_command_.mediapipe_initialized);
    drawStatus("AprilTag", state_command_.apriltag_initialized);
    drawStatus("IMU", state_command_.imu_initialized);
    
    // Instructions
    y_offset += 20;
    cv::putText(display, "Press 't' for TRACKING", 
                cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(255, 255, 0), 2);
    y_offset += line_height;
    cv::putText(display, "Press 'q' to QUIT", 
                cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(255, 255, 0), 2);
    
    display.copyTo(frame);
}

void IdleState::cleanup() {
    std::cout << "IdleState: Cleaning up...\n";
    
    if (imu_server_) imu_server_.reset();
    if (imu_tracker_) imu_tracker_.reset();
    if (apriltag_utils_) apriltag_utils_.reset();
    if (pipe_utils_) pipe_utils_.reset();
    if (fastsam_utils_) fastsam_utils_.reset();
    if (midas_utils_) midas_utils_.reset();
    
    std::cout << "IdleState: Cleanup complete\n";
}
