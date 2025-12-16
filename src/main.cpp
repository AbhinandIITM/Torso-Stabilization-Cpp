#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include "include/idle_state.hpp"
#include "include/state_command.hpp"
#include "include/tracking_state.hpp"

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "  Torso Stabilization System\n";
    std::cout << "========================================\n\n";

    std::string midas_model_path = "src/models/dpt_swin2_tiny_256_torchscript.pt";

    // Parse command line arguments
    std::string calib_path = "src/calib_2.yaml";
    
    int camera_id = 2;

    if (argc > 1) camera_id = std::atoi(argv[1]);
    if (argc > 2) calib_path = argv[2];
    if (argc > 3) midas_model_path = argv[3];

    std::cout << "Configuration:\n";
    std::cout << "  Camera ID: " << camera_id << "\n";
    std::cout << "  Calibration: " << calib_path << "\n";
    std::cout << "  Midas Model: " << midas_model_path << "\n\n";

    // Open camera
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera " << camera_id << "\n";
        return -1;
    }

    std::cout << "Camera opened successfully\n";
    std::cout << "Resolution: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) 
              << "x" << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << "\n\n";

    try {
        // Create StateCommand
        StateCommand state_command;

        // Initialize tracking state with configuration
        std::cout << "Initializing TrackingState module...\n";
        torso_stabilization::TrackingState::TrackingConfig tracking_config;
        
        // Configure trajectory analyzer
        tracking_config.trajectory_config.trajectory_window_ms = 500.0f;
        tracking_config.trajectory_config.min_movement_threshold_cm = 2.0f;
        tracking_config.trajectory_config.velocity_magnitude_threshold_cm_per_s = 5.0f;
        
        // Configure object scoring
        tracking_config.scoring_config.angle_weight = 0.6f;
        tracking_config.scoring_config.distance_weight = 0.4f;
        tracking_config.scoring_config.max_reach_distance_m = 0.8f;
        tracking_config.scoring_config.reach_direction_angle_deg = 45.0f;
        
        // Configure finalization
        tracking_config.object_finalization_duration_ms = 350.0f;
        tracking_config.min_focus_frames = 8;
        tracking_config.focus_angle_threshold_deg = 20.0f;
        tracking_config.min_score_to_start_finalization = 0.5f;
        
        auto tracking_state = std::make_shared<torso_stabilization::TrackingState>(tracking_config);
        std::cout << "TrackingState initialized successfully\n\n";

        // Create and run IdleState
        std::cout << "Starting IdleState...\n";
        IdleState idle_state(state_command, cap, calib_path, midas_model_path, tracking_state);
        
        std::cout << "Running main loop. Press 'q' in any window to quit.\n\n";
        idle_state.run();
        
        std::cout << "\nShutting down...\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        cap.release();
        return -1;
    }

    cap.release();
    cv::destroyAllWindows();
    
    std::cout << "Shutdown complete.\n";
    return 0;
}
