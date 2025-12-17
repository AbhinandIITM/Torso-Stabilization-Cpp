#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include "include/idle_state.hpp"
#include "include/state_command.hpp"
#include "include/tracking_state.hpp"

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "  Torso Stabilization System v2.0\n";
    std::cout << "========================================\n\n";

    // Configuration paths
    std::string midas_model_path = "src/models/dpt_swin2_tiny_256_torchscript.pt";
    std::string calib_path = "src/calib_2.yaml";
    int camera_id = 2;  // Default to camera 2

    // Parse command line arguments
    if (argc > 1) camera_id = std::atoi(argv[1]);
    if (argc > 2) calib_path = argv[2];
    if (argc > 3) midas_model_path = argv[3];

    std::cout << "Configuration:\n";
    std::cout << "  Camera ID: " << camera_id << "\n";
    std::cout << "  Calibration: " << calib_path << "\n";
    std::cout << "  MiDaS Model: " << midas_model_path << "\n\n";

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
        // Create shared state command
        StateCommand state_command;
        state_command.current_state = SystemState::IDLE;

        // State machine loop
        while (state_command.current_state != SystemState::SHUTDOWN) {
            
            switch (state_command.current_state) {
                case SystemState::IDLE: {
                    std::cout << "\n========================================\n";
                    std::cout << "  ENTERING IDLE STATE\n";
                    std::cout << "========================================\n\n";
                    
                    IdleState idle_state(state_command, cap, calib_path, midas_model_path);
                    SystemState next_state = idle_state.run();
                    state_command.current_state = next_state;
                    break;
                }
                
                case SystemState::TRACKING: {
                    std::cout << "\n========================================\n";
                    std::cout << "  ENTERING TRACKING STATE\n";
                    std::cout << "========================================\n\n";
                    
                    TrackingState tracking_state(state_command, cap);
                    SystemState next_state = tracking_state.run();
                    state_command.current_state = next_state;
                    break;
                }
                
                case SystemState::ERROR: {
                    std::cerr << "\n========================================\n";
                    std::cerr << "  ERROR STATE\n";
                    std::cerr << "========================================\n";
                    std::cerr << "Error: " << state_command.error_message << "\n\n";
                    std::cerr << "Press any key to retry or ESC to quit\n";
                    
                    char key = cv::waitKey(0);
                    if (key == 27) { // ESC
                        state_command.current_state = SystemState::SHUTDOWN;
                    } else {
                        state_command.current_state = SystemState::IDLE;
                        state_command.error_message.clear();
                    }
                    break;
                }
                
                case SystemState::SHUTDOWN:
                    // Will exit loop
                    break;
            }
        }
        
        std::cout << "\nShutting down...\n";
        
        // Cleanup shared components
        if (state_command.midas_utils) delete state_command.midas_utils;
        if (state_command.fastsam_utils) delete state_command.fastsam_utils;
        if (state_command.apriltag_utils) delete state_command.apriltag_utils;
        if (state_command.mediapipe_utils) delete state_command.mediapipe_utils;
        if (state_command.imu_server) delete state_command.imu_server;
        if (state_command.imu_tracker) delete state_command.imu_tracker;

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
        cap.release();
        return -1;
    }

    cap.release();
    cv::destroyAllWindows();
    
    std::cout << "Shutdown complete.\n";
    return 0;
}
