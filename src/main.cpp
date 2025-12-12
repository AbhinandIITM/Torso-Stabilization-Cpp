#include <opencv2/opencv.hpp>
#include <iostream>
#include "idle_state.hpp"
#include "state_command.hpp"

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "  Torso Stabilization System\n";
    std::cout << "========================================\n\n";

    // Parse command line arguments
    std::string calib_path = "config/calibration.yaml";
    std::string midas_model_path = "models/midas_model.pt";
    int camera_id = 0;

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

        // Create and run IdleState
        std::cout << "Starting IdleState...\n";
        IdleState idle_state(state_command, cap, calib_path, midas_model_path);
        
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
