#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include "include/idle_state.hpp"
#include "include/state_command.hpp"
#include "include/tracking_state.hpp"

int main(int argc, char** argv) {
    // Config and concise greeting
    std::string midas_model_path = "src/models/dpt_swin2_tiny_256_torchscript.pt";
    std::string calib_path = "src/calib_2.yaml";
    int camera_id = 2;

    if (argc > 1) camera_id = std::atoi(argv[1]);
    if (argc > 2) calib_path = argv[2];
    if (argc > 3) midas_model_path = argv[3];

    std::cout << ">> Torso System v2.0 | Cam: " << camera_id << " | Calib: " << calib_path << std::endl;

    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        std::cerr << "[Error] Camera " << camera_id << " unavailable." << std::endl;
        return -1;
    }

    try {
        StateCommand state_command;
        state_command.current_state = SystemState::IDLE;

        while (state_command.current_state != SystemState::SHUTDOWN) {
            switch (state_command.current_state) {
                case SystemState::IDLE: {
                    IdleState idle_state(state_command, cap, calib_path, midas_model_path);
                    state_command.current_state = idle_state.run();
                    break;
                }
                case SystemState::TRACKING: {
                    TrackingState tracking_state(state_command, cap);
                    state_command.current_state = tracking_state.run();
                    break;
                }
                case SystemState::ERROR: {
                    std::cerr << "[Status] Error: " << state_command.error_message << " (ESC to quit)" << std::endl;
                    if (cv::waitKey(0) == 27) state_command.current_state = SystemState::SHUTDOWN;
                    else { state_command.current_state = SystemState::IDLE; state_command.error_message.clear(); }
                    break;
                }
                case SystemState::SHUTDOWN: break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[Fatal] " << e.what() << std::endl;
        return -1;
    }

    cap.release();
    cv::destroyAllWindows();
    std::cout << ">> Shutdown complete." << std::endl;
    return 0;
}
