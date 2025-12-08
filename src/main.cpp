#include <iostream>
#include <opencv2/opencv.hpp>
#include "state_command.hpp"



int main() {
    // Create StateCommand object
    StateCommand SC;

    // OpenCV video capture
    cv::VideoCapture cap(2);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    int state = 0;

    if (state == 0) {
        SC.state = PickState::IDLE;
        IdleState idle_state(SC,cap);
        idle_state.run();
    } 
    
    // else if (state == 1) {
    //     SC.state = PickState::OBJECT;
    //     ObjectState obj_state(cap, SC);
    //     std::cout << "start run " << std::endl;
    //     obj_state.run();
    // }

    return 0;
}
