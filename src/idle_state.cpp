#include "include/idle_state.hpp"
#include <iostream>
#include <chrono>
#include <sstream>  
IdleState::IdleState(StateCommand& state_command,
                     cv::VideoCapture& cap,
                     const std::string& calib_path,
                     const std::string& midas_model_path,
                     std::shared_ptr<torso_stabilization::TrackingState> tracking_state)
    : state_command_(state_command),
      cap_(cap),
      calib_path_(calib_path),
      midas_model_path_(midas_model_path),
      tracking_state_(tracking_state) {
    
    std::cout << "IdleState: Initializing...\n";
    initialize();
}

void IdleState::initialize() {
    // Initialize MediaPipe HandLandmarker
    pipe_utils = std::make_unique<utils::HandLandmarkerMP>();
    std::cout << "  - MediaPipe HandLandmarker initialized\n";

    // Initialize MIDAS depth estimation
    midas_utils = std::make_unique<MiDaSDepth>(midas_model_path_);
    std::cout << "  - MIDAS depth initialized\n";

    // Initialize IMU tracker and server
    imu_tracker = std::make_unique<IMUTracker>();  // NO utils:: prefix
    imu_server = std::make_unique<utils::IMUServer>(8080);  // Updated constructor
    std::cout << "  - IMU Server initialized\n";

    // Load camera calibration
    cv::FileStorage fs(calib_path_, cv::FileStorage::READ);
    if (fs.isOpened()) {
        fs["camera_matrix"] >> camera_matrix_;
        fs["distortion_coefficients"] >> dist_coeffs_;
        fs.release();
        std::cout << "  - Camera calibration loaded\n";
    } else {
        std::cerr << "Warning: Could not load calibration file\n";
    }

    std::cout << "IdleState: Initialization complete\n\n";
}

void IdleState::run() {
    std::cout << "IdleState: Starting main loop\n";
    
    int frame_count = 0;
    
    while (true) {
        // Capture frame
        if (!cap_.read(current_frame_)) {
            std::cerr << "Error: Failed to capture frame\n";
            break;
        }

        frame_count++;

        // Process frame with tracking state
        processFrame(frame_count);

        // Display frame
        cv::imshow("Torso Stabilization - Idle State", current_frame_);

        // Check for exit
        char key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {  // 'q' or ESC
            std::cout << "Quit requested\n";
            break;
        }

        // Print status every 30 frames
        if (frame_count % 30 == 0) {
            std::cout << "Frame " << frame_count 
                      << " - Tracking State: " 
                      << static_cast<int>(tracking_state_->GetCurrentState()) 
                      << "\n";
        }
    }

    cleanup();
}

void IdleState::processFrame(int frame_count) {
    // Get current timestamp
    int64_t timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // ============================================================================
    // 1. Detect hands with MediaPipe
    // ============================================================================
    utils::HandLandmarks hands = pipe_utils->Detect(current_frame_);
    
    torso_stabilization::HandFrame hand_frame;
    hand_frame.timestamp_ns = timestamp_ns;
    
    // Get smoothed index finger tip (primary tracking point)
    auto tip = pipe_utils->GetSmoothedIndexTip(current_frame_);
    
    if (tip.has_value() && !hands.empty()) {
        hand_frame.detected = true;
        
        // Convert MediaPipe landmarks to TrackingState format
        // Assuming first hand detected
        // Convert MediaPipe landmarks to TrackingState format
        // Assuming first hand detected
        const auto& first_hand = hands.begin()->second;
        hand_frame.landmarks.clear();
        hand_frame.landmarks.reserve(first_hand.size());

        for (const auto& pt : first_hand) {
            torso_stabilization::HandLandmark landmark;
            landmark.position = Eigen::Vector3f(
                pt.x * current_frame_.cols,  // Convert normalized to pixels
                pt.y * current_frame_.rows,
                0.0f  // Z-depth if available, otherwise 0
            );
            landmark.visibility = 1.0f;  // TODO: Get actual visibility from MediaPipe if available
            hand_frame.landmarks.push_back(landmark);
        }

        
        hand_frame.handedness = 0.5f;  // TODO: Get actual handedness from MediaPipe
        
        // Visualize on frame
        cv::circle(current_frame_, *tip, 8, cv::Scalar(0, 0, 255), -1);  // Red dot
        cv::putText(current_frame_, "Hand Detected", cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    } else {
        hand_frame.detected = false;
    }
    
    // Draw all hand landmarks for visualization
    for (const auto& [hand_id, landmarks] : hands) {
        for (const auto& pt : landmarks) {
            int x = static_cast<int>(pt.x * current_frame_.cols);
            int y = static_cast<int>(pt.y * current_frame_.rows);
            cv::circle(current_frame_, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);  // Green dots
        }
    }

    // ============================================================================
    // 2. Get detected objects (TODO: integrate FastSAM + depth)
    // ============================================================================
    std::vector<torso_stabilization::DetectedObject> objects;
    
    // TODO: Run FastSAM segmentation
    // TODO: Get depth map from MiDaS
    // TODO: Combine into DetectedObject list

    // ============================================================================
    // 3. Process with tracking state machine
    // ============================================================================
    auto state = tracking_state_->ProcessFrame(hand_frame, objects, timestamp_ns);

    // ============================================================================
    // 4. Handle state transitions and visualization
    // ============================================================================
    switch (state) {
        case torso_stabilization::TrackingState::State::WAITING:
            // Waiting for hand detection
            cv::putText(current_frame_, "State: WAITING", cv::Point(10, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
            break;

        case torso_stabilization::TrackingState::State::TRACKING:
            // Hand is being tracked
            {
                auto direction = tracking_state_->GetReachDirection();
                
                // Visualize reach direction
                cv::putText(current_frame_, "State: TRACKING", cv::Point(10, 60),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 165, 0), 2);
                
                if (tip.has_value()) {
                    // Draw direction vector
                    cv::Point start(tip->x, tip->y);
                    cv::Point end(tip->x + direction.x() * 100, 
                                  tip->y + direction.y() * 100);
                    cv::arrowedLine(current_frame_, start, end, 
                                    cv::Scalar(255, 0, 255), 2);
                }
            }
            break;

        case torso_stabilization::TrackingState::State::OBJECT_FINALIZING:
            // Object is being finalized
            {
                float progress = tracking_state_->GetFinalizationProgress();
                
                cv::putText(current_frame_, "State: FINALIZING", cv::Point(10, 60),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 165, 255), 2);
                
                // Draw progress bar
                int bar_width = 200;
                int bar_height = 20;
                cv::Point bar_start(10, 80);
                cv::rectangle(current_frame_, bar_start, 
                              cv::Point(bar_start.x + bar_width, bar_start.y + bar_height),
                              cv::Scalar(255, 255, 255), 2);
                cv::rectangle(current_frame_, bar_start,
                              cv::Point(bar_start.x + (int)(bar_width * progress), 
                                        bar_start.y + bar_height),
                              cv::Scalar(0, 255, 255), -1);
                
                if (frame_count % 10 == 0) {
                    std::cout << "Finalizing object: " << (progress * 100) << "%\n";
                }
            }
            break;

        case torso_stabilization::TrackingState::State::OBJECT_FINALIZED:
            // Target object confirmed!
            {
                const auto* target = tracking_state_->GetTargetObject(objects);
                if (target) {
                    std::cout << "🎯 TARGET CONFIRMED: Object " << target->object_id << "\n";
                    
                    cv::putText(current_frame_, "TARGET LOCKED!", cv::Point(10, 60),
                                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 3);
                    
                    // TODO: Send to lift detection module
                }
                tracking_state_->Reset();
            }
            break;

        default:
            break;
    }
    
    // Display frame info
    std::stringstream ss;
    ss << "Frame: " << frame_count << " | FPS: " << (int)cap_.get(cv::CAP_PROP_FPS);
    cv::putText(current_frame_, ss.str(), cv::Point(10, current_frame_.rows - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}



void IdleState::cleanup() {
    std::cout << "IdleState: Cleaning up...\n";
    
    // Stop IMU server
    if (imu_server) {
        imu_server.reset();
    }

    // Release other resources
    pipe_utils.reset();
    midas_utils.reset();
    imu_tracker.reset();

    std::cout << "IdleState: Cleanup complete\n";
}
