#ifndef IDLE_STATE_HPP_
#define IDLE_STATE_HPP_

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include "state_command.hpp"
#include "utils/mp_utils.hpp"
#include "utils/midas_utils.hpp"
#include "utils/IMU_server.hpp"
#include "utils/IMU_tracker.hpp"
#include "utils/apriltag_utils.hpp"
#include "utils/fastsam_utils.hpp"

/**
 * @brief IdleState - Initialization and component verification state
 * 
 * Responsibilities:
 * - Initialize all required components (MiDaS, FastSAM, MediaPipe, AprilTag, IMU)
 * - Verify each component works correctly
 * - Display initialization status to user
 * - Share initialized components with other states
 * - Transition to TrackingState when all components are ready
 */
class IdleState {
 public:
  IdleState(StateCommand& state_command,
            cv::VideoCapture& cap,
            const std::string& calib_path,
            const std::string& midas_model_path);

  // Run idle state - returns next state to transition to
  SystemState run();
  
  // Release ownership of components (they'll be managed by StateCommand)
  void releaseComponents();

 private:
  StateCommand& state_command_;
  cv::VideoCapture& cap_;
  std::string calib_path_;
  std::string midas_model_path_;

  // Utility objects (will be transferred to StateCommand)
  std::unique_ptr<utils::HandLandmarkerMP> pipe_utils_;
  std::unique_ptr<MiDaSDepth> midas_utils_;
  std::unique_ptr<FastSAMSegmenter> fastsam_utils_;
  std::unique_ptr<ApriltagUtils> apriltag_utils_;
  std::unique_ptr<utils::IMUServer> imu_server_;
  std::unique_ptr<IMUTracker> imu_tracker_;

  // Camera calibration data
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  
  // Helper methods
  bool initializeComponents();
  bool verifyComponents();
  void displayStatus(const cv::Mat& frame);
  void cleanup();
};

#endif  // IDLE_STATE_HPP_
