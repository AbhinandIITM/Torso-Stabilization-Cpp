#ifndef IDLE_STATE_HPP_
#define IDLE_STATE_HPP_

#include <opencv2/opencv.hpp>
#include <open3d/Open3D.h>
#include <memory>
#include <string>
#include "state_command.hpp"
#include "tracking_state.hpp"
#include "utils/mp_utils.hpp"
#include "utils/midas_utils.hpp"
#include "utils/IMU_server.hpp"
#include "utils/IMU_tracker.hpp"
#include "utils/apriltag_utils.hpp"
#include "utils/fastsam_utils.hpp"


class IdleState {
 public:
  IdleState(StateCommand& state_command,
            cv::VideoCapture& cap,
            const std::string& calib_path,
            const std::string& midas_model_path,
            std::shared_ptr<torso_stabilization::TrackingState> tracking_state);

  void run();

 private:
  StateCommand& state_command_;
  cv::VideoCapture& cap_;
  std::string calib_path_;
  std::string midas_model_path_;
  std::shared_ptr<torso_stabilization::TrackingState> tracking_state_;

  // Utility objects with correct namespaces
  std::unique_ptr<utils::HandLandmarkerMP> pipe_utils;
  std::unique_ptr<MiDaSDepth> midas_utils;
  std::unique_ptr<FastSAMSegmenter> fastsam_utils_;
  std::unique_ptr<utils::IMUServer> imu_server;  // IN utils namespace
  std::unique_ptr<IMUTracker> imu_tracker;       // NOT in namespace!

  // Camera calibration data
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  
  // Frame data
  cv::Mat current_frame_;
  
  // Helper methods
  void initialize();
  void processFrame(int frame_count);
  void cleanup();
};

#endif  // IDLE_STATE_HPP_
