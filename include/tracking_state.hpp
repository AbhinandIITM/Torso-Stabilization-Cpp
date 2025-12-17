#ifndef TRACKING_STATE_HPP_
#define TRACKING_STATE_HPP_

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <optional>
#include "state_command.hpp"
#include "utils/mp_utils.hpp"
#include "utils/midas_utils.hpp"
#include "utils/apriltag_utils.hpp"
#include "utils/fastsam_utils.hpp"

/**
 * @brief TrackingState - Active tracking using AprilTag, MiDaS, FastSAM and MediaPipe
 * 
 * Responsibilities:
 * - Continuously track AprilTag for pose estimation
 * - Use MiDaS for depth estimation
 * - Segment objects with FastSAM
 * - Detect and visualize hand skeletons with MediaPipe
 * - Update 3D position of tracked objects
 * - Provide real-time tracking visualization
 */
class TrackingState {
 public:
  TrackingState(StateCommand& state_command,
                cv::VideoCapture& cap);

  // Run tracking state - returns next state to transition to
  SystemState run();

 private:
  StateCommand& state_command_;
  cv::VideoCapture& cap_;
  
  // References to shared components (from StateCommand)
  MiDaSDepth* midas_utils_;
  ApriltagUtils* apriltag_utils_;
  utils::HandLandmarkerMP* mediapipe_utils_;
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  
  // Frame data
  cv::Mat current_frame_;
  
  // FastSAM result
  FastSAMResult fastsam_result_;
  
  // Hand landmarks from MediaPipe
  utils::HandLandmarks hand_landmarks_;
  
  // Smoothed index fingertip
  std::optional<cv::Point> smoothed_index_tip_;
  
  // Tracking state
  bool tracking_active_;
  int frames_without_detection_;
  static constexpr int MAX_FRAMES_WITHOUT_DETECTION = 30;
  
  // Helper methods
  void processFrame();
  bool detectAprilTag();
  void estimateDepth();
  void segmentObjects();
  void detectHands();
  void updateTracking();
  void visualizeTracking();
  void drawHandSkeleton(const std::vector<utils::HandPoint>& landmarks);
  bool checkExitCondition();
};

#endif  // TRACKING_STATE_HPP_
