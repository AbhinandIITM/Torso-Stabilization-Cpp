#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <mutex>
#include <chrono>
#include <opencv2/core.hpp>

namespace utils {

struct HandPoint {
  float x;
  float y;
};

using HandLandmarks = std::map<int, std::vector<HandPoint>>;

class HandLandmarkerMP {
 public:
  explicit HandLandmarkerMP(const std::string& model_path = "mediapipe/models/hand_landmarker.task",
                            int max_num_hands = 1);
  ~HandLandmarkerMP();

  // Non-blocking update: sends frame to background AI thread
  void Update(const cv::Mat& frame_bgr);

  // Thread-safe getters
  HandLandmarks GetLatestLandmarks();
  float GetLatestInferenceMs(); // New: returns actual AI latency
  std::optional<cv::Point> GetSmoothedIndexTip(int width, int height);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  // Result storage and sync
  std::mutex result_mutex_;
  HandLandmarks latest_hands_;
  
  // Latency tracking
  std::mutex latency_mutex_;
  float latest_inference_ms_ = 0.0f;
  std::map<uint64_t, std::chrono::steady_clock::time_point> timestamp_map_;

  // Smoothing state
  float smoothing_factor_ = 0.3f;
  std::optional<cv::Point> last_tip_;
};

}  // namespace utils