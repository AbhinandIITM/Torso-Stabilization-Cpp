#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <optional>

#include <opencv2/core.hpp>

namespace utils {

struct HandPoint {
  float x;
  float y;
};

using HandLandmarks = std::map<int, std::vector<HandPoint>>;

class HandLandmarkerMP {
 public:
  explicit HandLandmarkerMP(const std::string& model_path,
                            int max_num_hands = 2);
  ~HandLandmarkerMP();

  // Existing API
  HandLandmarks Detect(const cv::Mat& frame_bgr);

  // ✅ Python-equivalent API
  // Returns smoothed pixel coordinates of extended index fingertip
  // std::nullopt if no hand detected
  std::optional<cv::Point> GetSmoothedIndexTip(const cv::Mat& frame_bgr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  // Smoothing state (equivalent to Python version)
  float smoothing_factor_ = 0.3f;
  std::optional<cv::Point> last_tip_;
};

}  // namespace utils
