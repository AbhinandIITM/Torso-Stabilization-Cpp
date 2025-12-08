#include "utils/mp_utils.h"

#include <cmath>
#include <cstring>

#include <opencv2/imgproc.hpp>

#include "absl/log/log.h"

#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_result.h"

namespace mp = mediapipe;
namespace hl = mediapipe::tasks::vision::hand_landmarker;
namespace containers = mediapipe::tasks::components::containers;

namespace utils {

using containers::Landmark;
using containers::NormalizedLandmarks;

// ---------- PIMPL storage ----------

struct HandLandmarkerMP::Impl {
  std::unique_ptr<hl::HandLandmarker> landmarker;
};

// ---------- Lifecycle ----------

HandLandmarkerMP::HandLandmarkerMP(const std::string& model_path,
                                   int max_num_hands)
    : impl_(std::make_unique<Impl>()) {
  auto options = std::make_unique<hl::HandLandmarkerOptions>();
  options->base_options.model_asset_path = model_path;
  options->num_hands = max_num_hands;

  auto landmarker_or = hl::HandLandmarker::Create(std::move(options));
  if (!landmarker_or.ok()) {
    LOG(FATAL) << "Failed to create HandLandmarker: "
               << landmarker_or.status();
  }

  impl_->landmarker = std::move(landmarker_or.value());
}

HandLandmarkerMP::~HandLandmarkerMP() = default;

// ---------- Basic detection API ----------

HandLandmarks HandLandmarkerMP::Detect(const cv::Mat& frame_bgr) {
  HandLandmarks output;

  if (frame_bgr.empty()) return output;

  // BGR -> RGB
  cv::Mat frame_rgb;
  cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB);

  // Wrap into ImageFrame
  auto image_frame = std::make_shared<mp::ImageFrame>(
      mp::ImageFormat::SRGB,
      frame_rgb.cols,
      frame_rgb.rows,
      /*alignment_boundary=*/1);

  std::memcpy(
      image_frame->MutablePixelData(),
      frame_rgb.data,
      frame_rgb.total() * frame_rgb.elemSize());

  mp::Image image(image_frame);

  auto result_or = impl_->landmarker->Detect(image);
  if (!result_or.ok()) {
    LOG(WARNING) << "HandLandmarker detect failed: "
                 << result_or.status();
    return output;
  }

  const auto& hands = result_or.value().hand_landmarks;

  int hand_id = 0;
  for (const NormalizedLandmarks& hand : hands) {
    std::vector<HandPoint> points;
    points.reserve(hand.landmarks.size());

    for (const auto& lm : hand.landmarks) {
      points.push_back({lm.x, lm.y});
    }

    output.emplace(hand_id++, std::move(points));
  }

  return output;
}

// ---------- Python-style get_smoothed_tip ----------

std::optional<cv::Point>
HandLandmarkerMP::GetSmoothedIndexTip(const cv::Mat& frame_bgr) {
  HandLandmarks hands = Detect(frame_bgr);
  if (hands.empty()) {
    return std::nullopt;
  }

  // Use first detected hand
  const auto& landmarks = hands.begin()->second;
  if (landmarks.size() < 9) {
    return std::nullopt;
  }

  const int start_idx = 5;  // base of index finger
  const int end_idx   = 8;  // tip of index finger

  const auto& start = landmarks[start_idx];
  const auto& end   = landmarks[end_idx];

  // Direction vector
  float vx = end.x - start.x;
  float vy = end.y - start.y;

  float norm = std::sqrt(vx * vx + vy * vy);
  if (norm < 1e-6f) {
    return std::nullopt;
  }

  vx /= norm;
  vy /= norm;

  // Extend fingertip
  constexpr float EXTEND_FACTOR = 0.1f;
  float ex = end.x + vx * EXTEND_FACTOR;
  float ey = end.y + vy * EXTEND_FACTOR;

  int w = frame_bgr.cols;
  int h = frame_bgr.rows;

  cv::Point extended_pixel(
      static_cast<int>(ex * w),
      static_cast<int>(ey * h));

  // Initialize smoothing
  if (!last_tip_) {
    last_tip_ = extended_pixel;
  }

  // Exponential smoothing
  cv::Point smoothed(
      static_cast<int>(last_tip_->x * (1.0f - smoothing_factor_) +
                       extended_pixel.x * smoothing_factor_),
      static_cast<int>(last_tip_->y * (1.0f - smoothing_factor_) +
                       extended_pixel.y * smoothing_factor_));

  // Clamp to image bounds
  smoothed.x = std::max(0, std::min(smoothed.x, w - 1));
  smoothed.y = std::max(0, std::min(smoothed.y, h - 1));

  last_tip_ = smoothed;
  return smoothed;
}

}  // namespace utils
