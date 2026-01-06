#include "utils/mp_utils.hpp"
#include <cmath>
#include <cstring>
#include <opencv2/imgproc.hpp>
#include "absl/log/log.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.h"

namespace mp = mediapipe;
namespace hl = mediapipe::tasks::vision::hand_landmarker;

namespace utils {

struct HandLandmarkerMP::Impl {
  std::unique_ptr<hl::HandLandmarker> landmarker;
};

HandLandmarkerMP::HandLandmarkerMP(const std::string& model_path, int max_num_hands)
    : impl_(std::make_unique<Impl>()) {
    
    auto options = std::make_unique<hl::HandLandmarkerOptions>();
    options->base_options.model_asset_path = model_path;
    options->num_hands = max_num_hands;
    options->running_mode = mediapipe::tasks::vision::core::RunningMode::LIVE_STREAM;
    options->base_options.delegate = mediapipe::tasks::core::BaseOptions::Delegate::CPU;

    options->result_callback = [this](absl::StatusOr<hl::HandLandmarkerResult> result, 
                                      const mp::Image& image, uint64_t timestamp_ms) {
        if (!result.ok()) return;

        // 1. Calculate Actual Inference Latency
        {
            std::lock_guard<std::mutex> lock(this->latency_mutex_);
            if (this->timestamp_map_.count(timestamp_ms)) {
                auto start_time = this->timestamp_map_[timestamp_ms];
                auto end_time = std::chrono::steady_clock::now();
                this->latest_inference_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                this->timestamp_map_.erase(timestamp_ms);
            }
        }

        // 2. Store Landmarks
        std::lock_guard<std::mutex> lock(this->result_mutex_);
        this->latest_hands_.clear();
        int hand_id = 0;
        for (const auto& hand : result.value().hand_landmarks) {
            std::vector<HandPoint> points;
            for (const auto& lm : hand.landmarks) {
                points.push_back({lm.x, lm.y});
            }
            this->latest_hands_[hand_id++] = std::move(points);
        }
    };

    auto landmarker_or = hl::HandLandmarker::Create(std::move(options));
    if (!landmarker_or.ok()) LOG(FATAL) << "Failed to create HandLandmarker: " << landmarker_or.status();
    impl_->landmarker = std::move(landmarker_or.value());
}

HandLandmarkerMP::~HandLandmarkerMP() = default;

void HandLandmarkerMP::Update(const cv::Mat& frame_bgr) {
  if (frame_bgr.empty()) return;

  cv::Mat resized_rgb;
  cv::resize(frame_bgr, resized_rgb, cv::Size(256, 256));
  cv::cvtColor(resized_rgb, resized_rgb, cv::COLOR_BGR2RGB);

  auto image_frame = std::make_shared<mp::ImageFrame>(mp::ImageFormat::SRGB, resized_rgb.cols, resized_rgb.rows, 1);
  std::memcpy(image_frame->MutablePixelData(), resized_rgb.data, resized_rgb.total() * resized_rgb.elemSize());

  auto now = std::chrono::steady_clock::now();
  uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  
  {
      std::lock_guard<std::mutex> lock(latency_mutex_);
      timestamp_map_[ms] = now;
  }

  impl_->landmarker->DetectAsync(mp::Image(image_frame), ms);
}

HandLandmarks HandLandmarkerMP::GetLatestLandmarks() {
    std::lock_guard<std::mutex> lock(result_mutex_);
    return latest_hands_;
}

float HandLandmarkerMP::GetLatestInferenceMs() {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    return latest_inference_ms_;
}

std::optional<cv::Point> HandLandmarkerMP::GetSmoothedIndexTip(int width, int height) {
    auto hands = GetLatestLandmarks();
    if (hands.empty()) return std::nullopt;
    const auto& landmarks = hands.begin()->second;
    if (landmarks.size() < 9) return std::nullopt;

    const auto& start = landmarks[5];
    const auto& end = landmarks[8];
    float vx = end.x - start.x;
    float vy = end.y - start.y;
    float norm = std::sqrt(vx*vx + vy*vy);
    if (norm < 1e-6f) return std::nullopt;

    float ex = end.x + (vx/norm) * 0.1f;
    float ey = end.y + (vy/norm) * 0.1f;
    cv::Point current_pt(static_cast<int>(ex * width), static_cast<int>(ey * height));

    if (!last_tip_) last_tip_ = current_pt;
    cv::Point smoothed(
        static_cast<int>(last_tip_->x * (1.0f - smoothing_factor_) + current_pt.x * smoothing_factor_),
        static_cast<int>(last_tip_->y * (1.0f - smoothing_factor_) + current_pt.y * smoothing_factor_)
    );
    last_tip_ = smoothed;
    return smoothed;
}

} // namespace utils